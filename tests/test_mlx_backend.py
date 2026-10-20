"""MLX backend parity with the torch reference on a tiny FP32 model; skipped without MLX."""
import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from yue2 import mlx_backend, nar  # noqa: E402
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM  # noqa: E402
from yue2.protocol import ABC_END, CODEC_OFFSET, EOD, MUSIC_END, VOCAB_SIZE, Sampling  # noqa: E402
from yue2.sampling import generate_tokens  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def mlx_cpu():
    # MLX GPU FP32 matmuls use reduced (TF32-like) precision; the CPU device checks the math exactly.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(scope="module")
def models(mlx_cpu):
    # The full vocabulary keeps the protocol's phase token ranges valid.
    with torch.random.fork_rng():
        torch.manual_seed(907)
        config = YuE2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            vocab_size=VOCAB_SIZE, max_position_embeddings=256, max_latent_frames=128)
        reference = YuE2ForCausalLM(config).eval()
        # Wide logit gaps keep top-k/top-p sets stable under FP32 rounding differences.
        torch.nn.init.normal_(reference.lm_head.weight, std=1.0)
    return reference, mlx_backend.from_torch(reference)


def as_torch(array):
    return torch.from_numpy(np.array(array.astype(mx.float32)))


@torch.inference_mode()
def test_full_and_cached_hidden_states_match_torch(models):
    reference, model = models
    ids = [EOD, 17, 4093, 151000, ABC_END, CODEC_OFFSET + 5, CODEC_OFFSET + 99, 33]
    expected = reference.model(input_ids=torch.tensor([ids]), position_ids=torch.arange(len(ids))[None],
                               use_cache=False)[0]
    torch.testing.assert_close(as_torch(model.hidden(mx.array([ids]))), expected, atol=2e-5, rtol=2e-5)
    cache = mlx_backend.KVCache(model, len(ids))
    rows = [model.hidden(mx.array([ids[:3]]), cache)] + [model.hidden(mx.array([[t]]), cache) for t in ids[3:]]
    torch.testing.assert_close(as_torch(mx.concatenate(rows, axis=1)), expected, atol=2e-5, rtol=2e-5)
    with pytest.raises(ValueError, match="capacity"):
        model.hidden(mx.array([[1]]), cache)
    with pytest.raises(ValueError, match="Chunked prefill"):
        model.hidden(mx.array([[1, 2]]), cache)


@pytest.mark.parametrize("phase,cfg,legacy", [("abc", 1.0, False), ("semantic", 1.0, False),
                                               ("semantic", 1.5, False), ("semantic", 1.01, True)])
def test_generated_tokens_match_torch_eager(models, phase, cfg, legacy):
    reference, model = models
    prefix = [EOD, 11, 12, 13, 14]
    negative = [EOD, 21, 22] if cfg != 1.0 else None
    sampling = Sampling(temperature=1.0, top_p=0.9, top_k=30, repetition_penalty=1.2,
                        penalty_window=8, min_tokens=6, max_tokens=24)
    expected, _, expected_truncated = generate_tokens(reference, prefix, sampling, 5, phase, negative=negative,
                                                      cfg_scale=cfg, legacy_off=legacy, use_cuda_graph=False)
    seen = []
    actual, timing, truncated = mlx_backend.generate_tokens(model, prefix, sampling, 5, phase, negative=negative,
                                                            cfg_scale=cfg, legacy_off=legacy,
                                                            on_token=lambda p, t: seen.append(t))
    assert actual == expected and truncated == expected_truncated
    low, high = (0, EOD) if phase == "abc" else (CODEC_OFFSET, CODEC_OFFSET + 32768)
    assert all(low <= token < high for token in actual) and len(actual) >= sampling.min_tokens
    assert seen[:len(actual)] == actual and timing["cfg_branches"] == (1 if cfg == 1.0 else 2)
    assert timing["backend_actual"] == "mlx" and timing["output_tokens"] == len(seen)


def test_generation_checks_protocol_limits(models):
    _, model = models
    with pytest.raises(ValueError, match="exceeds 24576"):
        mlx_backend.generate_tokens(model, [EOD] * 20000, Sampling(max_tokens=9000), 1, "semantic")
    short = Sampling(min_tokens=0, max_tokens=4)
    with pytest.raises(ValueError, match="negative prefix"):
        mlx_backend.generate_tokens(model, [EOD], short, 1, "semantic", cfg_scale=1.5)
    with pytest.raises(InterruptedError):
        mlx_backend.generate_tokens(model, [EOD], short, 1, "abc", cancelled=lambda: True)


@pytest.mark.parametrize("visible", [0, 3])
@torch.inference_mode()
def test_velocity_and_midpoint_solve_match_torch(models, visible):
    reference, model = models
    noise = torch.randn((7, 64), generator=torch.Generator().manual_seed(42))
    chunk = nar.Chunk([EOD, 5, 6, CODEC_OFFSET + 3, CODEC_OFFSET + 8, MUSIC_END], noise, nar_cond_end=visible)
    expected, actual = nar.CachedNAR(reference, chunk), mlx_backend.CachedNAR(model, chunk)
    for raw_t in [20.0, 0.0, -2.3]:
        torch.testing.assert_close(as_torch(actual.velocity(mx.array(noise.numpy()), raw_t)),
                                   expected.velocity(noise, raw_t), atol=2e-5, rtol=2e-5)
    progress = []
    solved = actual.solve(steps=4, on_progress=lambda done, total: progress.append((done, total)))
    torch.testing.assert_close(solved, expected.solve(steps=4), atol=5e-5, rtol=5e-5)
    assert solved.dtype == torch.float32 and progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_synthesis_uses_original_chunks_and_noise(models):
    reference, model = models
    prefix, codec = [EOD, 7, 8], list(range(40))
    expected = nar.synthesize(reference, prefix, codec, 3, steps=2, context=50)
    actual = mlx_backend.synthesize(model, prefix, codec, 3, steps=2, context=50)
    assert len(nar.song_chunks(prefix, codec, 3, 50)) == 2 and actual.shape == (40, 64)
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)
