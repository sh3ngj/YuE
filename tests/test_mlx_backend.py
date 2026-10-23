"""MLX backend parity with the torch reference on a tiny FP32 model; skipped without MLX."""
import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from yue2 import mlx_backend, nar  # noqa: E402
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM  # noqa: E402
from yue2.protocol import ABC_END, CODEC_OFFSET, EOD, MUSIC_END, VOCAB_SIZE, Sampling  # noqa: E402
from yue2.sampling import distribution, generate_tokens  # noqa: E402


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
                                                            cfg_scale=cfg, legacy_off=legacy, sampler="torch",
                                                            on_token=lambda p, t: seen.append(t))
    assert actual == expected and truncated == expected_truncated
    low, high = (0, EOD) if phase == "abc" else (CODEC_OFFSET, CODEC_OFFSET + 32768)
    assert all(low <= token < high for token in actual) and len(actual) >= sampling.min_tokens
    assert seen[:len(actual)] == actual and timing["cfg_branches"] == (1 if cfg == 1.0 else 2)
    assert timing["backend_actual"] == "mlx" and timing["output_tokens"] == len(seen)
    assert timing["sampler"] == "torch_cpu"


def phase_local(phase, native):
    return native if phase == "abc" else native - CODEC_OFFSET


@pytest.mark.parametrize("phase", ["abc", "semantic"])
@pytest.mark.parametrize("sampling", [
    Sampling(temperature=1.0, top_p=0.9, top_k=30, repetition_penalty=1.2, penalty_window=8, min_tokens=6),
    Sampling(temperature=0.7, top_p=0.95, top_k=100, repetition_penalty=1.005, penalty_window=100, min_tokens=0),
    Sampling(temperature=1.3, top_p=1.0, top_k=5, repetition_penalty=1.0, penalty_window=1, min_tokens=0),
    Sampling(temperature=0.0, top_p=0.9, top_k=30, repetition_penalty=1.2, penalty_window=50, min_tokens=0),
])
def test_device_filters_match_torch_distribution(phase, sampling):
    allowed = torch.from_numpy(mlx_backend.PHASE_TOKENS[phase])
    generator = torch.Generator().manual_seed(11)
    compact = torch.randn(len(allowed), generator=generator) * 4  # FP32: no ties
    # Repeats exercise the frequency-powered penalty, the planner's guard against loops.
    history = [int(allowed[i]) for i in torch.randint(0, 200, (12,), generator=generator)] + [int(allowed[3])] * 3
    full = torch.full((1, VOCAB_SIZE), float("-inf"))
    full[0, allowed] = compact
    device = mlx_backend.DeviceSampler(sampling, phase, seed=1)
    for step in (0, 7):
        expected = distribution(full, sampling, history, step, phase)[0, allowed]
        recent = mx.array([phase_local(phase, t) for t in history[-sampling.penalty_window:]], dtype=mx.int32)
        actual = as_torch(device.scores(mx.array(compact.numpy()), recent, step))
        assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
        finite = torch.isfinite(expected)
        torch.testing.assert_close(actual[finite], expected[finite], atol=1e-5, rtol=1e-5)


def test_device_draws_follow_the_filtered_distribution():
    device = mlx_backend.DeviceSampler(Sampling(min_tokens=0), "semantic", seed=3)
    values = np.full(device.size, -np.inf, dtype=np.float32)
    values[[4, 9, 20]] = [0.0, 1.0, 2.0]
    scores = mx.array(values)
    draws = np.array([device.sample(scores).item() for _ in range(3000)])
    expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0]), -1).numpy()
    np.testing.assert_allclose([np.mean(draws == i) for i in (4, 9, 20)], expected, atol=0.03)


@pytest.mark.parametrize("cfg", [1.0, 1.5])
def test_device_generation_is_seeded_and_follows_the_protocol(models, cfg):
    _, model = models
    sampling = Sampling(temperature=1.0, top_p=0.9, top_k=30, repetition_penalty=1.2,
                        penalty_window=8, min_tokens=6, max_tokens=24)
    negative = [EOD, 21, 22] if cfg != 1.0 else None
    runs = []
    for seed in (5, 5, 6):
        seen = []
        tokens, timing, truncated = mlx_backend.generate_tokens(model, [EOD, 11, 12], sampling, seed, "semantic",
                                                                negative=negative, cfg_scale=cfg,
                                                                on_token=lambda p, t: seen.append(t))
        assert all(CODEC_OFFSET <= t < CODEC_OFFSET + 32768 for t in tokens) and len(tokens) >= 6
        assert seen[:len(tokens)] == tokens and timing["output_tokens"] == len(seen)
        assert timing["sampler"] == "mlx_device" and timing["cfg_branches"] == (1 if cfg == 1.0 else 2)
        assert truncated == (len(tokens) == sampling.max_tokens and seen[-1] != MUSIC_END)
        runs.append(tokens)
    assert runs[0] == runs[1] and runs[0] != runs[2]


def test_device_request_for_historical_off_uses_torch_sampling(models):
    reference, model = models
    sampling = Sampling(min_tokens=3, max_tokens=12)
    expected, _, _ = generate_tokens(reference, [EOD, 9], sampling, 2, "semantic", negative=[EOD],
                                     cfg_scale=1.01, legacy_off=True, use_cuda_graph=False)
    actual, timing, _ = mlx_backend.generate_tokens(model, [EOD, 9], sampling, 2, "semantic", negative=[EOD],
                                                    cfg_scale=1.01, legacy_off=True)
    assert actual == expected and timing["sampler"] == "torch_cpu"
    assert timing["sampler_fallback_reason"] == "historical_off_sampling_uses_torch"
    with pytest.raises(ValueError, match="sampler"):
        mlx_backend.generate_tokens(model, [EOD], sampling, 2, "semantic", sampler="gpu")


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
