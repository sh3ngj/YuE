"""MLX VAE decoder parity with the torch decoder on tiny FP32 checkpoints; skipped without MLX."""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

mx = pytest.importorskip("mlx.core")

from yue2.mlx_vae import MLXVAEDecoder, Upsample  # noqa: E402
from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig  # noqa: E402
from yue2.pipeline import YuE2Pipeline  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def mlx_cpu():
    # MLX GPU FP32 matmuls use reduced (TF32-like) precision; the CPU device checks the math exactly.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def export(directory, latent_dim=2, seed=231):
    # The released six-stage strides, padding and SnakeBeta topology with two channels.
    common = dict(channels=2, c_mults=[1, 1, 1, 1, 1, 1], strides=[2, 2, 4, 4, 5, 6], use_snake=True)
    config = YuE2VAEConfig(encoder_config=dict(common, in_channels=2, latent_dim=4),
                           decoder_config=dict(common, out_channels=2, latent_dim=latent_dim,
                                               snake_type="vanilla", final_tanh=False), latent_dim=latent_dim)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = YuE2VAE(config)
        for name, parameter in model.named_parameters():
            if name.endswith((".alpha", ".beta")):  # nonzero log-scale SnakeBeta parameters
                parameter.data.normal_(0, 0.3)
    config.save_pretrained(directory)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(directory / "model.safetensors"))
    return model


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("vae")
    return export(directory), MLXVAEDecoder(directory)


@pytest.mark.parametrize("frames", [1, 3, 17])
def test_full_decode_and_lengths_match_torch(models, frames):
    reference, decoder = models
    latent = torch.randn(1, 2, frames, generator=torch.Generator().manual_seed(frames))
    expected = reference.decode(latent)[0].T.numpy()
    actual = decoder.decode(latent[0].T.numpy())
    assert actual.shape == expected.shape == (decoder.natural_output_length(frames), 2) == (1920 * frames - 64, 2)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("frames,core", [(1, 1), (15, 7), (33, 16), (65, 32)])
def test_tiled_decode_matches_torch_tiles_and_halo_rules(models, frames, core):
    reference, decoder = models
    latent = torch.randn(1, 2, frames, generator=torch.Generator().manual_seed(core))
    expected = reference.decode_tiled(latent, core_frames=core, halo_frames=16)[0].T.numpy()
    progress = []
    actual = decoder.decode_tiled(latent[0].T.numpy(), core_frames=core, halo_frames=16,
                                  on_progress=lambda done, total: progress.append((done, total)))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    tiles = -(-frames // core)
    assert progress == [(i + 1, tiles) for i in range(tiles)]
    assert decoder.required_halo(core) == reference.required_halo(core)
    with pytest.raises(ValueError, match="halo_frames"):
        decoder.decode_tiled(latent[0].T.numpy(), core_frames=core, halo_frames=decoder.required_halo(core) - 1)


@pytest.mark.skipif(not mx.metal.is_available(), reason="No Metal GPU")
def test_polyphase_upsampling_keeps_fp32_on_the_gpu():
    # MLX's GPU conv_transpose1d computes FP32 at TF32-like precision; the polyphase form must not.
    rng = np.random.default_rng(0)
    inputs, outputs, stride = 256, 128, 5
    weight = rng.standard_normal((inputs, outputs, 2 * stride)).astype(np.float32)
    bias = rng.standard_normal(outputs).astype(np.float32)
    norm = np.linalg.norm(weight.reshape(inputs, -1), axis=1).reshape(inputs, 1, 1).astype(np.float32)
    layer = Upsample({"up.weight_v": weight, "up.weight_g": norm, "up.bias": bias}, "up", stride)
    x = rng.standard_normal((1, 2000, inputs)).astype(np.float32)
    with mx.stream(mx.gpu):
        actual = np.array(layer(mx.array(x)))
    expected = torch.nn.functional.conv_transpose1d(
        torch.from_numpy(x).double().transpose(1, 2), torch.from_numpy(weight).double(),
        torch.from_numpy(bias).double(), stride=stride, padding=-(-stride // 2)).transpose(1, 2).numpy()
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() < 1e-5 * np.abs(expected).max()


def test_checkpoint_validation(models, tmp_path):
    reference, _ = models
    with pytest.raises(ValueError, match="latents"):
        models[1].decode(np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        models[1].decode(np.full((4, 2), np.nan, dtype=np.float32))
    export(tmp_path)
    state = {k: v for k, v in reference.state_dict().items()}
    save_file({**state, "decoder.extra": torch.zeros(1)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="unexpected"):
        MLXVAEDecoder(tmp_path)
    save_file({k: v.double() if k.startswith("decoder.layers.0.") else v for k, v in state.items()},
              str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="FP32"):
        MLXVAEDecoder(tmp_path)


def test_pipeline_routes_decoding_to_mlx(tmp_path):
    reference = export(tmp_path, latent_dim=64)
    pipe = SimpleNamespace(progress=False, vae_dir=tmp_path, _mlx_vae=None, vae_core_frames=4)
    pipe._status = MethodType(YuE2Pipeline._status, pipe)
    latent = torch.randn(9, 64, generator=torch.Generator().manual_seed(5))
    expected = reference.decode_tiled(latent.T[None], core_frames=4, halo_frames=16)[0].clamp(-1, 1).T.numpy()
    for value in (latent.numpy(), latent.T[None].numpy()):
        actual = YuE2Pipeline._decode_mlx(pipe, value, full=False, vae=None)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        assert actual.dtype == np.float32 and np.abs(actual).max() <= 1
    full = YuE2Pipeline._decode_mlx(pipe, latent.numpy(), full=True, vae=None)
    np.testing.assert_allclose(full, reference.decode(latent.T[None])[0].clamp(-1, 1).T.numpy(), rtol=2e-5, atol=2e-6)
    with pytest.raises(ValueError, match="Expected latents"):
        YuE2Pipeline._decode_mlx(pipe, np.zeros((2, 3, 4), dtype=np.float32), full=False, vae=None)
