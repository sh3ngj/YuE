"""MLX twin of the YuE2 Oobleck VAE decoder: FP32, same tensors, tiles and exact crops.

Transposed convolutions run as one regular convolution over their stride
phases. On the GPU, MLX's conv_transpose1d computes FP32 at reduced (TF32-like)
precision, whereas its regular convolutions keep FP32 at the decoder's shapes.
Base imports never require MLX; YuE2Pipeline(vae_backend="mlx") imports this module.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np
from safetensors import safe_open


def _weight_norm(state, prefix):
    """torch weight_norm(dim=0): v * g / ||v|| over every dimension but the first."""
    v = state.pop(prefix + ".weight_v").astype(np.float64)
    g = state.pop(prefix + ".weight_g").astype(np.float64)
    norm = np.sqrt(np.square(v).sum(axis=tuple(range(1, v.ndim)), keepdims=True))
    return (v * (g / norm)).astype(np.float32)


def _snake(x, alpha, inverse_beta):
    return x + inverse_beta * mx.square(mx.sin(x * alpha))


# One fused kernel instead of five passes over the decoder's largest activations.
_fused_snake = mx.compile(_snake, shapeless=True)


class Snake:
    def __init__(self, state, prefix):
        # Log-scale SnakeBeta: x + 1 / (exp(beta) + 1e-9) * sin(x * exp(alpha))^2.
        self.alpha = mx.exp(mx.array(state.pop(prefix + ".alpha")))
        self.inverse_beta = 1.0 / (mx.exp(mx.array(state.pop(prefix + ".beta"))) + 0.000000001)

    def __call__(self, x):
        return _fused_snake(x, self.alpha, self.inverse_beta)

    def support(self, low, high):
        return low, high

    def length(self, frames):
        return frames


class Conv:
    def __init__(self, state, prefix, *, padding=0, dilation=1, bias=True):
        weight = _weight_norm(state, prefix)  # torch [out, in, kernel]
        self.weight = mx.array(np.ascontiguousarray(weight.transpose(0, 2, 1)))
        self.bias = mx.array(state.pop(prefix + ".bias")) if bias else None
        self.padding, self.dilation, self.kernel = padding, dilation, weight.shape[2]

    def __call__(self, x):
        y = mx.conv1d(x, self.weight, padding=self.padding, dilation=self.dilation)
        return y if self.bias is None else y + self.bias

    def support(self, low, high):
        return low - self.padding, high - self.padding + self.dilation * (self.kernel - 1)

    def length(self, frames):
        return frames + 2 * self.padding - self.dilation * (self.kernel - 1)


class Upsample:
    """ConvTranspose1d whose kernel is a multiple of its stride, as one polyphase convolution."""

    def __init__(self, state, prefix, stride):
        weight = _weight_norm(state, prefix)  # torch [in, out, kernel]
        inputs, outputs, kernel = weight.shape
        if kernel % stride:
            raise ValueError("Polyphase upsampling needs a kernel that is a multiple of its stride")
        self.taps, self.stride, self.outputs = kernel // stride, stride, outputs
        self.kernel, self.padding = kernel, math.ceil(stride / 2)
        # Phase r, tap m of the regular convolution uses transposed tap (taps - 1 - m) * stride + r.
        phases = weight.reshape(inputs, outputs, self.taps, stride)[:, :, ::-1, :]
        self.weight = mx.array(np.ascontiguousarray(
            phases.transpose(3, 1, 2, 0).reshape(stride * outputs, self.taps, inputs)))
        self.bias = mx.array(state.pop(prefix + ".bias"))

    def __call__(self, x):
        batch, frames, _ = x.shape
        span = (frames + self.taps - 1) * self.stride
        y = mx.conv1d(x, self.weight, padding=self.taps - 1).reshape(batch, span, self.outputs)
        return y[:, self.padding:span - self.padding] + self.bias

    def support(self, low, high):
        return -(-(low + self.padding - (self.kernel - 1)) // self.stride), (high + self.padding) // self.stride

    def length(self, frames):
        return (frames - 1) * self.stride - 2 * self.padding + self.kernel


class Residual:
    def __init__(self, state, prefix, dilation):
        self.layers = [Snake(state, f"{prefix}.layers.0"),
                       Conv(state, f"{prefix}.layers.1", padding=3 * dilation, dilation=dilation),
                       Snake(state, f"{prefix}.layers.2"), Conv(state, f"{prefix}.layers.3")]

    def __call__(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
        return x + residual

    def support(self, low, high):
        a, b = low, high
        for layer in reversed(self.layers):
            a, b = layer.support(a, b)
        return min(a, low), max(b, high)

    def length(self, frames):
        return frames


class MLXVAEDecoder:
    """Decoder half of YuE2VAE in MLX; input latents [T,64], output FP32 audio [samples,channels]."""

    def __init__(self, directory):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        decoder = config["decoder_config"]
        if (not decoder.get("use_snake") or decoder.get("snake_type", "vanilla") != "vanilla"
                or any(decoder.get(name) for name in ("antialias_activation", "use_nearest_upsample", "use_filter"))):
            raise ValueError("Unsupported option for the released decoder")
        state = self._decoder_tensors(directory)
        if any(value.dtype != np.float32 for value in state.values()):
            raise ValueError("VAE export contains tensors that are not FP32")
        mults, strides = [1] + list(decoder["c_mults"]), decoder["strides"]
        self.layers = [Conv(state, "decoder.layers.0", padding=3)]
        index = 1
        for i in range(len(mults) - 1, 0, -1):
            block = f"decoder.layers.{index}.layers"
            self.layers += [Snake(state, f"{block}.0"), Upsample(state, f"{block}.1", strides[i - 1])]
            self.layers += [Residual(state, f"{block}.{2 + j}", dilation) for j, dilation in enumerate((1, 3, 9))]
            index += 1
        self.layers += [Snake(state, f"decoder.layers.{index}"),
                        Conv(state, f"decoder.layers.{index + 1}", padding=3, bias=False)]
        if state:
            raise ValueError(f"VAE tensor mismatch: unexpected={sorted(state)}")
        self.final_tanh = decoder.get("final_tanh", True)
        self.ratio, self.latent_dim = int(config["downsampling_ratio"]), int(config["latent_dim"])
        self.core_frames = int(config.get("decode_core_frames", 1024))
        self.halo_frames = int(config.get("decode_halo_frames", 16))
        mx.eval([array for layer in self._flat() for array in vars(layer).values() if isinstance(array, mx.array)])

    @classmethod
    def from_pretrained(cls, name):
        """A local export or Hub repository, resolved like YuE2VAE.from_pretrained."""
        path = Path(name).expanduser()
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(name), allow_patterns=[
                "config.json", "*.safetensors", "*.safetensors.index.json"]))
        return cls(path)

    @staticmethod
    def _decoder_tensors(directory):
        index = directory / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            files = sorted({name for key, name in mapping.items() if key.startswith("decoder.")})
        else:
            files = ["model.safetensors"]
        state = {}
        for name in files:
            with safe_open(directory / name, framework="np") as handle:
                for key in handle.keys():
                    if key.startswith("decoder."):
                        if key in state:
                            raise ValueError(f"Duplicate VAE tensor: {key}")
                        state[key] = handle.get_tensor(key)
        return state

    def _flat(self):
        for layer in self.layers:
            yield from layer.layers if isinstance(layer, Residual) else (layer,)

    def natural_output_length(self, frames):
        if int(frames) < 1:
            raise ValueError("frames must be positive")
        length = int(frames)
        for layer in self.layers:
            length = layer.length(length)
        return length

    def required_halo(self, core_frames=None):
        core_frames = self.core_frames if core_frames is None else core_frames
        low, high = 0, core_frames * self.ratio - 1
        for layer in reversed(self.layers):
            low, high = layer.support(low, high)
        return max(0, -low, high - core_frames + 1)

    def _latent(self, latent):
        latent = np.asarray(latent, dtype=np.float32)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim or latent.shape[0] < 1:
            raise ValueError(f"Expected nonempty [T,{self.latent_dim}] latents")
        if not np.isfinite(latent).all():
            raise ValueError("VAE latents contain non-finite values")
        return latent

    def _decode(self, latent):
        x = mx.array(latent)[None]
        for layer in self.layers:
            x = layer(x)
        return (mx.tanh(x) if self.final_tanh else x)[0]

    def decode(self, latent):
        """Full waveform, FP32 [1920*T-64, channels], without clipping."""
        return np.array(self._decode(self._latent(latent)))

    def decode_tiled(self, latent, core_frames=None, halo_frames=None,
                     on_progress: Callable[[int, int], None] | None = None):
        """YuE2VAE.decode_tiled's tiles and exact cores: no crossfade, natural end length."""
        latent = self._latent(latent)
        core_frames = self.core_frames if core_frames is None else core_frames
        halo_frames = self.halo_frames if halo_frames is None else halo_frames
        if not isinstance(core_frames, int) or core_frames < 1:
            raise ValueError("core_frames must be a positive integer")
        required = self.required_halo(core_frames)
        if not isinstance(halo_frames, int) or halo_frames < required:
            raise ValueError(f"halo_frames must be at least {required} for this decoder")
        frames = len(latent)
        total = self.natural_output_length(frames)
        audio = None
        tiles = (frames + core_frames - 1) // core_frames
        for tile_index, start in enumerate(range(0, frames, core_frames)):
            end = min(frames, start + core_frames)
            left, right = max(0, start - halo_frames), min(frames, end + halo_frames)
            tile = self._decode(latent[left:right])
            out_start, out_end = start * self.ratio, min(end * self.ratio, total)
            crop_start = (start - left) * self.ratio
            crop = np.array(tile[crop_start:crop_start + out_end - out_start])
            if crop.shape[0] != out_end - out_start:
                raise RuntimeError("VAE tile did not cover its requested output core")
            if audio is None:
                audio = np.empty((total, crop.shape[1]), dtype=np.float32)
            audio[out_start:out_end] = crop
            del tile, crop
            if on_progress is not None:
                on_progress(tile_index + 1, tiles)
        return audio
