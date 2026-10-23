"""Optional Apple Silicon MLX backend for the YuE2 AR–NAR network.

The network runs in MLX. Token sampling applies the eager path's filters either
in MLX with its own RNG (default) or with the eager path's CPU arithmetic and
torch RNG. Acoustic flow matching reuses the full-song CPU noise draw, midpoint
schedule and model-dtype state updates. Base imports never require MLX;
YuE2Pipeline(backend="mlx") imports this module on demand.
"""
from __future__ import annotations

import json
import math
import time
from numbers import Integral
from pathlib import Path
from typing import Callable, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from .nar import Chunk, song_chunks
from .protocol import ABC_END, CODEC_OFFSET, CODEC_SIZE, CONTEXT, EOD, MUSIC_END
from .sampling import distribution

# Tokens each phase may sample; distribution() sets every other logit to -inf.
PHASE_TOKENS = {
    "abc": np.concatenate([np.arange(EOD), [ABC_END]]),
    "semantic": np.concatenate([np.arange(CODEC_OFFSET, CODEC_OFFSET + CODEC_SIZE), [MUSIC_END]]),
}
TORCH_DTYPES = {mx.bfloat16: torch.bfloat16, mx.float16: torch.float16, mx.float32: torch.float32}
MLX_DTYPES = {value: key for key, value in TORCH_DTYPES.items()}


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.heads, self.kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
        self.dim, self.base = config["head_dim"], float(config["rope_theta"])
        self.q_proj = nn.Linear(hidden, self.heads * self.dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.dim, hidden, bias=False)
        self.q_norm = nn.RMSNorm(self.dim, eps=eps)
        self.k_norm = nn.RMSNorm(self.dim, eps=eps)

    def project(self, x, offset):
        """Q [B,H,L,D] and K/V [B,KV,L,D], head-normalized, RoPE at offset..offset+L-1."""
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, length, self.heads, self.dim))
        k = self.k_norm(self.k_proj(x).reshape(batch, length, self.kv_heads, self.dim))
        v = self.v_proj(x).reshape(batch, length, self.kv_heads, self.dim)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        # Non-traditional RoPE rotates the two halves, as _apply_rotary does.
        rope = dict(traditional=False, base=self.base, scale=1.0, offset=offset)
        return mx.fast.rope(q, self.dim, **rope), mx.fast.rope(k, self.dim, **rope), v

    def attend(self, q, k, v, mask=None):
        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.dim ** -0.5, mask=mask)
        batch, _, length, _ = h.shape
        return self.o_proj(h.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, inner = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, inner, bias=False)
        self.up_proj = nn.Linear(hidden, inner, bias=False)
        self.down_proj = nn.Linear(inner, hidden, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """AR attention/MLP plus their NAR copies; attribute names match the checkpoint."""

    def __init__(self, config):
        super().__init__()
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.input_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.mlp = MLP(config)
        self.nar_input_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.nar_self_attn = Attention(config)
        self.nar_pre_mlp_layernorm = nn.RMSNorm(hidden, eps=eps)
        self.nar_mlp = MLP(config)


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [DecoderLayer(config) for _ in range(config["num_hidden_layers"])]
        self.norm = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = [nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size)]
        self.frequency_embedding_size = frequency_embedding_size


class AudioPositionEmbedding(nn.Module):
    def __init__(self, max_frames, hidden_size):
        super().__init__()
        self.pe = mx.zeros((max_frames, hidden_size))


class YuE2MLX(nn.Module):
    """MLX twin of YuE2ForCausalLM for inference; loads the same tensors by name."""

    def __init__(self, config):
        super().__init__()
        if config.get("latent_type", "vae") != "vae":
            raise ValueError("YuE2 inference supports only latent_type='vae'")
        self.vocab_size, self.max_latent_frames = config["vocab_size"], config["max_latent_frames"]
        self.max_positions = config["max_position_embeddings"]
        self.timestep_shift = float(config["timestep_shift"])
        self.model = Backbone(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        self.llm2vae = nn.Linear(config["hidden_size"], config["latent_dim"])
        self.vae2llm = nn.Linear(config["latent_dim"], config["hidden_size"])
        self.time_embedder = TimestepEmbedder(config["hidden_size"])
        self.latent_pos_embed = AudioPositionEmbedding(config["max_latent_frames"], config["hidden_size"])

    @property
    def dtype(self):
        return self.lm_head.weight.dtype

    def hidden(self, ids, cache=None):
        """Final normed AR hidden states; with a cache, appends at its current length."""
        offset = cache.length if cache is not None else 0
        if offset and ids.shape[1] > 1:
            raise ValueError("Chunked prefill is not supported; prefill once, then decode one token at a time")
        x = self.model.embed_tokens(ids)
        mask = "causal" if ids.shape[1] > 1 else None
        for index, layer in enumerate(self.model.layers):
            attention = layer.self_attn
            q, k, v = attention.project(layer.input_layernorm(x), offset)
            if cache is not None:
                k, v = cache.update(index, k, v)
            x = x + attention.attend(q, k, v, mask)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.model.norm(x)

    def time_embedding(self, raw_t):
        """TimestepEmbedder of the sigmoid-shifted time, sharing the torch scalar arithmetic."""
        dtype = TORCH_DTYPES[self.dtype]
        sigmoid = torch.sigmoid(torch.tensor(raw_t, dtype=dtype))
        shift = self.timestep_shift
        shifted = shift * sigmoid / (1 + (shift - 1) * sigmoid)
        half = self.time_embedder.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        args = shifted.float().reshape(1, 1) * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype)
        first, _, second = self.time_embedder.mlp
        return second(nn.silu(first(mx.array(emb.float().numpy()).astype(self.dtype))))


def load(model_dir):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = YuE2MLX(config)
    model.load_weights(str(model_dir / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    return model


def from_torch(module):
    """MLX copy of a torch YuE2ForCausalLM with the same tensors and dtypes."""
    model = YuE2MLX(module.config.to_dict())
    weights = [(name, mx.array(tensor.detach().float().cpu().numpy()).astype(MLX_DTYPES[tensor.dtype]))
               for name, tensor in module.state_dict().items()]
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


class KVCache:
    """Preallocated AR keys/values for one branch; returns the used prefix."""

    def __init__(self, model, capacity):
        attention = model.model.layers[0].self_attn
        shape = (1, attention.kv_heads, capacity, attention.dim)
        self.keys = [mx.zeros(shape, model.dtype) for _ in model.model.layers]
        self.values = [mx.zeros(shape, model.dtype) for _ in model.model.layers]
        self.capacity, self.length = capacity, 0

    def update(self, index, k, v):
        end = self.length + k.shape[2]
        if end > self.capacity:
            raise ValueError(f"KV cache capacity {self.capacity} exceeded by {end}; generation was not shortened")
        self.keys[index][..., self.length:end, :] = k
        self.values[index][..., self.length:end, :] = v
        if index == len(self.keys) - 1:
            self.length = end
        return self.keys[index][..., :end, :], self.values[index][..., :end, :]


class DeviceSampler:
    """sampling.distribution's filters on phase-local logits in MLX, with a request-local MLX RNG.

    The min-length, windowed repetition penalty, temperature, top-k (ties survive) and
    shifted top-p rules run in FP32 as in the torch sampler. Tied values at the top-p
    boundary may sort differently, and draws come from MLX, so a seed does not reproduce
    the torch sampler's tokens.
    """

    def __init__(self, sampling, phase, seed):
        self.sampling = sampling
        self.size = len(PHASE_TOKENS[phase])
        self.end = self.size - 1  # Both phases list their end token last.
        self.positions = mx.arange(self.size)
        self.key = mx.random.key(seed)

    def scores(self, logits, recent, step):
        """Filtered FP32 scores; recent holds phase-local IDs of the penalty window, or None."""
        sampling, blocked = self.sampling, mx.array(float("-inf"))
        scores = logits.astype(mx.float32)
        if step < sampling.min_tokens:
            scores = mx.where(self.positions == self.end, blocked, scores)
        if sampling.repetition_penalty != 1.0 and recent is not None:
            frequency = mx.zeros((self.size,), dtype=mx.float32).at[recent].add(1.0)
            alpha = mx.power(sampling.repetition_penalty, frequency)
            scores = mx.where(scores < 0, scores * alpha, scores / alpha)
        if sampling.temperature == 0:
            return scores
        if sampling.temperature != 1:
            scores = scores / sampling.temperature
        threshold = mx.min(mx.topk(scores, min(sampling.top_k, self.size)))
        scores = mx.where(scores < threshold, blocked, scores)
        if sampling.top_p < 1:
            order = mx.argsort(-scores)
            values = scores[order]
            probabilities = mx.softmax(values, precise=True)
            removed = (mx.cumsum(probabilities) - probabilities > sampling.top_p) & (self.positions >= 1)
            scores = mx.put_along_axis(scores, order, mx.where(removed, blocked, values), axis=-1)
        return scores

    def sample(self, scores):
        if self.sampling.temperature == 0:
            return mx.argmax(scores)
        self.key, key = mx.random.split(self.key)
        return mx.random.categorical(scores, key=key)


def generate_tokens(model, prefix, sampling, seed, phase, negative=None, cfg_scale=1.0,
                    legacy_off=False, cancelled=None, on_token=None, sampler="device"):
    """MLX twin of sampling.generate_tokens with the same checks, filters and timing record.

    sampler="torch" keeps the eager path's CPU sampling arithmetic and torch RNG, so equal
    logits give the torch backend's tokens. sampler="device" samples with DeviceSampler and
    queues each next decode step before reading the current token. Historical off sampling
    keeps its BF16 torch arithmetic, so it always uses the torch sampler.
    """
    if phase not in PHASE_TOKENS:
        raise ValueError("phase must be abc or semantic")
    if sampler not in ("device", "torch"):
        raise ValueError("sampler must be device or torch")
    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    if negative is not None and len(negative) + sampling.max_tokens > CONTEXT:
        raise ValueError("Negative prefix + generation budget exceeds context")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")
    fallback = None
    if sampler == "device" and legacy_off:
        sampler, fallback = "torch", "historical_off_sampling_uses_torch"
    tokens = PHASE_TOKENS[phase]
    # Only allowed rows of lm_head are evaluated; the others would be masked to -inf.
    head = mx.take(model.lm_head.weight, mx.array(tokens), axis=0)

    def logits(ids, cache):
        return (model.hidden(ids, cache)[:, -1] @ head.T)[0]

    def mix(conditional, unconditional=None):
        # Preserve historical model-dtype CFG subtraction/multiply/add before upcast.
        return conditional if unconditional is None else unconditional + cfg_scale * (conditional - unconditional)

    start = time.perf_counter()
    branches = [KVCache(model, len(prefix) + sampling.max_tokens)]
    prefilled = [logits(mx.array([prefix], dtype=mx.int32), branches[0])]
    if cfg_scale != 1.0:
        branches.append(KVCache(model, len(negative) + sampling.max_tokens))
        prefilled.append(logits(mx.array([negative], dtype=mx.int32), branches[1]))
    mx.eval(prefilled)
    prefill_seconds = time.perf_counter() - start

    def decode(ids):
        return mix(*(logits(ids, cache) for cache in branches))

    history, first, eos = [], None, False
    end = ABC_END if phase == "abc" else MUSIC_END

    def accept(token):
        """Record one sampled native token; True when it ends the phase."""
        nonlocal first, eos
        if first is None:
            first = time.perf_counter() - start
        if on_token is not None:
            on_token(phase, token)
        if token == end:
            eos = True
            return True
        history.append(token)
        return False

    if sampler == "torch":
        # The two stages deliberately reset their request-local seed, matching the preset.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        allowed = torch.from_numpy(tokens)
        # vLLM's symbolic processor receives FP32 logits; historical off uses BF16.
        score_dtype = TORCH_DTYPES[model.dtype] if legacy_off else torch.float32
        current = mix(*prefilled)
        for step in range(sampling.max_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError(f"Cancelled during {phase}")
            scores = torch.full((1, model.vocab_size), float("-inf"), dtype=score_dtype)
            scores[0, allowed] = torch.from_numpy(np.array(current.astype(mx.float32))).to(score_dtype)
            scores = distribution(scores, sampling, history, step, phase, legacy_off)
            if sampling.temperature == 0:
                next_id = scores.argmax(-1, keepdim=True)
            else:
                next_id = torch.multinomial(scores.softmax(-1), 1, generator=generator)
            if accept(int(next_id.item())):
                break
            if step + 1 < sampling.max_tokens:
                current = decode(mx.array([[history[-1]]], dtype=mx.int32))
    else:
        device = DeviceSampler(sampling, phase, seed)
        native, recent = mx.array(tokens, dtype=mx.int32), None
        token = device.sample(device.scores(mix(*prefilled), recent, 0))
        mx.async_eval(token)
        for step in range(sampling.max_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError(f"Cancelled during {phase}")
            following = None
            if step + 1 < sampling.max_tokens:
                # Queue the next step before reading this token; an end token wastes that one step.
                recent = token[None] if recent is None else mx.concatenate([recent, token[None]])
                recent = recent[-sampling.penalty_window:]
                following = device.sample(device.scores(decode(native[token].reshape(1, 1)), recent, step + 1))
                mx.async_eval(following)
            if accept(int(tokens[token.item()])):
                break
            token = following
    seconds = time.perf_counter() - start
    count = len(history) + int(eos)
    timing = {"seconds": seconds, "prefill_seconds": prefill_seconds, "ttft_seconds": first,
              "output_tokens": count, "content_tokens": len(history), "output_tps": count / seconds,
              "prefix_tokens": len(prefix), "cfg_branches": len(branches), "execution": "mlx",
              "attention": "mlx_fast_sdpa", "backend_actual": "mlx", "mlx_version": mx.__version__,
              "sampler": "mlx_device" if sampler == "device" else "torch_cpu"}
    if fallback:
        timing["sampler_fallback_reason"] = fallback
    return history, timing, not eos


def _raw_time(t):
    return torch.logit(torch.tensor(t, dtype=torch.float64, device="cpu")).clamp(-20, 20).item()


class CachedNAR:
    """MLX twin of nar.CachedNAR: one AR prefill per original chunk, then the midpoint solve."""

    def __init__(self, model, chunk: Chunk):
        self.model, self.chunk = model, chunk
        noise = chunk.noise
        if noise.ndim != 2 or noise.shape[1] != 64 or len(noise) < 1:
            raise ValueError("Expected nonempty acoustic noise [frames,64]")
        if not torch.isfinite(noise).all():
            raise ValueError("Acoustic noise contains non-finite values")
        self.ar_length, self.nar_length = len(chunk.ar_tokens), len(noise) + 2
        if self.ar_length < 1 or min(chunk.ar_tokens) < 0 or max(chunk.ar_tokens) >= model.vocab_size:
            raise ValueError("AR prefix is empty or outside the model vocabulary")
        if self.ar_length + self.nar_length > model.max_positions:
            raise ValueError("Original acoustic chunk exceeds the model context")
        if chunk.nar_cond_end < 0:
            raise ValueError("nar_cond_end must be nonnegative")
        self.visible_length = min(chunk.nar_cond_end, self.ar_length) if chunk.nar_cond_end else self.ar_length
        local = mx.minimum(mx.arange(self.nar_length), model.max_latent_frames - 1)
        self.pos_emb = model.latent_pos_embed.pe[local][None]
        self.cache = []
        self._prefill()

    def _prefill(self):
        backbone = self.model.model
        x = backbone.embed_tokens(mx.array([self.chunk.ar_tokens], dtype=mx.int32))
        for layer in backbone.layers:
            q, k, v = layer.self_attn.project(layer.input_layernorm(x), 0)
            self.cache.append((k[:, :, :self.visible_length], v[:, :, :self.visible_length]))
            x = x + layer.self_attn.attend(q, k, v, "causal")
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        # Evaluating only the cache leaves the unused final-layer output uncomputed.
        mx.eval(self.cache)

    def velocity(self, state, raw_t):
        model = self.model
        if tuple(state.shape) != tuple(self.chunk.noise.shape):
            raise ValueError("ODE state shape changed")
        x = model.vae2llm(mx.pad(state, [(1, 1), (0, 0)])[None])
        x = x + model.time_embedding(raw_t)[None]
        x = x + self.pos_emb
        for layer, (ar_k, ar_v) in zip(model.model.layers, self.cache):
            attention = layer.nar_self_attn
            q, k, v = attention.project(layer.nar_input_layernorm(x), self.ar_length)
            x = x + attention.attend(q, mx.concatenate([ar_k, k], axis=2), mx.concatenate([ar_v, v], axis=2))
            x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        return model.llm2vae(model.model.norm(x))[0, 1:-1]

    def solve(self, steps=32, cancelled: Callable[[], bool] | None = None,
              on_progress: Callable[[int, int], None] | None = None):
        """Midpoint ODE in the model dtype, as nar.CachedNAR.solve; returns CPU FP32 latents."""
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("steps must be a positive integer")
        state = mx.array(self.chunk.noise.numpy()).astype(self.model.dtype)
        dt = 1.0 / steps
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            first = self.velocity(state, _raw_time(t))
            mid = state - first * (dt / 2)
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            state = state - self.velocity(mid, _raw_time(t - dt / 2)) * dt
            mx.eval(state)
            if on_progress is not None:
                on_progress(step + 1, int(steps))
        result = torch.from_numpy(np.array(state.astype(mx.float32)))
        if not torch.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self):
        self.cache.clear()
        self.pos_emb = None


def synthesize(model, prefix: Sequence[int], codec: Sequence[int], seed: int, steps=32, context=CONTEXT,
               cancelled=None, on_progress: Callable[[int, int], None] | None = None):
    """MLX twin of nar.synthesize: CPU FP32 [frames,64] latents from serially solved chunks."""
    chunks = song_chunks(prefix, codec, seed, context)
    output = []
    for chunk_index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled before acoustic prefill")
        engine = CachedNAR(model, chunk)
        try:
            progress = None
            if on_progress is not None:
                def progress(completed, total):
                    on_progress(chunk_index * total + completed, total * len(chunks))
            output.append(engine.solve(steps, cancelled, on_progress=progress))
        finally:
            engine.close()
        del engine
    return torch.cat(output, dim=0)
