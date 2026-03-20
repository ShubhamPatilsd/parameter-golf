"""
BitNet experiment for parameter golf.

Replaces CastedLinear with BitLinear (ternary weights: {-1, 0, +1}).
Scales model_dim from 512 -> 768 to use the ~4x compression headroom.

Ternary packs 4 values per byte (2 bits each) vs 1 value per byte (int8),
so we can fit ~4x more parameters in the same 16MB budget.

Compare:
  Baseline: dim=512, ~17M params, int8 -> ~8MB compressed
  BitNet:   dim=768, ~38M params, ternary 2-bit -> ~10MB before zlib

Usage (quick convergence test, compare val_bpb against baseline):
  ITERATIONS=200 VAL_LOSS_EVERY=50 python3 train_bitnet_mlx.py
  ITERATIONS=200 VAL_LOSS_EVERY=50 python3 train_gpt_mlx.py
"""
from __future__ import annotations

import glob
import math
import os
import pickle
import struct
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

COMPUTE_DTYPE = mx.bfloat16

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight",
)

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 5))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model: dim=512 matches baseline for local testing; scale to 768+ on H100s
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

# ==============================================================================
# DATA LOADING (unchanged from train_gpt_mlx.py)
# ==============================================================================

def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return tokens.astype(np.int32, copy=False)


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _next_file(self):
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self._next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


class TokenLoader:
    def __init__(self, pattern: str):
        self.stream = TokenStream(pattern)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = np.concatenate([load_data_shard(f) for f in files])
    usable = ((tokens.size - 1) // seq_len) * seq_len
    return tokens[: usable + 1]

# ==============================================================================
# BITLINEAR — ternary weights with straight-through estimator
# ==============================================================================

class BitLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1}.
    - Full-precision latent weights are maintained for the optimizer.
    - During forward: quantize to ternary on the fly (absmean scale).
    - Gradients flow through via straight-through estimator.
    - At save time: weights are packed as 2 bits each (4x smaller than int8).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        std = 1.0 / math.sqrt(in_dim)
        self.weight = mx.random.normal((out_dim, in_dim), dtype=mx.float32) * std

    def __call__(self, x: mx.array) -> mx.array:
        w = self.weight  # float32 latent weights
        scale = mx.mean(mx.abs(w)) + 1e-5
        w_norm = w / scale
        # Ternary quantize: round to {-1, 0, +1}
        w_q = mx.clip(mx.round(w_norm), -1.0, 1.0)
        # Straight-through: gradient flows as if no quantization
        w_eff = w_norm + mx.stop_gradient(w_q - w_norm)
        return x @ (w_eff * scale).astype(x.dtype).T

# ==============================================================================
# MODEL
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    return (x.T if transposed else x).astype(g.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = BitLinear(dim, dim)
        self.c_k = BitLinear(dim, kv_dim)
        self.c_v = BitLinear(dim, kv_dim)
        self.proj = BitLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return self.proj(y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = BitLinear(dim, hidden)
        self.proj = BitLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc(x))
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * self.attn(rms_norm(x))
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float, qk_gain_init: float):
        super().__init__()
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init) for _ in range(num_layers)]
        # Zero-init output projections
        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std).astype(COMPUTE_DTYPE)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return rms_norm(x)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits = self.logit_softcap * mx.tanh((x @ self.tok_emb.weight.astype(x.dtype).T) / self.logit_softcap)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

# ==============================================================================
# OPTIMIZER (Muon + Adam, same as baseline)
# ==============================================================================

class Muon:
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict, grads: dict, step: int, lr_mul: float) -> dict:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out = {}
        for k in self.keys:
            p, g = params[k], grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = zeropower_newtonschulz5(g + momentum * buf, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
            out[k] = p - lr * (g_eff * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2
            and not any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k == "skip_weights" or (
                k.startswith("blocks.") and (
                    p.ndim < 2 or any(pat in k for pat in CONTROL_TENSOR_NAME_PATTERNS)
                )
            )
        ]
        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(learning_rate=args.tied_embed_lr, betas=[args.beta1, args.beta2], eps=args.adam_eps)
        self.adam_scalar = optim.Adam(learning_rate=args.scalar_lr, betas=[args.beta1, args.beta2], eps=args.adam_eps)

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))
        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(self.adam_embed.apply_gradients({self.embed_key: grads[self.embed_key]}, {self.embed_key: params[self.embed_key]}))
        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        updated.update(self.adam_scalar.apply_gradients({k: grads[k] for k in self.scalar_keys}, {k: params[k] for k in self.scalar_keys}))
        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# TERNARY COMPRESSION — 2 bits per weight
# ==============================================================================

def pack_ternary(arr: np.ndarray) -> tuple[np.ndarray, float, tuple, int]:
    """Quantize float array to {-1, 0, +1} and pack 4 values per byte."""
    shape = arr.shape
    flat = arr.flatten().astype(np.float32)
    n = len(flat)
    scale = float(np.mean(np.abs(flat))) + 1e-5
    q = np.clip(np.round(flat / scale), -1, 1).astype(np.int8)
    # Map {-1,0,1} -> {0,1,2} so we can use 2 bits
    u = (q + 1).astype(np.uint8)
    # Pad to multiple of 4
    pad = (4 - n % 4) % 4
    if pad:
        u = np.pad(u, (0, pad))
    packed = (u[0::4] | (u[1::4] << 2) | (u[2::4] << 4) | (u[3::4] << 6)).astype(np.uint8)
    return packed, scale, shape, n


def unpack_ternary(packed: np.ndarray, scale: float, shape: tuple, n: int) -> np.ndarray:
    """Unpack 2-bit ternary weights back to float."""
    u = np.empty(len(packed) * 4, dtype=np.uint8)
    u[0::4] = packed & 0x03
    u[1::4] = (packed >> 2) & 0x03
    u[2::4] = (packed >> 4) & 0x03
    u[3::4] = (packed >> 6) & 0x03
    q = u[:n].astype(np.float32) - 1.0  # {0,1,2} -> {-1,0,1}
    return (q * scale).reshape(shape)


def is_bitlinear_weight(name: str) -> bool:
    """Identify BitLinear weight tensors (the ones we compress as ternary)."""
    return name.endswith(".weight") and any(
        layer in name for layer in ["c_q", "c_k", "c_v", "attn.proj", "mlp.fc", "mlp.proj"]
    )


def quantize_bitnet_state_dict(flat_state: dict[str, mx.array]) -> tuple[dict, dict]:
    """
    Compress BitLinear weights as 2-bit ternary.
    Everything else (embeddings, scales, etc.) uses the same int8+fp16 scheme as baseline.
    """
    ternary = {}   # name -> (packed_bytes, scale, shape, n)
    passthrough = {}  # name -> np.ndarray (fp16 or fp32)
    stats = {"ternary_bytes": 0, "passthrough_bytes": 0, "total_params": 0}

    for name, arr in flat_state.items():
        a = np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)
        stats["total_params"] += a.size

        if is_bitlinear_weight(name) and a.ndim == 2 and a.size > 1024:
            packed, scale, shape, n = pack_ternary(a)
            ternary[name] = (packed, scale, shape, n)
            stats["ternary_bytes"] += packed.nbytes + 8  # 8 bytes for scale+n
        else:
            # Small tensors / embeddings / scalars: store as fp16
            pt = a.astype(np.float16)
            passthrough[name] = pt
            stats["passthrough_bytes"] += pt.nbytes

    return {"ternary": ternary, "passthrough": passthrough}, stats


def dequantize_bitnet_state_dict(obj: dict) -> dict[str, mx.array]:
    out = {}
    for name, (packed, scale, shape, n) in obj["ternary"].items():
        f32 = unpack_ternary(packed, scale, shape, n)
        out[name] = mx.array(f32, dtype=mx.float32)
    for name, arr in obj["passthrough"].items():
        out[name] = mx.array(arr.astype(np.float32))
    return out

# ==============================================================================
# EVALUATION
# ==============================================================================

def build_sentencepiece_luts(sp, vocab_size: int):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros(table_size, dtype=np.int16)
    has_leading_space_lut = np.zeros(table_size, dtype=np.bool_)
    is_boundary_token_lut = np.ones(table_size, dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_loss = mx.array(0.0, dtype=mx.float32)
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_seq_start in range(0, total_seqs, val_batch_seqs):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        ct = float(y.size)
        total_loss = total_loss + compiled_loss(x, y).astype(mx.float32) * ct
        prev_ids, tgt_ids = x_np.reshape(-1), y_np.reshape(-1)
        b = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        b += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).astype(np.int16)
        total_tokens += ct
        total_bytes += float(b.astype(np.float64).sum())
    total_loss = total_loss / total_tokens
    mx.eval(total_loss)
    val_loss = float(total_loss.item())
    val_bpb = (val_loss / math.log(2.0)) * (total_tokens / total_bytes)
    return val_loss, val_bpb

# ==============================================================================
# GRADIENT HELPERS
# ==============================================================================

def accumulate_flat_grads(accum, grads_tree, scale):
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


def loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad):
    # Sub-chunk microbatch to limit peak memory (critical on Mac unified memory)
    microbatch = args.microbatch_tokens
    max_chunk = args.mlx_max_microbatch_tokens
    seq_len = args.train_seq_len
    # Build list of chunk sizes that sum to microbatch, each <= max_chunk
    usable = (microbatch // seq_len) * seq_len
    chunk_tokens = max((max_chunk // seq_len) * seq_len, seq_len)
    chunks = []
    remaining = usable
    while remaining > 0:
        c = min(remaining, chunk_tokens)
        chunks.append(c)
        remaining -= c
    total_tokens = float(sum(chunks))
    loss_val = mx.array(0.0, dtype=mx.float32)
    accum = None
    for ct in chunks:
        x, y = train_loader.next_batch(ct, seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_val = loss_val + loss.astype(mx.float32) * scale
        accum = accumulate_flat_grads(accum, grads, scale)
    return loss_val, tree_unflatten(list(accum.items()))


def clip_grad_tree(grads_tree, max_norm):
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = sum(float(np.sum(np.square(np.array(g.astype(mx.float32), dtype=np.float32)))) for g in flat.values())
    total_norm = math.sqrt(max(total_sq, 0.0))
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}_bitnet.txt"
    print(logfile)

    def log(msg: str, console: bool = True):
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    log(f"run_id:{args.run_id}")
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch: {args.vocab_size} vs {int(sp.vocab_size())}")

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size)

    mx.random.seed(args.seed)
    train_loader = TokenLoader(args.train_files)

    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
    )
    opt = SplitOptimizers(model, args)

    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    n_bitlinear = sum(
        int(np.prod(p.shape)) for name, p in tree_flatten(model.parameters())
        if is_bitlinear_weight(name) and p.ndim == 2
    )
    log(f"model_params:{n_params} bitlinear_params:{n_bitlinear} "
        f"dim:{args.model_dim} layers:{args.num_layers} heads:{args.num_heads}/{args.num_kv_heads}")
    log(f"estimated_ternary_bytes:{n_bitlinear * 2 // 8:,} "
        f"estimated_baseline_bytes:{n_params:,} (int8 pre-zlib)")
    log(f"iterations:{args.iterations} val_loss_every:{args.val_loss_every} seq_len:{args.train_seq_len}")

    # Warmup
    if args.warmup_steps > 0:
        for ws in range(args.warmup_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            mx.eval(loss, grads)
            mx.synchronize()
        log(f"warmup done ({args.warmup_steps} steps)")
        train_loader = TokenLoader(args.train_files)

    # Training loop
    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            val_loss, val_bpb = eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms")
            t0 = time.perf_counter()

        if last_step:
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_val = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()

        approx_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        step += 1

        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            log(f"step:{step}/{args.iterations} train_loss:{train_loss_val:.4f} "
                f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")

        if max_wallclock_ms is not None and stop_after_step is None and approx_ms >= max_wallclock_ms:
            stop_after_step = step

    # Serialize + ternary roundtrip eval
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    quant_obj, quant_stats = quantize_bitnet_state_dict(flat_state)
    raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(raw, level=9)
    code_bytes = len(Path(__file__).read_text(encoding="utf-8").encode("utf-8"))

    quant_path = out_dir / f"{args.run_id}_bitnet_model.ptz"
    quant_path.write_bytes(compressed)

    log(f"ternary_params:{quant_stats['total_params']} "
        f"ternary_weight_bytes_raw:{quant_stats['ternary_bytes']:,} "
        f"passthrough_bytes:{quant_stats['passthrough_bytes']:,}")
    log(f"compressed_model_bytes:{len(compressed):,} code_bytes:{code_bytes:,} "
        f"total_submission_bytes:{len(compressed) + code_bytes:,}")
    log(f"16MB_limit:16000000 headroom:{16_000_000 - len(compressed) - code_bytes:,}")

    # Roundtrip: decompress, load ternary weights, re-evaluate
    loaded = pickle.loads(zlib.decompress(quant_path.read_bytes()))
    recovered = dequantize_bitnet_state_dict(loaded)
    model.update(tree_unflatten(list(recovered.items())))
    q_val_loss, q_val_bpb = eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log(f"final_ternary_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    log(f"final_ternary_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
