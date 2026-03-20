"""
BitNet experiment for parameter golf — PyTorch/CUDA version for Colab/RunPod.

Setup in Colab:
    !git clone https://github.com/openai/parameter-golf.git
    %cd parameter-golf
    !pip install sentencepiece
    !python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
    !python train_bitnet_colab.py

Quick test (200 steps, ~2 min on T4):
    ITERATIONS=200 VAL_LOSS_EVERY=50 TRAIN_BATCH_TOKENS=65536 python train_bitnet_colab.py

Compare against baseline:
    ITERATIONS=200 VAL_LOSS_EVERY=50 TRAIN_BATCH_TOKENS=65536 torchrun --standalone --nproc_per_node=1 train_gpt.py

What this tests:
    - dim=512, same as baseline, to verify BitNet converges at all
    - Once confirmed: set MODEL_DIM=768 to test the real bet (4x more params, same compressed size)
"""
from __future__ import annotations

import glob
import io
import math
import os
import random
import struct
import sys
import time
import uuid
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    # Repeat the same num_layers blocks this many times in the forward pass.
    # num_recurrent_repeats=1 = normal (no recurrence).
    # num_recurrent_repeats=3 = 3x effective depth, same parameters, same compressed size.
    num_recurrent_repeats = int(os.environ.get("NUM_RECURRENT_REPEATS", 1))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

# ==============================================================================
# MUON OPTIMIZER (same as baseline)
# ==============================================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(momentum).add_(g)
                g_eff = g.add(buf, alpha=momentum) if nesterov else buf
                g_eff = zeropower_via_newtonschulz5(g_eff, steps=backend_steps)
                g_eff *= max(1, g_eff.size(0) / g_eff.size(1)) ** 0.5
                p.add_(g_eff.to(dtype=p.dtype), alpha=-lr)

# ==============================================================================
# BITLINEAR — ternary weights with straight-through estimator
# ==============================================================================

class BitLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1}.
    Full-precision latent weights are maintained for the optimizer.
    During forward: absmean scale, round to {-1,0,+1}, straight-through for grads.
    At save time: pack 4 values per byte (2 bits each) — 4x smaller than int8.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.frozen = False  # set True after loading recovered ternary weights
        if bias:
            raise NotImplementedError("BitLinear does not support bias")

    def forward(self, x: Tensor) -> Tensor:
        if self.frozen:
            # Weights are already exactly {-1,0,+1}*scale — use directly, no re-quantization
            return F.linear(x, self.weight.to(x.dtype))
        w = self.weight.float()
        scale = w.abs().mean() + 1e-5
        w_norm = w / scale
        # Quantize to {-1, 0, +1}
        w_q = w_norm.round().clamp(-1, 1)
        # Straight-through estimator: gradient flows as if no quantization
        w_eff = w_norm + (w_q - w_norm).detach()
        return F.linear(x, (w_eff * scale).to(x.dtype))

# ==============================================================================
# MODEL
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype), self._sin_cached.to(dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = BitLinear(dim, dim)
        self.c_k = BitLinear(dim, kv_dim)
        self.c_v = BitLinear(dim, kv_dim)
        self.proj = BitLinear(dim, dim)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))
        return self.proj(y.transpose(1, 2).contiguous().reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        self.fc = BitLinear(dim, dim * mlp_mult)
        self.proj = BitLinear(dim * mlp_mult, dim)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))
        self.resid_mix = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        x = x + self.attn_scale.to(x.dtype)[None, None, :] * self.attn(self.attn_norm(x))
        x = x + self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, args: Hyperparameters):
        super().__init__()
        self.logit_softcap = args.logit_softcap
        self.num_recurrent_repeats = args.num_recurrent_repeats
        self.tok_emb = nn.Embedding(args.vocab_size, args.model_dim)
        self.num_encoder_layers = args.num_layers // 2
        self.num_decoder_layers = args.num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, args.model_dim))
        self.blocks = nn.ModuleList([
            Block(args.model_dim, args.num_heads, args.num_kv_heads, args.mlp_mult, args.rope_base, args.qk_gain_init)
            for _ in range(args.num_layers)
        ])
        self.final_norm = RMSNorm()
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=args.tied_embed_init_std)
        for b in self.blocks:
            nn.init.zeros_(b.attn.proj.weight)
            nn.init.zeros_(b.mlp.proj.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = F.rms_norm(self.tok_emb(input_ids), (self.tok_emb.embedding_dim,))
        x0 = x
        # Run the same blocks num_recurrent_repeats times.
        # Each repeat: encoder half stores skips, decoder half consumes them.
        # Skip connections are re-used across repeats (same weights).
        for _ in range(self.num_recurrent_repeats):
            skips = []
            for i in range(self.num_encoder_layers):
                x = self.blocks[i](x, x0)
                skips.append(x)
            for i in range(self.num_decoder_layers):
                if skips:
                    x = x + self.skip_weights[i].to(x.dtype)[None, None, :] * skips.pop()
                x = self.blocks[self.num_encoder_layers + i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        logits = F.linear(x, self.tok_emb.weight)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return F.cross_entropy(logits.float(), target_ids.reshape(-1))

# ==============================================================================
# TERNARY COMPRESSION — 2 bits per weight, 4 values per byte
# ==============================================================================

BITLINEAR_WEIGHT_KEYS = ("c_q.weight", "c_k.weight", "c_v.weight", "attn.proj.weight", "mlp.fc.weight", "mlp.proj.weight")

def is_bitlinear_weight(name: str) -> bool:
    return any(name.endswith(k) for k in BITLINEAR_WEIGHT_KEYS)


def pack_ternary(arr: np.ndarray) -> tuple[np.ndarray, float, tuple, int]:
    shape = arr.shape
    flat = arr.flatten().astype(np.float32)
    n = len(flat)
    scale = float(np.mean(np.abs(flat))) + 1e-5
    q = np.clip(np.round(flat / scale), -1, 1).astype(np.int8)
    u = (q + 1).astype(np.uint8)  # {-1,0,1} -> {0,1,2}
    pad = (4 - n % 4) % 4
    if pad:
        u = np.pad(u, (0, pad))
    packed = (u[0::4] | (u[1::4] << 2) | (u[2::4] << 4) | (u[3::4] << 6)).astype(np.uint8)
    return packed, scale, shape, n


def unpack_ternary(packed: np.ndarray, scale: float, shape: tuple, n: int) -> np.ndarray:
    u = np.empty(len(packed) * 4, dtype=np.uint8)
    u[0::4] = packed & 0x03
    u[1::4] = (packed >> 2) & 0x03
    u[2::4] = (packed >> 4) & 0x03
    u[3::4] = (packed >> 6) & 0x03
    q = u[:n].astype(np.float32) - 1.0  # {0,1,2} -> {-1,0,1}
    return (q * scale).reshape(shape)


def quantize_bitnet_state_dict(state_dict: dict[str, Tensor]) -> tuple[dict, dict]:
    ternary, passthrough, scales_fp16 = {}, {}, {}
    stats = {"ternary_params": 0, "passthrough_params": 0, "ternary_bytes": 0, "passthrough_bytes": 0}
    for name, t in state_dict.items():
        arr = t.detach().cpu().float().numpy()
        if is_bitlinear_weight(name) and arr.ndim == 2:
            packed, scale, shape, n = pack_ternary(arr)
            ternary[name] = packed
            scales_fp16[name] = np.float16(scale)
            stats["ternary_params"] += n
            stats["ternary_bytes"] += packed.nbytes + 2  # 2 bytes for fp16 scale
        else:
            pt = arr.astype(np.float16)
            passthrough[name] = pt
            stats["passthrough_params"] += arr.size
            stats["passthrough_bytes"] += pt.nbytes
    obj = {"ternary": ternary, "scales": scales_fp16, "passthrough": passthrough}
    return obj, stats


def dequantize_bitnet_state_dict(obj: dict) -> dict[str, Tensor]:
    out = {}
    for name, packed in obj["ternary"].items():
        scale = float(obj["scales"][name])
        # Shape and n were stored implicitly — recover from packed size
        # We store shape separately; here we need a workaround:
        # Actually we need to store shape. Let's handle below.
        out[name] = torch.from_numpy(packed.astype(np.float32))  # placeholder, fixed below
    for name, arr in obj["passthrough"].items():
        out[name] = torch.from_numpy(arr.astype(np.float32))
    return out


# Fixed version that stores shape+n
def quantize_bitnet_state_dict_v2(state_dict: dict[str, Tensor]) -> tuple[dict, dict]:
    ternary_packed, ternary_meta, passthrough = {}, {}, {}
    stats = {"ternary_params": 0, "passthrough_params": 0, "ternary_bytes": 0, "passthrough_bytes": 0}
    for name, t in state_dict.items():
        arr = t.detach().cpu().float().numpy()
        if is_bitlinear_weight(name) and arr.ndim == 2:
            packed, scale, shape, n = pack_ternary(arr)
            ternary_packed[name] = packed
            ternary_meta[name] = {"scale": float(scale), "shape": list(shape), "n": n}
            stats["ternary_params"] += n
            stats["ternary_bytes"] += packed.nbytes + 16
        else:
            pt = arr.astype(np.float16)
            passthrough[name] = pt
            stats["passthrough_params"] += arr.size
            stats["passthrough_bytes"] += pt.nbytes
    obj = {"ternary_packed": ternary_packed, "ternary_meta": ternary_meta, "passthrough": passthrough}
    return obj, stats


def dequantize_bitnet_state_dict_v2(obj: dict) -> dict[str, Tensor]:
    out = {}
    for name, packed in obj["ternary_packed"].items():
        meta = obj["ternary_meta"][name]
        arr = unpack_ternary(packed, meta["scale"], tuple(meta["shape"]), meta["n"])
        out[name] = torch.from_numpy(arr).float()
    for name, arr in obj["passthrough"].items():
        out[name] = torch.from_numpy(arr.astype(np.float32))
    return out

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=256 * 4)
    return torch.from_numpy(tokens.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self.file_idx = (self.file_idx + 1) % len(self.files)
                self.tokens = load_data_shard(self.files[self.file_idx])
                self.pos = 0
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos: self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def next_batch(stream: TokenStream, total_tokens: int, seq_len: int, device: torch.device):
    chunk = stream.take(total_tokens + 1)
    x = chunk[:-1].reshape(-1, seq_len).to(device=device, dtype=torch.int64)
    y = chunk[1:].reshape(-1, seq_len).to(device=device, dtype=torch.int64)
    return x, y


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No val files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files])
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]

# ==============================================================================
# VALIDATION
# ==============================================================================

def build_sentencepiece_luts(sp, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros(table_size, dtype=np.int16)
    has_leading_space_np = np.zeros(table_size, dtype=np.bool_)
    is_boundary_token_np = np.ones(table_size, dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def eval_val(args, model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut):
    local_batch_seqs = args.val_batch_size // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for seq_start in range(0, total_seqs, local_batch_seqs):
            seq_end = min(seq_start + local_batch_seqs, total_seqs)
            raw_start = seq_start * args.train_seq_len
            raw_end = seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y).detach()
            ct = float(y.numel())
            val_loss_sum += loss.to(torch.float64) * ct
            val_token_count += ct
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
    val_loss = float((val_loss_sum / val_token_count).item())
    val_bpb = float((val_loss / math.log(2.0)) * (val_token_count / val_byte_count).item())
    model.train()
    return val_loss, val_bpb

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — run on Colab/RunPod with GPU runtime")

    args = Hyperparameters()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{args.run_id}_bitnet.txt"
    print(logfile)

    def log(msg: str):
        print(msg)
        with open(logfile, "a") as f:
            print(msg, file=f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f"VOCAB_SIZE mismatch: {args.vocab_size} vs tokenizer {int(sp.vocab_size())}")

    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)

    # Build model
    model = GPT(args).to(device).bfloat16()
    # BitLinear latent weights stay in float32 for optimizer quality
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.weight.data = m.weight.data.float()
    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)

    # Optimizer split: BitLinear weights (2D matrices) -> Muon, rest -> Adam
    CONTROL_PATTERNS = ("attn_scale", "mlp_scale", "resid_mix", "q_gain", "skip_weight")
    block_params = list(model.blocks.named_parameters())
    matrix_params = [p for name, p in block_params if p.ndim == 2 and not any(pat in name for pat in CONTROL_PATTERNS)]
    scalar_params = [p for name, p in block_params if p.ndim < 2 or any(pat in name for pat in CONTROL_PATTERNS)]
    if model.skip_weights.numel() > 0:
        scalar_params.append(model.skip_weights)

    opt_tok = torch.optim.Adam([{"params": [model.tok_emb.weight], "lr": args.tied_embed_lr, "base_lr": args.tied_embed_lr}],
                                betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opt_muon = Muon(matrix_params, lr=args.matrix_lr, momentum=args.muon_momentum, backend_steps=args.muon_backend_steps)
    for g in opt_muon.param_groups:
        g["base_lr"] = args.matrix_lr
    opt_scalar = torch.optim.Adam([{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
                                   betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizers = [opt_tok, opt_muon, opt_scalar]

    n_params = sum(p.numel() for p in model.parameters())
    n_ternary = sum(p.numel() for name, p in model.named_parameters() if is_bitlinear_weight(name))
    effective_layers = args.num_layers * args.num_recurrent_repeats
    log(f"model_params:{n_params} ternary_params:{n_ternary} dim:{args.model_dim} "
        f"layers:{args.num_layers} repeats:{args.num_recurrent_repeats} effective_layers:{effective_layers}")
    log(f"estimated_ternary_bytes:{n_ternary * 2 // 8:,} vs int8_bytes:{n_params:,}")
    log(f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} seq_len:{args.train_seq_len}")

    train_stream = TokenStream(args.train_files)
    grad_scale = 1.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            ws = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if ws <= step else 1.0
        step_ms = elapsed_ms / max(step, 1)
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        warmdown_ms = args.warmdown_iters * step_ms
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    def zero_grad():
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    # Warmup
    if args.warmup_steps > 0:
        model.train()
        for ws in range(args.warmup_steps):
            zero_grad()
            x, y = next_batch(train_stream, args.train_batch_tokens, args.train_seq_len, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = compiled_model(x, y)
            loss.backward()
            for opt in optimizers:
                opt.step()
            zero_grad()
            if (ws + 1) % 5 == 0 or ws + 1 == args.warmup_steps:
                log(f"warmup_step:{ws+1}/{args.warmup_steps}")
        train_stream = TokenStream(args.train_files)

    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(args, compiled_model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
            log(f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms")
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        zero_grad()
        x, y = next_batch(train_stream, args.train_batch_tokens, args.train_seq_len, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compiled_model(x, y)
        loss.backward()

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        opt_muon.param_groups[0]["momentum"] = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g.get("base_lr", g["lr"]) * scale
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad()

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if args.train_log_every > 0 and (step <= 5 or step % args.train_log_every == 0):
            log(f"step:{step}/{args.iterations} train_loss:{loss.item():.4f} "
                f"train_time:{approx_ms:.0f}ms step_avg:{approx_ms / step:.2f}ms")
        if max_wallclock_ms is not None and stop_after_step is None and approx_ms >= max_wallclock_ms:
            stop_after_step = step

    # Serialize with ternary compression
    quant_obj, quant_stats = quantize_bitnet_state_dict_v2(model.state_dict())
    buf = io.BytesIO()
    import pickle
    buf.write(pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL))
    compressed = zlib.compress(buf.getvalue(), level=9)
    code_bytes = len(Path(__file__).read_text(encoding="utf-8").encode("utf-8"))

    with open("final_bitnet_model.ptz", "wb") as f:
        f.write(compressed)

    log(f"ternary_params:{quant_stats['ternary_params']:,} passthrough_params:{quant_stats['passthrough_params']:,}")
    log(f"ternary_raw_bytes:{quant_stats['ternary_bytes']:,} passthrough_bytes:{quant_stats['passthrough_bytes']:,}")
    log(f"compressed_model_bytes:{len(compressed):,} code_bytes:{code_bytes:,} "
        f"total:{len(compressed) + code_bytes:,} limit:16000000 "
        f"headroom:{16_000_000 - len(compressed) - code_bytes:,}")

    # Roundtrip: load ternary weights back, re-evaluate
    with open("final_bitnet_model.ptz", "rb") as f:
        recovered_obj = pickle.loads(zlib.decompress(f.read()))
    recovered_state = dequantize_bitnet_state_dict_v2(recovered_obj)
    model.load_state_dict(recovered_state, strict=True)
    # Freeze all BitLinear layers so forward() uses recovered weights directly
    # without re-quantizing (weights are already exactly {-1,0,+1}*scale)
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.frozen = True
    torch.cuda.synchronize()
    q_val_loss, q_val_bpb = eval_val(args, compiled_model, device, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
    log(f"final_ternary_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    log(f"final_ternary_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
