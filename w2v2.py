"""wav2vec 2.0 XLS-R 300m backbone (no fairseq dependency), batched + trainable.

The modules mirror the fairseq module and parameter names, so the released
checkpoint state dict loads directly. Three properties matter for fine-tuning:
(a) `_Wav2Vec2SSL.forward` takes `n_samples` [B] and builds the frame padding
mask; (b) the encoder zeroes padded frames with `masked_fill` (autograd-safe,
the in-place `x[mask] = 0` breaks backward through a trainable encoder);
(c) optional per-block activation checkpointing (`encoder.grad_ckpt`, set by
features.W2V2Front from cfg.grad_ckpt). `load_xlsr` returns an all-frozen
eval-mode module; the trainable subset (the encoder) is chosen afterwards in
`features.set_trainable`.
Conv extractor: 7 strided convs, total stride 320 samples = 20 ms, receptive
field 400 samples, no padding, so frame t covers samples [320t, 320t+400) and
T' = (n_samples-400)//320 + 1. `layer=24` (default) = output after the final
encoder LayerNorm; `0 <= layer < 24` = hidden state after transformer layer
`layer` (0 = input to layer 1, i.e. after the pos_conv addition), without the
final LN.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PRETRAIN_ONLY = ("quantizer", "project_q", "final_proj", "target_glu",
                 "label_embs_concat", "input_quantizer", "project_inp")


class _TransposeLast(nn.Module):
    def forward(self, x):
        return x.transpose(-1, -2)


class _Fp32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        out = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return out.type_as(x)


class _SamePad(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


class _MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, key_padding_mask=None):
        # x: [T, B, C]
        tgt_len, bsz, _ = x.size()
        q = self.q_proj(x).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(x).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(x).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            attn = attn.view(bsz, self.num_heads, tgt_len, tgt_len)
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn = attn.view(bsz * self.num_heads, tgt_len, tgt_len)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).type_as(q)
        out = torch.bmm(attn, v)
        out = out.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        return self.out_proj(out)


class _TransformerSentenceEncoderLayer(nn.Module):
    """One transformer block (pre-LN / layer_norm_first variant used by XLS-R)."""

    def __init__(self, embed_dim: int, ffn_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = _MultiheadAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = _Fp32LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = _Fp32LayerNorm(embed_dim)

    def forward(self, x, padding_mask=None):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, key_padding_mask=padding_mask)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


class _ConvFeatureExtractionModel(nn.Module):
    """Strided 1D conv stack on raw audio. `layer_norm` mode + conv_bias=True
    matches the XLS-R checkpoint."""

    def __init__(self, conv_layers, conv_bias: bool = True):
        super().__init__()

        def block(n_in, n_out, k, stride):
            conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
            nn.init.kaiming_normal_(conv.weight)
            return nn.Sequential(
                conv,
                nn.Dropout(p=0.0),
                nn.Sequential(_TransposeLast(), _Fp32LayerNorm(n_out, elementwise_affine=True), _TransposeLast()),
                nn.GELU(),
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
        for conv in self.conv_layers:
            x = conv(x)
        return x


def _make_conv_pos(embed_dim: int, kernel_size: int, groups: int) -> nn.Module:
    conv = nn.Conv1d(
        embed_dim, embed_dim, kernel_size=kernel_size,
        padding=kernel_size // 2, groups=groups,
    )
    std = math.sqrt(4.0 / (kernel_size * embed_dim))
    nn.init.normal_(conv.weight, mean=0.0, std=std)
    nn.init.constant_(conv.bias, 0.0)
    conv = nn.utils.weight_norm(conv, name="weight", dim=2)
    return nn.Sequential(conv, _SamePad(kernel_size), nn.GELU())


def _pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -2) -> Tuple[torch.Tensor, int]:
    seqlen = x.size(dim)
    rem = seqlen % multiple
    if rem == 0:
        return x, 0
    pad = multiple - rem
    pad_offset = [0] * (-1 - dim) * 2 + [0, pad]
    return F.pad(x, pad_offset, value=0), pad


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_layers: int,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        required_seq_len_multiple: int = 2,
    ):
        super().__init__()
        self.required_seq_len_multiple = required_seq_len_multiple
        self.pos_conv = _make_conv_pos(embed_dim, conv_pos, conv_pos_groups)
        self.layers = nn.ModuleList([
            _TransformerSentenceEncoderLayer(embed_dim, ffn_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.layer_norm = _Fp32LayerNorm(embed_dim)
        self.grad_ckpt = False          # per-block activation checkpointing (training only)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None,
                layer: Optional[int] = None):
        # x: [B, T, C]; padding_mask [B, T] bool, True = padded; layer None or
        # == len(self.layers): final-LN output; 0 <= layer < len(self.layers):
        # hidden state after block `layer`, no final LN
        n = len(self.layers)
        layer = n if layer is None else layer
        assert 0 <= layer <= n, f"layer {layer} not in [0, {n}]"
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)   # out of place: autograd-safe

        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)

        x, pad_len = _pad_to_multiple(x, self.required_seq_len_multiple, dim=-2)
        if pad_len > 0:
            if padding_mask is None:
                padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            else:
                padding_mask, _ = _pad_to_multiple(padding_mask, self.required_seq_len_multiple, dim=-1)
            padding_mask[:, -pad_len:] = True

        x = x.transpose(0, 1)  # [B, T, C] -> [T, B, C]
        ckpt = self.grad_ckpt and torch.is_grad_enabled()
        for blk in self.layers[:layer]:
            if ckpt:
                x = checkpoint(blk, x, padding_mask, use_reentrant=False)
            else:
                x = blk(x, padding_mask=padding_mask)
        x = x.transpose(0, 1)  # back to [B, T, C]

        # layer_norm_first: final LN is applied after the stack
        if layer == n:
            x = self.layer_norm(x)

        if pad_len > 0:
            x = x[:, :-pad_len]
        return x


class _Wav2Vec2SSL(nn.Module):
    """Inference-only XLS-R 300m backbone. Submodule names mirror the fairseq
    checkpoint so its state dict loads directly."""

    def __init__(self):
        super().__init__()
        conv_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
        self.conv_geom = [(k, s) for _, k, s in conv_layers]   # (kernel, stride) per conv
        self.feature_extractor = _ConvFeatureExtractionModel(conv_layers, conv_bias=True)
        self.post_extract_proj = nn.Linear(512, 1024)
        self.layer_norm = _Fp32LayerNorm(512)
        # checkpoint stores a `mask_emb` parameter (unused at inference)
        self.mask_emb = nn.Parameter(torch.FloatTensor(1024).uniform_())
        self.encoder = _TransformerEncoder(
            embed_dim=1024, ffn_dim=4096, num_heads=16, num_layers=24,
            conv_pos=128, conv_pos_groups=16, required_seq_len_multiple=2,
        )

    def frames(self, n_samples: torch.Tensor) -> torch.Tensor:
        """Sample counts [B] -> conv-extractor frame counts [B]; composes to
        (n - 400)//320 + 1 for the XLS-R geometry."""
        t = n_samples
        for k, s in self.conv_geom:
            t = torch.div(t - k, s, rounding_mode="floor") + 1
        return t

    def forward(self, source: torch.Tensor, layer: int = 24,
                n_samples: Optional[torch.Tensor] = None) -> torch.Tensor:
        # source: [B, N] raw waveform at 16kHz (already per-utt standardized,
        # zero-padded past n_samples[i]); n_samples LongTensor [B] or None (no mask)
        feats = self.feature_extractor(source)        # [B, 512, T'_max]
        feats = feats.transpose(1, 2)                 # [B, T'_max, 512]
        feats = self.layer_norm(feats)
        feats = self.post_extract_proj(feats)         # [B, T'_max, 1024]
        padding_mask = None
        if n_samples is not None:                     # True = padded frame t >= T'_i
            t = torch.arange(feats.shape[1], device=feats.device)
            padding_mask = t[None] >= self.frames(n_samples.to(feats.device))[:, None]
        return self.encoder(feats, padding_mask=padding_mask, layer=layer)   # [B, T'_max, 1024]


def load_xlsr(path: str) -> _Wav2Vec2SSL:
    """Fairseq XLS-R checkpoint -> frozen eval-mode _Wav2Vec2SSL (fp32, cpu).
    Only fairseq pretraining-only keys may be unexpected; no key may be missing."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    m = _Wav2Vec2SSL()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in unexpected if not k.startswith(PRETRAIN_ONLY)]
    if missing or bad:
        raise RuntimeError(f"{path}: missing {missing} unexpected {bad}")
    return m.eval().requires_grad_(False)


if __name__ == "__main__":  # load test + stride check on a random 1 s input
    import sys                                              # argv[1] = xlsr2_300m.pt
    sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["model"]
    m = load_xlsr(sys.argv[1])
    own = set(m.state_dict())
    print(f"ckpt keys {len(sd)}  loaded {len(own & set(sd))}  "
          f"ignored {len(set(sd) - own)}  params {sum(p.numel() for p in m.parameters())/1e6:.2f} M")
    x = torch.randn(1, 16000)
    x = F.layer_norm(x, x.shape[1:])
    with torch.no_grad():
        y = m(x)
    print(f"in {tuple(x.shape)} -> out {tuple(y.shape)}  "
          f"(16000-400)//320+1 == T': {(16000 - 400)//320 + 1 == y.shape[1]}")
    # batched (zero-padded + n_samples mask) vs single forward, cpu fp32, loaded ckpt
    torch.manual_seed(0)
    ns = [16000, 27200]                                     # 1.0 s and 1.7 s
    xs = [F.layer_norm(torch.randn(n), (n,)) for n in ns]
    X = torch.zeros(2, max(ns))
    for i, xi in enumerate(xs):
        X[i, :len(xi)] = xi
    with torch.no_grad():
        yb = m(X, n_samples=torch.tensor(ns))               # [2, T'_max, 1024]
        for i, xi in enumerate(xs):
            yi = m(xi[None])[0]                             # [T'_i, 1024]
            Ti = yi.shape[0]
            print(f"item {i}: n {ns[i]} T' {Ti} (mask says {int(m.frames(torch.tensor([ns[i]]))[0])})  "
                  f"batched-vs-single max abs diff on valid frames "
                  f"{(yb[i, :Ti] - yi).abs().max().item():.3e}  median |y| {yi.abs().median().item():.3e}")
