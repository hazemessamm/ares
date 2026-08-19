from __future__ import annotations

import math
from typing import Optional, Tuple

import einops
import torch
import torch.nn.functional as F
from torch import nn


def soft_capping(x: torch.Tensor, capping_value: float) -> torch.Tensor:
    return capping_value * torch.tanh(x / capping_value)


def non_trainable_rms_norm(
    x: torch.FloatTensor,
    eps: float = 1e-6,
) -> torch.FloatTensor:
    return F.rms_norm(
        x,
        normalized_shape=(x.shape[-1],),
        eps=eps,
        weight=None,
    )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base_frequency: int = 10000,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale
        inv_freq = 1.0 / (
            base_frequency ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _compute_cos_sin(
        self,
        bsz: int,
        seq_length: int,
        device: torch.device,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            t = (
                torch.arange(
                    seq_length,
                    device=device,
                    dtype=self.inv_freq.dtype,
                )
                .unsqueeze(0)
                .expand(bsz, -1)
            )
        else:
            t = position_ids.to(dtype=self.inv_freq.dtype, device=device)

        t = t * self.scale
        freqs = torch.einsum("bi,j->bij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        cos_emb = emb.cos()[:, None, :, :]
        sin_emb = emb.sin()[:, None, :, :]
        return cos_emb, sin_emb

    def forward(
        self,
        q: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = q.shape[0]
        seq_length = q.shape[1]
        device = q.device
        return self._compute_cos_sin(
            bsz,
            seq_length,
            device,
            position_ids=position_ids,
        )


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    attn_bias: Optional[torch.Tensor] = None,
    capping_value: Optional[float] = None,
    training: bool = True,
) -> torch.Tensor:
    scaling = q.shape[-1] ** -0.5
    q = q * scaling
    attn_weights = torch.matmul(q, k.transpose(-2, -1))

    if capping_value is not None and capping_value > 0.0:
        attn_weights = soft_capping(
            attn_weights,
            capping_value=capping_value,
        )

    if attn_bias is not None:
        attn_weights = attn_weights + attn_bias

    weights_dtype = attn_weights.dtype
    attn_weights = nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    )
    attn_weights = nn.functional.dropout(
        attn_weights,
        dropout_p,
        training=training,
    )
    attn_weights = attn_weights.to(weights_dtype)
    return attn_weights @ v


def grouped_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    attn_bias: Optional[torch.Tensor] = None,
    capping_value: Optional[float] = None,
    training: bool = True,
) -> torch.Tensor:
    scaling = q.shape[-1] ** -0.5
    q = q * scaling
    attn_weights = torch.matmul(
        q,
        k.unsqueeze(2).transpose(-2, -1),
    )

    if capping_value is not None and capping_value > 0.0:
        attn_weights = soft_capping(
            attn_weights,
            capping_value=capping_value,
        )

    if attn_bias is not None:
        attn_weights = attn_weights + attn_bias.unsqueeze(1)

    weights_dtype = attn_weights.dtype
    attn_weights = nn.functional.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32,
    )
    attn_weights = nn.functional.dropout(
        attn_weights,
        dropout_p,
        training=training,
    )
    attn_weights = attn_weights.to(weights_dtype)
    return attn_weights @ v.unsqueeze(2)


def repeat_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    num_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = torch.repeat_interleave(k, num_groups, dim=1)
    v = torch.repeat_interleave(v, num_groups, dim=1)
    return k, v


class GroupedKVAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        capping_value: float | None = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.qk_norm = qk_norm
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        self.dropout = dropout
        self.bias = bias
        self.head_dim = embed_dim // num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.capping_value = capping_value

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(
            embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=bias,
        )
        self.v_proj = nn.Linear(
            embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=bias,
        )
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = 1 / math.sqrt(2)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)

        if self.bias:
            nn.init.constant_(self.o_proj.bias, 0.0)
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        q: torch.FloatTensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cos_sin_emb: Optional[
            Tuple[torch.FloatTensor, torch.FloatTensor]
        ] = None,
    ) -> torch.FloatTensor:
        del position_ids
        xq = einops.rearrange(
            self.q_proj(q),
            "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        xk = einops.rearrange(
            self.k_proj(q),
            "b t (h d) -> b h t d",
            h=self.num_kv_heads,
        )
        xv = einops.rearrange(
            self.v_proj(q),
            "b t (h d) -> b h t d",
            h=self.num_kv_heads,
        )

        if self.qk_norm:
            xq = non_trainable_rms_norm(xq)
            xk = non_trainable_rms_norm(xk)

        if cos_sin_emb is not None:
            cos_emb, sin_emb = cos_sin_emb
            xq = apply_rotary_pos_emb(xq, cos_emb, sin_emb)
            xk = apply_rotary_pos_emb(xk, cos_emb, sin_emb)
        else:
            raise ValueError("cos_sin_emb must be provided")

        if self.num_kv_groups > 1:
            xq = einops.rearrange(
                xq,
                "b (h g) t d -> b h g t d",
                h=self.num_kv_heads,
                g=self.num_kv_groups,
            )
            attn_output = grouped_scaled_dot_product_attention(
                q=xq,
                k=xk,
                v=xv,
                dropout_p=self.dropout if self.training else 0.0,
                attn_bias=attention_bias,
                capping_value=self.capping_value,
                training=self.training,
            )
            attn_output = einops.rearrange(
                attn_output,
                "b h g t d -> b t (h g d)",
            )
        else:
            attn_output = scaled_dot_product_attention(
                q=xq,
                k=xk,
                v=xv,
                dropout_p=self.dropout if self.training else 0.0,
                attn_bias=attention_bias,
                capping_value=self.capping_value,
                training=self.training,
            )
            attn_output = einops.rearrange(
                attn_output,
                "b h t d -> b t (h d)",
            )

        return self.o_proj(attn_output)


class RepeatKVAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        capping_value: float | None = None,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.qk_norm = qk_norm
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        self.dropout = dropout
        self.bias = bias
        self.head_dim = embed_dim // num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.capping_value = capping_value

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(
            embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=bias,
        )
        self.v_proj = nn.Linear(
            embed_dim,
            self.num_kv_heads * self.head_dim,
            bias=bias,
        )
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = 1 / math.sqrt(2)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)

        if self.bias:
            nn.init.constant_(self.o_proj.bias, 0.0)
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)

    def forward(
        self,
        q: torch.FloatTensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cos_sin_emb: Optional[
            Tuple[torch.FloatTensor, torch.FloatTensor]
        ] = None,
    ) -> torch.FloatTensor:
        del position_ids
        xq = einops.rearrange(
            self.q_proj(q),
            "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        xk = einops.rearrange(
            self.k_proj(q),
            "b t (h d) -> b h t d",
            h=self.num_kv_heads,
        )
        xv = einops.rearrange(
            self.v_proj(q),
            "b t (h d) -> b h t d",
            h=self.num_kv_heads,
        )

        if self.qk_norm:
            xq = non_trainable_rms_norm(xq)
            xk = non_trainable_rms_norm(xk)

        if cos_sin_emb is not None:
            cos_emb, sin_emb = cos_sin_emb
            xq = apply_rotary_pos_emb(xq, cos_emb, sin_emb)
            xk = apply_rotary_pos_emb(xk, cos_emb, sin_emb)
        else:
            raise ValueError("cos_sin_emb must be provided")

        if self.num_kv_groups > 1:
            xk, xv = repeat_kv(xk, xv, self.num_kv_groups)

        scaled_attention_bias = None
        if attention_bias is not None:
            scaled_attention_bias = attention_bias.unsqueeze(1)

        attn_output = scaled_dot_product_attention(
            q=xq,
            k=xk,
            v=xv,
            dropout_p=self.dropout if self.training else 0.0,
            attn_bias=scaled_attention_bias,
            capping_value=self.capping_value,
            training=self.training,
        )
        attn_output = einops.rearrange(
            attn_output,
            "b h t d -> b t (h d)",
        )
        return self.o_proj(attn_output)


def build_attention_bias(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    causal_mask = torch.full(
        (seq_len, seq_len),
        fill_value=torch.finfo(dtype).min,
        device=device,
        dtype=dtype,
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    return causal_mask.unsqueeze(0).expand(batch_size, -1, -1)


def compare_one_case(
    *,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_kv_heads: int,
    qk_norm: bool,
    capping_value: Optional[float],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    grouped_attn = GroupedKVAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
        bias=True,
        capping_value=capping_value,
        qk_norm=qk_norm,
    ).to(device=device, dtype=dtype)
    repeat_attn = RepeatKVAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dropout=0.0,
        bias=True,
        capping_value=capping_value,
        qk_norm=qk_norm,
    ).to(device=device, dtype=dtype)

    repeat_attn.load_state_dict(grouped_attn.state_dict())
    grouped_attn.eval()
    repeat_attn.eval()

    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
    rotary = RotaryEmbedding(dim=embed_dim // num_heads).to(device)
    cos_sin_emb = rotary(x)
    attention_bias = build_attention_bias(batch_size, seq_len, device, dtype)

    with torch.no_grad():
        grouped_out = grouped_attn(
            x,
            attention_bias=attention_bias,
            cos_sin_emb=cos_sin_emb,
        )
        repeated_out = repeat_attn(
            x,
            attention_bias=attention_bias,
            cos_sin_emb=cos_sin_emb,
        )

    max_abs_diff = (grouped_out - repeated_out).abs().max().item()
    allclose = torch.allclose(grouped_out, repeated_out, atol=1e-6, rtol=1e-5)

    print(
        "case:",
        {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "qk_norm": qk_norm,
            "capping_value": capping_value,
            "dtype": str(dtype),
            "max_abs_diff": max_abs_diff,
            "allclose": allclose,
        },
    )

    if not allclose:
        raise AssertionError(
            "Grouped KV attention and repeated KV attention diverged "
            f"(max_abs_diff={max_abs_diff})"
        )


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    cases = [
        {
            "batch_size": 2,
            "seq_len": 16,
            "embed_dim": 128,
            "num_heads": 8,
            "num_kv_heads": 2,
            "qk_norm": False,
            "capping_value": None,
        },
        {
            "batch_size": 2,
            "seq_len": 16,
            "embed_dim": 128,
            "num_heads": 8,
            "num_kv_heads": 2,
            "qk_norm": True,
            "capping_value": 30.0,
        },
        {
            "batch_size": 1,
            "seq_len": 11,
            "embed_dim": 96,
            "num_heads": 6,
            "num_kv_heads": 3,
            "qk_norm": True,
            "capping_value": 10.0,
        },
    ]

    for case in cases:
        compare_one_case(
            device=device,
            dtype=dtype,
            **case,
        )

    print("All grouped-KV and repeat-KV comparisons matched.")


if __name__ == "__main__":
    main()
