from __future__ import annotations
import einops
import torch
from typing import Optional, Tuple
from torch import nn
import math
from ares.models.utils import soft_capping
from ares.models.layers import non_trainable_rms_norm
from ares.models.rotary import apply_rotary_pos_emb


def scaled_dot_product_attention(
    q,
    k,
    v,
    dropout_p=0.0,
    attn_bias=None,
    capping_value=None,
    training=True,
):
    scaling = q.shape[-1] ** -0.5
    q = q * scaling
    attn_weights = torch.matmul(q, k.transpose(-2, -1))

    if capping_value is not None and capping_value > 0.0:
        attn_weights = soft_capping(attn_weights, capping_value=capping_value)

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
    q,
    k,
    v,
    dropout_p=0.0,
    attn_bias=None,
    capping_value=None,
    training=True,
):
    scaling = q.shape[-1] ** -0.5
    q = q * scaling
    attn_weights = torch.matmul(
        q,
        k.unsqueeze(2).transpose(-2, -1),
    )

    if capping_value is not None and capping_value > 0.0:
        attn_weights = soft_capping(attn_weights, capping_value=capping_value)

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
    """
    Repeat K/V tensors along the head dimension for grouped-query attention.
    """
    k = torch.repeat_interleave(k, num_groups, dim=1)
    v = torch.repeat_interleave(v, num_groups, dim=1)
    return k, v


class MultiHeadAttention(nn.Module):
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

    def reset_parameters(self):
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
        if self.qk_norm:
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
