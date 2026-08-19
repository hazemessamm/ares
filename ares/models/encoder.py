from torch import nn
from typing import Optional, Tuple

from ares.models.attention import MultiHeadAttention
from ares.models.layers import get_norm, FeedForward
from ares.models.expert_choice_router import ExpertChoiceRouting
import torch
from ares.models.config import AresConfig
from ares.models.soft_router import SoftRouter
import logging

logger = logging.getLogger("ares")


class EncoderLayer(nn.Module):
    def __init__(
        self,
        config: AresConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout=config.attn_dropout,
            bias=config.bias,
            capping_value=config.attn_capping_value,
            qk_norm=config.qk_norm,
        )

        self._using_moe = False

        if config.moe_interleaved:
            if layer_idx % 2 == 0:
                self.ff = FeedForward(
                    embed_dim=config.embed_dim,
                    ff_dim=config.ff_dim,
                    bias=config.bias,
                    activation=config.activation,
                    gated=config.gated,
                    ff_norm_type=config.ff_norm_type,
                )
            else:
                if config.moe_type == "expert_choice":
                    self.ff = ExpertChoiceRouting(
                        embed_dim=config.embed_dim,
                        num_experts=config.num_experts,
                        expert_capacity_factor=config.expert_capacity_factor,
                        ff_dim=config.ff_dim,
                        activation=config.activation,
                        gated=config.gated,
                        bias=config.bias,
                        noise_scale=config.moe_noise_level,
                        norm_type=config.ff_norm_type,
                    )
                elif config.moe_type == "soft_router":
                    self.ff = SoftRouter(
                        embed_dim=config.embed_dim,
                        num_experts=config.num_experts,
                        num_slots=config.moe_num_slots,
                        ff_dim=config.ff_dim,
                        activation=config.activation,
                        gated=config.gated,
                        ff_norm_type=config.ff_norm_type,
                        normalize=config.moe_normalize,
                        noise_scale=config.moe_noise_level,
                        bias=config.bias,
                    )
                else:
                    raise ValueError(f"Invalid MOE type: {config.moe_type}")
                self._using_moe = True
        else:
            if (
                config.moe_after_num_layers is not None
                and layer_idx >= config.moe_after_num_layers
            ):  # noqa
                if config.moe_type == "expert_choice":
                    self.ff = ExpertChoiceRouting(
                        embed_dim=config.embed_dim,
                        num_experts=config.num_experts,
                        expert_capacity_factor=config.expert_capacity_factor,
                        ff_dim=config.ff_dim,
                        activation=config.activation,
                        gated=config.gated,
                        bias=config.bias,
                        noise_scale=config.moe_noise_level,
                        norm_type=config.ff_norm_type,
                    )
                elif config.moe_type == "soft_router":
                    self.ff = SoftRouter(
                        embed_dim=config.embed_dim,
                        num_experts=config.num_experts,
                        num_slots=config.moe_num_slots,
                        ff_dim=config.ff_dim,
                        activation=config.activation,
                        gated=config.gated,
                        ff_norm_type=config.ff_norm_type,
                        normalize=config.moe_normalize,
                        noise_scale=config.moe_noise_level,
                        bias=config.bias,
                    )
                else:
                    raise ValueError(f"Invalid MOE type: {config.moe_type}")
                self._using_moe = True
            else:
                self.ff = FeedForward(
                    embed_dim=config.embed_dim,
                    ff_dim=config.ff_dim,
                    bias=config.bias,
                    activation=config.activation,
                    gated=config.gated,
                    ff_norm_type=config.ff_norm_type,
                )

        self.pre_attention_norm = get_norm(
            config.norm_type,
            config.embed_dim,
        )
        self.post_attention_norm = get_norm(
            config.norm_type,
            config.embed_dim,
        )

    def forward(
        self,
        query: torch.FloatTensor,
        padding_mask: Optional[torch.LongTensor] = None,
        attention_bias: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cos_sin_emb: Optional[
            Tuple[torch.FloatTensor, torch.FloatTensor]
        ] = None,
    ):
        ff_kwargs = {"padding_mask": padding_mask} if self._using_moe else {}
        x = query
        x = x + self.self_attention(
            q=self.pre_attention_norm(x),
            attention_bias=attention_bias,
            position_ids=position_ids,
            cos_sin_emb=cos_sin_emb,
        )
        x = x + self.ff(self.post_attention_norm(x), **ff_kwargs)
        return x
