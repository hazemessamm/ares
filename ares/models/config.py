from __future__ import annotations

from transformers import PretrainedConfig
from typing import Union


class AresConfig(PretrainedConfig):
    model_type = "ares"

    def __init__(
        self,
        embed_dim: int = 1024,
        vocab_size: int = 30,
        num_heads: int = 16,
        num_layers: int = 12,
        moe_after_num_layers: Union[int, None] = 6,
        num_experts: int = 4,
        expert_capacity_factor: float = 2.0,
        ff_dim: int = 4096,
        activation: str = "silu",
        gated: bool = True,
        attn_dropout: float = 0.0,
        bias: bool = False,
        attn_capping_value: float | None = None,
        logits_capping_value: float | None = None,
        qk_norm: bool = False,
        norm_type: str = "rms",
        ff_norm_type: str | None = None,
        moe_noise_level: float = 0.05,
        moe_type: str = "expert_choice",
        num_kv_heads: int = 16,
        rope_frequency: int = 10000,
        rope_scale: float = 1.0,
        moe_normalize: bool = False,
        moe_num_slots: int = 16,
        moe_interleaved: bool = False,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_capacity_factor = expert_capacity_factor
        self.ff_dim = ff_dim
        self.activation = activation
        self.gated = gated
        self.attn_dropout = attn_dropout
        self.bias = bias
        self.norm_type = norm_type
        self.ff_norm_type = ff_norm_type
        self.moe_after_num_layers = moe_after_num_layers
        self.vocab_size = vocab_size
        self.attn_capping_value = attn_capping_value
        self.logits_capping_value = logits_capping_value
        self.qk_norm = qk_norm
        self.moe_noise_level = moe_noise_level
        self.num_kv_heads = num_kv_heads
        self.rope_frequency = rope_frequency
        self.rope_scale = rope_scale
        self.moe_type = moe_type
        self.moe_normalize = moe_normalize
        self.moe_num_slots = moe_num_slots
        self.pad_token_id = pad_token_id
        self.moe_interleaved = moe_interleaved
