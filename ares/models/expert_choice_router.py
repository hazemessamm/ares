from __future__ import annotations
import logging
import math
from typing import Optional
import torch
from torch import nn

from ares.models.layers import get_norm
from ares.models.utils import jitter_noise

logger = logging.getLogger("ares")


def compute_k(
    num_tokens: int, expert_capacity_factor: float, num_experts: int
):
    # Using math.ceil to avoid having k = 0 in case num_tokens = 1.
    if num_experts == 0:  # num_tokens is guaranteed > 0 by user
        return 0
    k = math.ceil((num_tokens * expert_capacity_factor) / num_experts)
    # torch.topk handles k=0 gracefully, returning empty tensors.
    return min(k, num_tokens)


def prepare_padding_mask(padding_mask: torch.Tensor):
    if padding_mask is not None:
        assert padding_mask.ndim == 2, "Padding mask must be 2D"
        assert padding_mask.dtype in [
            torch.bool,
            torch.long,
        ], "Padding mask must be bool or long"  # noqa
        padding_mask = padding_mask.view(-1, 1).bool()
        return padding_mask.logical_not()


class ExpertChoiceRouter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        expert_capacity_factor: float = 2.0,
        bias: bool = False,
        noise_scale: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.expert_capacity_factor = expert_capacity_factor
        self.noise_scale = noise_scale
        self.bias = bias
        self.router = nn.Linear(embed_dim, num_experts, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.router.weight, a=-0.01, b=0.01)
        if self.bias:
            nn.init.zeros_(self.router.bias)

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: Optional[torch.LongTensor] = None,
    ):
        if x.ndim > 2:
            x = x.view((-1, x.shape[-1]))

        num_tokens = x.shape[0]
        router_logits = self.router(x)

        if self.training:
            router_logits = jitter_noise(router_logits, self.noise_scale)

        if padding_mask is not None:
            padding_mask = prepare_padding_mask(padding_mask)
            router_logits = router_logits.masked_fill(
                padding_mask,
                torch.finfo(router_logits.dtype).min,
            )

        # Clamp before softmax to avoid NaN
        # when all logits are -inf (all padding).
        # Compute in fp32 for numerical stability, then cast back to the
        # compute dtype so downstream multiplies / einsums don't hit a dtype
        # mismatch when the model runs in bf16 without autocast.
        router_probs = nn.functional.softmax(
            router_logits,
            dim=0,
            dtype=torch.float32,
        ).to(router_logits.dtype).T

        top_k = compute_k(
            num_tokens=num_tokens,
            expert_capacity_factor=self.expert_capacity_factor,
            num_experts=self.num_experts,
        )

        tokens_prob, tokens_indices = torch.topk(
            router_probs,
            k=top_k,
            dim=-1,
        )
        return tokens_prob, tokens_indices


class ExpertChoiceMLP(nn.Module):
    def __init__(
        self,
        num_experts: int,
        embed_dim: int,
        ff_dim: int,
        bias: bool = True,
        activation: str = "silu",
        gated: bool = False,
        ff_norm_type: str | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.bias = bias
        self.activation_name = activation
        self.gated = gated
        self.norm_type = ff_norm_type

        # Stack all experts' weights
        self.w1 = nn.Parameter(torch.empty(num_experts, ff_dim, embed_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, embed_dim, ff_dim))

        if bias:
            self.b1 = nn.Parameter(torch.zeros(num_experts, ff_dim))
            self.b2 = nn.Parameter(torch.zeros(num_experts, embed_dim))
        else:
            self.b1 = None
            self.b2 = None

        if gated:
            self.g = nn.Parameter(torch.empty(num_experts, ff_dim, embed_dim))
        else:
            self.g = None

        self.norm = get_norm(ff_norm_type, ff_dim)

        self.activation_fn = getattr(nn.functional, activation)
        self.reset_parameters()

    def reset_parameters(self):
        mean = 0
        std = (2 / (self.ff_dim + self.embed_dim)) ** 0.5
        nn.init.normal_(self.w1, mean=mean, std=std)
        nn.init.normal_(self.w2, mean=mean, std=std)

        if self.gated:
            nn.init.normal_(self.g, mean=mean, std=std)

    def forward(self, tokens_per_expert):
        # tokens_per_expert: [num_experts, k, embed_dim]
        up = torch.einsum("efd,ekd->ekf", self.w1, tokens_per_expert)
        if self.bias:
            up = up + self.b1.unsqueeze(1)
        if self.norm is not None:
            up = self.norm(up)

        if self.gated:
            gates = torch.einsum("efd,ekd->ekf", self.g, tokens_per_expert)
            up = up * self.activation_fn(gates)
        else:
            up = self.activation_fn(up)

        down = torch.einsum("edf,ekf->ekd", self.w2, up)
        if self.bias:
            down = down + self.b2.unsqueeze(1)
        return down.contiguous()


class ExpertChoiceRouting(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        expert_capacity_factor: float = 2.0,
        ff_dim: int = 2048,
        activation: str = "silu",
        gated: bool = True,
        bias: bool = False,
        noise_scale: float = 0.0,
        norm_type: str = "rms",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.expert_capacity_factor = expert_capacity_factor

        self.router = ExpertChoiceRouter(
            embed_dim=embed_dim,
            num_experts=num_experts,
            expert_capacity_factor=expert_capacity_factor,
            noise_scale=noise_scale,
        )

        self.experts = ExpertChoiceMLP(
            num_experts=num_experts,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            bias=bias,
            activation=activation,
            gated=gated,
            ff_norm_type=norm_type,
        )

    def forward(
        self,
        x: torch.FloatTensor,
        padding_mask: Optional[torch.LongTensor] = None,
    ):
        original_input_shape = x.shape
        output = torch.zeros_like(x)

        if x.ndim > 2:
            x = x.view((-1, x.shape[-1]))
            output = output.view_as(x)

        tokens_probs, tokens_indices = self.router(
            x=x, padding_mask=padding_mask
        )

        # tokens_probs shape: (num_experts, k)
        # tokens_indices shape: (num_experts, k)

        flat_indices = tokens_indices.reshape(-1)  # [num_experts * k]

        # non_padded = padding_mask.sum()
        # unique_selected = torch.unique(flat_indices).shape[0]
        # real_drop_rate = 1 - unique_selected / non_padded
        # logger.info(f"real_drop_rate: {real_drop_rate}, non_padded: {non_padded}, unique_selected: {unique_selected}")

        flat_probs = tokens_probs.reshape(-1, 1)  # [num_experts * k, 1]

        dispatched_tokens = x[flat_indices]  # [num_experts * k, embed_dim]
        tokens_per_expert = dispatched_tokens.view(
            self.num_experts, -1, self.embed_dim
        )
        probs_per_expert = flat_probs.view(self.num_experts, -1, 1)

        expert_outputs = self.experts(tokens_per_expert)
        weighted_outputs = (expert_outputs * probs_per_expert).to(x.dtype)

        weighted_outputs = weighted_outputs.view(
            -1,
            weighted_outputs.shape[-1],
        )

        output = output.index_add(0, flat_indices, weighted_outputs)

        return output.view(original_input_shape)
