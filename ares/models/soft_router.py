from __future__ import annotations
from torch import nn
import torch

from typing import Tuple, Union, Optional

from ares.models.utils import jitter_noise
from ares.models.layers import get_norm


def softmax(x: torch.Tensor, dim: Union[int, Tuple[int, ...]]):
    if isinstance(dim, int):
        dim = (dim,)
    # To make this function safe under mixed precision, ensure that:
    # - The subtraction and exponentiation are done in float32 to avoid
    # overflow/underflow.
    # - The denominator summation is also in float32.
    # - The final division is cast back to the input dtype if needed.
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    max_val = torch.amax(x, dim=dim, keepdim=True)
    x_exp = torch.exp(x - max_val)
    denom = torch.sum(x_exp, dim=dim, keepdim=True, dtype=torch.float32)
    result = x_exp / denom
    return result.to(orig_dtype)


class SoftMoEFeedForward(nn.Module):
    def __init__(
        self,
        num_experts: int,
        embed_dim: int,
        ff_dim: int,
        bias: bool = True,
        activation: str = "silu",
        gated: bool = False,
        ff_norm_type: str | None = "rms",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.gated = gated
        self.bias = bias

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
        self.activation_name = activation
        self.activation_fn = getattr(nn.functional, activation)
        self.reset_parameters()

    def reset_parameters(self):
        mean = 0
        std = (2 / (self.ff_dim + self.embed_dim)) ** 0.5
        nn.init.normal_(self.w1, mean=mean, std=std)
        nn.init.normal_(self.w2, mean=mean, std=std)

        if self.gated:
            nn.init.normal_(self.g, mean=mean, std=std)

    def forward(self, x):
        up = torch.einsum("efd, besd -> besf", self.w1, x)
        if self.bias:
            up = up + self.b1.unsqueeze(1)
        if self.norm is not None:
            up = self.norm(up)
        if self.gated:
            gates = torch.einsum("efd, besd -> besf", self.g, x)
            up = up * self.activation_fn(gates)
        else:
            up = self.activation_fn(up)
        down = torch.einsum("edf, besf -> besd", self.w2, up)
        if self.bias:
            down = down + self.b2.unsqueeze(1)
        return down.contiguous()


def prepare_padding_mask(padding_mask: torch.Tensor):
    if padding_mask is not None:
        # `padding_mask` is (batch, seq_len) with `True` for valid tokens
        # and `False` for padding. We want to mask out the padding, so we
        # invert the mask.
        assert padding_mask.ndim == 2, "Padding mask must be 2D"
        assert padding_mask.dtype in [
            torch.bool,
            torch.long,
        ], "Padding mask must be bool or long"  # noqa
        padding_mask = padding_mask[..., None, None].bool()
        return padding_mask.logical_not()


class SoftRouter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        num_slots: int,
        ff_dim: int = 2048,
        activation: str = "silu",
        gated: bool = True,
        ff_norm_type: str | None = "rms",
        normalize: bool = False,
        noise_scale: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.normalize = normalize
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.noise_scale = noise_scale
        self.phi = nn.Parameter(
            torch.empty((embed_dim, num_experts, num_slots)),
            requires_grad=True,
        )
        self.experts = SoftMoEFeedForward(
            num_experts=num_experts,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            bias=bias,
            activation=activation,
            gated=gated,
            ff_norm_type=ff_norm_type,
        )

        if self.normalize:
            self.scaler = nn.Parameter(torch.ones([1]))

        self.dispatch_weights_ = None
        self.combine_weights_ = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.phi)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1)
            phi = nn.functional.normalize(self.phi, dim=0) * self.scaler
        else:
            phi = self.phi

        # logits shape: batch, seqlen, num_experts, slots
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        if self.training:
            logits = jitter_noise(logits, self.noise_scale)

        if padding_mask is not None:
            # `padding_mask` is (batch, seq_len) with `True` for valid tokens
            # and `False` for padding. We want to mask out the padding, so we
            # invert the mask.
            mask = prepare_padding_mask(padding_mask)
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)

        # Compute softmax in fp32 for numerical stability, then cast back to
        # the compute dtype so downstream einsums don't hit a dtype mismatch
        # when the model runs in bf16 without autocast. Under fp32 this is
        # a no-op; under autocast the cast was previously happening implicitly
        # at the einsum boundary, so behavior is unchanged in either case.
        dispatch_weights = nn.functional.softmax(
            logits, dim=1, dtype=torch.float32
        ).to(logits.dtype)
        combine_weights = softmax(logits, dim=(2, 3))
        # was used when doing analysis on the dispatch and combine weights.
        # Not a very clean way of analyzing the weights.
        # self.dispatch_weights_ = dispatch_weights.detach()
        # self.combine_weights_ = combine_weights.detach()
        # Xs shape: batch_size, num_experts, slots, embed_dim
        Xs = torch.einsum("bmd,bmnp->bnpd", x, dispatch_weights)

        expert_outputs = self.experts(Xs)
        combined_outputs = torch.einsum(
            "bnpd,bmnp->bmd",
            expert_outputs,
            combine_weights,
        )
        return combined_outputs.to(x.dtype)
