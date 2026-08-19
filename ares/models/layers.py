from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Embedding):
    def reset_parameters(self):
        # Better initialization
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim**-0.5)
        self._fill_padding_idx_with_zero()


class ScaleNorm(nn.Module):
    def __init__(self, scale: int, eps: float = 1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.tensor([scale**0.5]), requires_grad=True)
        self.eps = eps

    def forward(self, inputs: torch.FloatTensor):
        out = self.g / inputs.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return inputs * out


def non_trainable_rms_norm(x: torch.FloatTensor, eps: float = 1e-6):
    return F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=eps, weight=None)


def get_norm(norm_type: str | None, embed_dim: int):
    if norm_type is None:
        return None

    if norm_type == "rms":
        return nn.RMSNorm(embed_dim)
    elif norm_type == "scale":
        return ScaleNorm(embed_dim)
    else:
        raise ValueError(f"Invalid norm type: {norm_type}")


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_dim: int,
        bias: bool = True,
        activation: str = "silu",
        gated: bool = False,
        ff_norm_type: str | None = "rms",
    ):
        """Feedforward layer that contains 2 linear layers,
        activation function and a dropout layer.

        Args:
            embed_dim (int): Embedding dimension.
            ff_dim (int): Intermediate output dimensions
            bias (bool): Whether to apply bias or not.
                Defaults to True.
            activation (str, optional): Activation function name,
                (should exist in `torch.nn.functional`). Defaults to "gelu".
            gated (bool, optional): Whether to use this activation function as
                a gated activation function or not. Defaults to False.
        """
        super().__init__()
        self.activation = getattr(F, activation)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.bias = bias
        self.gated = gated
        self.w1 = nn.Linear(embed_dim, ff_dim, bias=bias)
        self.w2 = nn.Linear(ff_dim, embed_dim, bias=bias)

        self.norm = get_norm(ff_norm_type, ff_dim)

        if gated:
            self.g = nn.Linear(embed_dim, self.ff_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        mean = 0
        std = (2 / (self.ff_dim + self.embed_dim)) ** 0.5
        nn.init.normal_(self.w1.weight, mean=mean, std=std)
        nn.init.normal_(self.w2.weight, mean=mean, std=std)

        if self.gated:
            nn.init.normal_(self.g.weight, mean=mean, std=std)

        if self.bias:
            nn.init.constant_(self.w1.bias, 0.0)
            nn.init.constant_(self.w2.bias, 0.0)
            if self.gated:
                nn.init.constant_(self.g.bias, 0.0)

    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        up = self.w1(embeddings)

        if self.norm is not None:
            up = self.norm(up)

        if self.gated:
            gates = self.g(embeddings)
            up = up * self.activation(gates)

        down = self.w2(up)
        return down
