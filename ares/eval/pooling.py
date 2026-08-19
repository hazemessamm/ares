import torch
import torch.nn as nn
from typing import Optional


class AveragePooling1D(nn.Module):
    def __init__(self, embed_dim=None):
        super().__init__()
        self.embed_dim = embed_dim

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask.to(device=x.device).bool()
        if special_tokens_mask is not None:
            non_special_mask = special_tokens_mask.to(device=x.device).bool()
            valid_mask = (
                non_special_mask if valid_mask is None else valid_mask & non_special_mask
            )
        if valid_mask is not None:
            x = x * valid_mask.unsqueeze(-1).to(dtype=x.dtype)
            denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            return x.sum(dim=1) / denom.to(dtype=x.dtype)
        return x.mean(dim=1)


class MaxPooling1D(nn.Module):
    def __init__(self, embed_dim=None):
        super().__init__()
        self.embed_dim = embed_dim

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        special_tokens_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask.to(device=x.device).bool()
        if special_tokens_mask is not None:
            non_special_mask = special_tokens_mask.to(device=x.device).bool()
            valid_mask = (
                non_special_mask if valid_mask is None else valid_mask & non_special_mask
            )
        if valid_mask is not None:
            all_invalid = ~valid_mask.any(dim=1, keepdim=True)
            masked_x = x.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
            pooled = masked_x.amax(dim=1)
            return pooled.masked_fill(all_invalid, 0.0)
        return x.amax(dim=1)


class AttentionPooling1D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        bias: bool = False,
    ):
        """
        Attention Pooling 1D.

        Args:
            embed_dim: The dimension of the embeddings.
            bias: Whether to use a bias in the linear layer.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = nn.Linear(embed_dim, 1, bias=bias)

    @property
    def output_dim(self) -> int:
        return self.embed_dim

    def forward(
        self,
        x: torch.FloatTensor,
        attention_mask: torch.FloatTensor | None = None,
        special_tokens_mask: torch.FloatTensor | None = None,
    ):
        scores = self.weight(x).squeeze(-1)

        valid_mask = None
        if attention_mask is not None:
            valid_mask = attention_mask.to(device=x.device).bool()
        if special_tokens_mask is not None:
            non_special_mask = special_tokens_mask.to(device=x.device).bool()
            valid_mask = (
                non_special_mask if valid_mask is None else valid_mask & non_special_mask
            )
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, float("-inf"))
            all_padding = ~valid_mask.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(all_padding, 0.0)
            probs = nn.functional.softmax(scores, dim=-1)
            probs = probs.masked_fill(all_padding, 0.0)
        else:
            probs = nn.functional.softmax(scores, dim=-1)

        return torch.sum(x * probs.unsqueeze(-1), dim=1)


SUPPORTED_POOLERS = {
    "avg": AveragePooling1D,
    "max": MaxPooling1D,
    "attn": AttentionPooling1D,
}


def get(identifier: str, **kwargs):
    if identifier not in SUPPORTED_POOLERS:
        raise ValueError(f"Invalid pooler identifier: {identifier}")
    return SUPPORTED_POOLERS[identifier](**kwargs)
