import numpy as np
import torch
from typing import Optional

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
except ImportError:
    xm = None
    xr = None

import math

# Maximum average NLL before exp() to prevent overflow to inf
_MAX_NLL_FOR_EXP = 100.0


@torch.no_grad()
def compute_accuracy_components(
    predictions,
    targets,
    apply_argmax: bool = True,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute accuracy components for efficient accumulation.
    Returns num_correct and num_valid instead of final accuracy.
    """
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    if apply_argmax:
        predictions = predictions.argmax(dim=-1)

    is_valid = targets != ignore_index
    num_correct = ((predictions == targets) & is_valid).sum()
    num_valid = is_valid.sum()
    return num_correct, num_valid


@torch.no_grad()
def compute_accuracy(
    predictions,
    targets,
    apply_argmax: bool = True,
    ignore_index: int = -100,
):
    num_correct, num_valid = compute_accuracy_components(
        predictions, targets, apply_argmax, ignore_index
    )
    if num_valid == 0:
        return 0.0
    return (num_correct / num_valid).item()


def _get_world_size() -> int:
    """Get actual world size, handling SPMD where xr.world_size() returns 1."""
    # In SPMD mode xr.world_size() returns 1. Use global_runtime_device_count
    # which gives the actual number of TPU chips.
    if xr is None:
        raise ImportError("torch_xla is not installed")
    return xr.global_runtime_device_count()


class Accuracy:
    """
    XLA-optimized accuracy metric.

    Uses two sets of accumulators:
    - XLA tensors for per-batch accumulation (no sync)
    - Python scalars for distributed reduction (updated in compute())
    """

    def __init__(self):
        if xr is None:
            raise ImportError("torch_xla is not installed")
        # Python scalars for distributed reduction
        self.num_correct = 0.0
        self.num_valid = 0.0
        # XLA tensors for per-batch accumulation
        self._num_correct_xla: Optional[torch.Tensor] = None
        self._num_valid_xla: Optional[torch.Tensor] = None

    def update(
        self,
        predictions,
        targets,
        apply_argmax: bool = True,
        ignore_index: int = -100,
    ):
        """Accumulate on XLA tensors (no sync)."""
        num_correct, num_valid = compute_accuracy_components(
            predictions, targets, apply_argmax, ignore_index
        )
        if self._num_correct_xla is None:
            self._num_correct_xla = num_correct.clone()
            self._num_valid_xla = num_valid.clone()
        else:
            self._num_correct_xla = self._num_correct_xla + num_correct
            self._num_valid_xla = self._num_valid_xla + num_valid

    def update_from_components(
        self,
        num_correct: torch.Tensor,
        num_valid: torch.Tensor,
    ):
        """Accumulate precomputed accuracy components on XLA tensors."""
        if self._num_correct_xla is None:
            self._num_correct_xla = num_correct.clone()
            self._num_valid_xla = num_valid.clone()
        else:
            self._num_correct_xla = self._num_correct_xla + num_correct
            self._num_valid_xla = self._num_valid_xla + num_valid

    def _flush_xla_to_scalars(self):
        """Transfer XLA accumulators to Python scalars."""
        if self._num_correct_xla is not None:
            self.num_correct += self._num_correct_xla.item()
            self.num_valid += self._num_valid_xla.item()
            self._num_correct_xla = None
            self._num_valid_xla = None

    def reset(self):
        """Reset all accumulators."""
        self.num_correct = 0.0
        self.num_valid = 0.0
        self._num_correct_xla = None
        self._num_valid_xla = None

    def local_compute(self):
        """Compute local accuracy (single device)."""
        self._flush_xla_to_scalars()
        if self.num_valid == 0:
            return 0.0
        return self.num_correct / self.num_valid

    def compute(self):
        """Compute global accuracy across all chips."""
        self._flush_xla_to_scalars()

        num_correct = self.num_correct
        num_valid = self.num_valid

        if _get_world_size() > 1:
            num_correct = xm.mesh_reduce("acc_correct", num_correct, sum)
            num_valid = xm.mesh_reduce("acc_valid", num_valid, sum)

        if num_valid == 0:
            return 0.0
        return num_correct / num_valid


class _NLLAccumulator:
    """
    Base class for NLL-based metrics (loss and perplexity).

    Accumulates total negative log-likelihood and token counts using
    XLA tensors for lazy batching and Python scalars for distributed
    reduction.
    """

    def __init__(self):
        if xr is None:
            raise ImportError("torch_xla is not installed")
        # Python scalars for distributed reduction
        self.total_nll = 0.0
        self.num_tokens = 0.0
        # XLA tensors for per-batch accumulation
        self._total_nll_xla: Optional[torch.Tensor] = None
        self._num_tokens_xla: Optional[torch.Tensor] = None

    def update(self, loss, labels, ignore_index: int = -100):
        """Accumulate on XLA tensors (no sync)."""
        valid_tokens = (labels != ignore_index).sum()

        # Ensure loss is a tensor with explicit dtype for XLA compatibility
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(
                loss, dtype=torch.float32, device=labels.device
            )

        batch_nll = loss * valid_tokens.float()

        if self._total_nll_xla is None:
            self._total_nll_xla = batch_nll.clone()
            self._num_tokens_xla = valid_tokens.clone()
        else:
            self._total_nll_xla = self._total_nll_xla + batch_nll
            self._num_tokens_xla = self._num_tokens_xla + valid_tokens

    def _flush_xla_to_scalars(self):
        """Transfer XLA accumulators to Python scalars."""
        if self._total_nll_xla is not None:
            self.total_nll += self._total_nll_xla.item()
            self.num_tokens += self._num_tokens_xla.item()
            self._total_nll_xla = None
            self._num_tokens_xla = None

    def reset(self):
        """Reset all accumulators."""
        self.total_nll = 0.0
        self.num_tokens = 0.0
        self._total_nll_xla = None
        self._num_tokens_xla = None

    def _get_reduced_nll_and_tokens(self, reduce_tag: str):
        """Flush and optionally reduce across devices."""
        self._flush_xla_to_scalars()

        total_nll = self.total_nll
        num_tokens = self.num_tokens

        if _get_world_size() > 1:
            total_nll = xm.mesh_reduce(f"{reduce_tag}_nll", total_nll, sum)
            num_tokens = xm.mesh_reduce(
                f"{reduce_tag}_tokens", num_tokens, sum
            )

        return total_nll, num_tokens


class Perplexity(_NLLAccumulator):
    """XLA-optimized perplexity metric."""

    def local_compute(self):
        """Compute local perplexity (single device)."""
        self._flush_xla_to_scalars()
        if self.num_tokens == 0:
            return 0.0
        avg_nll = min(self.total_nll / self.num_tokens, _MAX_NLL_FOR_EXP)
        return math.exp(avg_nll)

    def compute(self):
        """Compute global perplexity across all chips."""
        total_nll, num_tokens = self._get_reduced_nll_and_tokens("ppl")
        if num_tokens == 0:
            return 0.0
        avg_nll = min(total_nll / num_tokens, _MAX_NLL_FOR_EXP)
        return math.exp(avg_nll)


class Loss(_NLLAccumulator):
    """XLA-optimized loss metric."""

    def local_compute(self):
        """Compute local loss (single device)."""
        self._flush_xla_to_scalars()
        if self.num_tokens == 0:
            return 0.0
        return self.total_nll / self.num_tokens

    def compute(self):
        """Compute global average loss across all chips."""
        total_nll, num_tokens = self._get_reduced_nll_and_tokens("loss")
        if num_tokens == 0:
            return 0.0
        return total_nll / num_tokens
