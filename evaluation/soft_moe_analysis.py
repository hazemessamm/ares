import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ares.models.soft_router import SoftRouter
from ares.models.model import Ares
from ares.tokenization.protein_tokenizer import AresProteinTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from dataclasses import dataclass
from tqdm import tqdm

import json
import pickle
from pathlib import Path


AA_PROPERTIES = {
    "hydrophobic": set("AILMFWVP"),
    "polar": set("STNQ"),
    "positive": set("RHK"),
    "negative": set("DE"),
    "aromatic": set("FWY"),
    "small": set("GAS"),
    "cysteine": set("C"),
}


# Weight types captured from the SoftRouter
#   - "dispatch": softmax over seq_len; dispatch_weights[b, s, e, p] is the
#     contribution of token s to slot p of expert e. Summing over slots gives
#     a "cumulative attention mass" placed on token s by expert e's slots.
#     NOTE: these weights do NOT sum to 1 per token.
#
#   - "combine": softmax over (experts, slots) per token; combine_weights[b, s, e, p] # noqa
#     is the fraction of token s's output that comes from slot p of expert e.
#     Summing over slots gives the fraction of token s's output coming from
#     expert e. These weights DO sum to 1 per token.
WEIGHT_TYPES = ("dispatch", "combine")


# =============================================================================
# Base class: handles hook registration and capture of both weight tensors.
# =============================================================================


class SoftMoEBaseAnalyzer:
    """
    Base class that registers hooks on all SoftRouter modules and captures
    BOTH dispatch and combine weights per forward pass.

    The SoftRouter must expose `dispatch_weights_` and `combine_weights_`
    as attributes populated during forward() (uncomment those lines in
    SoftRouter.forward).

    Subclasses implement:
        _init_state()
        _process(layer_name, weight_type, weights_summed_over_slots,
                 token_ids, attention_mask)
        compute()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List = []
        self.router_names: List[str] = []
        self.routers: List[SoftRouter] = []

        # _last[layer_name][weight_type] -> tensor of shape (B, S, E)
        self._last: Dict[str, Dict[str, torch.Tensor]] = {}

        for name, module in model.named_modules():
            if isinstance(module, SoftRouter):
                self.router_names.append(name)
                self.routers.append(module)

        if not self.routers:
            raise ValueError("No SoftRouter modules found in the model.")

        self.num_experts = self.routers[0].num_experts
        self.num_layers = len(self.routers)

        self._init_state()
        self._register_hooks()

    def _register_hooks(self):
        for name, router in zip(self.router_names, self.routers):
            hook = router.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        def hook_fn(module: SoftRouter, input, output):
            entry = {}
            if getattr(module, "dispatch_weights_", None) is not None:
                # (B, S, E, P) -> (B, S, E) by summing over slots
                entry["dispatch"] = module.dispatch_weights_.sum(
                    dim=-1
                ).detach()
            if getattr(module, "combine_weights_", None) is not None:
                entry["combine"] = module.combine_weights_.sum(dim=-1).detach()
            self._last[layer_name] = entry

        return hook_fn

    def update(self, token_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Call after each forward pass."""
        token_ids_cpu = token_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()
        for layer_name in self.router_names:
            entry = self._last.get(layer_name)
            if entry is None:
                continue
            for weight_type, tensor in entry.items():
                tensor_cpu = tensor.cpu()
                self._process(
                    layer_name,
                    weight_type,
                    tensor_cpu,
                    token_ids_cpu,
                    attention_mask_cpu,
                )
        self._last.clear()

    def _init_state(self):
        raise NotImplementedError

    def _process(
        self, layer_name, weight_type, weights, token_ids, attention_mask
    ):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def reset(self):
        self._init_state()
        self._last.clear()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# Helper: build a (vocab_size,) lookup tensor mapping token_id -> property_index # noqa
# Used to vectorize token -> property classification.
# Tokens belonging to no property get -1. Tokens in ignore_ids also get -1.
# If a token belongs to multiple property groups (e.g. F is hydrophobic AND
# aromatic), we expand it to multiple (token_id, property_idx) entries and
# apply them additively in _process, so overlap is handled correctly.
# =============================================================================


def _build_property_entries(
    id_to_aa: Dict[int, str],
    property_groups: Dict[str, set],
    ignore_ids: Set[int],
) -> List[Tuple[int, int]]:
    """
    Returns a list of (token_id, property_index) pairs.
    A token appears multiple times if it belongs to multiple property groups.
    """
    group_order = list(property_groups.keys())
    group_to_idx = {g: i for i, g in enumerate(group_order)}
    entries = []
    for tid, aa in id_to_aa.items():
        if tid in ignore_ids:
            continue
        for g, aas in property_groups.items():
            if aa in aas:
                entries.append((tid, group_to_idx[g]))
    return entries, group_order


# =============================================================================
# 1. Per-expert amino acid preferences (vectorized, both weight types)
# =============================================================================


class ExpertAminoAcidAnalyzer(SoftMoEBaseAnalyzer):
    """
    For each (layer, expert, amino_acid):
        - mean_weight: average summed-over-slots weight for tokens of this AA
        - baseline: average summed-over-slots weight across all valid tokens
        - specialization_ratio: mean_weight / baseline

    Produces results for BOTH "dispatch" and "combine" weight types.
    """

    def __init__(
        self,
        model: nn.Module,
        id_to_aa: Dict[int, str],
        ignore_ids: Optional[Set[int]] = None,
    ):
        self.id_to_aa = id_to_aa
        self.ignore_ids = ignore_ids or set()
        # Build list of valid amino acids seen in vocab
        self._aa_list = sorted(
            set(
                aa
                for tid, aa in id_to_aa.items()
                if tid not in self.ignore_ids
            )
        )
        self._aa_to_idx = {aa: i for i, aa in enumerate(self._aa_list)}
        # Build lookup: token_id -> aa_idx (-1 if should be ignored or not an AA) # noqa
        max_tid = max(id_to_aa.keys()) if id_to_aa else 0
        self._tid_to_aa_idx = torch.full((max_tid + 1,), -1, dtype=torch.long)
        for tid, aa in id_to_aa.items():
            if tid in self.ignore_ids:
                continue
            if aa in self._aa_to_idx:
                self._tid_to_aa_idx[tid] = self._aa_to_idx[aa]
        self._max_tid = max_tid
        super().__init__(model)

    def _init_state(self):
        num_aa = len(self._aa_list)
        # Per (weight_type, layer): (num_experts, num_aa) sums and counts
        self.sums: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {
                name: torch.zeros(
                    self.num_experts, num_aa, dtype=torch.float64
                )
                for name in self.router_names
            }
            for wt in WEIGHT_TYPES
        }
        self.counts: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {
                name: torch.zeros(num_aa, dtype=torch.long)
                for name in self.router_names
            }
            for wt in WEIGHT_TYPES
        }
        # Baseline: per (weight_type, layer): (num_experts,) sum, scalar count
        self.baseline_sums: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {
                name: torch.zeros(self.num_experts, dtype=torch.float64)
                for name in self.router_names
            }
            for wt in WEIGHT_TYPES
        }
        self.baseline_counts: Dict[str, Dict[str, int]] = {
            wt: {name: 0 for name in self.router_names} for wt in WEIGHT_TYPES
        }

    def _process(
        self, layer_name, weight_type, weights, token_ids, attention_mask
    ):
        # weights: (B, S, E), token_ids: (B, S), attention_mask: (B, S)
        B, S, E = weights.shape
        flat_weights = weights.reshape(-1, E).to(torch.float64)  # (B*S, E)
        flat_tokens = token_ids.reshape(-1)  # (B*S,)
        flat_mask = attention_mask.reshape(-1).bool()  # (B*S,)

        # Clamp token ids to fit the lookup table; any token id beyond max_tid
        # will be treated as -1 (not an amino acid).
        safe_tokens = flat_tokens.clamp(max=self._max_tid)
        aa_idx = self._tid_to_aa_idx[safe_tokens]  # (B*S,)
        aa_idx = aa_idx.masked_fill(~flat_mask, -1)
        # Also mask any original token_id that exceeded max_tid
        oversize = flat_tokens > self._max_tid
        aa_idx = aa_idx.masked_fill(oversize, -1)

        valid = aa_idx >= 0  # (B*S,)
        if not valid.any():
            return

        valid_weights = flat_weights[valid]  # (N_valid, E)
        valid_aa_idx = aa_idx[valid]  # (N_valid,)

        # Baseline (over all valid tokens regardless of AA)
        self.baseline_sums[weight_type][layer_name] += valid_weights.sum(dim=0)
        self.baseline_counts[weight_type][layer_name] += int(
            valid.sum().item()
        )

        # Per-AA sums via index_add
        # sums shape: (E, num_aa). We add valid_weights[n, e] to sums[e, aa_idx[n]]. # noqa
        sums = self.sums[weight_type][layer_name]  # (E, num_aa)
        # Transpose for convenience: accumulate per-expert along aa axis
        # Easier: use scatter_add across the aa dimension
        # For each expert e independently: sums[e].index_add_(0, valid_aa_idx, valid_weights[:, e]) # noqa
        # We can do this in one shot with index_add on axis=1 of sums.T
        sums_t = sums.t().contiguous()  # (num_aa, E)
        sums_t.index_add_(0, valid_aa_idx, valid_weights)
        self.sums[weight_type][layer_name] = sums_t.t().contiguous()

        # Per-AA counts
        counts = self.counts[weight_type][layer_name]  # (num_aa,)
        ones = torch.ones_like(valid_aa_idx, dtype=torch.long)
        counts.index_add_(0, valid_aa_idx, ones)

    def compute(
        self,
    ) -> Dict[str, Dict[str, Dict[int, Dict[str, Dict[str, float]]]]]:
        """
        Returns:
            {weight_type: {layer_name: {expert: {aa: {
                "mean_weight", "baseline", "specialization_ratio"
            }}}}}
        """
        results = {wt: {} for wt in WEIGHT_TYPES}
        for wt in WEIGHT_TYPES:
            for name in self.router_names:
                sums = self.sums[wt][name]  # (E, num_aa)
                counts = self.counts[wt][name]  # (num_aa,)
                base_sum = self.baseline_sums[wt][name]  # (E,)
                base_count = self.baseline_counts[wt][name]
                results[wt][name] = {}
                baseline = (
                    (base_sum / base_count).tolist()
                    if base_count > 0
                    else [0.0] * self.num_experts
                )
                for e in range(self.num_experts):
                    results[wt][name][e] = {}
                    for aa_i, aa in enumerate(self._aa_list):
                        c = int(counts[aa_i].item())
                        if c > 0:
                            mean_w = float(sums[e, aa_i].item() / c)
                        else:
                            mean_w = 0.0
                        b = baseline[e]
                        ratio = (mean_w / b) if b > 0 else 0.0
                        results[wt][name][e][aa] = {
                            "mean_weight": mean_w,
                            "baseline": float(b),
                            "specialization_ratio": ratio,
                        }
        return results


# =============================================================================
# 2. Per-expert amino acid property group preferences
# =============================================================================


class ExpertPropertyAnalyzer(SoftMoEBaseAnalyzer):
    """
    For each (layer, expert, property_group):
        - mean_weight, baseline, specialization_ratio
    Handles AA overlap (F, W in both hydrophobic and aromatic) correctly:
    a token contributes to every property group it belongs to.
    """

    def __init__(
        self,
        model: nn.Module,
        id_to_aa: Dict[int, str],
        ignore_ids: Optional[Set[int]] = None,
        property_groups: Optional[Dict[str, set]] = None,
    ):
        self.id_to_aa = id_to_aa
        self.ignore_ids = ignore_ids or set()
        self.property_groups = property_groups or AA_PROPERTIES

        # Build a (vocab_size, num_groups) bool membership table
        max_tid = max(id_to_aa.keys()) if id_to_aa else 0
        self._group_order = list(self.property_groups.keys())
        num_groups = len(self._group_order)
        self._membership = torch.zeros(
            max_tid + 1, num_groups, dtype=torch.bool
        )
        # Separately track whether a token is a valid (non-ignored) AA at all
        self._is_valid_aa = torch.zeros(max_tid + 1, dtype=torch.bool)
        for tid, aa in id_to_aa.items():
            if tid in self.ignore_ids:
                continue
            self._is_valid_aa[tid] = True
            for g_i, g in enumerate(self._group_order):
                if aa in self.property_groups[g]:
                    self._membership[tid, g_i] = True
        self._max_tid = max_tid
        super().__init__(model)

    def _init_state(self):
        num_groups = len(self._group_order)
        self.sums: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {
                name: torch.zeros(
                    self.num_experts, num_groups, dtype=torch.float64
                )
                for name in self.router_names
            }
            for wt in WEIGHT_TYPES
        }
        self.counts: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {
                name: torch.zeros(num_groups, dtype=torch.long)
                for name in self.router_names
            }
            for wt in WEIGHT_TYPES
        }
        self.baseline_sums: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {
                name: torch.zeros(self.num_experts, dtype=torch.float64)
                for name in self.router_names
            }
            for wt in WEIGHT_TYPES
        }
        self.baseline_counts: Dict[str, Dict[str, int]] = {
            wt: {name: 0 for name in self.router_names} for wt in WEIGHT_TYPES
        }

    def _process(
        self, layer_name, weight_type, weights, token_ids, attention_mask
    ):
        B, S, E = weights.shape
        flat_weights = weights.reshape(-1, E).to(torch.float64)  # (N, E)
        flat_tokens = token_ids.reshape(-1)  # (N,)
        flat_mask = attention_mask.reshape(-1).bool()  # (N,)
        # N = flat_weights.shape[0]

        # Clamp and mask out-of-range token ids
        oversize = flat_tokens > self._max_tid
        safe_tokens = flat_tokens.clamp(max=self._max_tid)

        is_valid_aa = (
            self._is_valid_aa[safe_tokens] & flat_mask & (~oversize)
        )  # (N,)
        if not is_valid_aa.any():
            return

        # Baseline over all valid AA tokens
        valid_weights = flat_weights[is_valid_aa]  # (N_valid, E)
        self.baseline_sums[weight_type][layer_name] += valid_weights.sum(dim=0)
        self.baseline_counts[weight_type][layer_name] += int(
            is_valid_aa.sum().item()
        )

        # Per-group: (N, num_groups) boolean membership for valid tokens
        membership = self._membership[safe_tokens]  # (N, num_groups)
        membership = membership & is_valid_aa.unsqueeze(
            -1
        )  # mask invalid rows

        # For each group: sum weights where membership[:, g] is True
        # Vectorized: for each g, sums[:, g] += flat_weights[membership[:, g]].sum(dim=0) # noqa
        # Equivalent to: sums += flat_weights.T @ membership.float()
        mem_f = membership.to(torch.float64)  # (N, num_groups)
        # flat_weights.T (E, N) @ mem_f (N, num_groups) -> (E, num_groups)
        self.sums[weight_type][layer_name] += flat_weights.t() @ mem_f

        # Counts per group
        self.counts[weight_type][layer_name] += membership.sum(dim=0).to(
            torch.long
        )

    def compute(self):
        """
        Returns:
            {weight_type: {layer_name: {expert: {group: {
                "mean_weight", "baseline", "specialization_ratio"
            }}}}}
        """
        results = {wt: {} for wt in WEIGHT_TYPES}
        for wt in WEIGHT_TYPES:
            for name in self.router_names:
                sums = self.sums[wt][name]  # (E, num_groups)
                counts = self.counts[wt][name]  # (num_groups,)
                base_sum = self.baseline_sums[wt][name]
                base_count = self.baseline_counts[wt][name]
                results[wt][name] = {}
                baseline = (
                    (base_sum / base_count).tolist()
                    if base_count > 0
                    else [0.0] * self.num_experts
                )
                for e in range(self.num_experts):
                    results[wt][name][e] = {}
                    for g_i, g in enumerate(self._group_order):
                        c = int(counts[g_i].item())
                        if c > 0:
                            mean_w = float(sums[e, g_i].item() / c)
                        else:
                            mean_w = 0.0
                        b = baseline[e]
                        ratio = (mean_w / b) if b > 0 else 0.0
                        results[wt][name][e][g] = {
                            "mean_weight": mean_w,
                            "baseline": float(b),
                            "specialization_ratio": ratio,
                        }
        return results


# =============================================================================
# 3. Positional preferences (N-terminal, interior, C-terminal)
# =============================================================================


class ExpertPositionalAnalyzer(SoftMoEBaseAnalyzer):
    """
    For each (layer, expert, relative_position_bin):
        - mean_weight, baseline, specialization_ratio
    """

    def __init__(self, model: nn.Module, num_bins: int = 5):
        self.num_bins = num_bins
        super().__init__(model)

    def _init_state(self):
        self.bin_labels = [
            f"{int(i / self.num_bins * 100)}-{int((i + 1) / self.num_bins * 100)}%"  # noqa
            for i in range(self.num_bins)
        ]
        self.sums: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {
                name: torch.zeros(
                    self.num_experts, self.num_bins, dtype=torch.float64
                )
                for name in self.router_names
            }
            for wt in WEIGHT_TYPES
        }
        self.counts: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {
                name: torch.zeros(self.num_bins, dtype=torch.long)
                for name in self.router_names
            }
            for wt in WEIGHT_TYPES
        }
        self.baseline_sums: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {
                name: torch.zeros(self.num_experts, dtype=torch.float64)
                for name in self.router_names
            }
            for wt in WEIGHT_TYPES
        }
        self.baseline_counts: Dict[str, Dict[str, int]] = {
            wt: {name: 0 for name in self.router_names} for wt in WEIGHT_TYPES
        }

    def _process(
        self, layer_name, weight_type, weights, token_ids, attention_mask
    ):
        # weights: (B, S, E)
        B, S, E = weights.shape
        valid_lens = attention_mask.sum(dim=1).to(torch.long)  # (B,)
        # Build bin_idx tensor of shape (B, S), with -1 for invalid positions
        bin_idx = torch.full((B, S), -1, dtype=torch.long)
        for b in range(B):
            L = int(valid_lens[b].item())
            if L == 0:
                continue
            positions = torch.arange(L, dtype=torch.float64)
            rel_pos = positions / L
            bins = (
                (rel_pos * self.num_bins)
                .to(torch.long)
                .clamp(max=self.num_bins - 1)
            )
            bin_idx[b, :L] = bins
        # Flatten
        flat_weights = weights.reshape(-1, E).to(torch.float64)
        flat_bin = bin_idx.reshape(-1)
        valid = flat_bin >= 0
        if not valid.any():
            return

        valid_weights = flat_weights[valid]
        valid_bins = flat_bin[valid]

        # Baseline
        self.baseline_sums[weight_type][layer_name] += valid_weights.sum(dim=0)
        self.baseline_counts[weight_type][layer_name] += int(
            valid.sum().item()
        )

        # Per-bin sums via index_add on transposed view (matches pattern used above) # noqa
        sums = self.sums[weight_type][layer_name]  # (E, num_bins)
        sums_t = sums.t().contiguous()  # (num_bins, E)
        sums_t.index_add_(0, valid_bins, valid_weights)
        self.sums[weight_type][layer_name] = sums_t.t().contiguous()

        counts = self.counts[weight_type][layer_name]
        ones = torch.ones_like(valid_bins, dtype=torch.long)
        counts.index_add_(0, valid_bins, ones)

    def compute(self):
        results = {wt: {} for wt in WEIGHT_TYPES}
        for wt in WEIGHT_TYPES:
            for name in self.router_names:
                sums = self.sums[wt][name]
                counts = self.counts[wt][name]
                base_sum = self.baseline_sums[wt][name]
                base_count = self.baseline_counts[wt][name]
                results[wt][name] = {}
                baseline = (
                    (base_sum / base_count).tolist()
                    if base_count > 0
                    else [0.0] * self.num_experts
                )
                for e in range(self.num_experts):
                    results[wt][name][e] = {}
                    for bi, label in enumerate(self.bin_labels):
                        c = int(counts[bi].item())
                        if c > 0:
                            mean_w = float(sums[e, bi].item() / c)
                        else:
                            mean_w = 0.0
                        b = baseline[e]
                        ratio = (mean_w / b) if b > 0 else 0.0
                        results[wt][name][e][label] = {
                            "mean_weight": mean_w,
                            "baseline": float(b),
                            "specialization_ratio": ratio,
                        }
        return results


# =============================================================================
# 4. Dispatch / combine weight heatmap collector
# =============================================================================


class DispatchHeatmapCollector(SoftMoEBaseAnalyzer):
    """
    Collects per-sequence heatmaps for BOTH weight types.
    Each stored entry: {weights, sequence, length}. `weights` is (seq_len, E).
    """

    def __init__(
        self,
        model: nn.Module,
        id_to_aa: Dict[int, str],
        max_sequences: int = 50,
    ):
        self.id_to_aa = id_to_aa
        self.max_sequences = max_sequences
        super().__init__(model)

    def _init_state(self):
        # heatmaps[weight_type][layer_name] -> list of dicts
        self.heatmaps: Dict[str, Dict[str, List[Dict]]] = {
            wt: {name: [] for name in self.router_names} for wt in WEIGHT_TYPES
        }
        # Count sequences separately per weight type so dispatch and combine
        # get paired entries for the same sequences.
        self.counts: Dict[str, int] = {wt: 0 for wt in WEIGHT_TYPES}

    def _process(
        self, layer_name, weight_type, weights, token_ids, attention_mask
    ):
        if self.counts[weight_type] >= self.max_sequences:
            return
        B = token_ids.shape[0]
        for b in range(B):
            if self.counts[weight_type] >= self.max_sequences:
                break
            valid_len = int(attention_mask[b].sum().item())
            w = weights[b, :valid_len].numpy()
            seq = [
                self.id_to_aa.get(int(token_ids[b, s].item()), "?")
                for s in range(valid_len)
            ]
            self.heatmaps[weight_type][layer_name].append(
                {
                    "weights": w,
                    "sequence": seq,
                    "length": valid_len,
                }
            )
            self.counts[weight_type] += 1

    def compute(self):
        """Returns {weight_type: {layer_name: [{weights, sequence, length}, ...]}}""" # noqa
        return self.heatmaps


# =============================================================================
# 5. Expert knockout analysis (unchanged — doesn't use hooks)
# =============================================================================


class ExpertKnockoutAnalyzer:
    """
    Zeros out one expert at a time and measures loss increase.
    Does NOT use hooks.
    """

    def __init__(
        self,
        model: nn.Module,
        forward_fn,
    ):
        self.model = model
        self.forward_fn = forward_fn

        self.router_names: List[str] = []
        self.routers: List[SoftRouter] = []
        for name, module in model.named_modules():
            if isinstance(module, SoftRouter):
                self.router_names.append(name)
                self.routers.append(module)

        if not self.routers:
            raise ValueError("No SoftRouter modules found in the model.")

        self.num_experts = self.routers[0].num_experts

    def run(
        self,
        dataloader,
        device: torch.device,
    ) -> Dict[str, Dict[int, float]]:
        self.model.eval()
        results = {}

        for layer_name, router in zip(self.router_names, self.routers):
            baseline_losses = []
            knockout_losses = {e: [] for e in range(self.num_experts)}

            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.no_grad():
                    baseline = self.forward_fn(
                        self.model, input_ids, attention_mask
                    ).item()
                baseline_losses.append(baseline)

                for expert_idx in range(self.num_experts):
                    saved = {}
                    for pname, param in router.experts.named_parameters():
                        if (
                            param.dim() >= 1
                            and param.shape[0] == self.num_experts
                        ):
                            saved[pname] = param.data[expert_idx].clone()
                            param.data[expert_idx].zero_()

                    with torch.no_grad():
                        ko_loss = self.forward_fn(
                            self.model, input_ids, attention_mask
                        ).item()
                    knockout_losses[expert_idx].append(ko_loss)

                    for pname, param in router.experts.named_parameters():
                        if pname in saved:
                            param.data[expert_idx].copy_(saved[pname])

            mean_baseline = np.mean(baseline_losses)
            results[layer_name] = {
                e: np.mean(knockout_losses[e]) - mean_baseline
                for e in range(self.num_experts)
            }

        return results


class UniRef30(Dataset):
    def __init__(
        self, max_examples: Optional[int] = None, split: str = "train"
    ):
        self.data = load_dataset(
            "hazemessam/sprot",
            data_files={split: split + ".parquet"},
            streaming=False,
        )[split]
        self.max_examples = (
            max_examples if max_examples is not None else len(self.data)
        )

    def __len__(self):
        return self.max_examples

    def __getitem__(self, idx):
        return {"sequence": self.data[idx]["sequence"]}


@dataclass
class Collator:
    tokenizer: AresProteinTokenizer

    def __call__(self, batch):
        b = defaultdict(list)
        for example in batch:
            for k, v in example.items():
                b[k].append(v)
        encoded = self.tokenizer(
            b["sequence"],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        return encoded


def save_results(
    results: dict,
    path: str,
    fmt: str = "json",
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(path, "w") as f:
            json.dump(convert(results), f, indent=2)

    elif fmt == "pickle":
        with open(path, "wb") as f:
            pickle.dump(results, f)

    else:
        raise ValueError(f"Unknown format: {fmt}")


def main():
    model = Ares.from_pretrained(
        "HazemLab/ares-softmoe-4b-consecutive", device_map="cuda:1"
    )
    tokenizer = AresProteinTokenizer()

    id_to_aa = {v: k for k, v in tokenizer.get_vocab().items()}
    ignore_ids = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
    }

    aa_analyzer = ExpertAminoAcidAnalyzer(model, id_to_aa, ignore_ids)
    property_analyzer = ExpertPropertyAnalyzer(model, id_to_aa, ignore_ids)
    positional_analyzer = ExpertPositionalAnalyzer(model, num_bins=5)
    heatmap_collector = DispatchHeatmapCollector(
        model, id_to_aa, max_sequences=10
    )

    dataset = UniRef30(max_examples=1000)
    collator = Collator(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collator,
        shuffle=True,
        generator=torch.manual_seed(42),
    )
    device = next(model.parameters()).device

    model.eval()
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
        aa_analyzer.update(input_ids, attention_mask)
        property_analyzer.update(input_ids, attention_mask)
        positional_analyzer.update(input_ids, attention_mask)
        heatmap_collector.update(input_ids, attention_mask)

    aa_results = aa_analyzer.compute()
    property_results = property_analyzer.compute()
    positional_results = positional_analyzer.compute()
    heatmaps = heatmap_collector.compute()

    output_dir = "analysis_outputs"
    save_results(aa_results, f"{output_dir}/amino_acid_preferences.json")
    save_results(property_results, f"{output_dir}/property_preferences.json")
    save_results(
        positional_results, f"{output_dir}/positional_preferences.json"
    )
    save_results(heatmaps, f"{output_dir}/dispatch_heatmaps.pkl", fmt="pickle")

    aa_analyzer.remove()
    property_analyzer.remove()
    positional_analyzer.remove()
    heatmap_collector.remove()


if __name__ == "__main__":
    main()
