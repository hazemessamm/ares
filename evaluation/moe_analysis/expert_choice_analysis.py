"""Analysis utilities for ExpertChoiceRouting layers.

Mirrors `soft_moe_analysis.py` but operates on the discrete top-k routing
decisions made by `ExpertChoiceRouter`. For each layer we reconstruct two
dense (B, S, E) tensors per forward pass:

    "selection": binary 0/1 — did expert e pick token (b, s)?
    "weighted":  router_prob if selected, 0 otherwise.

These are direct analogues of the SoftMoE "dispatch" / "combine" tensors
(after the slot dimension is reduced), so the per-AA / per-property /
per-position analyses port over almost verbatim. A few additional analyzers
(coverage, co-occurrence) are included because expert-choice routing has
properties that don't exist in the soft variant: tokens can be dropped
entirely or selected by multiple experts.

To enable capture, the analyzers flip `router.capture_routing = True` on
every `ExpertChoiceRouter`. The hook then reads `tokens_prob_` and
`tokens_indices_`, scatters them into dense (B, S, E) matrices on the GPU,
moves the result to CPU, and clears the GPU-side cache so it doesn't
accumulate across layers.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ares.models.expert_choice_router import (
    ExpertChoiceRouter,
    ExpertChoiceRouting,
)


AA_PROPERTIES = {
    "hydrophobic": set("AILMFWV"),
    "polar_uncharged": set("STNQY"),
    "positive": set("RHK"),
    "negative": set("DE"),
    "charged": set("RHKDE"),
    "aromatic": set("FWYH"),
    "small": set("GAST"),
    "cysteine": set("C"),
    "proline": set("P"),
}


# Weight types captured per layer.
#
# These are SEMANTICALLY DIFFERENT signals and the same metric (mean /
# baseline / ratio) means different things on each. Always read the
# descriptions before interpreting "specialization_ratio".
WEIGHT_TYPES = ("selection", "weighted")

WEIGHT_DESCRIPTIONS = {
    "selection": (
        "Selection is a binary 0/1 indicator: did expert e pick token "
        "(b, s) into one of its top-k slots? mean_weight is the fraction "
        "of tokens of the given class that get selected by expert e, and "
        "specialization_ratio compares that fraction against expert e's "
        "overall selection rate (which is bounded by capacity_factor / E "
        "across the whole batch). Treats every selection equally; ignores "
        "the router's confidence."
    ),
    "weighted": (
        "Weighted = router softmax probability if selected, else 0. "
        "specialization_ratio here couples selection rate with the "
        "router's confidence, analogous to SoftMoE's 'combine' weights, "
        "and is the value most directly proportional to the expert's "
        "contribution to the layer output."
    ),
}

PROPERTY_OVERLAP_NOTE = (
    "Property groups intentionally overlap (e.g. F is in both 'hydrophobic' "
    "and 'aromatic'). High specialization in two correlated groups is often "
    "driven by a shared subset of residues; cross-check against the per-AA "
    "results in `amino_acid_preferences.json` before drawing conclusions."
)


def _safe_log_ratio(mean_w: float, baseline: float) -> Optional[float]:
    """log(mean / baseline). Returns None when either side is non-positive."""
    if mean_w <= 0.0 or baseline <= 0.0:
        return None
    return float(math.log(mean_w) - math.log(baseline))


def _entry(
    mean_w: float,
    baseline: float,
    count: int,
    min_count: int,
) -> Dict[str, object]:
    ratio = (mean_w / baseline) if baseline > 0 else 0.0
    return {
        "mean_weight": mean_w,
        "baseline": float(baseline),
        "specialization_ratio": ratio,
        "log_ratio": _safe_log_ratio(mean_w, baseline),
        "count": int(count),
        "low_support": int(count) < int(min_count),
    }


# =============================================================================
# Base class: registers hooks on every ExpertChoiceRouting and reconstructs
# (B, S, E) selection / weighted matrices per forward pass.
# =============================================================================

class ExpertChoiceBaseAnalyzer:
    """
    Subclasses implement:
        _init_state()
        _process(layer_name, weight_type, weights, token_ids, attention_mask)
            where weights is (B, S, E), already on CPU
        compute()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: List = []
        self.router_names: List[str] = []
        self.routings: List[ExpertChoiceRouting] = []

        # _last[layer_name][weight_type] -> CPU tensor (B, S, E)
        self._last: Dict[str, Dict[str, torch.Tensor]] = {}

        for name, module in model.named_modules():
            if isinstance(module, ExpertChoiceRouting):
                self.router_names.append(name)
                self.routings.append(module)

        if not self.routings:
            raise ValueError(
                "No ExpertChoiceRouting modules found in the model."
            )

        self.num_experts = self.routings[0].num_experts
        self.num_layers = len(self.routings)

        # Turn on routing-decision capture. The hook resets the buffers
        # after reading them so production memory pressure stays minimal.
        for routing in self.routings:
            routing.router.capture_routing = True

        self._init_state()
        self._register_hooks()

    def _register_hooks(self):
        for name, routing in zip(self.router_names, self.routings):
            hook = routing.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)

    def _make_hook(self, layer_name: str):
        # Multiple analyzer instances may register hooks on the same routing
        # module. The first one to fire after a forward (when
        # `tokens_prob_` / `tokens_indices_` are still populated) does the
        # scatter + GPU->CPU transfer, then clears the GPU caches. Later
        # analyzer hooks reuse the cached CPU tensors via a shared
        # module-level attribute.
        def hook_fn(module: ExpertChoiceRouting, input, output):
            inner = module.router
            tokens_prob = getattr(inner, "tokens_prob_", None)
            tokens_indices = getattr(inner, "tokens_indices_", None)
            has_fresh = tokens_prob is not None and tokens_indices is not None

            if has_fresh:
                # input[0] is x with original shape (B, S, D). The router
                # internally flattens to (B*S, D); tokens_indices contains
                # flat indices into that flattened view.
                x = input[0]
                if x.ndim == 3:
                    B, S, _ = x.shape
                elif x.ndim == 2:
                    # Already flat; we don't know B/S, treat the whole thing
                    # as a single "batch" of length B*S.
                    B, S = 1, x.shape[0]
                else:
                    raise RuntimeError(
                        f"Unexpected input rank {x.ndim} for ExpertChoiceRouting."
                    )

                E, k = tokens_indices.shape
                device = tokens_indices.device

                # Scatter top-k decisions into dense (B*S, E) matrices.
                # Each (e, j) slot writes to row tokens_indices[e, j],
                # column e. tokens_indices is unique per row (topk result),
                # and writes across different e land in different columns,
                # so there are no collisions.
                e_idx = torch.arange(
                    E, device=device, dtype=torch.long
                ).unsqueeze(1).expand(E, k)

                selection = torch.zeros(
                    B * S, E, dtype=torch.float32, device=device
                )
                weighted = torch.zeros(
                    B * S, E, dtype=torch.float32, device=device
                )
                selection[tokens_indices, e_idx] = 1.0
                weighted[tokens_indices, e_idx] = tokens_prob.to(torch.float32)

                cache = {
                    "selection": selection.view(B, S, E).detach().cpu(),
                    "weighted": weighted.view(B, S, E).detach().cpu(),
                }

                inner.tokens_prob_ = None
                inner.tokens_indices_ = None
                module._analyzer_summed_cache = cache

            cache = getattr(module, "_analyzer_summed_cache", None)
            if cache:
                self._last[layer_name] = cache

        return hook_fn

    def update(self, token_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Call after each forward pass.

        The captured tensors are already on CPU, so this just dispatches
        them into `_process` for accumulation.
        """
        token_ids_cpu = token_ids.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu()
        for layer_name in self.router_names:
            entry = self._last.get(layer_name)
            if entry is None:
                continue
            for weight_type, tensor in entry.items():
                self._process(
                    layer_name, weight_type,
                    tensor, token_ids_cpu, attention_mask_cpu,
                )
        self._last.clear()

    def _init_state(self):
        raise NotImplementedError

    def _process(self, layer_name, weight_type, weights, token_ids, attention_mask):
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
        for routing in self.routings:
            routing.router.capture_routing = False
            routing.router.tokens_prob_ = None
            routing.router.tokens_indices_ = None
            if hasattr(routing, "_analyzer_summed_cache"):
                routing._analyzer_summed_cache = None


# =============================================================================
# 1. Per-expert amino acid preferences
# =============================================================================

class ExpertAminoAcidAnalyzer(ExpertChoiceBaseAnalyzer):
    """
    For each (layer, expert, amino_acid):
        - mean_weight: average (selection or weighted) value across tokens
                       whose amino-acid is `aa`
        - baseline:    same average over all valid tokens
        - specialization_ratio: mean_weight / baseline

    For "selection", `mean_weight` is the fraction of tokens of the given AA
    that get picked by the expert.
    """

    def __init__(
        self,
        model: nn.Module,
        id_to_aa: Dict[int, str],
        ignore_ids: Optional[Set[int]] = None,
    ):
        self.id_to_aa = id_to_aa
        self.ignore_ids = ignore_ids or set()
        self._aa_list = sorted(set(
            aa for tid, aa in id_to_aa.items()
            if tid not in self.ignore_ids
        ))
        self._aa_to_idx = {aa: i for i, aa in enumerate(self._aa_list)}
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
        self.sums: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {name: torch.zeros(self.num_experts, num_aa, dtype=torch.float64)
                 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }
        self.counts: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {name: torch.zeros(num_aa, dtype=torch.long)
                 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }
        self.baseline_sums: Dict[str, Dict[str, torch.Tensor]] = {
            wt: {name: torch.zeros(self.num_experts, dtype=torch.float64)
                 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }
        self.baseline_counts: Dict[str, Dict[str, int]] = {
            wt: {name: 0 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }

    def _process(self, layer_name, weight_type, weights, token_ids, attention_mask):
        B, S, E = weights.shape
        flat_weights = weights.reshape(-1, E).to(torch.float64)
        flat_tokens = token_ids.reshape(-1)
        flat_mask = attention_mask.reshape(-1).bool()

        safe_tokens = flat_tokens.clamp(max=self._max_tid)
        aa_idx = self._tid_to_aa_idx[safe_tokens]
        aa_idx = aa_idx.masked_fill(~flat_mask, -1)
        oversize = flat_tokens > self._max_tid
        aa_idx = aa_idx.masked_fill(oversize, -1)

        valid = aa_idx >= 0
        if not valid.any():
            return

        valid_weights = flat_weights[valid]
        valid_aa_idx = aa_idx[valid]

        self.baseline_sums[weight_type][layer_name] += valid_weights.sum(dim=0)
        self.baseline_counts[weight_type][layer_name] += int(valid.sum().item())

        sums = self.sums[weight_type][layer_name]
        sums_t = sums.t().contiguous()
        sums_t.index_add_(0, valid_aa_idx, valid_weights)
        self.sums[weight_type][layer_name] = sums_t.t().contiguous()

        counts = self.counts[weight_type][layer_name]
        ones = torch.ones_like(valid_aa_idx, dtype=torch.long)
        counts.index_add_(0, valid_aa_idx, ones)

    def compute(self, min_count: int = 0) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {wt: {} for wt in WEIGHT_TYPES}
        results["_metadata"] = {
            "weight_descriptions": WEIGHT_DESCRIPTIONS,
            "min_count": int(min_count),
        }
        for wt in WEIGHT_TYPES:
            for name in self.router_names:
                sums = self.sums[wt][name]
                counts = self.counts[wt][name]
                base_sum = self.baseline_sums[wt][name]
                base_count = self.baseline_counts[wt][name]
                results[wt][name] = {}
                baseline = (
                    (base_sum / base_count).tolist()
                    if base_count > 0 else [0.0] * self.num_experts
                )
                for e in range(self.num_experts):
                    results[wt][name][e] = {}
                    for aa_i, aa in enumerate(self._aa_list):
                        c = int(counts[aa_i].item())
                        mean_w = float(sums[e, aa_i].item() / c) if c > 0 else 0.0
                        b = baseline[e]
                        results[wt][name][e][aa] = _entry(mean_w, b, c, min_count)
        return results


# =============================================================================
# 2. Per-expert amino acid property group preferences
# =============================================================================

class ExpertPropertyAnalyzer(ExpertChoiceBaseAnalyzer):
    """
    For each (layer, expert, property_group): mean_weight, baseline,
    specialization_ratio. Handles overlapping groups (a token contributes
    to every group it belongs to).
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

        max_tid = max(id_to_aa.keys()) if id_to_aa else 0
        self._group_order = list(self.property_groups.keys())
        num_groups = len(self._group_order)
        self._membership = torch.zeros(max_tid + 1, num_groups, dtype=torch.bool)
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
        self.sums = {
            wt: {name: torch.zeros(self.num_experts, num_groups, dtype=torch.float64)
                 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }
        self.counts = {
            wt: {name: torch.zeros(num_groups, dtype=torch.long)
                 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }
        self.baseline_sums = {
            wt: {name: torch.zeros(self.num_experts, dtype=torch.float64)
                 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }
        self.baseline_counts = {
            wt: {name: 0 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }

    def _process(self, layer_name, weight_type, weights, token_ids, attention_mask):
        B, S, E = weights.shape
        flat_weights = weights.reshape(-1, E).to(torch.float64)
        flat_tokens = token_ids.reshape(-1)
        flat_mask = attention_mask.reshape(-1).bool()

        oversize = flat_tokens > self._max_tid
        safe_tokens = flat_tokens.clamp(max=self._max_tid)
        is_valid_aa = self._is_valid_aa[safe_tokens] & flat_mask & (~oversize)
        if not is_valid_aa.any():
            return

        valid_weights = flat_weights[is_valid_aa]
        self.baseline_sums[weight_type][layer_name] += valid_weights.sum(dim=0)
        self.baseline_counts[weight_type][layer_name] += int(is_valid_aa.sum().item())

        membership = self._membership[safe_tokens]
        membership = membership & is_valid_aa.unsqueeze(-1)
        mem_f = membership.to(torch.float64)
        self.sums[weight_type][layer_name] += flat_weights.t() @ mem_f
        self.counts[weight_type][layer_name] += membership.sum(dim=0).to(torch.long)

    def compute(self, min_count: int = 0) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {wt: {} for wt in WEIGHT_TYPES}
        results["_metadata"] = {
            "weight_descriptions": WEIGHT_DESCRIPTIONS,
            "property_groups": {
                g: sorted(self.property_groups[g]) for g in self._group_order
            },
            "overlap_note": PROPERTY_OVERLAP_NOTE,
            "min_count": int(min_count),
        }
        for wt in WEIGHT_TYPES:
            for name in self.router_names:
                sums = self.sums[wt][name]
                counts = self.counts[wt][name]
                base_sum = self.baseline_sums[wt][name]
                base_count = self.baseline_counts[wt][name]
                results[wt][name] = {}
                baseline = (
                    (base_sum / base_count).tolist()
                    if base_count > 0 else [0.0] * self.num_experts
                )
                for e in range(self.num_experts):
                    results[wt][name][e] = {}
                    for g_i, g in enumerate(self._group_order):
                        c = int(counts[g_i].item())
                        mean_w = float(sums[e, g_i].item() / c) if c > 0 else 0.0
                        b = baseline[e]
                        results[wt][name][e][g] = _entry(mean_w, b, c, min_count)
        return results


# =============================================================================
# 3. Positional preferences (relative position bins)
# =============================================================================

class ExpertPositionalAnalyzer(ExpertChoiceBaseAnalyzer):
    """
    For each (layer, expert, relative_position_bin): mean_weight, baseline,
    specialization_ratio.
    """

    def __init__(self, model: nn.Module, num_bins: int = 5):
        self.num_bins = num_bins
        super().__init__(model)

    def _init_state(self):
        self.bin_labels = [
            f"{int(i / self.num_bins * 100)}-{int((i + 1) / self.num_bins * 100)}%"
            for i in range(self.num_bins)
        ]
        self.sums = {
            wt: {name: torch.zeros(self.num_experts, self.num_bins, dtype=torch.float64)
                 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }
        self.counts = {
            wt: {name: torch.zeros(self.num_bins, dtype=torch.long)
                 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }
        self.baseline_sums = {
            wt: {name: torch.zeros(self.num_experts, dtype=torch.float64)
                 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }
        self.baseline_counts = {
            wt: {name: 0 for name in self.router_names}
            for wt in WEIGHT_TYPES
        }

    def _process(self, layer_name, weight_type, weights, token_ids, attention_mask):
        B, S, E = weights.shape
        valid_lens = attention_mask.sum(dim=1).to(torch.long)
        bin_idx = torch.full((B, S), -1, dtype=torch.long)
        for b in range(B):
            L = int(valid_lens[b].item())
            if L == 0:
                continue
            positions = torch.arange(L, dtype=torch.float64)
            rel_pos = positions / L
            bins = (rel_pos * self.num_bins).to(torch.long).clamp(max=self.num_bins - 1)
            bin_idx[b, :L] = bins

        flat_weights = weights.reshape(-1, E).to(torch.float64)
        flat_bin = bin_idx.reshape(-1)
        valid = flat_bin >= 0
        if not valid.any():
            return

        valid_weights = flat_weights[valid]
        valid_bins = flat_bin[valid]

        self.baseline_sums[weight_type][layer_name] += valid_weights.sum(dim=0)
        self.baseline_counts[weight_type][layer_name] += int(valid.sum().item())

        sums = self.sums[weight_type][layer_name]
        sums_t = sums.t().contiguous()
        sums_t.index_add_(0, valid_bins, valid_weights)
        self.sums[weight_type][layer_name] = sums_t.t().contiguous()

        counts = self.counts[weight_type][layer_name]
        ones = torch.ones_like(valid_bins, dtype=torch.long)
        counts.index_add_(0, valid_bins, ones)

    def compute(self, min_count: int = 0) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {wt: {} for wt in WEIGHT_TYPES}
        results["_metadata"] = {
            "weight_descriptions": WEIGHT_DESCRIPTIONS,
            "num_bins": self.num_bins,
            "min_count": int(min_count),
        }
        for wt in WEIGHT_TYPES:
            for name in self.router_names:
                sums = self.sums[wt][name]
                counts = self.counts[wt][name]
                base_sum = self.baseline_sums[wt][name]
                base_count = self.baseline_counts[wt][name]
                results[wt][name] = {}
                baseline = (
                    (base_sum / base_count).tolist()
                    if base_count > 0 else [0.0] * self.num_experts
                )
                for e in range(self.num_experts):
                    results[wt][name][e] = {}
                    for bi, label in enumerate(self.bin_labels):
                        c = int(counts[bi].item())
                        mean_w = float(sums[e, bi].item() / c) if c > 0 else 0.0
                        b = baseline[e]
                        results[wt][name][e][label] = _entry(mean_w, b, c, min_count)
        return results


# =============================================================================
# 4. Token coverage analyzer (drop rate / over-subscription)
# =============================================================================

class TokenCoverageAnalyzer(ExpertChoiceBaseAnalyzer):
    """
    Per layer:
        - drop_rate:             fraction of valid tokens NOT picked by any expert
        - mean_experts_per_token: average number of experts that pick a valid token
        - coverage_histogram:    distribution over {0, 1, 2, ..., num_experts}
                                 of "how many experts picked this token".

    This analyzer only consumes the "selection" weight type since binary
    counts are what's meaningful here.
    """

    def _init_state(self):
        bins = self.num_experts + 1  # 0, 1, ..., num_experts
        self.histogram: Dict[str, torch.Tensor] = {
            name: torch.zeros(bins, dtype=torch.long)
            for name in self.router_names
        }
        self.total_tokens: Dict[str, int] = {
            name: 0 for name in self.router_names
        }

    def _process(self, layer_name, weight_type, weights, token_ids, attention_mask):
        if weight_type != "selection":
            return
        B, S, E = weights.shape
        flat_mask = attention_mask.reshape(-1).bool()
        flat_weights = weights.reshape(-1, E)
        valid_weights = flat_weights[flat_mask]  # (N_valid, E)
        if valid_weights.numel() == 0:
            return
        # Number of experts selecting each valid token.
        per_token_count = valid_weights.sum(dim=1).to(torch.long)  # (N_valid,)
        per_token_count = per_token_count.clamp(max=self.num_experts)
        hist = torch.bincount(per_token_count, minlength=self.num_experts + 1)
        self.histogram[layer_name] += hist
        self.total_tokens[layer_name] += int(flat_mask.sum().item())

    def compute(self):
        results: Dict[str, Dict] = {}
        for name in self.router_names:
            hist = self.histogram[name]
            total = self.total_tokens[name]
            if total == 0:
                results[name] = {
                    "drop_rate": 0.0,
                    "mean_experts_per_token": 0.0,
                    "coverage_histogram": {i: 0 for i in range(self.num_experts + 1)},
                    "coverage_fractions": {i: 0.0 for i in range(self.num_experts + 1)},
                }
                continue
            drop_rate = float(hist[0].item()) / total
            counts = torch.arange(self.num_experts + 1, dtype=torch.float64)
            mean_eps = float((hist.to(torch.float64) * counts).sum().item() / total)
            results[name] = {
                "drop_rate": drop_rate,
                "mean_experts_per_token": mean_eps,
                "coverage_histogram": {
                    i: int(hist[i].item()) for i in range(self.num_experts + 1)
                },
                "coverage_fractions": {
                    i: float(hist[i].item()) / total
                    for i in range(self.num_experts + 1)
                },
            }
        return results


# =============================================================================
# 5. Expert co-occurrence analyzer
# =============================================================================

class ExpertCoOccurrenceAnalyzer(ExpertChoiceBaseAnalyzer):
    """
    Per layer, accumulates a (num_experts, num_experts) matrix where
    entry [i, j] is the number of valid tokens picked by BOTH expert i
    and expert j. The diagonal is just the per-expert selection count.

    `compute()` returns both raw counts and a normalized "Jaccard-like"
    fraction = co_occurrence[i, j] / (selected_by_i + selected_by_j - co_occurrence[i, j]).

    Only operates on the "selection" weight type.
    """

    def _init_state(self):
        self.co_occurrence: Dict[str, torch.Tensor] = {
            name: torch.zeros(
                self.num_experts, self.num_experts, dtype=torch.float64
            )
            for name in self.router_names
        }
        self.total_tokens: Dict[str, int] = {
            name: 0 for name in self.router_names
        }

    def _process(self, layer_name, weight_type, weights, token_ids, attention_mask):
        if weight_type != "selection":
            return
        B, S, E = weights.shape
        flat_mask = attention_mask.reshape(-1).bool()
        flat_weights = weights.reshape(-1, E)
        valid_weights = flat_weights[flat_mask].to(torch.float64)
        if valid_weights.numel() == 0:
            return
        # selection.T @ selection counts pairwise co-occurrence over tokens.
        self.co_occurrence[layer_name] += valid_weights.t() @ valid_weights
        self.total_tokens[layer_name] += int(flat_mask.sum().item())

    def compute(self):
        results: Dict[str, Dict] = {}
        for name in self.router_names:
            co = self.co_occurrence[name]
            E = self.num_experts
            diag = torch.diagonal(co)  # (E,) per-expert selection counts
            jaccard = torch.zeros_like(co)
            for i in range(E):
                for j in range(E):
                    union = diag[i] + diag[j] - co[i, j]
                    jaccard[i, j] = (co[i, j] / union) if union > 0 else 0.0
            results[name] = {
                "co_occurrence_counts": co.tolist(),
                "selections_per_expert": diag.tolist(),
                "jaccard": jaccard.tolist(),
                "total_valid_tokens": self.total_tokens[name],
            }
        return results


# =============================================================================
# 6. Per-sequence routing heatmap collector
# =============================================================================

class ExpertChoiceHeatmapCollector(ExpertChoiceBaseAnalyzer):
    """
    Collects per-sequence routing heatmaps. For each captured sequence,
    stores a (seq_len, num_experts) selection mask AND the corresponding
    weighted matrix, alongside the AA letters.
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
        self.heatmaps: Dict[str, Dict[str, List[Dict]]] = {
            wt: {name: [] for name in self.router_names} for wt in WEIGHT_TYPES
        }
        self.counts: Dict[str, int] = {wt: 0 for wt in WEIGHT_TYPES}

    def _process(self, layer_name, weight_type, weights, token_ids, attention_mask):
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
            self.heatmaps[weight_type][layer_name].append({
                "weights": w,
                "sequence": seq,
                "length": valid_len,
            })
            self.counts[weight_type] += 1

    def compute(self):
        return self.heatmaps


# =============================================================================
# 7. Expert knockout analysis (no hooks; mirrors soft-MoE version)
# =============================================================================

class ExpertKnockoutAnalyzer:
    """
    Per-expert ablation analysis. For each (layer, expert), zeroes the
    expert's routing probabilities so its expert MLP contributes nothing
    to the layer output, then measures the change in
    `forward_fn(model, input_ids, attention_mask)` (typically a loss).

    Implementation: rather than zeroing the expert's PARAMETERS (which
    leaves the expert producing junk output that still gets weighted into
    the residual stream and measures "corruption robustness", not an
    ablation), we set `routing.router.knockout_expert = e` so the router
    itself zeros that expert's combine weights.
    """

    def __init__(self, model: nn.Module, forward_fn):
        self.model = model
        self.forward_fn = forward_fn

        self.router_names: List[str] = []
        self.routings: List[ExpertChoiceRouting] = []
        for name, module in model.named_modules():
            if isinstance(module, ExpertChoiceRouting):
                self.router_names.append(name)
                self.routings.append(module)

        if not self.routings:
            raise ValueError(
                "No ExpertChoiceRouting modules found in the model."
            )

        self.num_experts = self.routings[0].num_experts

    def run(
        self,
        dataloader,
        device: torch.device,
    ) -> Dict[str, Dict[str, object]]:
        """
        Returns:
            {layer_name: {
                "baseline_loss": float,
                "delta_per_expert": {expert_idx: float, ...},
                "relative_delta_per_expert": {expert_idx: float, ...},
            }, ...}
        """
        self.model.eval()
        results = {}

        for layer_name, routing in zip(self.router_names, self.routings):
            baseline_losses = []
            knockout_losses = {e: [] for e in range(self.num_experts)}

            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                assert routing.router.knockout_expert is None, (
                    "knockout_expert should be cleared before each batch"
                )
                with torch.no_grad():
                    baseline = self.forward_fn(
                        self.model, input_ids, attention_mask
                    ).item()
                baseline_losses.append(baseline)

                for expert_idx in range(self.num_experts):
                    routing.router.knockout_expert = expert_idx
                    try:
                        with torch.no_grad():
                            ko_loss = self.forward_fn(
                                self.model, input_ids, attention_mask
                            ).item()
                    finally:
                        routing.router.knockout_expert = None
                    knockout_losses[expert_idx].append(ko_loss)

            mean_baseline = float(np.mean(baseline_losses))
            deltas = {
                e: float(np.mean(knockout_losses[e]) - mean_baseline)
                for e in range(self.num_experts)
            }
            relative = {
                e: (deltas[e] / mean_baseline) if mean_baseline != 0 else 0.0
                for e in range(self.num_experts)
            }
            results[layer_name] = {
                "baseline_loss": mean_baseline,
                "delta_per_expert": deltas,
                "relative_delta_per_expert": relative,
            }

        return results


# =============================================================================
# Entry point
# =============================================================================

from ares.models.model import Ares
from ares.tokenization.protein_tokenizer import AresProteinTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from dataclasses import dataclass
from tqdm import tqdm

import json
import pickle
from pathlib import Path


class UniRef30(Dataset):
    def __init__(self, max_examples: Optional[int] = None, split: str = "train"):
        self.data = load_dataset(
            "hazemessam/sprot",
            data_files={split: split + ".parquet"},
            streaming=False,
        )[split]
        self.max_examples = max_examples if max_examples is not None else len(self.data)

    def __len__(self):
        return self.max_examples

    def __getitem__(self, idx):
        return {"sequence": self.data[idx]["sequence"]}


class Sprot(Dataset):
    # Swissprot
    def __init__(self, max_examples: Optional[int] = None, split: str = "train", max_length = 1024):
        self.data = load_dataset(
            "hazemessam/sprot",
            data_files={split: split + ".parquet"},
            streaming=False,
        )[split]
        self.max_examples = max_examples if max_examples is not None else len(self.data)
        counter = 0
        self.sequences = []
        for example in self.data:
            if len(example["sequence"]) <= max_length:
                self.sequences.append(example["sequence"])
                counter += 1
            if max_examples is not None and counter >= max_examples:
                break
        
        if max_examples is not None and len(self.sequences) < max_examples:
            raise ValueError(f"Not enough sequences found in {split} to reach max_examples")

    def __len__(self):
        return self.max_examples

    def __getitem__(self, idx):
        return {"sequence": self.sequences[idx]}


@dataclass
class Collator:
    tokenizer: AresProteinTokenizer

    def __call__(self, batch):
        b = defaultdict(list)
        for example in batch:
            for k, v in example.items():
                b[k].append(v)
        encoded = self.tokenizer(
            b["sequence"], return_tensors="pt", padding=True, truncation=False,
        )
        return encoded


def save_results(results: dict, path: str, fmt: str = "json"):
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
        "HazemLab/ares-expert-choice-4b-interleaved-150K", device_map="cuda:1"
    )
    tokenizer = AresProteinTokenizer()

    id_to_aa = {v: k for k, v in tokenizer.get_vocab().items()}
    # Drop ambiguity codes from the vocabulary used for AA analyses.
    # B = Asx (N or D), Z = Glx (Q or E), J = Xle (L or I), X = Xaa (unknown).
    # These are annotation placeholders, not real residues. U (Sec) and
    # O (Pyl) ARE real residues but very rare; we keep them and rely on
    # the `low_support` flag to mark them.
    AMBIGUITY_CODES = {"B", "Z", "J", "X"}
    ambiguity_ids = {
        tid for tid, aa in id_to_aa.items() if aa in AMBIGUITY_CODES
    }
    ignore_ids = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
    } | ambiguity_ids

    aa_analyzer = ExpertAminoAcidAnalyzer(model, id_to_aa, ignore_ids)
    property_analyzer = ExpertPropertyAnalyzer(model, id_to_aa, ignore_ids)
    positional_analyzer = ExpertPositionalAnalyzer(model, num_bins=5)
    coverage_analyzer = TokenCoverageAnalyzer(model)
    cooccurrence_analyzer = ExpertCoOccurrenceAnalyzer(model)
    heatmap_collector = ExpertChoiceHeatmapCollector(
        model, id_to_aa, max_sequences=10
    )

    dataset = Sprot(max_examples=10000)
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
        coverage_analyzer.update(input_ids, attention_mask)
        cooccurrence_analyzer.update(input_ids, attention_mask)
        heatmap_collector.update(input_ids, attention_mask)

    MIN_SUPPORT = 50
    aa_results = aa_analyzer.compute(min_count=MIN_SUPPORT)
    property_results = property_analyzer.compute(min_count=MIN_SUPPORT)
    positional_results = positional_analyzer.compute(min_count=MIN_SUPPORT)
    coverage_results = coverage_analyzer.compute()
    cooccurrence_results = cooccurrence_analyzer.compute()
    heatmaps = heatmap_collector.compute()

    output_dir = "expert_choice_analysis_outputs"
    save_results(aa_results, f"{output_dir}/amino_acid_preferences.json")
    save_results(property_results, f"{output_dir}/property_preferences.json")
    save_results(positional_results, f"{output_dir}/positional_preferences.json")
    save_results(coverage_results, f"{output_dir}/token_coverage.json")
    save_results(cooccurrence_results, f"{output_dir}/expert_co_occurrence.json")
    save_results(heatmaps, f"{output_dir}/routing_heatmaps.pkl", fmt="pickle")

    aa_analyzer.remove()
    property_analyzer.remove()
    positional_analyzer.remove()
    coverage_analyzer.remove()
    cooccurrence_analyzer.remove()
    heatmap_collector.remove()


if __name__ == "__main__":
    main()
