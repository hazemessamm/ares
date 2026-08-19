"""Tier-2 diagnostic for the L7 anomaly: single-layer expert knockout.

Asks the central question that Tier-1 (read-only) diagnostics cannot
answer: are the "redundant cluster" experts at the anomalous layer
actually load-bearing, or are they dispensable?

Method
------
For each expert e at the target layer:
  1. Set ``routing.router.knockout_expert = e``. The router itself
     zeroes that expert's combine column post-topk, so its expert MLP
     contributes nothing to the layer output. (We do NOT zero the
     expert's parameters — that leaves a still-routed-to expert
     producing junk and measures corruption, not ablation.)
  2. Run a *masked* LM forward pass (15% random masking, same mask
     used for the un-ablated baseline so the comparison is paired).
  3. Record cross-entropy on the masked positions only.

We then cross-reference each expert with its mean off-diagonal Jaccard
from the existing ``expert_co_occurrence.json``, classify experts into
"redundant" (top half by mean Jaccard at this layer) vs "isolated"
(bottom half), and report:

  - Per-expert delta loss and relative delta loss.
  - Mean delta within each group → tests "redundant ⇒ dispensable".
  - A scatter of mean Jaccard vs. relative delta — the strongest single
    figure for the paper if the trend is monotonic.

Outputs
-------
  - ``<output-dir>/<layer_short>_knockout.json``  — full numerical results
  - ``<output-dir>/<layer_short>_knockout_bars.pdf``  — per-expert bars
  - ``<output-dir>/<layer_short>_knockout_scatter.pdf`` — Jaccard vs Δ

Usage
-----
    python l7_knockout.py --layer L7 --num-batches 200 --device cuda:1
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

from ares.models.model import Ares
from ares.models.expert_choice_router import ExpertChoiceRouting
from ares.tokenization.protein_tokenizer import AresProteinTokenizer

from visualize_analysis import apply_style, _short_layer_label, load_json


# =============================================================================
# Data
# =============================================================================

class UniRef30(Dataset):
    def __init__(self, max_examples: Optional[int] = None, split: str = "train"):
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
            b["sequence"], return_tensors="pt", padding=True, truncation=False,
        )
        return encoded


# =============================================================================
# Masked-LM batch construction
# =============================================================================

def make_mlm_batch(
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    *,
    tokenizer: AresProteinTokenizer,
    mask_prob: float = 0.15,
    seed: int,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Return (masked_input_ids, labels) for a standard MLM loss.

    Special tokens (pad/cls/eos/mask) and any token outside attention
    are *never* masked. ``labels`` is -100 everywhere except the masked
    positions, where it equals the original token.

    A per-batch seed makes the mask reproducible across the (baseline +
    32 ablations) forward passes, so the only thing that varies is the
    expert that's been knocked out.
    """
    device = input_ids.device
    special_ids = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
    }

    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = torch.rand(input_ids.shape, generator=g).to(device)

    is_special = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        if sid is not None:
            is_special |= input_ids == sid

    eligible = (attention_mask.bool()) & (~is_special)
    mask = (rand < mask_prob) & eligible

    if not mask.any():
        # Force at least one masked position (rare but possible for tiny
        # sequences). Pick the first eligible token.
        flat_eligible = eligible.flatten()
        if flat_eligible.any():
            first = int(flat_eligible.nonzero(as_tuple=False)[0].item())
            flat_mask = mask.flatten().clone()
            flat_mask[first] = True
            mask = flat_mask.view_as(mask)

    masked_input_ids = input_ids.clone()
    masked_input_ids[mask] = tokenizer.mask_token_id

    labels = torch.full_like(input_ids, -100)
    labels[mask] = input_ids[mask]
    return masked_input_ids, labels


def make_forward_fn(tokenizer: AresProteinTokenizer, mask_prob: float = 0.15):
    """Closure that returns the MLM loss tensor for a given batch.

    The seed is derived from the *content* of the batch so that the
    baseline pass and all 32 knockout passes use the same mask.
    """
    def _seed(input_ids: torch.LongTensor) -> int:
        # Cheap, deterministic, batch-dependent seed.
        return int(input_ids.sum().item()) & 0x7FFFFFFF

    def forward_fn(model, input_ids, attention_mask):
        masked, labels = make_mlm_batch(
            input_ids, attention_mask,
            tokenizer=tokenizer,
            mask_prob=mask_prob,
            seed=_seed(input_ids),
        )
        out = model(masked, attention_mask=attention_mask, labels=labels)
        return out.loss

    return forward_fn


# =============================================================================
# Single-layer knockout
# =============================================================================

def find_routing_modules(model: nn.Module) -> Dict[str, ExpertChoiceRouting]:
    """Map module name -> ExpertChoiceRouting instance."""
    return {
        name: m for name, m in model.named_modules()
        if isinstance(m, ExpertChoiceRouting)
    }


def resolve_target_layer(
    target: str, routings: Dict[str, ExpertChoiceRouting],
) -> str:
    """Allow either the full module name (``layers.7.ff``) or the short
    label (``L7``). Raises if the target isn't a routing module."""
    if target in routings:
        return target
    matches = [
        name for name in routings if _short_layer_label(name) == target
    ]
    if len(matches) == 1:
        return matches[0]
    available = sorted(routings, key=lambda n: _short_layer_label(n))
    raise SystemExit(
        f"target layer {target!r} not found among {len(routings)} routing "
        f"modules. Available: {[_short_layer_label(n) for n in available]} "
        f"(full names: {available})"
    )


def single_layer_knockout(
    model: nn.Module,
    routing: ExpertChoiceRouting,
    dataloader: DataLoader,
    forward_fn,
    device: torch.device,
) -> Dict[str, object]:
    """Run knockout for every expert at *one* routing module.

    Returns a dict with per-expert paired baseline/knockout losses (one
    entry per batch) plus the corresponding deltas.
    """
    model.eval()
    num_experts = routing.num_experts
    paired_baseline: List[float] = []
    paired_knockout: Dict[int, List[float]] = {
        e: [] for e in range(num_experts)
    }

    assert routing.router.knockout_expert is None, (
        "knockout_expert must be cleared before starting"
    )

    for batch in tqdm(dataloader, desc=f"knockout {_short_layer_label_safe(routing)}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            base = forward_fn(model, input_ids, attention_mask).item()
        if not np.isfinite(base):
            # Degenerate batch (all-special sequences, etc). Skip it
            # entirely so it doesn't contaminate any expert's row.
            continue
        paired_baseline.append(base)

        for e in range(num_experts):
            routing.router.knockout_expert = e
            try:
                with torch.no_grad():
                    ko = forward_fn(model, input_ids, attention_mask).item()
            finally:
                routing.router.knockout_expert = None
            paired_knockout[e].append(ko)

    paired_baseline_arr = np.asarray(paired_baseline, dtype=np.float64)
    mean_baseline = float(paired_baseline_arr.mean()) if paired_baseline_arr.size else 0.0

    per_expert: Dict[int, Dict[str, float]] = {}
    for e in range(num_experts):
        ko_arr = np.asarray(paired_knockout[e], dtype=np.float64)
        # Paired delta — per-batch (knockout - baseline), then averaged.
        # This is more stable than the difference of means because it
        # cancels per-batch variance (long vs short sequences, etc).
        deltas = ko_arr - paired_baseline_arr
        delta_mean = float(deltas.mean()) if deltas.size else 0.0
        delta_se = (
            float(deltas.std(ddof=1) / np.sqrt(deltas.size))
            if deltas.size > 1 else 0.0
        )
        per_expert[e] = {
            "mean_knockout_loss": float(ko_arr.mean()) if ko_arr.size else 0.0,
            "delta": delta_mean,
            "relative_delta": (delta_mean / mean_baseline) if mean_baseline else 0.0,
            "delta_se": delta_se,
            "n_batches": int(deltas.size),
        }

    return {
        "baseline_loss": mean_baseline,
        "n_batches": int(paired_baseline_arr.size),
        "per_expert": per_expert,
    }


def _short_layer_label_safe(routing: ExpertChoiceRouting) -> str:
    # `routing` itself doesn't carry its module name; the caller looked
    # it up. This helper is just a fallback for tqdm description and
    # falls back to the class name if no name is attached.
    return getattr(routing, "_layer_label", routing.__class__.__name__)


# =============================================================================
# Redundancy grouping (uses existing co-occurrence JSON)
# =============================================================================

def expert_redundancy(
    cooc: dict, target_layer: str,
) -> Tuple[np.ndarray, Tuple[List[int], List[int]]]:
    """Return:
      mean_jaccard[e]: per-expert mean off-diagonal Jaccard at the layer.
      (redundant_idx, isolated_idx): split by median (top half =
      "redundant", bottom half = "isolated").
    """
    if target_layer not in cooc:
        # Allow the caller to pass a short label.
        matches = [k for k in cooc if _short_layer_label(k) == target_layer]
        if len(matches) == 1:
            target_layer = matches[0]
        else:
            raise KeyError(
                f"layer {target_layer!r} not found in co-occurrence JSON. "
                f"Available: {list(cooc.keys())}"
            )
    J = np.asarray(cooc[target_layer]["jaccard"], dtype=np.float64)
    E = J.shape[0]
    Joff = J.copy()
    np.fill_diagonal(Joff, np.nan)
    mean_j = np.nanmean(Joff, axis=1)
    median = float(np.median(mean_j))
    redundant = [int(e) for e in range(E) if mean_j[e] >= median]
    isolated = [int(e) for e in range(E) if mean_j[e] < median]
    return mean_j, (redundant, isolated)


# =============================================================================
# Plotters
# =============================================================================

GROUP_COLORS = {
    "redundant": "#c0392b",
    "isolated": "#2c3e50",
}


def plot_knockout_bars(
    knockout: Dict[str, object],
    mean_jaccard: np.ndarray,
    groups: Tuple[List[int], List[int]],
    *,
    target_short: str,
    save_path: Path,
) -> None:
    """Bar chart: relative Δ loss per expert, sorted by expert idx,
    bars colored by redundancy group."""
    redundant, isolated = groups
    per_expert = knockout["per_expert"]
    experts = sorted(int(k) for k in per_expert.keys())
    rel = np.array([per_expert[e]["relative_delta"] for e in experts])
    se = np.array([per_expert[e]["delta_se"] for e in experts])
    base = float(knockout["baseline_loss"])
    rel_se = se / base if base else np.zeros_like(se)

    colors = [
        GROUP_COLORS["redundant"] if e in set(redundant)
        else GROUP_COLORS["isolated"]
        for e in experts
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.bar(
        experts, rel * 100, yerr=rel_se * 100,
        color=colors, edgecolor="white", linewidth=0.4,
        capsize=2, error_kw={"linewidth": 0.6, "alpha": 0.6},
    )
    ax.axhline(0, color="black", linewidth=0.6)

    # Group means as horizontal reference lines.
    for grp_name, idx in (("redundant", redundant), ("isolated", isolated)):
        if not idx:
            continue
        m = float(np.mean([per_expert[e]["relative_delta"] for e in idx])) * 100
        ax.axhline(
            m, color=GROUP_COLORS[grp_name], linestyle=":", linewidth=1.0,
            alpha=0.9,
        )
        ax.text(
            len(experts) - 0.5, m, f"  {grp_name} mean: {m:+.2f}%",
            color=GROUP_COLORS[grp_name], va="bottom", ha="right",
            fontsize=8,
        )

    ax.set_xticks(experts)
    ax.set_xticklabels([str(e) for e in experts], fontsize=6)
    ax.set_xlabel("expert index")
    ax.set_ylabel(r"relative $\Delta$ loss (\%)")
    ax.set_title(
        f"{target_short} per-expert knockout — "
        f"baseline MLM loss = {base:.3f}"
    )

    handles = [
        mpl.patches.Patch(color=GROUP_COLORS["redundant"], label="redundant cluster"),
        mpl.patches.Patch(color=GROUP_COLORS["isolated"], label="isolated"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_jaccard_vs_delta(
    knockout: Dict[str, object],
    mean_jaccard: np.ndarray,
    groups: Tuple[List[int], List[int]],
    *,
    target_short: str,
    save_path: Path,
) -> None:
    """Scatter: per-expert mean Jaccard vs. relative Δ loss.

    Tests the hypothesis that redundant experts are dispensable. A
    significant negative correlation (high Jaccard ⇒ small Δ) supports
    cluster collapse; flat or positive correlation supports a redundant
    ensemble.
    """
    redundant, isolated = groups
    per_expert = knockout["per_expert"]
    experts = sorted(int(k) for k in per_expert.keys())
    rel = np.array([per_expert[e]["relative_delta"] for e in experts]) * 100
    j = mean_jaccard[experts]

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    redundant_set = set(redundant)
    for e, je, re_ in zip(experts, j, rel):
        c = GROUP_COLORS["redundant"] if e in redundant_set else GROUP_COLORS["isolated"]
        ax.scatter(je, re_, color=c, edgecolor="white", linewidth=0.5, s=30, zorder=3)
        ax.annotate(
            str(e), (je, re_), textcoords="offset points",
            xytext=(3, 2), fontsize=5, color="#2c3e50", alpha=0.8,
        )

    if len(experts) >= 3 and np.std(j) > 0:
        # Pearson r and a fit line.
        r = float(np.corrcoef(j, rel)[0, 1])
        slope, intercept = np.polyfit(j, rel, 1)
        xs = np.linspace(float(j.min()), float(j.max()), 50)
        ax.plot(
            xs, slope * xs + intercept,
            color="#7f8c8d", linestyle="--", linewidth=0.8, zorder=1,
        )
        ax.text(
            0.02, 0.97, f"$r = {r:+.2f}$",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
        )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("mean off-diagonal Jaccard at layer")
    ax.set_ylabel(r"relative $\Delta$ loss (\%)")
    ax.set_title(f"{target_short} — redundancy vs. impact")

    handles = [
        mpl.patches.Patch(color=GROUP_COLORS["redundant"], label="redundant"),
        mpl.patches.Patch(color=GROUP_COLORS["isolated"], label="isolated"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def run(
    *,
    target_layer_arg: str,
    num_batches: int,
    output_dir: Path,
    ec_dir: Path,
    device: str,
    checkpoint: str,
    seed: int,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    cooc = load_json(ec_dir / "expert_co_occurrence.json")

    print(f"loading {checkpoint} on {device}…")
    model = Ares.from_pretrained(checkpoint, device_map=device)
    tokenizer = AresProteinTokenizer()

    routings = find_routing_modules(model)
    if not routings:
        raise SystemExit("No ExpertChoiceRouting modules found in the model.")
    target_layer = resolve_target_layer(target_layer_arg, routings)
    target_short = _short_layer_label(target_layer)
    routing = routings[target_layer]
    routing._layer_label = target_short  # for tqdm prefix

    print(
        f"target layer: {target_layer}  ({target_short}, "
        f"{routing.num_experts} experts)"
    )

    mean_jaccard, (redundant, isolated) = expert_redundancy(cooc, target_layer)
    print(f"  redundant experts (top half by Jaccard): {redundant}")
    print(f"  isolated  experts (bot half by Jaccard): {isolated}")

    dataset = UniRef30(max_examples=num_batches)
    collator = Collator(tokenizer)
    dataloader = DataLoader(
        dataset, batch_size=1, collate_fn=collator,
        shuffle=True, generator=torch.manual_seed(seed),
    )
    model_device = next(model.parameters()).device
    forward_fn = make_forward_fn(tokenizer, mask_prob=0.30)

    knockout = single_layer_knockout(
        model, routing, dataloader, forward_fn, model_device,
    )

    summary = {
        "target_layer": target_layer,
        "target_short": target_short,
        "checkpoint": checkpoint,
        "num_experts": int(routing.num_experts),
        "n_batches": int(knockout["n_batches"]),
        "baseline_loss": knockout["baseline_loss"],
        "groups": {
            "redundant": list(map(int, redundant)),
            "isolated": list(map(int, isolated)),
        },
        "mean_jaccard_per_expert": [float(x) for x in mean_jaccard.tolist()],
        "per_expert": {
            int(e): {
                **{k: float(v) if not isinstance(v, int) else int(v)
                   for k, v in d.items()},
                "mean_jaccard": float(mean_jaccard[int(e)]),
                "group": ("redundant" if int(e) in set(redundant) else "isolated"),
            }
            for e, d in knockout["per_expert"].items()
        },
    }

    rel_redundant = [
        summary["per_expert"][e]["relative_delta"] for e in redundant
    ]
    rel_isolated = [
        summary["per_expert"][e]["relative_delta"] for e in isolated
    ]
    summary["group_means"] = {
        "redundant_mean_relative_delta": float(np.mean(rel_redundant))
        if rel_redundant else 0.0,
        "isolated_mean_relative_delta": float(np.mean(rel_isolated))
        if rel_isolated else 0.0,
    }
    rel_array = np.array([
        summary["per_expert"][e]["relative_delta"] for e in range(routing.num_experts)
    ])
    if rel_array.std() > 0 and mean_jaccard.std() > 0:
        summary["jaccard_vs_delta_pearson_r"] = float(
            np.corrcoef(mean_jaccard, rel_array)[0, 1]
        )

    json_path = output_dir / f"{target_short}_knockout.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    plot_knockout_bars(
        knockout, mean_jaccard, (redundant, isolated),
        target_short=target_short,
        save_path=output_dir / f"{target_short}_knockout_bars.pdf",
    )
    plot_jaccard_vs_delta(
        knockout, mean_jaccard, (redundant, isolated),
        target_short=target_short,
        save_path=output_dir / f"{target_short}_knockout_scatter.pdf",
    )

    # Human-readable digest.
    print(f"\n=== {target_short} knockout ===")
    print(f"  baseline MLM loss: {summary['baseline_loss']:.4f}  "
          f"(n_batches={summary['n_batches']})")
    print(f"  redundant group mean rel-Δ: "
          f"{summary['group_means']['redundant_mean_relative_delta']*100:+.2f}%  "
          f"(n={len(redundant)})")
    print(f"  isolated  group mean rel-Δ: "
          f"{summary['group_means']['isolated_mean_relative_delta']*100:+.2f}%  "
          f"(n={len(isolated)})")
    if "jaccard_vs_delta_pearson_r" in summary:
        r = summary["jaccard_vs_delta_pearson_r"]
        verdict = (
            "redundant ⇒ dispensable (cluster collapse)"
            if r < -0.2 else
            "redundant ⇒ load-bearing (redundant ensemble)"
            if r > 0.2 else
            "weak / no relationship"
        )
        print(f"  Pearson r(jaccard, rel-Δ): {r:+.3f}  →  {verdict}")
    print("\n  Top-5 most-impactful experts (largest rel-Δ):")
    sorted_e = sorted(
        summary["per_expert"].items(),
        key=lambda kv: kv[1]["relative_delta"], reverse=True,
    )
    for e, d in sorted_e[:5]:
        print(f"    E{int(e):>2}  rel-Δ={d['relative_delta']*100:+.2f}%  "
              f"(group={d['group']}, J̄={d['mean_jaccard']:.3f})")
    print("\n  Bottom-5 (smallest rel-Δ):")
    for e, d in sorted_e[-5:]:
        print(f"    E{int(e):>2}  rel-Δ={d['relative_delta']*100:+.2f}%  "
              f"(group={d['group']}, J̄={d['mean_jaccard']:.3f})")
    print(f"\n  Wrote {json_path.name} + 2 PDFs to {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tier-2: per-expert knockout at one MoE layer.",
    )
    p.add_argument(
        "--layer", default="L7",
        help="Target layer (e.g. 'L7' or 'layers.7.ff').",
    )
    p.add_argument("--num-batches", type=int, default=200,
                   help="Number of sequences (batch_size=1) for the loop.")
    p.add_argument(
        "--ec-dir", type=Path,
        default=Path(__file__).parent / "expert_choice_analysis_outputs",
        help="Directory containing expert_co_occurrence.json (for redundancy grouping).",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "figures" / "anomaly" / "mask_prob_30",
    )
    p.add_argument("--device", default="cuda:1")
    p.add_argument(
        "--checkpoint",
        default="HazemLab/ares-expert-choice-4b-interleaved-150K",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run(
        target_layer_arg=args.layer,
        num_batches=args.num_batches,
        output_dir=args.output_dir,
        ec_dir=args.ec_dir,
        device=args.device,
        checkpoint=args.checkpoint,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
