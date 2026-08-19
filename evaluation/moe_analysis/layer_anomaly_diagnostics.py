"""Decompose a "weird" layer's routing behaviour using only the existing
analysis JSONs (no model re-run required).

Given a target layer like L7 (the one with anomalously high drop rate +
high pairwise Jaccard in the expert-choice analysis), this script
extracts:

  1. Per-AA "mean experts per token"  — which residues are attractors
     (over-subscribed) vs dropouts (under-subscribed) at the target layer.
  2. Per-position "mean experts per token" — does the imbalance correlate
     with N-/C-terminal regions or interior?
  3. The top-K most-redundant expert PAIRS at the target layer (highest
     off-diagonal Jaccard). If those pairs are consistent across layers,
     the routing has "twin" experts.
  4. Per-expert "selection share" at the target layer — is the load
     concentrated on a few experts?

Each metric is also computed for at least one "normal" reference layer so
you have a side-by-side comparison for the paper.

Outputs both a JSON summary and a small set of PDF figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from visualize_analysis import (
    apply_style, load_json, _layer_sort_key, _short_layer_label,
    CANONICAL_AA,
)


# =============================================================================
# Aggregation helpers
# =============================================================================

def mean_experts_per_class(
    aa_results: dict,
    layer: str,
    classes: Sequence[str],
    weight_type: str = "selection",
) -> Dict[str, float]:
    """For each class (AA or property or position bin), return the
    expected number of experts that pick a token of that class:
        sum over experts of mean_weight[layer, expert, class].
    For weight_type="selection" this is literally
    "mean experts per token of this class".
    """
    inner = aa_results[weight_type][layer]
    out: Dict[str, float] = {}
    for cls in classes:
        total = 0.0
        for e_str, by_class in inner.items():
            entry = by_class.get(cls)
            if entry is None:
                continue
            total += float(entry["mean_weight"])
        out[cls] = total
    return out


def expert_selection_shares(
    aa_results: dict,
    layer: str,
    weight_type: str = "selection",
) -> Dict[int, float]:
    """For each expert at `layer`, return its overall mean selection
    rate (i.e. `baseline` from the per-AA file, or a count-weighted mean
    of `mean_weight` across classes if older outputs lack `count`)."""
    inner = aa_results[weight_type][layer]
    out: Dict[int, float] = {}
    for e_str, by_class in inner.items():
        # `baseline` is per-expert and identical across classes — use it
        # if present (it's the analyzer's own mean-weight-over-all-tokens).
        any_entry = next(iter(by_class.values()), None)
        if any_entry is not None and "baseline" in any_entry:
            out[int(e_str)] = float(any_entry["baseline"])
            continue
        total_w = 0.0
        total_c = 0
        for entry in by_class.values():
            c = int(entry.get("count", 1))
            if c == 0:
                continue
            total_w += float(entry["mean_weight"]) * c
            total_c += c
        out[int(e_str)] = (total_w / total_c) if total_c > 0 else 0.0
    return out


def top_redundant_pairs(
    cooc: dict,
    layer: str,
    top_k: int = 5,
) -> List[Tuple[int, int, float]]:
    """Return the `top_k` (i, j, jaccard) tuples with highest off-diagonal
    Jaccard at `layer`. Symmetric, so we report each pair once with i < j.
    """
    J = np.array(cooc[layer]["jaccard"])
    E = J.shape[0]
    pairs: List[Tuple[int, int, float]] = []
    for i in range(E):
        for j in range(i + 1, E):
            pairs.append((i, j, float(J[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_k]


# =============================================================================
# Plotters
# =============================================================================

def _diverging_norm(values: np.ndarray, center: float):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        rng = 1.0
    else:
        rng = float(max(np.nanmax(np.abs(finite - center)), 1e-3))
    return mpl.colors.TwoSlopeNorm(vmin=center - rng, vcenter=center, vmax=center + rng)


def plot_class_dropout_comparison(
    layer_to_class_mean: Dict[str, Dict[str, float]],
    classes: Sequence[str],
    *,
    target_layer: str,
    title: str,
    save_path: Path,
    capacity_factor: float = 2.0,
) -> None:
    """Bar comparison: per-class mean experts/token at the target layer
    vs the (median across non-target) reference, with a capacity line."""
    layers = list(layer_to_class_mean.keys())
    if target_layer not in layers:
        return
    ref_layers = [L for L in layers if L != target_layer]
    target_values = np.array([layer_to_class_mean[target_layer][c] for c in classes])
    ref_matrix = np.array([
        [layer_to_class_mean[L][c] for c in classes] for L in ref_layers
    ])
    ref_median = np.median(ref_matrix, axis=0) if ref_matrix.size else np.zeros_like(target_values)
    ref_iqr = (
        np.percentile(ref_matrix, 75, axis=0) - np.percentile(ref_matrix, 25, axis=0)
        if ref_matrix.size else np.zeros_like(target_values)
    )

    x = np.arange(len(classes))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(5.0, 0.4 * len(classes) + 2.0), 3.2))
    ax.bar(x - width/2, ref_median, width, yerr=ref_iqr / 2,
           color="#7f8c8d", label=f"other layers (median ± IQR/2, n={len(ref_layers)})",
           capsize=2, edgecolor="white", linewidth=0.4)
    target_short = _short_layer_label(target_layer)
    ax.bar(x + width/2, target_values, width,
           color="#c0392b", label=f"{target_short}",
           edgecolor="white", linewidth=0.4)
    ax.axhline(capacity_factor, color="#2c3e50", linestyle=":", linewidth=0.8,
               label=f"capacity factor = {capacity_factor:g}")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("mean experts per token")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_expert_load(
    layer_to_shares: Dict[str, Dict[int, float]],
    *,
    target_layer: str,
    save_path: Path,
) -> None:
    """For each layer, sorted-descending bar of per-expert selection shares.
    Highlights the target layer to show whether load is concentrated."""
    layers = sorted(layer_to_shares.keys(), key=_layer_sort_key)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for L in layers:
        shares = sorted(layer_to_shares[L].values(), reverse=True)
        x = np.arange(len(shares))
        is_target = (L == target_layer)
        ax.plot(
            x, shares,
            color="#c0392b" if is_target else "#bdc3c7",
            linewidth=2.0 if is_target else 0.9,
            alpha=1.0 if is_target else 0.7,
            label=_short_layer_label(L) if is_target else None,
            zorder=3 if is_target else 1,
        )
    ax.set_xlabel("expert rank (within layer)")
    ax.set_ylabel("mean selection rate")
    ax.set_title("Per-layer expert load (sorted, descending)")
    ax.legend(loc="upper right")
    fig.savefig(save_path)
    plt.close(fig)


def plot_redundant_pairs(
    cooc: dict,
    *,
    target_layer: str,
    save_path: Path,
    top_k: int = 8,
) -> None:
    """For the target layer, draw a small E×E heatmap with the top-k
    redundant pairs annotated."""
    if target_layer not in cooc:
        return
    J = np.array(cooc[target_layer]["jaccard"])
    E = J.shape[0]
    J_disp = J.copy()
    np.fill_diagonal(J_disp, np.nan)

    cmap = plt.get_cmap("rocket" if "rocket" in plt.colormaps() else "viridis").copy()
    cmap.set_bad(color="#1a1a1a")

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    vmax = max(float(np.nanpercentile(J_disp[np.isfinite(J_disp)], 99)), 1e-3)
    im = ax.imshow(J_disp, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(E))
    ax.set_yticks(range(E))
    ax.set_xticklabels(range(E), fontsize=5)
    ax.set_yticklabels(range(E), fontsize=5)
    ax.set_title(f"{_short_layer_label(target_layer)} — top {top_k} redundant expert pairs")

    pairs = top_redundant_pairs(cooc, target_layer, top_k=top_k)
    for i, j, score in pairs:
        ax.scatter([j, i], [i, j], facecolors="none",
                   edgecolors="white", linewidths=0.8, s=18)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="max")
    cbar.set_label("Jaccard")
    cbar.outline.set_linewidth(0.5)

    # Annotate the top 5 pairs in a side text panel
    text_lines = [f"top {top_k} pairs (i, j, J):"] + [
        f"  ({i:>2}, {j:>2}) → {s:.3f}" for i, j, s in pairs
    ]
    ax.text(
        1.18, 1.0, "\n".join(text_lines),
        transform=ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=7,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def run(
    ec_dir: Path,
    target_layer: str,
    output_dir: Path,
    capacity_factor: float,
) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    aa = load_json(ec_dir / "amino_acid_preferences.json")
    prop = load_json(ec_dir / "property_preferences.json")
    pos = load_json(ec_dir / "positional_preferences.json")
    cov = load_json(ec_dir / "token_coverage.json")
    cooc = load_json(ec_dir / "expert_co_occurrence.json")

    layers = sorted(
        (k for k in aa["selection"].keys() if not k.startswith("_")),
        key=_layer_sort_key,
    )
    if target_layer not in layers:
        # Allow short label like "L7"
        match = [L for L in layers if _short_layer_label(L) == target_layer]
        if match:
            target_layer = match[0]
        else:
            raise SystemExit(
                f"target layer {target_layer!r} not found. "
                f"Available: {[_short_layer_label(L) for L in layers]}"
            )

    target_short = _short_layer_label(target_layer)
    summary: Dict[str, object] = {
        "target_layer": target_layer,
        "target_short": target_short,
        "drop_rate": cov[target_layer]["drop_rate"],
        "mean_experts_per_token": cov[target_layer]["mean_experts_per_token"],
    }

    # 1. Per-AA mean experts per token
    aa_per_layer = {
        L: mean_experts_per_class(aa, L, CANONICAL_AA, "selection")
        for L in layers
    }
    plot_class_dropout_comparison(
        aa_per_layer, CANONICAL_AA,
        target_layer=target_layer,
        title=f"Per-AA routing load — {target_short} vs other layers",
        save_path=output_dir / f"{target_short}_per_aa_load.pdf",
        capacity_factor=capacity_factor,
    )

    # Rank AAs by deviation from the median reference
    ref_layers = [L for L in layers if L != target_layer]
    ref_med_aa = {
        c: float(np.median([aa_per_layer[L][c] for L in ref_layers]))
        for c in CANONICAL_AA
    }
    aa_dev = {
        c: aa_per_layer[target_layer][c] - ref_med_aa[c]
        for c in CANONICAL_AA
    }
    summary["per_aa"] = {
        "target_mean_experts": aa_per_layer[target_layer],
        "reference_median_mean_experts": ref_med_aa,
        "deviation_from_reference": aa_dev,
        "most_overrouted_aas": sorted(aa_dev.items(), key=lambda t: t[1], reverse=True)[:5],
        "most_dropped_aas": sorted(aa_dev.items(), key=lambda t: t[1])[:5],
    }

    # 2. Per-position mean experts per token
    pos_bins = list(pos["selection"][layers[0]]["0"].keys())
    pos_per_layer = {
        L: mean_experts_per_class(pos, L, pos_bins, "selection")
        for L in layers
    }
    plot_class_dropout_comparison(
        pos_per_layer, pos_bins,
        target_layer=target_layer,
        title=f"Per-position routing load — {target_short} vs other layers",
        save_path=output_dir / f"{target_short}_per_position_load.pdf",
        capacity_factor=capacity_factor,
    )
    summary["per_position"] = {
        "target_mean_experts": pos_per_layer[target_layer],
        "reference_median_mean_experts": {
            b: float(np.median([pos_per_layer[L][b] for L in ref_layers]))
            for b in pos_bins
        },
    }

    # 3. Per-property mean experts per token
    prop_md = prop.get("_metadata", {})
    prop_groups = list(prop_md.get("property_groups", {}).keys()) or [
        "hydrophobic", "polar_uncharged", "positive", "negative",
        "charged", "aromatic", "small", "cysteine", "proline",
    ]
    prop_per_layer = {
        L: mean_experts_per_class(prop, L, prop_groups, "selection")
        for L in layers
    }
    plot_class_dropout_comparison(
        prop_per_layer, prop_groups,
        target_layer=target_layer,
        title=f"Per-property routing load — {target_short} vs other layers",
        save_path=output_dir / f"{target_short}_per_property_load.pdf",
        capacity_factor=capacity_factor,
    )
    summary["per_property"] = {
        "target_mean_experts": prop_per_layer[target_layer],
        "reference_median_mean_experts": {
            g: float(np.median([prop_per_layer[L][g] for L in ref_layers]))
            for g in prop_groups
        },
    }

    # 4. Expert load distribution (sorted descending per layer)
    shares = {
        L: expert_selection_shares(aa, L, "selection")
        for L in layers
    }
    plot_expert_load(
        shares,
        target_layer=target_layer,
        save_path=output_dir / f"{target_short}_expert_load.pdf",
    )
    target_shares_sorted = sorted(shares[target_layer].values(), reverse=True)
    summary["expert_load"] = {
        "target_shares_sorted": target_shares_sorted,
        "load_concentration_top1": float(target_shares_sorted[0] / sum(target_shares_sorted))
        if sum(target_shares_sorted) > 0 else 0.0,
        "load_concentration_top4": float(sum(target_shares_sorted[:4]) / sum(target_shares_sorted))
        if sum(target_shares_sorted) > 0 else 0.0,
    }

    # 5. Most-redundant expert pairs at target layer
    pairs = top_redundant_pairs(cooc, target_layer, top_k=10)
    plot_redundant_pairs(
        cooc,
        target_layer=target_layer,
        save_path=output_dir / f"{target_short}_redundant_pairs.pdf",
        top_k=10,
    )
    summary["redundant_pairs"] = [
        {"i": i, "j": j, "jaccard": s} for i, j, s in pairs
    ]

    # Save JSON summary
    with open(output_dir / f"{target_short}_diagnostics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print a human-readable digest
    print(f"\n=== {target_short} diagnostics ===")
    print(f"  drop_rate: {summary['drop_rate']:.3f}")
    print(f"  mean experts/token: {summary['mean_experts_per_token']:.3f}")
    print(f"\n  Most over-routed AAs (target - reference median):")
    for c, d in summary["per_aa"]["most_overrouted_aas"]:
        print(f"    {c}: +{d:+.3f}  (target {aa_per_layer[target_layer][c]:.2f}, ref {ref_med_aa[c]:.2f})")
    print(f"\n  Most dropped AAs:")
    for c, d in summary["per_aa"]["most_dropped_aas"]:
        print(f"    {c}: {d:+.3f}  (target {aa_per_layer[target_layer][c]:.2f}, ref {ref_med_aa[c]:.2f})")
    print(f"\n  Top-3 redundant expert pairs:")
    for i, j, s in pairs[:3]:
        print(f"    (E{i:>2}, E{j:>2}) Jaccard = {s:.3f}")
    print(f"\n  Load concentration: top-1 expert = {summary['expert_load']['load_concentration_top1']*100:.1f}%, "
          f"top-4 = {summary['expert_load']['load_concentration_top4']*100:.1f}%")
    print(f"\n  Wrote {len(list(output_dir.glob(f'{target_short}_*.pdf')))} PDFs and "
          f"{target_short}_diagnostics.json to {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnose a single anomalous layer using existing analysis JSONs.",
    )
    p.add_argument(
        "--ec-dir", type=Path,
        default=Path(__file__).parent / "expert_choice_analysis_outputs",
    )
    p.add_argument("--layer", default="L7", help="Layer to analyse (e.g. 'L7' or 'layers.7.ff').")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).parent / "figures" / "anomaly")
    p.add_argument("--capacity-factor", type=float, default=2.0)
    args = p.parse_args()
    run(args.ec_dir, args.layer, args.output_dir, args.capacity_factor)


if __name__ == "__main__":
    main()
