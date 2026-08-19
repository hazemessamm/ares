"""LaTeX-style figure generator for Expert Choice analysis outputs.

Reads the JSON artifacts produced by `expert_choice_analysis.py` and
writes publication-ready vector PDFs (property / amino-acid / positional
specialization, token coverage, expert co-occurrence, and optional
expert knockout).

Usage:
    python -m ares.evaluation.visualize_expert_choice \\
        --input-dir  ares/evaluation/expert_choice_analysis_outputs \\
        --output-dir ares/evaluation/figures
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from visualize_analysis import (
    _layer_sort_key,
    _short_layer_label,
    apply_style,
    load_json,
    plot_aa_specialization,
    plot_knockout,
    plot_positional_specialization,
    plot_property_specialization,
)


# =============================================================================
# Expert Choice-specific plotters
# =============================================================================

def plot_token_coverage(
    coverage: dict,
    save_dir: Path,
) -> None:
    """Stacked bar chart of coverage_fractions per layer."""
    layers = sorted(coverage.keys(), key=_layer_sort_key)
    if not layers:
        return
    bins = sorted((int(b) for b in coverage[layers[0]]["coverage_fractions"].keys()))
    num_bins = len(bins)

    # Truncate to where there's meaningful mass — anything past the
    # 99.5 % cumulative mass becomes a single "more" bucket so we
    # don't waste plot real estate on near-empty tail bins.
    fractions_full = np.array([
        [coverage[L]["coverage_fractions"][str(b)] for b in bins]
        for L in layers
    ])
    cumulative = fractions_full.cumsum(axis=1)
    last_meaningful = max(
        int(np.searchsorted(cumulative.mean(axis=0), 0.995)) + 1,
        2,
    )
    fractions = fractions_full[:, :last_meaningful]
    if last_meaningful < num_bins:
        tail = fractions_full[:, last_meaningful:].sum(axis=1, keepdims=True)
        fractions = np.concatenate([fractions, tail], axis=1)
        labels = [str(b) for b in bins[:last_meaningful]] + [
            f"$\\geq${bins[last_meaningful]}"
        ]
    else:
        labels = [str(b) for b in bins[:last_meaningful]]

    fig, ax = plt.subplots(figsize=(max(5.0, 0.45 * len(layers) + 2.0), 3.5))
    bottoms = np.zeros(len(layers))
    cmap = plt.get_cmap("rocket_r" if "rocket_r" in plt.colormaps() else "viridis")
    colors = cmap(np.linspace(0.05, 0.95, fractions.shape[1]))
    x = np.arange(len(layers))
    for i in range(fractions.shape[1]):
        ax.bar(
            x, fractions[:, i], bottom=bottoms,
            color=colors[i], edgecolor="white", linewidth=0.4,
            label=f"{labels[i]} expert{'s' if labels[i] != '1' else ''}",
        )
        bottoms += fractions[:, i]
    ax.set_xticks(x)
    ax.set_xticklabels([_short_layer_label(L) for L in layers], rotation=45, ha="right")
    ax.set_ylabel("fraction of valid tokens")
    ax.set_ylim(0, 1)
    ax.set_title("Token coverage — fraction picked by $k$ experts")
    ax.legend(
        title="$k$",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        ncol=1,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "token_coverage.pdf")
    plt.close(fig)

    # Companion: drop_rate + mean_experts_per_token line plot.
    drop_rate = [coverage[L]["drop_rate"] for L in layers]
    mept = [coverage[L]["mean_experts_per_token"] for L in layers]
    fig, ax1 = plt.subplots(figsize=(max(4.5, 0.4 * len(layers) + 1.8), 2.8))
    color1 = "#c0392b"
    color2 = "#2c3e50"
    ax1.plot(x, drop_rate, "o-", color=color1, linewidth=1.5, markersize=4, label="drop rate")
    ax1.set_ylabel("drop rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels([_short_layer_label(L) for L in layers], rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)
    ax2.plot(x, mept, "s--", color=color2, linewidth=1.2, markersize=4,
             label="mean experts/token")
    ax2.set_ylabel("mean experts / token")

    ax1.set_title("Routing imbalance per layer")
    handles = [
        plt.Line2D([0], [0], color=color1, marker="o", linestyle="-", label="drop rate"),
        plt.Line2D([0], [0], color=color2, marker="s", linestyle="--", label="mean experts/token"),
    ]
    ax1.legend(handles=handles, loc="best")
    fig.savefig(save_dir / "drop_rate.pdf")
    plt.close(fig)


def plot_co_occurrence(
    cooccurrence: dict,
    save_dir: Path,
    *,
    layers_per_row: int = 5,
) -> None:
    """Small multiples: one E×E Jaccard heatmap per layer."""
    layers = sorted(cooccurrence.keys(), key=_layer_sort_key)
    if not layers:
        return
    n = len(layers)
    cols = layers_per_row
    rows = math.ceil(n / cols)
    cmap = plt.get_cmap("rocket" if "rocket" in plt.colormaps() else "viridis")

    # Determine global vmax so panels are comparable.
    all_off_diag = []
    for L in layers:
        J = np.array(cooccurrence[L]["jaccard"])
        E = J.shape[0]
        mask = ~np.eye(E, dtype=bool)
        all_off_diag.append(J[mask])
    vmax = float(np.percentile(np.concatenate(all_off_diag), 99))
    vmax = max(vmax, 1e-3)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(2.0 * cols + 0.6, 2.0 * rows + 0.4),
        squeeze=False,
    )
    for ax in axes.flatten():
        ax.set_axis_off()

    im = None
    for idx, L in enumerate(layers):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.set_axis_on()
        J = np.array(cooccurrence[L]["jaccard"])
        # Mask diagonal (always 1, dominates the colorbar).
        J_disp = J.copy()
        np.fill_diagonal(J_disp, np.nan)
        cmap_local = cmap.copy()
        cmap_local.set_bad(color="#1a1a1a")
        im = ax.imshow(J_disp, cmap=cmap_local, vmin=0, vmax=vmax,
                       interpolation="nearest", aspect="equal")
        ax.set_title(_short_layer_label(L), pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

    if im is not None:
        cbar = fig.colorbar(
            im, ax=axes,
            fraction=0.02, pad=0.02, shrink=0.7, extend="max",
        )
        cbar.set_label("Jaccard (off-diagonal)")
        cbar.outline.set_linewidth(0.5)
    fig.suptitle("Pairwise expert co-selection (Jaccard)", y=1.02)

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "co_occurrence.pdf")
    plt.close(fig)


# =============================================================================
# Orchestration
# =============================================================================

def run(input_dir: Path, output_dir: Path) -> None:
    apply_style()
    if not input_dir.exists():
        raise FileNotFoundError(f"Expert Choice input dir does not exist: {input_dir}")

    sub = output_dir / "expert_choice"
    sub.mkdir(parents=True, exist_ok=True)

    plotters = [
        ("property_preferences.json", plot_property_specialization),
        ("amino_acid_preferences.json", plot_aa_specialization),
        ("positional_preferences.json", plot_positional_specialization),
    ]
    for fname, plotter in plotters:
        path = input_dir / fname
        if path.exists():
            plotter(load_json(path), sub, name_prefix="Expert Choice")
            print(f"[ec]    {fname:32s} -> {sub}")
        else:
            print(f"[ec]    {fname:32s} (missing, skipped)")

    cov_path = input_dir / "token_coverage.json"
    if cov_path.exists():
        plot_token_coverage(load_json(cov_path), sub)
        print(f"[ec]    {cov_path.name:32s} -> {sub}")

    co_path = input_dir / "expert_co_occurrence.json"
    if co_path.exists():
        plot_co_occurrence(load_json(co_path), sub)
        print(f"[ec]    {co_path.name:32s} -> {sub}")

    ko_path = input_dir / "expert_knockout.json"
    if ko_path.exists():
        plot_knockout(load_json(ko_path), sub)
        print(f"[ec]    {ko_path.name:32s} -> {sub}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from Expert Choice analysis outputs.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).parent / "expert_choice_analysis_outputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "figures",
    )
    args = parser.parse_args()
    run(input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
