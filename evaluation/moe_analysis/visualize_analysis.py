"""Shared figure infrastructure for MoE analysis visualizations.

This module is *not* a CLI. It exposes the publication style, layout
helpers, matrix builders, and per-analysis plot primitives that are
shared between the two top-level visualization scripts:

* `visualize_soft_moe.py`      — Soft MoE pipeline (with `--no-l2` flag)
* `visualize_expert_choice.py` — Expert Choice pipeline

Per-pipeline, only-one-side plots (token coverage, expert co-occurrence)
live in their respective scripts to keep this module pipeline-agnostic.

All figures use a serif (Computer Modern) font, vector PDF, and a tight
bounding box. Diverging fields (log-ratios) are plotted with `RdBu_r`
centered at zero; sequential fields use `viridis` / `rocket`. AAs are
filtered to the canonical 20 and `low_support` entries are masked
(rendered as light gray) so reviewers can immediately see which cells
are statistically unreliable.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# =============================================================================
# Style
# =============================================================================

CANONICAL_AA: Tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")

PUBLICATION_RC: Dict[str, object] = {
    "font.family": "serif",
    "font.serif": [
        "Computer Modern Roman",
        "CMU Serif",
        "Times New Roman",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "cm",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "image.cmap": "viridis",
    "pdf.fonttype": 42,  # editable text in vector output
    "ps.fonttype": 42,
}


def apply_style() -> None:
    mpl.rcParams.update(PUBLICATION_RC)


# =============================================================================
# Loading helpers
# =============================================================================

LAYER_RE = re.compile(r"(\d+)")


def _layer_sort_key(name: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in LAYER_RE.findall(name)) or (0,)


def _short_layer_label(name: str) -> str:
    """Turn 'layers.10.ff' (or similar) into 'L10'."""
    nums = LAYER_RE.findall(name)
    return f"L{nums[-1]}" if nums else name


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_layers(results: dict, weight_type: str) -> List[str]:
    inner = results[weight_type]
    layers = [k for k in inner.keys() if not k.startswith("_")]
    return sorted(layers, key=_layer_sort_key)


def get_weight_types(results: dict) -> List[str]:
    return [k for k in results.keys() if not k.startswith("_")]


def get_experts(results: dict, weight_type: str) -> List[int]:
    layers = get_layers(results, weight_type)
    if not layers:
        return []
    expert_keys = list(results[weight_type][layers[0]].keys())
    return sorted(int(k) for k in expert_keys)


# =============================================================================
# Matrix builders
# =============================================================================

def build_log_ratio_matrix(
    results: dict,
    weight_type: str,
    classes: Sequence[str],
    *,
    skip_low_support: bool = True,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build a (num_layers * num_experts, num_classes) matrix of log ratios.

    Returns the matrix, the row labels ("L<idx>.E<expert>"), and the
    list of layer names so callers can draw separators.
    """
    layers = get_layers(results, weight_type)
    experts = get_experts(results, weight_type)
    rows: List[List[float]] = []
    row_labels: List[str] = []
    inner = results[weight_type]
    for layer in layers:
        layer_short = _short_layer_label(layer)
        for e in experts:
            entry_dict = inner[layer].get(str(e), {})
            row = []
            for cls in classes:
                entry = entry_dict.get(cls)
                if entry is None:
                    row.append(np.nan)
                    continue
                if skip_low_support and entry.get("low_support"):
                    row.append(np.nan)
                    continue
                lr = entry.get("log_ratio")
                if lr is None:
                    # Fall back to log(specialization_ratio) for older runs
                    sr = entry.get("specialization_ratio")
                    lr = (math.log(sr) if (sr is not None and sr > 0) else np.nan)
                row.append(lr if lr is not None else np.nan)
            rows.append(row)
            row_labels.append(f"{layer_short}.E{e}")
    matrix = np.array(rows, dtype=float)
    return matrix, row_labels, layers


# =============================================================================
# Plotting primitives
# =============================================================================

def _diverging_cmap_for(matrix: np.ndarray) -> Tuple[mpl.colors.Colormap, TwoSlopeNorm]:
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd", alpha=1.0)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        absmax = 1.0
    else:
        absmax = float(np.nanpercentile(np.abs(finite), 99))
        absmax = max(absmax, 1e-3)
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0.0, vmax=absmax)
    return cmap, norm


def _draw_layer_separators(
    ax: plt.Axes,
    num_experts: int,
    num_layers: int,
) -> None:
    for i in range(1, num_layers):
        ax.axhline(i * num_experts - 0.5, color="white", linewidth=0.6)


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    title: str,
    save_path: Path,
    cbar_label: str = "log ratio",
    num_experts_per_layer: Optional[int] = None,
) -> None:
    cmap, norm = _diverging_cmap_for(matrix)
    n_rows, n_cols = matrix.shape

    # Sizing: ~0.16in per row, ~0.45in per column, plus padding.
    fig_w = max(4.0, 0.45 * n_cols + 1.6)
    fig_h = max(2.5, 0.16 * n_rows + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", rotation_mode="anchor")
    # Y-axis: only label the first expert of each layer to avoid clutter.
    if num_experts_per_layer:
        tick_positions = list(range(0, n_rows, num_experts_per_layer))
        tick_labels = [row_labels[i].split(".")[0] for i in tick_positions]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
        ax.set_ylabel("layer (each row $=$ one expert)")
        _draw_layer_separators(ax, num_experts_per_layer, n_rows // num_experts_per_layer)
    else:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels)

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, extend="both")
    cbar.set_label(cbar_label)
    cbar.outline.set_linewidth(0.5)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# =============================================================================
# Per-analysis plot routines (shared between pipelines)
# =============================================================================

_DEFAULT_PROPERTY_GROUPS: Tuple[str, ...] = (
    "hydrophobic", "polar_uncharged", "positive", "negative",
    "charged", "aromatic", "small", "cysteine", "proline",
)


def plot_property_specialization(
    results: dict,
    save_dir: Path,
    *,
    name_prefix: str,
    file_prefix: str = "property",
) -> None:
    """One PDF per weight type."""
    metadata = results.get("_metadata", {})
    group_order = list(
        metadata.get("property_groups", {}).keys()
    ) or list(_DEFAULT_PROPERTY_GROUPS)

    for wt in get_weight_types(results):
        matrix, row_labels, _ = build_log_ratio_matrix(results, wt, group_order)
        num_experts = len(get_experts(results, wt))
        plot_heatmap(
            matrix,
            row_labels,
            group_order,
            title=f"{name_prefix} — property specialization ({wt})",
            save_path=save_dir / f"{file_prefix}_{wt}.pdf",
            cbar_label=r"$\log(\mathrm{mean} / \mathrm{baseline})$",
            num_experts_per_layer=num_experts,
        )


def plot_aa_specialization(
    results: dict,
    save_dir: Path,
    *,
    name_prefix: str,
    file_prefix: str = "amino_acid",
    aa_order: Sequence[str] = CANONICAL_AA,
) -> None:
    for wt in get_weight_types(results):
        matrix, row_labels, _ = build_log_ratio_matrix(results, wt, aa_order)
        num_experts = len(get_experts(results, wt))
        plot_heatmap(
            matrix,
            row_labels,
            aa_order,
            title=f"{name_prefix} — amino-acid specialization ({wt})",
            save_path=save_dir / f"{file_prefix}_{wt}.pdf",
            cbar_label=r"$\log(\mathrm{mean} / \mathrm{baseline})$",
            num_experts_per_layer=num_experts,
        )


def plot_positional_specialization(
    results: dict,
    save_dir: Path,
    *,
    name_prefix: str,
    file_prefix: str = "positional",
) -> None:
    # Bin labels are the same across weight types
    wt0 = get_weight_types(results)[0]
    layers = get_layers(results, wt0)
    experts = get_experts(results, wt0)
    bin_labels = list(results[wt0][layers[0]][str(experts[0])].keys())

    for wt in get_weight_types(results):
        matrix, row_labels, _ = build_log_ratio_matrix(results, wt, bin_labels)
        plot_heatmap(
            matrix,
            row_labels,
            bin_labels,
            title=f"{name_prefix} — positional specialization ({wt})",
            save_path=save_dir / f"{file_prefix}_{wt}.pdf",
            cbar_label=r"$\log(\mathrm{mean} / \mathrm{baseline})$",
            num_experts_per_layer=len(experts),
        )


def plot_knockout(
    knockout: dict,
    save_dir: Path,
    *,
    file_name: str = "knockout.pdf",
) -> None:
    """Per-layer × per-expert heatmap of relative loss delta."""
    layers = sorted(knockout.keys(), key=_layer_sort_key)
    if not layers:
        return
    first = knockout[layers[0]]
    if "relative_delta_per_expert" not in first:
        return
    experts = sorted(int(k) for k in first["relative_delta_per_expert"].keys())
    matrix = np.array([
        [knockout[L]["relative_delta_per_expert"][str(e)] for e in experts]
        for L in layers
    ])

    cmap, norm = _diverging_cmap_for(matrix)
    fig_w = max(4.0, 0.4 * len(experts) + 1.6)
    fig_h = max(2.5, 0.3 * len(layers) + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(experts)))
    ax.set_xticklabels(experts)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([_short_layer_label(L) for L in layers])
    ax.set_xlabel("expert")
    ax.set_title("Knockout — relative $\\Delta$ loss per (layer, expert)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
    cbar.set_label("relative loss increase")
    cbar.outline.set_linewidth(0.5)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / file_name)
    plt.close(fig)
