"""Compare Soft MoE specialization between the L2 and no-L2 runs.

Reads the JSON artifacts produced by `soft_moe_analysis.py` for both
ablations (`analysis_outputs_l2/*_l2.json`,
`analysis_outputs_no_l2/*_no_l2.json`) and answers four questions:

1. **Specialization strength** — does L2 produce stronger or weaker
   `|log_ratio|` per (layer, expert, class)?
2. **Expert diversity within a layer** — do experts in a layer split
   the token space (high diversity) or collapse onto the same niche
   (high redundancy)?
3. **Cross-run agreement** — when L2 and no-L2 agree on *what* each
   expert prefers, the routing topology is preserved and L2 only
   rescales; when they disagree, L2 reorganizes specialization.
4. **Where the effect lives** — per-(expert, class) delta heatmaps so
   reviewers can see *which* experts and *which* classes shift the
   most.

Only the amino-acid and property analyses are compared; positional
specialization is excluded because the binning is sequence-dependent
and the per-bin ordering is not directly meaningful for ablation
deltas.

How to read these numbers
-------------------------
The L2 router (`SoftRouter(normalize=True)` in `ares/models/soft_router.py`)
L2-normalizes both tokens and slot anchors, then multiplies by a single
learnable scalar:

    logits = scaler · cos(x, φ)   ∈   [−scaler, +scaler]

So with L2 on:

* dispatch logits are **bounded**, which mechanically caps every
  per-cell `|log_ratio|` near zero — small values are an *expected
  consequence of the parameterization*, not evidence that routing has
  collapsed.
* routing can only partition tokens by **direction**; no-L2 also has
  the magnitude axis, so two classes that occupy similar direction
  cones in embedding space can be distinguished by no-L2 but not by L2.

That means the strength / diversity / redundancy metrics measure
**effective specialization expressivity** under each parameterization,
not routing health. The L2-fair metric `selectivity_per_layer` is
scale-invariant (in `[0, 1]`) and the most direct apples-to-apples
comparison.

Outputs:
    <output_dir>/comparison/
        summary.md                                 — headline numbers, one table per analysis
        strength_per_layer.pdf                     — mean |log_ratio| per layer (Q1)
        strength_distribution.pdf                  — overlaid |log_ratio| distributions (Q1)
        expert_diversity_<a>.pdf                   — # distinct argmax classes / layer,
                                                     one file per analysis (Q2)
        expert_redundancy.pdf                      — mean within-layer cosine sim (Q2)
        agreement_per_layer.pdf                    — Spearman correlation per layer (Q3)
        strength_delta_layer_class_<a>_<wt>.pdf    — (layer × class) heatmap of
                                                     |log_ratio|_L2 − |log_ratio|_no-L2,
                                                     experts averaged out (Q4)
        strength_delta_per_class_<a>_<wt>.pdf      — per-class box+strip of the same
                                                     strength delta (Q4 supplement)
        strength_per_class_<a>_<wt>.pdf            — two-line plot of mean |log_ratio|
                                                     per class for L2 vs no-L2 (Q1, Q4)
        selectivity_per_layer.pdf                  — L2-fair, scale-invariant
                                                     per-expert selectivity index
                                                     averaged per layer (Q1)

Usage:
    python -m ares.evaluation.compare_l2_vs_no_l2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from visualize_analysis import (
    CANONICAL_AA,
    _diverging_cmap_for,
    _short_layer_label,
    apply_style,
    get_experts,
    get_layers,
    get_weight_types,
    load_json,
)


# =============================================================================
# Analysis registry
# =============================================================================

_DEFAULT_PROPERTY_GROUPS: Tuple[str, ...] = (
    "hydrophobic", "polar_uncharged", "positive", "negative",
    "charged", "aromatic", "small", "cysteine", "proline",
)


def _property_classes(results: dict) -> List[str]:
    md = results.get("_metadata", {})
    groups = md.get("property_groups")
    if isinstance(groups, dict) and groups:
        return list(groups.keys())
    return list(_DEFAULT_PROPERTY_GROUPS)


def _aa_classes(_: dict) -> List[str]:
    return list(CANONICAL_AA)


_ANALYSES: List[Tuple[str, str, Callable[[dict], List[str]]]] = [
    ("amino_acid", "amino_acid_preferences", _aa_classes),
    ("property",   "property_preferences",   _property_classes),
]


def _pretty_analysis(key: str) -> str:
    """Human-readable analysis label for plot titles and reports."""
    return key.replace("_", " ").title()


# =============================================================================
# Loading
# =============================================================================

def load_run(input_dir: Path, suffix: str) -> Dict[str, dict]:
    """Load every analysis JSON in `input_dir` matching `<stem><suffix>.json`."""
    out: Dict[str, dict] = {}
    for key, stem, _ in _ANALYSES:
        path = input_dir / f"{stem}{suffix}.json"
        if path.exists():
            out[key] = load_json(path)
    return out


# =============================================================================
# Aligned matrix builder
# =============================================================================

def aligned_matrices(
    l2_results: dict,
    no_l2_results: dict,
    weight_type: str,
    classes: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[int]]:
    """Build paired (layer*expert, class) log-ratio matrices in identical order.

    A cell is masked (NaN) iff *either* run flags it `low_support` or
    its `log_ratio` is null. This way every metric we compute uses the
    same set of comparable cells in both runs.
    """
    common_layers = [
        L for L in get_layers(l2_results, weight_type)
        if L in get_layers(no_l2_results, weight_type)
    ]
    common_experts = sorted(
        set(get_experts(l2_results, weight_type))
        & set(get_experts(no_l2_results, weight_type))
    )
    rows_l2: List[List[float]] = []
    rows_no: List[List[float]] = []
    row_labels: List[str] = []
    inner_l2 = l2_results[weight_type]
    inner_no = no_l2_results[weight_type]
    for layer in common_layers:
        layer_short = _short_layer_label(layer)
        for e in common_experts:
            entry_l2 = inner_l2[layer].get(str(e), {})
            entry_no = inner_no[layer].get(str(e), {})
            r_l2: List[float] = []
            r_no: List[float] = []
            for cls in classes:
                e_l2 = entry_l2.get(cls)
                e_no = entry_no.get(cls)
                if (e_l2 is None or e_no is None
                        or e_l2.get("low_support") or e_no.get("low_support")):
                    r_l2.append(np.nan)
                    r_no.append(np.nan)
                    continue
                lr_l2 = e_l2.get("log_ratio")
                lr_no = e_no.get("log_ratio")
                r_l2.append(np.nan if lr_l2 is None else float(lr_l2))
                r_no.append(np.nan if lr_no is None else float(lr_no))
            rows_l2.append(r_l2)
            rows_no.append(r_no)
            row_labels.append(f"{layer_short}.E{e}")
    return (
        np.array(rows_l2, dtype=float),
        np.array(rows_no, dtype=float),
        row_labels,
        common_layers,
        common_experts,
    )


# =============================================================================
# Metrics
# =============================================================================

def strength_summary(mat: np.ndarray) -> Dict[str, float]:
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return {k: float("nan") for k in ("mean_abs", "p95_abs", "p99_abs", "max_abs")}
    abs_finite = np.abs(finite)
    return {
        "mean_abs": float(np.mean(abs_finite)),
        "p95_abs": float(np.percentile(abs_finite, 95)),
        "p99_abs": float(np.percentile(abs_finite, 99)),
        "max_abs": float(np.max(abs_finite)),
    }


def per_layer_mean_abs(mat: np.ndarray, num_experts: int) -> np.ndarray:
    n_rows, n_cols = mat.shape
    n_layers = n_rows // num_experts
    blocks = np.abs(mat).reshape(n_layers, num_experts, n_cols)
    return np.array([
        np.nanmean(b) if np.any(np.isfinite(b)) else np.nan for b in blocks
    ])


def per_layer_diversity(mat: np.ndarray, num_experts: int) -> np.ndarray:
    """Distinct argmax classes among experts per layer (peak |log_ratio|).

    *Caveat for L2 vs no-L2 comparisons:* L2-normalized routing
    partitions tokens by direction only, which is coarser than the
    direction-plus-magnitude partitioning available to no-L2. Multiple
    experts can therefore end up specializing on overlapping direction
    cones, which lowers this metric *mechanically* — not because
    routing has failed. Use `per_layer_selectivity` (scale-invariant)
    for the L2-fair comparison.
    """
    n_rows, _ = mat.shape
    n_layers = n_rows // num_experts
    out = np.zeros(n_layers, dtype=int)
    for L in range(n_layers):
        block = mat[L * num_experts:(L + 1) * num_experts]
        # All-NaN rows have no defined argmax; replace with -inf so they
        # don't get assigned to class 0 by accident.
        absblock = np.where(np.isfinite(block), np.abs(block), -np.inf)
        argmax = np.argmax(absblock, axis=1)
        # Drop experts where every cell was NaN (argmax is meaningless).
        valid = np.isfinite(absblock).any(axis=1)
        out[L] = len(set(argmax[valid].tolist()))
    return out


def per_layer_redundancy(mat: np.ndarray, num_experts: int) -> np.ndarray:
    """Mean off-diagonal cosine similarity among experts within a layer.

    NaNs are treated as zero so they don't drag the cosine — they simply
    contribute nothing to either side of the inner product. Higher values
    mean experts behave more similarly (more redundant routing).

    *Caveat for L2 vs no-L2 comparisons:* values near 1 on L2 dispatch
    indicate that the per-class dispatch *profiles* are nearly identical
    across experts. Under L2 routing this most likely reflects
    tightly-clustered slot anchors `φ̂_e` on the unit sphere rather than
    "experts have collapsed onto the same function" — they may still
    differentiate at finer directional resolution than the AA / property
    aggregation used here.
    """
    n_rows, _ = mat.shape
    n_layers = n_rows // num_experts
    out = np.full(n_layers, np.nan)
    for L in range(n_layers):
        block = mat[L * num_experts:(L + 1) * num_experts]
        block_clean = np.where(np.isfinite(block), block, 0.0)
        norms = np.linalg.norm(block_clean, axis=1, keepdims=True)
        if (norms < 1e-12).any():
            # If any row is effectively zero, redundancy is ill-defined.
            continue
        normed = block_clean / norms
        sim = normed @ normed.T
        mask = ~np.eye(num_experts, dtype=bool)
        out[L] = float(np.mean(sim[mask]))
    return out


def per_layer_selectivity(mat: np.ndarray, num_experts: int) -> np.ndarray:
    """Per-layer mean *selectivity index* across experts.

    For each (layer, expert), the selectivity index is

        SI = 1 − mean_c |log_ratio[c]| / max_c |log_ratio[c]|     ∈  [0, 1]

    where the mean and max are taken over the classes that have a
    finite log-ratio. SI = 0 means every class draws the same |log_ratio|
    (uniform expert), and SI ≈ 1 means a single class dominates the
    expert's response.

    The metric is **scale-invariant**: rescaling all log-ratios by a
    constant `k > 0` leaves SI unchanged. That makes it the L2-fair
    counterpart to `per_layer_mean_abs`, since L2 mechanically caps the
    range of `|log_ratio|` (the dispatch logits are `scaler · cos(x, φ)`,
    bounded by `|scaler|`), but it does **not** cap how *concentrated*
    the response is across classes.

    Returns a 1-D array of per-layer mean SIs, with NaN for layers
    where every expert had max `|log_ratio| < 1e-12` (no resolvable
    peak; the metric is undefined).
    """
    n_rows, _ = mat.shape
    n_layers = n_rows // num_experts
    out = np.full(n_layers, np.nan)
    for L in range(n_layers):
        block = np.abs(mat[L * num_experts:(L + 1) * num_experts])
        si_per_expert: List[float] = []
        for row in block:
            finite = row[np.isfinite(row)]
            if finite.size == 0:
                continue
            peak = float(finite.max())
            if peak < 1e-12:
                continue
            si_per_expert.append(1.0 - float(finite.mean()) / peak)
        if si_per_expert:
            out[L] = float(np.mean(si_per_expert))
    return out


def per_layer_spearman(
    mat_a: np.ndarray, mat_b: np.ndarray, num_experts: int,
) -> np.ndarray:
    n_rows, _ = mat_a.shape
    n_layers = n_rows // num_experts
    out = np.full(n_layers, np.nan)
    for L in range(n_layers):
        a = mat_a[L * num_experts:(L + 1) * num_experts].ravel()
        b = mat_b[L * num_experts:(L + 1) * num_experts].ravel()
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            continue
        rho, _ = spearmanr(a[mask], b[mask])
        if np.isfinite(rho):
            out[L] = float(rho)
    return out


def top_class_flip_rate(
    mat_a: np.ndarray, mat_b: np.ndarray, num_experts: int,
) -> Tuple[float, int, int]:
    """Fraction of experts whose argmax-class flips between runs.

    Returns (flip_rate, num_flips, num_compared).
    """
    n_rows, _ = mat_a.shape
    flips = 0
    compared = 0
    for r in range(n_rows):
        a = mat_a[r]
        b = mat_b[r]
        if not (np.isfinite(a).any() and np.isfinite(b).any()):
            continue
        ai = int(np.nanargmax(np.abs(np.where(np.isfinite(a), a, np.nan))))
        bi = int(np.nanargmax(np.abs(np.where(np.isfinite(b), b, np.nan))))
        compared += 1
        if ai != bi:
            flips += 1
    rate = (flips / compared) if compared else float("nan")
    return rate, flips, compared


# =============================================================================
# Plots
# =============================================================================

def _layer_xticks(ax: plt.Axes, layers: Sequence[str]) -> None:
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([_short_layer_label(L) for L in layers],
                       rotation=45, ha="right")


def plot_paired_lines(
    series: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, Sequence[str]]],
    *,
    title: str,
    ylabel: str,
    save_path: Path,
    hline: Optional[float] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Faceted (analysis × weight_type) line plot, two lines (L2, no-L2)."""
    if not series:
        return
    analyses = sorted({a for (a, _) in series})
    weight_types = sorted({w for (_, w) in series})
    n_rows, n_cols = len(weight_types), len(analyses)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols + 0.6, 2.4 * n_rows + 0.8),
        squeeze=False, sharey=True, constrained_layout=True,
    )
    for (a, w), (l2, no_l2, layers) in series.items():
        r = weight_types.index(w)
        c = analyses.index(a)
        ax = axes[r, c]
        x = np.arange(len(layers))
        ax.plot(x, l2, "o-", color="#c0392b", linewidth=1.5, markersize=3.5, label="L2")
        ax.plot(x, no_l2, "s--", color="#2c3e50", linewidth=1.2, markersize=3.5, label="no-L2")
        if hline is not None:
            ax.axhline(hline, color="#999", linewidth=0.6, linestyle=":")
        if ylim is not None:
            ax.set_ylim(*ylim)
        _layer_xticks(ax, layers)
        ax.set_title(f"{_pretty_analysis(a)} ({w})", pad=2)
        if c == 0:
            ax.set_ylabel(ylabel)
    axes[0, -1].legend(loc="best")
    fig.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_single_line(
    series: Dict[Tuple[str, str], Tuple[np.ndarray, Sequence[str]]],
    *,
    title: str,
    ylabel: str,
    save_path: Path,
    hline: Optional[float] = None,
    color: str = "#6c3483",
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Faceted (analysis × weight_type) plot of a *single* series per panel.
    Used for metrics that are themselves a comparison (e.g. Spearman ρ
    between L2 and no-L2)."""
    if not series:
        return
    analyses = sorted({a for (a, _) in series})
    weight_types = sorted({w for (_, w) in series})
    n_rows, n_cols = len(weight_types), len(analyses)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.8 * n_cols + 0.6, 2.4 * n_rows + 0.8),
        squeeze=False, sharey=True, constrained_layout=True,
    )
    for (a, w), (vals, layers) in series.items():
        r = weight_types.index(w)
        c = analyses.index(a)
        ax = axes[r, c]
        x = np.arange(len(layers))
        ax.plot(x, vals, "o-", color=color, linewidth=1.5, markersize=3.5)
        if hline is not None:
            ax.axhline(hline, color="#999", linewidth=0.6, linestyle=":")
        if ylim is not None:
            ax.set_ylim(*ylim)
        _layer_xticks(ax, layers)
        ax.set_title(f"{_pretty_analysis(a)} ({w})", pad=2)
        if c == 0:
            ax.set_ylabel(ylabel)
    fig.suptitle(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_strength_distribution(
    distributions: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
    *,
    save_path: Path,
) -> None:
    """Overlaid violin: |log_ratio| under L2 vs no-L2, faceted by analysis × wt."""
    if not distributions:
        return
    analyses = sorted({a for (a, _) in distributions})
    weight_types = sorted({w for (_, w) in distributions})
    n_rows, n_cols = len(weight_types), len(analyses)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols + 0.6, 2.2 * n_rows + 0.8),
        squeeze=False, sharey=True, constrained_layout=True,
    )
    for (a, w), (l2_vals, no_l2_vals) in distributions.items():
        r = weight_types.index(w)
        c = analyses.index(a)
        ax = axes[r, c]
        data = [np.abs(l2_vals[np.isfinite(l2_vals)]),
                np.abs(no_l2_vals[np.isfinite(no_l2_vals)])]
        if all(d.size > 0 for d in data):
            parts = ax.violinplot(data, positions=[0, 1], widths=0.7,
                                  showmeans=True, showmedians=False, showextrema=False)
            for pc, color in zip(parts["bodies"], ["#c0392b", "#2c3e50"]):
                pc.set_facecolor(color)
                pc.set_alpha(0.4)
                pc.set_edgecolor(color)
            parts["cmeans"].set_color("#222")
            parts["cmeans"].set_linewidth(1.2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["L2", "no-L2"])
        ax.set_title(f"{_pretty_analysis(a)} ({w})", pad=2)
        if c == 0:
            ax.set_ylabel(r"$|\log\,\mathrm{ratio}|$")
    fig.suptitle("Specialization strength distribution (per-cell |log ratio|)")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_strength_delta_layer_class(
    mat_l2: np.ndarray,
    mat_no_l2: np.ndarray,
    classes: Sequence[str],
    layers: Sequence[str],
    num_experts: int,
    *,
    title: str,
    save_path: Path,
) -> None:
    """Compact (num_layers × num_classes) heatmap of the
    *specialization-strength* difference between runs.

    Per cell we compute `|log_ratio|_L2 − |log_ratio|_no-L2`, then
    average over experts within the layer. Sign convention:

    - **negative (blue)** → no-L2 is *more specialized* at this
      `(layer, class)` than L2 (L2 erodes specialization here).
    - **positive (red)**  → L2 is *more specialized* at this
      `(layer, class)` than no-L2.

    This is the strength-comparison view, consistent with
    `strength_per_layer.pdf` / `strength_distribution.pdf`. (For the
    direction of preference vs avoidance, look at the per-run
    figures from `visualize_soft_moe.py`.)
    """
    strength_delta = np.abs(mat_l2) - np.abs(mat_no_l2)
    n_rows, n_cols = strength_delta.shape
    n_layers = n_rows // num_experts
    blocks = strength_delta.reshape(n_layers, num_experts, n_cols)
    with np.errstate(invalid="ignore"):
        agg = np.where(
            np.any(np.isfinite(blocks), axis=1),
            np.nanmean(blocks, axis=1),
            np.nan,
        )
    cmap, norm = _diverging_cmap_for(agg)

    fig_w = max(3.5, 0.28 * n_cols + 1.4)
    fig_h = max(2.0, 0.22 * n_layers + 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(agg, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([_short_layer_label(L) for L in layers])
    ax.set_ylabel("layer")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="both")
    cbar.set_label(r"$|\log\,\mathrm{ratio}|_{L2} - |\log\,\mathrm{ratio}|_{\mathrm{no\!-\!}L2}$")
    cbar.outline.set_linewidth(0.5)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_strength_delta_per_class(
    mat_l2: np.ndarray,
    mat_no_l2: np.ndarray,
    classes: Sequence[str],
    *,
    title: str,
    save_path: Path,
) -> None:
    """Per-class distribution of the strength delta
    `|log_ratio|_L2 − |log_ratio|_no-L2` across all (layer, expert)
    cells, drawn as a box plot with a jittered strip behind it.

    Same sign convention as `plot_strength_delta_layer_class`: a box
    sitting **below zero** means no-L2 is more specialized on that
    class than L2 (across most experts and layers).
    """
    strength_delta = np.abs(mat_l2) - np.abs(mat_no_l2)
    _, n_cols = strength_delta.shape
    data_per_class: List[np.ndarray] = [
        strength_delta[:, c][np.isfinite(strength_delta[:, c])] for c in range(n_cols)
    ]
    if not any(d.size > 0 for d in data_per_class):
        return

    fig_w = max(3.5, 0.28 * n_cols + 1.4)
    fig_h = 2.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    positions = np.arange(n_cols)

    rng = np.random.default_rng(0)
    for c, vals in enumerate(data_per_class):
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=vals.size)
        ax.scatter(c + jitter, vals, s=3, color="#888", alpha=0.35, linewidths=0)

    ax.boxplot(
        data_per_class, positions=positions, widths=0.55,
        showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="#d6eaf8", edgecolor="#1f618d", linewidth=0.8),
        medianprops=dict(color="#1f618d", linewidth=1.2),
        whiskerprops=dict(color="#1f618d", linewidth=0.8),
        capprops=dict(color="#1f618d", linewidth=0.8),
    )

    ax.axhline(0, color="#999", linewidth=0.6, linestyle=":")
    ax.set_xticks(positions)
    ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"$|\log\,\mathrm{ratio}|_{L2} - |\log\,\mathrm{ratio}|_{\mathrm{no\!-\!}L2}$")
    ax.set_title(title)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_strength_per_class(
    mat_l2: np.ndarray,
    mat_no_l2: np.ndarray,
    classes: Sequence[str],
    *,
    title: str,
    save_path: Path,
) -> None:
    """Two-line plot of mean specialization strength per class.

    For each class, plot `mean_{layer, expert} |log_ratio|` separately
    for L2 (red, solid) and no-L2 (dark blue, dashed). Behind each
    line, a shaded band spans the 25th–75th percentile of `|log_ratio|`
    across the (layer, expert) cells, so the absolute strengths and
    the L2 / no-L2 gap are both visible at a glance — analogous to
    `strength_per_layer.pdf` but indexed by class instead of layer.
    """
    abs_l2 = np.abs(mat_l2)
    abs_no = np.abs(mat_no_l2)
    _, n_cols = abs_l2.shape

    def _per_class_stats(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.full(n_cols, np.nan)
        q25 = np.full(n_cols, np.nan)
        q75 = np.full(n_cols, np.nan)
        for c in range(n_cols):
            col = mat[:, c]
            col = col[np.isfinite(col)]
            if col.size:
                mean[c] = col.mean()
                q25[c], q75[c] = np.percentile(col, [25, 75])
        return mean, q25, q75

    m_l2, l_l2, u_l2 = _per_class_stats(abs_l2)
    m_no, l_no, u_no = _per_class_stats(abs_no)

    fig_w = max(3.5, 0.28 * n_cols + 1.4)
    fig_h = 2.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    x = np.arange(n_cols)

    ax.fill_between(x, l_l2, u_l2, color="#c0392b", alpha=0.15, linewidth=0)
    ax.plot(x, m_l2, "o-", color="#c0392b", linewidth=1.5, markersize=4, label="L2")
    ax.fill_between(x, l_no, u_no, color="#2c3e50", alpha=0.15, linewidth=0)
    ax.plot(x, m_no, "s--", color="#2c3e50", linewidth=1.2, markersize=4, label="no-L2")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"mean $|\log\,\mathrm{ratio}|$")
    ax.set_title(title)
    ax.legend(loc="best")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


# =============================================================================
# Report
# =============================================================================

def _fmt(x: float, digits: int = 3) -> str:
    return "—" if not np.isfinite(x) else f"{x:.{digits}f}"


def write_summary(report_path: Path, sections: List[Dict]) -> None:
    """Write a markdown summary of the headline numbers."""
    lines: List[str] = []
    lines.append("# L2 vs no-L2 — Soft MoE specialization comparison\n")
    lines.append(
        "> All numbers are computed on `(layer × expert × class)` cells "
        "where **both** runs report a usable `log_ratio` "
        "(neither side is `low_support`).\n"
    )
    lines.append(
        "\n## How to read these numbers\n"
        "\n"
        "The L2 router (`SoftRouter(normalize=True)`) L2-normalizes both "
        "tokens and slot anchors and multiplies by a single learnable "
        "scalar, so dispatch logits reduce to "
        "`scaler · cos(x, φ) ∈ [−scaler, +scaler]`. Two consequences:\n"
        "\n"
        "1. Every per-cell `|log_ratio|` is **mechanically bounded near "
        "zero** under L2 — small specialization-strength values are "
        "expected from the parameterization, not evidence of a routing "
        "failure.\n"
        "2. L2 routing partitions tokens by **direction only**; no-L2 "
        "also has the magnitude axis available, so AAs that share a "
        "directional cone in embedding space can be split by no-L2 "
        "but not by L2.\n"
        "\n"
        "As a result, the strength / diversity / redundancy numbers "
        "below all measure **effective specialization expressivity** "
        "under each parameterization, not routing health. Use the "
        "`selectivity` block (scale-invariant, in `[0, 1]`) for the "
        "L2-fair apples-to-apples comparison.\n"
    )
    for sec in sections:
        a = sec["analysis"]
        w = sec["weight_type"]
        lines.append(f"\n## {_pretty_analysis(a)} — `{w}`\n")
        s_l2 = sec["strength_l2"]
        s_no = sec["strength_no_l2"]
        lines.append("**Specialization strength** (`|log_ratio|`):\n")
        lines.append("| | mean | p95 | p99 | max |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| L2 | {_fmt(s_l2['mean_abs'])} | {_fmt(s_l2['p95_abs'])} | {_fmt(s_l2['p99_abs'])} | {_fmt(s_l2['max_abs'])} |")
        lines.append(f"| no-L2 | {_fmt(s_no['mean_abs'])} | {_fmt(s_no['p95_abs'])} | {_fmt(s_no['p99_abs'])} | {_fmt(s_no['max_abs'])} |")
        delta_mean = s_l2["mean_abs"] - s_no["mean_abs"]
        rel = (delta_mean / s_no["mean_abs"] * 100.0) if s_no["mean_abs"] else float("nan")
        sign = "stronger" if delta_mean > 0 else "weaker"
        lines.append(
            f"\n→ L2 specialization is {_fmt(abs(delta_mean), 4)} "
            f"({_fmt(rel, 1)} %) {sign} on average than no-L2.\n"
        )
        lines.append(
            f"\n**Expert diversity** (distinct argmax classes per layer, "
            f"out of {sec['num_experts']} experts × {sec['num_classes']} classes):\n"
        )
        lines.append(
            f"- L2:    mean = {_fmt(np.nanmean(sec['diversity_l2']))}, "
            f"min = {int(np.nanmin(sec['diversity_l2']))}, "
            f"max = {int(np.nanmax(sec['diversity_l2']))}"
        )
        lines.append(
            f"- no-L2: mean = {_fmt(np.nanmean(sec['diversity_no_l2']))}, "
            f"min = {int(np.nanmin(sec['diversity_no_l2']))}, "
            f"max = {int(np.nanmax(sec['diversity_no_l2']))}"
        )
        lines.append(
            f"\n**Expert redundancy** (mean within-layer cosine similarity; "
            f"lower is better):\n"
        )
        lines.append(f"- L2:    mean = {_fmt(np.nanmean(sec['redundancy_l2']))}")
        lines.append(f"- no-L2: mean = {_fmt(np.nanmean(sec['redundancy_no_l2']))}")
        sel_l2 = float(np.nanmean(sec["selectivity_l2"]))
        sel_no = float(np.nanmean(sec["selectivity_no_l2"]))
        lines.append(
            f"\n**Specialization selectivity** "
            f"(`1 − mean / max` of `|log_ratio|` per expert, averaged over the layer; "
            f"scale-invariant in `[0, 1]`, so this is the **L2-fair** comparison):\n"
        )
        lines.append(f"- L2:    mean SI = {_fmt(sel_l2)}")
        lines.append(f"- no-L2: mean SI = {_fmt(sel_no)}")
        if np.isfinite(sel_l2) and np.isfinite(sel_no):
            sign = "more" if sel_l2 > sel_no else "less"
            lines.append(
                f"  → on this metric L2 experts are *{sign}* selective "
                f"than no-L2 (Δ = {_fmt(sel_l2 - sel_no, 4)})."
            )
        lines.append(
            f"\n**Cross-run agreement** (Spearman ρ between flattened "
            f"`(expert × class)` log-ratio matrices, per layer):\n"
        )
        lines.append(f"- mean ρ across layers = {_fmt(np.nanmean(sec['spearman']))}")
        lines.append(f"- min ρ = {_fmt(np.nanmin(sec['spearman']))}")
        lines.append(
            f"\n**Top-class flips**: "
            f"{sec['flip_count']} / {sec['flip_compared']} "
            f"({_fmt(sec['flip_rate'] * 100.0, 1)} %) experts change "
            f"their most-preferred class between L2 and no-L2.\n"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


# =============================================================================
# Orchestration
# =============================================================================

def run(
    l2_dir: Path,
    no_l2_dir: Path,
    output_dir: Path,
) -> None:
    apply_style()
    if not l2_dir.exists():
        raise FileNotFoundError(f"L2 input dir not found: {l2_dir}")
    if not no_l2_dir.exists():
        raise FileNotFoundError(f"no-L2 input dir not found: {no_l2_dir}")

    out = output_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    l2_runs = load_run(l2_dir, "_l2")
    no_l2_runs = load_run(no_l2_dir, "_no_l2")

    # Series for paired-line plots, keyed by (analysis, weight_type)
    strength_per_layer: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, Sequence[str]]] = {}
    strength_dist: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    diversity_per_layer: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, Sequence[str]]] = {}
    redundancy_per_layer: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, Sequence[str]]] = {}
    selectivity_per_layer: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, Sequence[str]]] = {}
    spearman_per_layer: Dict[Tuple[str, str], Tuple[np.ndarray, Sequence[str]]] = {}

    sections: List[Dict] = []

    for analysis_key, _, classes_fn in _ANALYSES:
        if analysis_key not in l2_runs or analysis_key not in no_l2_runs:
            print(f"[cmp]   skipping {analysis_key} (missing in one or both runs)")
            continue
        l2_data = l2_runs[analysis_key]
        no_l2_data = no_l2_runs[analysis_key]
        classes = classes_fn(l2_data)

        common_wts = [
            w for w in get_weight_types(l2_data)
            if w in get_weight_types(no_l2_data)
        ]
        for wt in common_wts:
            l2_mat, no_mat, _, layers, experts = aligned_matrices(
                l2_data, no_l2_data, wt, classes,
            )
            num_experts = len(experts)
            if num_experts == 0 or l2_mat.size == 0:
                continue

            s_l2 = strength_summary(l2_mat)
            s_no = strength_summary(no_mat)
            pl_strength_l2 = per_layer_mean_abs(l2_mat, num_experts)
            pl_strength_no = per_layer_mean_abs(no_mat, num_experts)
            div_l2 = per_layer_diversity(l2_mat, num_experts)
            div_no = per_layer_diversity(no_mat, num_experts)
            red_l2 = per_layer_redundancy(l2_mat, num_experts)
            red_no = per_layer_redundancy(no_mat, num_experts)
            sel_l2 = per_layer_selectivity(l2_mat, num_experts)
            sel_no = per_layer_selectivity(no_mat, num_experts)
            rho = per_layer_spearman(l2_mat, no_mat, num_experts)
            flip_rate, flips, compared = top_class_flip_rate(
                l2_mat, no_mat, num_experts,
            )

            strength_per_layer[(analysis_key, wt)] = (pl_strength_l2, pl_strength_no, layers)
            strength_dist[(analysis_key, wt)] = (l2_mat.ravel(), no_mat.ravel())
            diversity_per_layer[(analysis_key, wt)] = (div_l2, div_no, layers)
            redundancy_per_layer[(analysis_key, wt)] = (red_l2, red_no, layers)
            selectivity_per_layer[(analysis_key, wt)] = (sel_l2, sel_no, layers)
            spearman_per_layer[(analysis_key, wt)] = (rho, layers)

            pretty = _pretty_analysis(analysis_key)
            plot_strength_delta_layer_class(
                l2_mat, no_mat, classes, layers, num_experts,
                title=(
                    f"Specialization strength — L2 vs no-L2 "
                    f"({pretty}, {wt})"
                ),
                save_path=out / f"strength_delta_layer_class_{analysis_key}_{wt}.pdf",
            )
            plot_strength_delta_per_class(
                l2_mat, no_mat, classes,
                title=(
                    f"Specialization strength per class — L2 vs no-L2 "
                    f"({pretty}, {wt})"
                ),
                save_path=out / f"strength_delta_per_class_{analysis_key}_{wt}.pdf",
            )
            plot_strength_per_class(
                l2_mat, no_mat, classes,
                title=f"Specialization strength per class ({pretty}, {wt})",
                save_path=out / f"strength_per_class_{analysis_key}_{wt}.pdf",
            )

            sections.append({
                "analysis": analysis_key,
                "weight_type": wt,
                "num_experts": num_experts,
                "num_classes": len(classes),
                "strength_l2": s_l2,
                "strength_no_l2": s_no,
                "diversity_l2": div_l2,
                "diversity_no_l2": div_no,
                "redundancy_l2": red_l2,
                "redundancy_no_l2": red_no,
                "selectivity_l2": sel_l2,
                "selectivity_no_l2": sel_no,
                "spearman": rho,
                "flip_rate": flip_rate,
                "flip_count": flips,
                "flip_compared": compared,
            })
            print(
                f"[cmp]   {analysis_key:11s} {wt:9s} "
                f"|Δmean|={s_l2['mean_abs'] - s_no['mean_abs']:+.4f} "
                f"ρ̄={np.nanmean(rho):+.3f} "
                f"flips={flips}/{compared}"
            )

    plot_paired_lines(
        strength_per_layer,
        title="Specialization strength per layer (mean $|\\log\\,\\mathrm{ratio}|$)",
        ylabel=r"mean $|\log\,\mathrm{ratio}|$",
        save_path=out / "strength_per_layer.pdf",
    )
    plot_strength_distribution(strength_dist, save_path=out / "strength_distribution.pdf")
    # Diversity is in absolute "# distinct classes", so the y-axis cap
    # differs by analysis (e.g. 20 for AA, 9 for property at 32 experts).
    # Split into one file per analysis so each panel uses the right range,
    # with a dotted reference line at the theoretical max and a y-limit
    # that runs the full 0 .. cap range plus a tiny bit of headroom.
    for analysis_key in sorted({a for (a, _) in diversity_per_layer}):
        subset = {k: v for k, v in diversity_per_layer.items() if k[0] == analysis_key}
        sec_for_a = next(s for s in sections if s["analysis"] == analysis_key)
        cap = min(sec_for_a["num_experts"], sec_for_a["num_classes"])
        plot_paired_lines(
            subset,
            title=(
                f"Expert diversity per layer — {_pretty_analysis(analysis_key)} "
                f"(distinct argmax classes, max = {cap})"
            ),
            ylabel="# distinct classes",
            save_path=out / f"expert_diversity_{analysis_key}.pdf",
            hline=cap,
            ylim=(0, cap + 0.5),
        )
    plot_paired_lines(
        redundancy_per_layer,
        title="Expert redundancy per layer (mean within-layer cosine similarity)",
        ylabel="mean cosine sim",
        save_path=out / "expert_redundancy.pdf",
        hline=0.0,
    )
    plot_paired_lines(
        selectivity_per_layer,
        title=(
            "Specialization selectivity per layer "
            "(scale-invariant; L2-fair)"
        ),
        ylabel="mean selectivity index",
        save_path=out / "selectivity_per_layer.pdf",
        ylim=(0, 1.0),
    )
    plot_single_line(
        spearman_per_layer,
        title="Cross-run agreement (Spearman $\\rho$, L2 vs no-L2)",
        ylabel=r"Spearman $\rho$",
        save_path=out / "agreement_per_layer.pdf",
        hline=0.0,
        ylim=(-0.05, 1.05),
    )

    write_summary(out / "summary.md", sections)
    print(f"[cmp]   summary    -> {out / 'summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Soft MoE L2 vs no-L2 specialization. Reads the "
            "JSON outputs of soft_moe_analysis.py from both ablations."
        ),
    )
    here = Path(__file__).parent
    parser.add_argument("--l2-dir", type=Path, default=here / "analysis_outputs_l2")
    parser.add_argument("--no-l2-dir", type=Path, default=here / "analysis_outputs_no_l2")
    parser.add_argument("--output-dir", type=Path, default=here / "figures_comparison")
    args = parser.parse_args()
    run(l2_dir=args.l2_dir, no_l2_dir=args.no_l2_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
