"""LaTeX-style figure generator for Soft MoE analysis outputs.

Reads the JSON artifacts produced by `soft_moe_analysis.py` and writes
publication-ready vector PDFs (property / amino-acid / positional
specialization, plus optional expert knockout).

Usage:
    # default L2 run
    python -m ares.evaluation.visualize_soft_moe \\
        --input-dir  ares/evaluation/analysis_outputs_l2 \\
        --output-dir ares/evaluation/figures_l2

    # no-L2 ablation: same script, with the --no-l2 flag
    python -m ares.evaluation.visualize_soft_moe --no-l2

The `--no-l2` flag changes the *defaults* (input dir, output dir, file
suffix) so a single command reproduces either ablation. It does not
affect plotting behavior — it just selects which set of artifacts to
read. The defaults match what `soft_moe_analysis.py` writes:
`analysis_outputs_l2/*_l2.json` and `analysis_outputs_no_l2/*_no_l2.json`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from visualize_analysis import (
    apply_style,
    load_json,
    plot_aa_specialization,
    plot_knockout,
    plot_positional_specialization,
    plot_property_specialization,
)


# Default paths for each ablation. Keep these in sync with the dirs
# `soft_moe_analysis.py` writes to (analysis_outputs_{l2,no_l2}/, with
# matching `_l2` / `_no_l2` filename suffixes).
_L2_DEFAULTS = {
    "input_dir": "analysis_outputs_l2",
    "output_dir": "figures_l2",
    "suffix": "_l2",
}
_NO_L2_DEFAULTS = {
    "input_dir": "analysis_outputs_no_l2",
    "output_dir": "figures_no_l2",
    "suffix": "_no_l2",
}


def _stem(name: str, suffix: str) -> str:
    """Return the analysis filename with or without the `_no_l2` suffix."""
    return f"{name}{suffix}.json"


def run(
    input_dir: Path,
    output_dir: Path,
    *,
    suffix: str = "",
) -> None:
    apply_style()
    if not input_dir.exists():
        raise FileNotFoundError(f"Soft MoE input dir does not exist: {input_dir}")

    sub = output_dir / "soft_moe"
    sub.mkdir(parents=True, exist_ok=True)

    plotters = [
        ("property_preferences", plot_property_specialization),
        ("amino_acid_preferences", plot_aa_specialization),
        ("positional_preferences", plot_positional_specialization),
    ]
    for stem, plotter in plotters:
        fname = _stem(stem, suffix)
        path = input_dir / fname
        if path.exists():
            plotter(load_json(path), sub, name_prefix="Soft MoE")
            print(f"[soft]  {fname:38s} -> {sub}")
        else:
            print(f"[soft]  {fname:38s} (missing, skipped)")

    ko_path = input_dir / _stem("expert_knockout", suffix)
    if ko_path.exists():
        plot_knockout(load_json(ko_path), sub)
        print(f"[soft]  {ko_path.name:38s} -> {sub}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-quality figures from Soft MoE analysis "
            "outputs. Use --no-l2 to switch to the no-L2 ablation defaults."
        ),
    )
    parser.add_argument(
        "--no-l2",
        action="store_true",
        help=(
            "Use no-L2 ablation defaults: read analysis_outputs_no_l2/*_no_l2.json "
            "and write to figures_no_l2/. Default (without this flag) reads "
            "analysis_outputs_l2/*_l2.json and writes to figures_l2/."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override the input directory (defaults depend on --no-l2).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory (defaults depend on --no-l2).",
    )
    args = parser.parse_args()

    defaults = _NO_L2_DEFAULTS if args.no_l2 else _L2_DEFAULTS
    here = Path(__file__).parent

    input_dir: Path = args.input_dir if args.input_dir is not None else here / defaults["input_dir"]
    output_dir: Path = args.output_dir if args.output_dir is not None else here / defaults["output_dir"]

    run(input_dir=input_dir, output_dir=output_dir, suffix=defaults["suffix"])


if __name__ == "__main__":
    main()
