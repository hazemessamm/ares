"""
ProteinGym zero-shot evaluation pipeline for masked protein language models.

Loads data directly from the official HuggingFace dataset maintained by
OATML-Markslab (the authors of the ProteinGym paper):

    https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v1

The DMS_substitutions configuration contains all variants across ~217
assays in a single dataframe with columns:

    mutated_sequence, target_seq, mutant, DMS_score, DMS_score_bin, DMS_id

The per-variant data (wildtype, mutation string, DMS_score) is all on HF.
The aggregation metadata (UniProt_ID, selection_type) is NOT on HF and
must come from the official reference CSV:

    https://marks.hms.harvard.edu/proteingym/DMS_substitutions.csv

(Or equivalently from the ProteinGym GitHub repo.) Pass its path as
`reference_csv`.

Protocol implemented
--------------------
1. Scoring: Meier et al. (2021) masked marginals.
   For each variant, mask ALL mutated positions in the wildtype simultaneously,
   run ONE forward pass, and compute
       score = sum_{i in mutated_positions}
               [log P(mut_aa_i | masked_wt) - log P(wt_aa_i | masked_wt)]

   Implementation note: variants that share the same set of mutated positions
   (this includes (a) all single-mut variants at the same position and
   (b) all multi-mut variants probing the same combinatorial site) share
   the same masked input, so we run one forward pass per unique mask-set
   and reuse the cached per-position log-probabilities across all variants
   in the set. This is mathematically identical to the per-variant scheme
   above; it just avoids redundant forward passes. On the full ProteinGym
   DMS_substitutions benchmark this collapses ~2.47M forward passes to
   ~280K (~9x).

2. Aggregation (three stages, matching Table A5 of the ProteinGym paper):
     (a) DMS level:       Spearman rho per assay
     (b) UniProt level:   mean across assays that share a UniProt_ID
     (c) Function level:  mean within each of
                              {Activity, Binding, Expression,
                               OrganismalFitness, Stability}
                          then mean across the five category means
                          -> final "All" score.

3. Two aggregation variants:
     - "standard": arithmetic mean at every stage (matches leaderboard).
     - "fisher":   same three-stage structure, means taken in Fisher-z space
                   and transformed back with tanh. Reduces the bias-toward-zero
                   of averaging correlation coefficients. Supplementary.

Model assumptions
-----------------
- HuggingFace-style tokenizer exposing:
      tokenizer(seq, return_tensors="pt")["input_ids"]
      tokenizer.mask_token_id
      tokenizer.convert_tokens_to_ids(aa)   # single-char AA -> token id
- HuggingFace-style model whose output exposes logits as `output["logits"]`
  with shape [batch, seq_len, vocab_size].
- The tokenizer prepends a BOS/CLS token (so protein position 1 maps to
  token index 1). If your tokenizer does NOT prepend BOS, pass `bos_offset=0`.

Sanity check before trusting Ares numbers
-----------------------------------------
Run ESM-2 650M through this pipeline and confirm the aggregate is near the
published ~0.41. If it's off by more than ~0.03 there's a pipeline bug.
"""

from __future__ import annotations
import os
from ares.models import Ares
from ares.tokenization import AresProteinTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer
import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FUNCTION_CATEGORIES = [
    "Activity",
    "Binding",
    "Expression",
    "OrganismalFitness",
    "Stability",
]

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")

# ProteinGym mutation format: <wt_aa><1-indexed_position><mut_aa>, e.g. "A1P".
# Multiple mutations in a single variant are joined by ":", e.g. "A1P:D2N".
_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


# ---------------------------------------------------------------------------
# Mutation parsing
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    wt_aa: str
    position: int   # 1-indexed, as in ProteinGym's `mutant` column
    mut_aa: str


def parse_mutant_string(mutant_str: str) -> list[Mutation]:
    mutations = []
    for token in mutant_str.split(":"):
        match = _MUTATION_RE.match(token.strip())
        if match is None:
            raise ValueError(f"Malformed mutation token: {token!r} in {mutant_str!r}")
        wt_aa, position, mut_aa = match.group(1), int(match.group(2)), match.group(3)
        mutations.append(Mutation(wt_aa=wt_aa, position=position, mut_aa=mut_aa))
    return mutations


# ---------------------------------------------------------------------------
# Masked-marginals scoring
# ---------------------------------------------------------------------------

def score_variant_batch(
    wildtype: str,
    mutant_strings: list[str],
    model,
    tokenizer,
    device: torch.device,
    bos_offset: int = 1,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Compute masked-marginals scores for all variants on a single wildtype.

    Variants that share the same set of mutated token positions share their
    masked input, so we deduplicate by mask-position-set and run one forward
    pass per unique set. The resulting per-variant scores are mathematically
    identical to running one forward pass per variant (each variant still
    masks all of its own positions and reads log p(mut) - log p(wt) at each
    of those positions); we just don't pay the cost of recomputing the same
    forward pass.

    Returns array of per-variant scores (higher = fitter under the model).
    """
    model.eval()
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError(
            "Tokenizer has no mask_token_id. Masked marginals requires a masked LM tokenizer."
        )

    # Parse + validate all variants up front.
    parsed: list[list[Mutation]] = []
    for s in mutant_strings:
        mutations = parse_mutant_string(s)
        for mut in mutations:
            if mut.position < 1 or mut.position > len(wildtype):
                raise ValueError(
                    f"Mutation {mut.wt_aa}{mut.position}{mut.mut_aa} outside "
                    f"wildtype length {len(wildtype)}"
                )
            if wildtype[mut.position - 1] != mut.wt_aa:
                raise ValueError(
                    f"Mutation {mut.wt_aa}{mut.position}{mut.mut_aa} inconsistent "
                    f"with wildtype: position {mut.position} is "
                    f"{wildtype[mut.position - 1]!r}, not {mut.wt_aa!r}"
                )
            if mut.mut_aa not in STANDARD_AAS or mut.wt_aa not in STANDARD_AAS:
                raise ValueError(
                    f"Non-standard amino acid in {mut.wt_aa}{mut.position}{mut.mut_aa}"
                )
        parsed.append(mutations)

    # Tokenize wildtype once.
    base_tokens = tokenizer(wildtype, return_tensors="pt")["input_ids"][0].to(device)
    seq_len = base_tokens.shape[0]

    # Resolve the 20 standard AAs to token ids in a fixed canonical order.
    aa_list = sorted(STANDARD_AAS)
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
    aa_token_id_list: list[int] = []
    for aa in aa_list:
        tid = tokenizer.convert_tokens_to_ids(aa)
        if tid is None or tid == tokenizer.unk_token_id:
            raise ValueError(
                f"Tokenizer does not recognize amino acid {aa!r} as a single token."
            )
        aa_token_id_list.append(tid)
    aa_token_ids = torch.tensor(aa_token_id_list, device=device, dtype=torch.long)

    # For each variant, compute its tuple of masked token indices (sorted, deduped).
    # This is the variant's "mask-set" and is the key we deduplicate on.
    variant_mask_sets: list[tuple[int, ...]] = []
    for mutations in parsed:
        idxs: set[int] = set()
        for mut in mutations:
            ti = mut.position - 1 + bos_offset
            if ti >= seq_len:
                raise ValueError(
                    f"Token index {ti} out of range for tokenized length {seq_len}. "
                    f"Check bos_offset."
                )
            idxs.add(ti)
        variant_mask_sets.append(tuple(sorted(idxs)))

    # Deduplicate mask-sets, preserving first-occurrence order for reproducibility.
    set_to_id: dict[tuple[int, ...], int] = {}
    unique_sets: list[tuple[int, ...]] = []
    for ms in variant_mask_sets:
        if ms not in set_to_id:
            set_to_id[ms] = len(unique_sets)
            unique_sets.append(ms)

    # Cache: for each unique mask-set, a dict {token_idx: 20-vector of log p over standard AAs}.
    cache: list[dict[int, np.ndarray]] = [None] * len(unique_sets)  # type: ignore[list-item]

    with torch.inference_mode():
        for batch_start in range(0, len(unique_sets), batch_size):
            batch_sets = unique_sets[batch_start : batch_start + batch_size]
            B = len(batch_sets)

            batch_inputs = base_tokens.unsqueeze(0).repeat(B, 1)
            for b, ms in enumerate(batch_sets):
                for ti in ms:
                    batch_inputs[b, ti] = mask_id

            logits = model(batch_inputs)["logits"]

            # Some accelerate device_map configurations can return logits on cpu
            # even when the inputs were on cuda; force them onto our target
            # device so all downstream indexing tensors share a device.
            if logits.device != device:
                logits = logits.to(device, non_blocking=True)

            # Only log_softmax at the masked positions, and only keep the 20 AA columns.
            # This avoids materializing log_softmax over the full [B, L, V] tensor.
            for b, ms in enumerate(batch_sets):
                idx_t = torch.tensor(ms, device=device, dtype=torch.long)
                pos_logits = logits[b].index_select(0, idx_t).float()   # [|ms|, V]
                lp = torch.log_softmax(pos_logits, dim=-1)              # [|ms|, V]
                lp_aa = lp.index_select(1, aa_token_ids).cpu().numpy()  # [|ms|, 20]
                cache[batch_start + b] = {ti: lp_aa[k] for k, ti in enumerate(ms)}

    # Vectorized score readout against the CPU-side cache (no GPU sync per mutation).
    scores = np.empty(len(parsed), dtype=np.float64)
    for i, mutations in enumerate(parsed):
        c = cache[set_to_id[variant_mask_sets[i]]]
        s = 0.0
        for mut in mutations:
            ti = mut.position - 1 + bos_offset
            row = c[ti]
            s += row[aa_to_idx[mut.mut_aa]] - row[aa_to_idx[mut.wt_aa]]
        scores[i] = s
    return scores


# ---------------------------------------------------------------------------
# Per-assay evaluation
# ---------------------------------------------------------------------------

def evaluate_assay(
    assay_df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    mutant_col: str = "mutant",
    dms_score_col: str = "DMS_score",
    target_seq_col: str = "target_seq",
    bos_offset: int = 1,
    batch_size: int = 8,
) -> tuple[float, int]:
    """
    Score a single DMS assay and return (spearman_rho, n_valid_variants).

    The assay dataframe must be already filtered to a single DMS_id
    (all rows share the same target_seq).
    """
    if len(assay_df) == 0:
        return float("nan"), 0

    required = {mutant_col, dms_score_col, target_seq_col}
    missing = required - set(assay_df.columns)
    if missing:
        raise KeyError(f"Assay dataframe missing columns: {missing}")

    unique_wildtypes = assay_df[target_seq_col].unique()
    if len(unique_wildtypes) != 1:
        raise ValueError(
            f"Assay contains {len(unique_wildtypes)} distinct target_seq values; "
            f"expected exactly 1."
        )
    wildtype = unique_wildtypes[0]

    mutant_strings = assay_df[mutant_col].astype(str).tolist()
    dms_scores = assay_df[dms_score_col].to_numpy(dtype=np.float64)

    predictions = score_variant_batch(
        wildtype=wildtype,
        mutant_strings=mutant_strings,
        model=model,
        tokenizer=tokenizer,
        device=device,
        bos_offset=bos_offset,
        batch_size=batch_size,
    )

    mask = np.isfinite(predictions) & np.isfinite(dms_scores)
    if mask.sum() < 3:
        return float("nan"), int(mask.sum())
    predictions = predictions[mask]
    dms_scores = dms_scores[mask]

    if np.all(predictions == predictions[0]) or np.all(dms_scores == dms_scores[0]):
        return float("nan"), int(mask.sum())

    rho, _ = spearmanr(predictions, dms_scores)
    return float(rho), int(mask.sum())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_proteingym_from_hf(
    hf_name: str = "OATML-Markslab/ProteinGym_v1",
    config_name: str = "DMS_substitutions",
    split: str = "train",
) -> pd.DataFrame:
    """
    Load the official ProteinGym DMS substitutions dataset from HuggingFace.

    Returns a single dataframe with all variants across all assays. The
    DMS_id column identifies which assay each row belongs to.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_name, data_dir=config_name, split=split)
    print(ds)
    df = ds.to_pandas()

    required = {"mutated_sequence", "target_seq", "mutant", "DMS_score", "DMS_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"HF dataset missing expected columns: {missing}")

    return df


def load_reference_metadata(reference_csv: str) -> pd.DataFrame:
    """
    Load the DMS_substitutions reference CSV. This provides the aggregation
    metadata that is NOT in the HF dataset:
      - UniProt_ID            (for UniProt-level aggregation)
      - coarse_selection_type (for function-category aggregation; contains
                               the five canonical categories: Activity,
                               Binding, Expression, OrganismalFitness,
                               Stability)

    Note: the reference CSV also has a `selection_type` column, but that
    contains fine-grained assay descriptions ("cDNA display proteolysis",
    "Growth", "FACS", etc.) with 36 distinct values and some nulls. It is
    NOT the column used for leaderboard aggregation. The correct column is
    `coarse_selection_type`.

    Official source:
        https://marks.hms.harvard.edu/proteingym/DMS_substitutions.csv
    or from the ProteinGym GitHub repo.

    This function renames `coarse_selection_type` to `selection_type` in
    the returned dataframe so the rest of the pipeline can use a single
    column name.
    """
    ref = pd.read_csv(reference_csv)
    required = {"DMS_id", "UniProt_ID", "coarse_selection_type"}
    missing = required - set(ref.columns)
    if missing:
        raise KeyError(
            f"Reference CSV missing columns {missing}. "
            f"Available columns: {sorted(ref.columns)}"
        )
    out = ref[["DMS_id", "UniProt_ID", "coarse_selection_type"]].copy()
    out = out.rename(columns={"coarse_selection_type": "selection_type"})

    # Sanity check: the five category values should match FUNCTION_CATEGORIES.
    observed = set(out["selection_type"].dropna().unique())
    expected = set(FUNCTION_CATEGORIES)
    unexpected = observed - expected
    if unexpected:
        raise ValueError(
            f"coarse_selection_type contains unexpected values {unexpected}. "
            f"Expected exactly {expected}."
        )
    return out


# ---------------------------------------------------------------------------
# Full-benchmark evaluation
# ---------------------------------------------------------------------------

def evaluate_all_assays(
    variants_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    bos_offset: int = 1,
    batch_size: int = 8,
    max_length: Optional[int] = None,
    dms_id_subset: Optional[list[str]] = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate every assay present in both variants_df and reference_df.

    Returns:
        per_assay_df: dataframe with columns
            DMS_id, UniProt_ID, selection_type, spearman, n_variants
        assay_status: dict with keys
            success: list of DMS_ids that produced a score
            fail:    list of {DMS_id, reason} records for assays not evaluated
    """
    # Inner join so we only evaluate assays that have both variants (on HF)
    # and metadata (in the reference CSV).
    merged = variants_df.merge(reference_df, on="DMS_id", how="inner")

    if dms_id_subset is not None:
        merged = merged[merged["DMS_id"].isin(dms_id_subset)]

    records = []
    success_ids: list[str] = []
    fail_records: list[dict[str, str]] = []
    # Group once instead of repeatedly filtering merged[merged["DMS_id"] == dms_id]
    # (which is O(N_total) per assay).
    grouped = list(merged.groupby("DMS_id", sort=False))
    if verbose:
        print(f"Evaluating {len(grouped)} assays...")

    for dms_id, assay_df in grouped:
        wildtype_len = len(assay_df["target_seq"].iloc[0])

        if max_length is not None and wildtype_len > max_length:
            reason = f"length {wildtype_len} > max_length {max_length}"
            fail_records.append({"DMS_id": dms_id, "reason": reason})
            if verbose:
                print(f"[skip] {dms_id}: {reason}")
            continue

        try:
            rho, n_valid = evaluate_assay(
                assay_df=assay_df,
                model=model,
                tokenizer=tokenizer,
                device=device,
                bos_offset=bos_offset,
                batch_size=batch_size,
            )
        except Exception as exc:
            fail_records.append({"DMS_id": dms_id, "reason": str(exc)})
            if verbose:
                print(f"[error] {dms_id}: {exc}")
            continue

        records.append({
            "DMS_id": dms_id,
            "UniProt_ID": assay_df["UniProt_ID"].iloc[0],
            "selection_type": assay_df["selection_type"].iloc[0],
            "spearman": rho,
            "n_variants": n_valid,
        })
        success_ids.append(dms_id)
        if verbose:
            print(f"{dms_id:60s}  rho = {rho:+.4f}  (n={n_valid})")

    per_assay_df = pd.DataFrame.from_records(
        records,
        columns=["DMS_id", "UniProt_ID", "selection_type", "spearman", "n_variants"],
    )
    assay_status = {"success": success_ids, "fail": fail_records}
    return per_assay_df, assay_status


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _fisher_z(rho: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    return np.arctanh(np.clip(rho, -1.0 + eps, 1.0 - eps))


def _fisher_mean(rhos: np.ndarray) -> float:
    rhos = np.asarray(rhos, dtype=np.float64)
    rhos = rhos[np.isfinite(rhos)]
    if rhos.size == 0:
        return float("nan")
    return float(np.tanh(np.mean(_fisher_z(rhos))))


def _arithmetic_mean(rhos: np.ndarray) -> float:
    rhos = np.asarray(rhos, dtype=np.float64)
    rhos = rhos[np.isfinite(rhos)]
    if rhos.size == 0:
        return float("nan")
    return float(np.mean(rhos))


def _aggregate(per_assay_df: pd.DataFrame, mean_fn: Callable[[np.ndarray], float]) -> dict:
    df = per_assay_df.dropna(subset=["spearman"]).copy()

    # Stage 2: UniProt level.
    uniprot_rows = []
    for (uniprot, sel), group in df.groupby(["UniProt_ID", "selection_type"]):
        uniprot_rows.append({
            "UniProt_ID": uniprot,
            "selection_type": sel,
            "uniprot_spearman": mean_fn(group["spearman"].to_numpy()),
            "n_assays": len(group),
        })
    uniprot_level = pd.DataFrame.from_records(uniprot_rows)

    # Stage 3: category level.
    category_means = {}
    for cat in FUNCTION_CATEGORIES:
        cat_df = uniprot_level[uniprot_level["selection_type"] == cat]
        if len(cat_df) == 0:
            category_means[cat] = float("nan")
        else:
            category_means[cat] = mean_fn(cat_df["uniprot_spearman"].to_numpy())

    # Final: mean of the five category means.
    category_values = np.array(
        [category_means[c] for c in FUNCTION_CATEGORIES], dtype=np.float64
    )
    final = mean_fn(category_values)

    return {
        "uniprot_level": uniprot_level,
        "category_means": category_means,
        "final_spearman": final,
    }


def aggregate_standard(per_assay_df: pd.DataFrame) -> dict:
    """Arithmetic mean at every stage. Headline number for the paper."""
    return _aggregate(per_assay_df, mean_fn=_arithmetic_mean)


def aggregate_fisher(per_assay_df: pd.DataFrame) -> dict:
    """Fisher-z mean at every stage. Supplementary."""
    return _aggregate(per_assay_df, mean_fn=_fisher_mean)


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------

def run_proteingym_evaluation(
    model,
    tokenizer,
    reference_csv: str,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
    hf_name: str = "OATML-Markslab/ProteinGym_v1",
    hf_config: str = "DMS_substitutions",
    bos_offset: int = 1,
    batch_size: int = 8,
    max_length: Optional[int] = 1024,
    dms_id_subset: Optional[list[str]] = None,
    verbose: bool = True,
    inference_dtype: Optional[torch.dtype] = None,
    compile_model: bool = False,
) -> dict:
    """
    Full pipeline: load data from HF, score every assay with masked marginals,
    aggregate under both standard and Fisher protocols, save results.

    Returns a dict with:
        per_assay: DataFrame of per-assay Spearmans
        assay_status: JSON-serializable success/fail assay records
        standard:  standard ProteinGym aggregation
        fisher:    Fisher-averaged aggregation
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    if inference_dtype is not None:
        # Run the model natively in this dtype rather than wrapping each forward
        # in torch.autocast. autocast keeps fp32 copies of intermediates inside
        # the ops it promotes (softmax, layernorm, reductions), which can
        # actually *increase* peak memory for an already-bf16 MoE model like
        # Ares. Casting once is simpler and strictly lighter on memory.
        try:
            model = model.to(dtype=inference_dtype)
        except Exception as exc:
            if verbose:
                print(
                    f"[warn] could not cast model to {inference_dtype}: {exc}. "
                    f"Continuing in the model's existing dtype."
                )

    if compile_model:
        if verbose:
            print("Compiling model with torch.compile (first batch will be slow)...")
        model = torch.compile(model)

    if verbose:
        print(f"Loading ProteinGym variants from HF: {hf_name} / {hf_config}")
    variants_df = load_proteingym_from_hf(hf_name=hf_name, config_name=hf_config)
    if verbose:
        print(f"  loaded {len(variants_df)} variants across "
              f"{variants_df['DMS_id'].nunique()} assays")

    if verbose:
        print(f"Loading reference metadata from {reference_csv}")
    reference_df = load_reference_metadata(reference_csv)

    per_assay_df, assay_status = evaluate_all_assays(
        variants_df=variants_df,
        reference_df=reference_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
        bos_offset=bos_offset,
        batch_size=batch_size,
        max_length=max_length,
        dms_id_subset=dms_id_subset,
        verbose=verbose,
    )

    standard = aggregate_standard(per_assay_df)
    fisher = aggregate_fisher(per_assay_df)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        per_assay_df.to_csv(
            os.path.join(output_dir, "per_assay_spearman.csv"), index=False
        )
        with open(os.path.join(output_dir, "assay_status.json"), "w", encoding="utf-8") as f:
            json.dump(assay_status, f, indent=2)
        standard["uniprot_level"].to_csv(
            os.path.join(output_dir, "uniprot_level_standard.csv"), index=False
        )
        fisher["uniprot_level"].to_csv(
            os.path.join(output_dir, "uniprot_level_fisher.csv"), index=False
        )
        summary = pd.DataFrame([
            {
                "aggregation": "standard_arithmetic",
                **standard["category_means"],
                "All": standard["final_spearman"],
            },
            {
                "aggregation": "fisher_z",
                **fisher["category_means"],
                "All": fisher["final_spearman"],
            },
        ])
        summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    if verbose:
        _print_summary("Standard ProteinGym aggregation (arithmetic)", standard)
        _print_summary("Fisher-z aggregation (supplementary)", fisher)

    return {
        "per_assay": per_assay_df,
        "assay_status": assay_status,
        "standard": standard,
        "fisher": fisher,
    }


def _print_summary(title: str, result: dict) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    for cat in FUNCTION_CATEGORIES:
        print(f"  {cat:20s}  {result['category_means'][cat]:+.4f}")
    print(f"  {'All':20s}  {result['final_spearman']:+.4f}")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    # ckpt = "HazemLab/ares-expert-choice-4b-interleaved-150K"
    # ckpt = "HazemLab/ares-softmoe-4b-consecutive-150K"
    ckpt = args.ckpt
    tokenizer = AresProteinTokenizer()
    # Loading the model in bf16 directly is the lightest path on memory.
    # Wrapping each forward in torch.autocast(bf16) on top of an already-bf16
    # MoE model wastes memory because autocast promotes some intermediates
    # back to fp32. Just let the model run natively in its loaded dtype.
    model = Ares.from_pretrained(ckpt, device_map="cuda:0", torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_proteingym_evaluation(
        model=model,
        tokenizer=tokenizer,
        reference_csv="./assets/DMS_substitutions.csv",
        output_dir="./proteingym_results/" + ckpt.split("/")[-1],
        device=device,
        bos_offset=1,   # 0 if your tokenizer does not prepend BOS/CLS
        batch_size=8,
        max_length=None,
        inference_dtype=None,  # model already loaded in bf16; no extra cast needed
        compile_model=True,   # set True for an extra ~1.3-1.8x once warm
        # dms_id_subset=["AICDA_HUMAN_Gajula_2014_3cycles", "B2L11_HUMAN_Dutta_2010_binding-Mcl-1"],  # for sanity runs
    )
    headline = results["standard"]["final_spearman"]
    fisher_v = results["fisher"]["final_spearman"]
    print(f"Standard: {headline:.4f}   Fisher: {fisher_v:.4f}")
