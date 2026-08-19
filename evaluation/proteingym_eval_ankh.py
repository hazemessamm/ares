"""
ProteinGym zero-shot evaluation for Ankh (T5-style encoder-decoder).

Ankh is a T5 encoder-decoder trained with sentinel-based span corruption:
masked positions in the encoder input are replaced with <extra_id_N> tokens,
and the decoder autoregressively reconstructs them. Masked marginals need
an encoder-decoder variant of the ESM scheme.

Scoring protocol implemented here
---------------------------------
For each variant, score each mutated position INDEPENDENTLY (same
"independent sites" assumption ESM uses on ProteinGym multi-mutant variants):

  For each mutation (wt_aa, pos, mut_aa):
      encoder_input  = wildtype with position `pos` replaced by <extra_id_0>
      decoder_input  = <pad><extra_id_0>
      forward pass -> decoder logits, shape [1, 2, vocab]
      read the logits at decoder position 1 (immediately after <extra_id_0>):
          lp = log_softmax(logits[0, 1, :])
      per-mutation score = lp[mut_aa_id] - lp[wt_aa_id]

  variant score = sum of per-mutation scores

Implementation note
-------------------
The encoder input only depends on the *position* that is masked, not on the
identity of the mutant AA. We therefore deduplicate across mutations by
position: one forward pass per unique mutated position, cache log p over the
20 standard AAs at that position, and every mutation (single or multi) reads
out from the cache. This is mathematically identical to running one forward
pass per mutation; it just avoids the ~19x redundant passes you otherwise
pay on a standard deep mutational scan.

Aggregation
-----------
Uses the same three-stage ProteinGym aggregation as the ESM/Ares pipeline:
DMS -> UniProt -> function category (coarse_selection_type) -> overall,
with both arithmetic and Fisher-z variants. Aggregation is imported from
proteingym_eval.py so both pipelines report comparable numbers.

Usage
-----
    import ankh
    from transformers import T5Tokenizer

    tokenizer = ankh.load_large_tokenizer()
    model, _  = ankh.load_large_model()
    model.eval()

    results = run_ankh_proteingym_evaluation(
        model=model,
        tokenizer=tokenizer,
        reference_csv="/path/to/DMS_substitutions.csv",
        output_dir="./proteingym_results/ankh_large/",
        batch_size=8,
        max_length=1024,
    )
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

# Reuse data loading, aggregation, and mutation parsing from the main pipeline.
# This keeps the two scripts numerically consistent on the aggregation side.
from proteingym_eval import (
    STANDARD_AAS,
    Mutation,
    parse_mutant_string,
    load_proteingym_from_hf,
    load_reference_metadata,
    aggregate_standard,
    aggregate_fisher,
    _print_summary,
)


# ---------------------------------------------------------------------------
# Ankh-specific tokenizer + encoder input helpers
# ---------------------------------------------------------------------------


def _resolve_ankh_token_ids(tokenizer) -> dict:
    """
    Resolve the token IDs Ankh needs: the 20 amino acids, <extra_id_0>, <pad>.
    Fails loudly if anything is missing rather than silently using <unk>.
    """
    ids = {}

    # Amino acids. Ankh's sentencepiece tokenizer treats single characters as
    # single tokens, but the token in the vocab may be prefixed with '▁'
    # (U+2581, sentencepiece's space marker). Try both.
    for aa in STANDARD_AAS:
        token_id = tokenizer.convert_tokens_to_ids(aa)
        if token_id is None or token_id == tokenizer.unk_token_id:
            token_id = tokenizer.convert_tokens_to_ids("\u2581" + aa)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(
                f"Tokenizer does not resolve amino acid {aa!r} to a single token "
                f"(tried both {aa!r} and '\\u2581{aa}'). Vocab size may be wrong."
            )
        ids[aa] = token_id

    sentinel_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    if sentinel_id is None or sentinel_id == tokenizer.unk_token_id:
        raise ValueError("Tokenizer has no <extra_id_0> token.")
    ids["<extra_id_0>"] = sentinel_id

    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer has no pad_token_id (needed as decoder start)."
        )
    ids["<pad>"] = tokenizer.pad_token_id

    return ids


def _tokenize_wildtype(wildtype: str, tokenizer) -> torch.Tensor:
    """
    Tokenize the wildtype exactly once for an assay.

    Ankh's tokenizer takes the raw residue string (no space-joining), emits
    one token per residue, and appends a single `</s>`. No BOS is added. So
    protein position `p` (1-indexed) maps to token index `p - 1`, and the
    final token at index `len(wildtype)` is always `</s>`.

    Returns the 1-D LongTensor of token ids.
    """
    encoded = tokenizer(wildtype, return_tensors="pt")["input_ids"][0]

    expected = len(wildtype) + 1  # residues + </s>
    if encoded.shape[0] != expected:
        raise ValueError(
            f"Unexpected Ankh tokenization: wildtype of length {len(wildtype)} "
            f"produced {encoded.shape[0]} tokens, expected {expected} "
            f"(residues + </s>, no BOS). Check tokenizer configuration."
        )
    return encoded


# ---------------------------------------------------------------------------
# Per-position cached scoring
# ---------------------------------------------------------------------------


def score_variant_batch_ankh(
    wildtype: str,
    mutant_strings: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """
    Compute Ankh masked-marginals scores for all variants on a single wildtype.

    Mutations that share the same masked position share their encoder input
    (and decoder input is position-independent), so we run one forward pass
    per unique position, cache log p over the 20 standard AAs at the sentinel
    output position, and every mutation reads from the cache. This is
    mathematically identical to scoring each mutation independently.

    Returns an array of per-variant scores (higher = fitter under the model).
    """
    model.eval()
    token_ids = _resolve_ankh_token_ids(tokenizer)
    sentinel_id = token_ids["<extra_id_0>"]
    pad_id = token_ids["<pad>"]

    # Parse + validate every variant up front.
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
                    f"Non-standard amino acid in "
                    f"{mut.wt_aa}{mut.position}{mut.mut_aa}"
                )
        parsed.append(mutations)

    # Tokenize the wildtype once. Ankh uses no BOS and a single trailing </s>,
    # so protein position p (1-indexed) maps directly to token index p - 1.
    base_tokens = _tokenize_wildtype(wildtype, tokenizer).to(device)
    seq_len = base_tokens.shape[0]

    # Canonical ordering for the 20 standard AAs, and their token ids.
    aa_list = sorted(STANDARD_AAS)
    aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}
    aa_token_id_list = [token_ids[aa] for aa in aa_list]

    # Collect every unique 1-indexed protein position actually mutated across
    # this assay's variants. Each contributes one forward pass.
    unique_positions: list[int] = []
    pos_to_cache_idx: dict[int, int] = {}
    for mutations in parsed:
        for mut in mutations:
            if mut.position not in pos_to_cache_idx:
                pos_to_cache_idx[mut.position] = len(unique_positions)
                unique_positions.append(mut.position)

    # Validate token indices. Ankh has no BOS, so protein position p (1-indexed)
    # maps to token index p - 1, and must be strictly before the trailing </s>
    # at index seq_len - 1.
    token_indices: list[int] = []
    for p in unique_positions:
        ti = p - 1
        if ti >= seq_len - 1:
            raise ValueError(
                f"Token index {ti} collides with or exceeds the </s> position "
                f"{seq_len - 1}. Check Ankh tokenizer configuration."
            )
        token_indices.append(ti)

    # Cache: one [20]-vector of log p over standard AAs per unique position.
    cache = np.empty((len(unique_positions), len(aa_list)), dtype=np.float64)

    aa_token_ids_t = torch.tensor(aa_token_id_list, device=device, dtype=torch.long)

    with torch.inference_mode():
        for batch_start in range(0, len(unique_positions), batch_size):
            batch_tis = token_indices[batch_start : batch_start + batch_size]
            B = len(batch_tis)

            encoder_input_ids = base_tokens.unsqueeze(0).repeat(B, 1)
            for b, ti in enumerate(batch_tis):
                encoder_input_ids[b, ti] = sentinel_id
            encoder_attention_mask = torch.ones_like(encoder_input_ids)

            # Decoder prompt for every row: <pad><extra_id_0>.
            # We want the logits at the position AFTER <extra_id_0>, which is
            # logits[:, -1, :] when the decoder input has length 2.
            decoder_input_ids = torch.tensor(
                [[pad_id, sentinel_id]] * B, dtype=torch.long, device=device
            )
            decoder_attention_mask = torch.ones_like(decoder_input_ids)

            outputs = model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            logits = outputs["logits"]

            if logits.device != device:
                logits = logits.to(device, non_blocking=True)

            # [B, 2, V] -> [B, V] at the position after <extra_id_0>, upcast
            # for a numerically stable log_softmax, then index the 20 AA ids.
            last_logits = logits[:, -1, :].float()
            lp = torch.log_softmax(last_logits, dim=-1)
            lp_aa = lp.index_select(1, aa_token_ids_t).cpu().numpy()  # [B, 20]

            cache[batch_start : batch_start + B] = lp_aa

    # Score readout against the CPU-side cache (no GPU sync per mutation).
    variant_scores = np.empty(len(parsed), dtype=np.float64)
    for i, mutations in enumerate(parsed):
        total = 0.0
        for mut in mutations:
            row = cache[pos_to_cache_idx[mut.position]]
            total += row[aa_to_idx[mut.mut_aa]] - row[aa_to_idx[mut.wt_aa]]
        variant_scores[i] = total
    return variant_scores


# ---------------------------------------------------------------------------
# Per-assay evaluation
# ---------------------------------------------------------------------------


def evaluate_assay_ankh(
    assay_df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    mutant_col: str = "mutant",
    dms_score_col: str = "DMS_score",
    target_seq_col: str = "target_seq",
    batch_size: int = 8,
) -> tuple[float, int]:
    """
    Score a single DMS assay with Ankh and return (spearman_rho, n_valid).

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
            f"Assay contains {len(unique_wildtypes)} distinct target_seq "
            f"values; expected exactly 1."
        )
    wildtype = unique_wildtypes[0]

    mutant_strings = assay_df[mutant_col].astype(str).tolist()
    dms_scores = assay_df[dms_score_col].to_numpy(dtype=np.float64)

    predictions = score_variant_batch_ankh(
        wildtype=wildtype,
        mutant_strings=mutant_strings,
        model=model,
        tokenizer=tokenizer,
        device=device,
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
# Full-benchmark evaluation
# ---------------------------------------------------------------------------


def evaluate_all_assays_ankh(
    variants_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
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
        print(f"Evaluating {len(grouped)} assays with Ankh...")

    for dms_id, assay_df in grouped:
        wildtype_len = len(assay_df["target_seq"].iloc[0])

        if max_length is not None and wildtype_len > max_length:
            reason = f"length {wildtype_len} > max_length {max_length}"
            fail_records.append({"DMS_id": dms_id, "reason": reason})
            if verbose:
                print(f"[skip] {dms_id}: {reason}")
            continue

        try:
            rho, n_valid = evaluate_assay_ankh(
                assay_df=assay_df,
                model=model,
                tokenizer=tokenizer,
                device=device,
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
# End-to-end runner
# ---------------------------------------------------------------------------


def run_ankh_proteingym_evaluation(
    model,
    tokenizer,
    reference_csv: str,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
    hf_name: str = "OATML-Markslab/ProteinGym_v1",
    hf_config: str = "DMS_substitutions",
    batch_size: int = 8,
    max_length: Optional[int] = 1024,
    dms_id_subset: Optional[list[str]] = None,
    verbose: bool = True,
    inference_dtype: Optional[torch.dtype] = None,
    compile_model: bool = False,
) -> dict:
    """
    Full Ankh pipeline: load data from HF, score every assay with the
    encoder-decoder sentinel protocol (with per-position caching), aggregate
    under both standard and Fisher protocols, save results.

    Returns a dict with:
        per_assay:    DataFrame of per-assay Spearmans
        assay_status: JSON-serializable success/fail assay records
        standard:     standard ProteinGym aggregation
        fisher:       Fisher-averaged aggregation
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if inference_dtype is not None:
        # Run the model natively in this dtype rather than wrapping each
        # forward in torch.autocast. For Ankh / T5 that's equivalent in
        # compute but strictly lighter on peak memory, since autocast keeps
        # fp32 copies of intermediates inside ops it promotes (softmax,
        # layernorm, reductions).
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
        print(
            f"  loaded {len(variants_df)} variants across "
            f"{variants_df['DMS_id'].nunique()} assays"
        )

    if verbose:
        print(f"Loading reference metadata from {reference_csv}")
    reference_df = load_reference_metadata(reference_csv)

    per_assay_df, assay_status = evaluate_all_assays_ankh(
        variants_df=variants_df,
        reference_df=reference_df,
        model=model,
        tokenizer=tokenizer,
        device=device,
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
        _print_summary("Ankh - Standard ProteinGym aggregation (arithmetic)", standard)
        _print_summary("Ankh - Fisher-z aggregation (supplementary)", fisher)

    return {
        "per_assay": per_assay_df,
        "assay_status": assay_status,
        "standard": standard,
        "fisher": fisher,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    ckpt = args.ckpt
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    results = run_ankh_proteingym_evaluation(
        model=model,
        tokenizer=tokenizer,
        reference_csv="./assets/DMS_substitutions.csv",
        output_dir="./proteingym_results/" + ckpt.split("/")[-1],
        device=device,
        batch_size=8,
        max_length=None,
        inference_dtype=None,   # set to torch.bfloat16 if your Ankh handles it
        compile_model=False,    # set True for an extra ~1.3-1.8x once warm
        # dms_id_subset=["AICDA_HUMAN_Gajula_2014_3cycles"],  # for sanity runs
    )
    headline = results["standard"]["final_spearman"]
    fisher_v = results["fisher"]["final_spearman"]
    print(f"Ankh Standard: {headline:.4f}   Fisher: {fisher_v:.4f}")
