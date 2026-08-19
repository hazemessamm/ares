"""
Steering experiment script for Ares Soft MoE.

Runs a small battery of steering experiments:
  1. Positive: up-weight a strongly specialized expert, measure probability
     shift on its associated amino acid property.
  2. Suppression: alpha < 1 on the same expert, expect shift in opposite
     direction.
  3. Negative control: steer a weakly specialized expert, expect little
     shift.

Outputs a JSON with per-(target, alpha) statistics.

USAGE
-----
    # Dry run (no real model, exercises code paths with random tensors)
    python steering_experiment.py --dry-run

    # Real run
    python steering_experiment.py

ADAPTATION REQUIRED
-------------------
The three callables near the top — `get_last_hidden_states`, `get_lm_logits`,
and `get_masked_lm_loss_head` — need to match how your Ares model exposes its
forward outputs. I've left them as best-guess defaults with clear error
messages; patch them if they don't match your model.
"""
from __future__ import annotations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Local imports — adapt paths if your project structure differs
from steering import (
    SteeringConfig,
    apply_steering,
    property_prob_mass,
    AA_PROPERTIES,
)


# =============================================================================
# PLACEHOLDER: sequences to run on. Replace with a file read / dataset load.
# =============================================================================

SEQUENCE = "MNYPAFSESEILNTSHERVGDPVAGTRSRMLGSGALTGSVIDQPAGDTWGDFYRDHPELQELNEDERLQFKIGVLEDAAQRFEATNIEVADALREEVGRLLLERAEPFAQAIAELRLKLNDSGLRSDEASVMEDKISDLELKYHQMMQYNPYVSGSTPYDREQLRQRAAETSDELDIWR"
# You can also set SEQUENCES = [...] to a list for batched evaluation.
SEQUENCES: Optional[List[str]] = None  # if None, uses [SEQUENCE]


# =============================================================================
# ADAPTERS — patch these to match your model's API
# =============================================================================


def get_last_hidden_states(model, input_ids, attention_mask):
    """
    Return the last hidden states of shape (B, S, D) from your Ares model.

    Common patterns:
        - HuggingFace style: return model(input_ids, attention_mask=attention_mask).last_hidden_state # noqa
        - Tuple return:      return model(input_ids, attention_mask=attention_mask)[0] # noqa
        - Direct tensor:     return model(input_ids, attention_mask=attention_mask) # noqa
    """
    out = model(input_ids, attention_mask=attention_mask)
    if torch.is_tensor(out):
        return out
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state

    if hasattr(out, "hidden_states"):
        return out.hidden_states[0]
    if isinstance(out, (tuple, list)) and torch.is_tensor(out[0]):
        return out[0]
    raise RuntimeError(
        f"Could not extract last hidden states from model output of type {type(out)}. "
        "Patch get_last_hidden_states() to match your model."
    )


def get_lm_logits(model, hidden_states):
    """
    Map last hidden states (B, S, D) to vocab logits (B, S, V) using the
    model's MLM head. Common attribute names: lm_head, mlm_head, cls, head.
    """
    for attr in (
        "lm_head",
        "mlm_head",
        "cls",
        "head",
        "decoder",
        "output",
        "output_layer",
    ):
        head = getattr(model, attr, None)
        if callable(head):
            return head(hidden_states)
    raise RuntimeError(
        "Could not find an LM head on the model. Patch get_lm_logits() to call "
        "your MLM head directly."
    )


# =============================================================================
# Mask construction
# =============================================================================


def random_mask_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_ids: set,
    mask_prob: float = 0.15,
    seed: int = 0,
) -> torch.Tensor:
    """
    Return a (B, S) bool tensor, True at positions that should be treated as
    masked for evaluation. Does NOT actually replace token ids in input_ids;
    we evaluate the model's prediction at these positions either way.
    """
    g = torch.Generator().manual_seed(seed)
    B, S = input_ids.shape
    mask = torch.zeros(B, S, dtype=torch.bool)
    for b in range(B):
        for s in range(S):
            if attention_mask[b, s].item() == 0:
                continue
            if int(input_ids[b, s].item()) in ignore_ids:
                continue
            if torch.rand(1, generator=g).item() < mask_prob:
                mask[b, s] = True
    return mask


def targeted_mask_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    id_to_aa: Dict[int, str],
    target_property: str,
    ignore_ids: set,
) -> torch.Tensor:
    """
    Return a (B, S) bool tensor, True at every position whose token is an
    amino acid in the target property group. Useful for "mask all cysteines,
    see if steering recovers them."
    """
    group_aas = AA_PROPERTIES[target_property]
    B, S = input_ids.shape
    mask = torch.zeros(B, S, dtype=torch.bool)
    for b in range(B):
        for s in range(S):
            if attention_mask[b, s].item() == 0:
                continue
            tid = int(input_ids[b, s].item())
            if tid in ignore_ids:
                continue
            aa = id_to_aa.get(tid)
            if aa in group_aas:
                mask[b, s] = True
    return mask


def apply_mask_to_input(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """Replace masked positions in input_ids with mask_token_id. Returns a copy."""
    out = input_ids.clone()
    out[mask] = mask_token_id
    return out


# =============================================================================
# Core experiment
# =============================================================================


def run_one_intervention(
    model: nn.Module,
    input_ids_masked: torch.Tensor,
    attention_mask: torch.Tensor,
    eval_positions: torch.Tensor,  # (B, S) bool
    id_to_aa: Dict[int, str],
    target_property: str,
    config: SteeringConfig,
    ignore_ids: set,
) -> Dict:
    """
    Run model with and without steering; return mean property mass at
    eval_positions, for baseline and steered runs, plus the delta.
    """
    model.eval()
    with torch.no_grad():
        hidden_base = get_last_hidden_states(
            model, input_ids_masked, attention_mask
        )
        logits_base = get_lm_logits(model, hidden_base)

        with apply_steering(model, config, position_mask=eval_positions):
            hidden_steer = get_last_hidden_states(
                model, input_ids_masked, attention_mask
            )
            logits_steer = get_lm_logits(model, hidden_steer)

    flat_eval = eval_positions.reshape(-1)
    V = logits_base.shape[-1]
    logits_base_at = logits_base.reshape(-1, V)[flat_eval]
    logits_steer_at = logits_steer.reshape(-1, V)[flat_eval]

    # Property mass shift
    mass_base = property_prob_mass(
        logits_base_at,
        id_to_aa,
        target_property,
        ignore_ids,
    )
    mass_steer = property_prob_mass(
        logits_steer_at,
        id_to_aa,
        target_property,
        ignore_ids,
    )

    # Top-1 accuracy for property recovery (did steering push more predictions
    # INTO the target property group?)
    vocab_size = V
    group_aas = AA_PROPERTIES[target_property]
    group_mask_vocab = torch.zeros(
        vocab_size, dtype=torch.bool, device=logits_base.device
    )
    for tid, aa in id_to_aa.items():
        if tid in ignore_ids or tid >= vocab_size:
            continue
        if aa in group_aas:
            group_mask_vocab[tid] = True

    pred_base = logits_base_at.argmax(dim=-1)
    pred_steer = logits_steer_at.argmax(dim=-1)
    in_group_base = group_mask_vocab[pred_base].float().mean().item()
    in_group_steer = group_mask_vocab[pred_steer].float().mean().item()

    return {
        "n_eval_positions": int(eval_positions.sum().item()),
        "baseline_mean_mass": float(mass_base.mean().item()),
        "steered_mean_mass": float(mass_steer.mean().item()),
        "mean_mass_shift": float((mass_steer - mass_base).mean().item()),
        "baseline_top1_in_group": in_group_base,
        "steered_top1_in_group": in_group_steer,
        "top1_in_group_shift": in_group_steer - in_group_base,
    }


def sanity_check_alpha_one(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_name: str,
    expert: int,
) -> None:
    """
    Run the model twice — once with no steering, once with alpha=1 steering
    on a target layer — and verify outputs match bit-for-bit (up to numerical
    noise). Catches integration bugs in the context manager.
    """
    model.eval()
    with torch.no_grad():
        hidden_a = get_last_hidden_states(model, input_ids, attention_mask)

        noop_config = SteeringConfig({layer_name: {expert: 1.0}})
        with apply_steering(model, noop_config):
            hidden_b = get_last_hidden_states(model, input_ids, attention_mask)

    max_abs_diff = (hidden_a - hidden_b).abs().max().item()
    print(f"  Sanity check (alpha=1.0 on {layer_name} expert {expert}):")
    print(f"    max |hidden_a - hidden_b| = {max_abs_diff:.2e}")
    if max_abs_diff > 1e-3:
        print(
            f"    WARNING: alpha=1 should be a no-op but outputs differ by {max_abs_diff:.2e}"
        )
        print(
            "    Most likely cause: _make_steered_forward doesn't exactly mirror the original forward."
        )
    else:
        print("    OK (below 1e-3 tolerance)")


# =============================================================================
# Intervention targets — pulled from property_preferences.json analysis
# =============================================================================

# Format: (label, layer_name, expert_idx, target_property, notes)
INTERVENTION_TARGETS = [
    # Strongest specializations where dispatch and combine agree
    (
        "L17_E9_cysteine",
        "layers.17.ff",
        9,
        "cysteine",
        "combine ratio 9.12, dispatch 5.13",
    ),
    (
        "L18_E20_aromatic",
        "layers.18.ff",
        20,
        "aromatic",
        "combine ratio 4.42, dispatch 4.24",
    ),
    (
        "L18_E26_cysteine",
        "layers.18.ff",
        26,
        "cysteine",
        "combine ratio 4.85, dispatch 4.09",
    ),
    (
        "L16_E9_cysteine",
        "layers.16.ff",
        9,
        "cysteine",
        "combine ratio 15.89 (strongest)",
    ),
    (
        "L16_E9_cysteine",
        "layers.16.ff",
        9,
        "positive",
        "combine ratio 15.89 (strongest)",
    ),
    (
        "L16_E9_cysteine",
        "layers.16.ff",
        9,
        "negative",
        "combine ratio 15.89 (strongest)",
    ),
    # Negative control: a weakly specialized expert (cysteine ratio ~1.0 in both)
    # You should pick this from your actual data — the one I suggest here is a
    # placeholder based on the low-end of your reported numbers.
    (
        "L19_E13_cysteine_control",
        "layers.19.ff",
        13,
        "cysteine",
        "negative control, low ratio",
    ),
]

ALPHA_SWEEP = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with a stub model on random tensors.",
    )
    parser.add_argument(
        "--model-id", default="HazemLab/ares-softmoe-4b-consecutive-150K"
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default="steering_results.json")
    parser.add_argument(
        "--masking",
        choices=["random", "targeted"],
        default="targeted",
        help="'targeted' masks all tokens of the target property; "
        "'random' masks 15%% uniformly.",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=50,
        help="Number of sequences to evaluate on (if using a real dataset).",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Model + tokenizer setup
    # -------------------------------------------------------------------------
    if args.dry_run:
        print("=== DRY RUN MODE (no real model) ===")
        print("Skipping model load; will exit before running real inference.")
        # We can't really run steering without SoftRouter layers present, so
        # dry-run exits after smoke-testing imports.
        _ = SteeringConfig({"layers.17.ff": {9: 5.0}})
        print("Imports OK; SteeringConfig constructible.")
        print("To run the real experiment, drop --dry-run.")
        return

    from ares.models.model import Ares
    from ares.tokenization.protein_tokenizer import AresProteinTokenizer

    print(f"Loading model: {args.model_id}")
    model = Ares.from_pretrained(args.model_id).to(args.device)
    model.eval()
    tokenizer = AresProteinTokenizer()

    id_to_aa = {v: k for k, v in tokenizer.get_vocab().items()}
    ignore_ids = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
    }
    mask_token_id = tokenizer.mask_token_id

    # -------------------------------------------------------------------------
    # Sequence batch — either the single placeholder or the multi-seq list
    # -------------------------------------------------------------------------
    sequences = SEQUENCES if SEQUENCES is not None else [SEQUENCE]
    print(f"Evaluating on {len(sequences)} sequence(s)")

    encoded = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)

    # -------------------------------------------------------------------------
    # Sanity check: alpha=1 should be a no-op
    # -------------------------------------------------------------------------
    print("\n=== Sanity check: alpha=1 should produce identical outputs ===")
    sanity_check_alpha_one(
        model,
        input_ids,
        attention_mask,
        layer_name="layers.17.ff",
        expert=9,
    )

    # -------------------------------------------------------------------------
    # Run interventions
    # -------------------------------------------------------------------------
    results = {
        "masking_mode": args.masking,
        "alpha_sweep": ALPHA_SWEEP,
        "num_sequences": len(sequences),
        "interventions": {},
    }

    print(
        f"\n=== Running {len(INTERVENTION_TARGETS)} interventions "
        f"× {len(ALPHA_SWEEP)} alpha values ==="
    )

    for label, layer, expert, target_prop, notes in INTERVENTION_TARGETS:
        print(
            f"\n[{label}] layer={layer} expert={expert} property={target_prop}"
        )
        print(f"  ({notes})")

        # Build eval positions and masked inputs
        if args.masking == "targeted":
            eval_positions = targeted_mask_positions(
                input_ids.cpu(),
                attention_mask.cpu(),
                id_to_aa,
                target_prop,
                ignore_ids,
            )
        else:
            eval_positions = random_mask_positions(
                input_ids.cpu(),
                attention_mask.cpu(),
                ignore_ids,
                mask_prob=0.20,
                seed=0,
            )
        eval_positions = eval_positions.to(args.device)
        input_ids_masked = apply_mask_to_input(
            input_ids,
            eval_positions,
            mask_token_id,
        )
        n_eval = int(eval_positions.sum().item())
        print(f"  {n_eval} evaluation positions")

        if n_eval == 0:
            print("  SKIP: no evaluation positions for this target property.")
            continue

        per_alpha = {}
        for alpha in ALPHA_SWEEP:
            config = SteeringConfig({layer: {expert: alpha}})
            stats = run_one_intervention(
                model,
                input_ids_masked,
                attention_mask,
                eval_positions,
                id_to_aa,
                target_prop,
                config,
                ignore_ids,
            )
            per_alpha[str(alpha)] = stats
            print(
                f"  alpha={alpha:>5.2f} | "
                f"mass: {stats['baseline_mean_mass']:.4f} -> {stats['steered_mean_mass']:.4f} "
                f"(Δ={stats['mean_mass_shift']:+.4f}) | "
                f"top1-in-group: {stats['baseline_top1_in_group']:.3f} -> "
                f"{stats['steered_top1_in_group']:.3f} "
                f"(Δ={stats['top1_in_group_shift']:+.3f})"
            )

        results["interventions"][label] = {
            "layer": layer,
            "expert": expert,
            "target_property": target_prop,
            "notes": notes,
            "per_alpha": per_alpha,
        }

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
