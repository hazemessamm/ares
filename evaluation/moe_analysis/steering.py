"""
Combine-weight steering for Soft MoE.

Temporarily replaces SoftRouter.forward with a wrapped version that multiplies
selected experts' combine weights by a scalar factor, then renormalizes per
token so combine weights still sum to 1 over (experts, slots).

Usage:
    steering = SteeringConfig({
        "layers.17.ff": {9: 5.0},           # alpha=5 for L17 expert 9
        "layers.18.ff": {20: 3.0, 26: 2.0}, # multiple experts per layer
    })
    with apply_steering(model, steering):
        outputs = model(input_ids, attention_mask=attention_mask)

Alpha > 1 emphasizes the target expert; 0 < alpha < 1 suppresses it.
alpha = 1 is a no-op (useful for sanity checks).
alpha = 0 effectively knocks the expert out of the combine step (but its
compute still runs).

Optionally restrict steering to specific positions (e.g. masked positions only):

    with apply_steering(model, steering, position_mask=masked_positions):
        ...

where position_mask has shape (batch, seq_len) with True at positions where
steering should be applied. Other positions keep their original combine weights.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from ares.models.soft_router import SoftRouter, softmax, prepare_padding_mask
from ares.models.utils import jitter_noise


@dataclass
class SteeringConfig:
    """
    Per-layer expert scaling factors.

    layer_expert_alphas[layer_name] is a dict {expert_index: alpha}.
    Any expert not listed for a layer gets alpha=1 (unchanged).
    Any layer not listed is not steered at all.
    """
    layer_expert_alphas: Dict[str, Dict[int, float]] = field(default_factory=dict)

    def scales_for_layer(self, layer_name: str, num_experts: int) -> Optional[torch.Tensor]:
        """
        Returns a (num_experts,) float tensor of per-expert scales, or None
        if this layer is not being steered.
        """
        alphas = self.layer_expert_alphas.get(layer_name)
        if not alphas:
            return None
        scales = torch.ones(num_experts, dtype=torch.float32)
        for e, a in alphas.items():
            if not (0 <= e < num_experts):
                raise ValueError(
                    f"Expert index {e} out of range [0, {num_experts}) for layer {layer_name}"
                )
            scales[e] = float(a)
        return scales


def _make_steered_forward(
    original_forward,
    module: SoftRouter,
    scales: torch.Tensor,
    position_mask: Optional[torch.Tensor],
):
    """
    Build a replacement forward() for this SoftRouter that applies combine
    steering. scales is (num_experts,) CPU tensor; we move to the module's
    device and dtype on first use.

    position_mask is (batch, seq_len) bool tensor or None. If provided, steering
    is applied only at positions where it is True; other positions receive the
    original (un-steered) combine weights.
    """
    def steered_forward(x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        # Recompute what the original forward does, up through combine_weights,
        # then intervene, then finish.
        if module.normalize:
            x_in = nn.functional.normalize(x, dim=-1)
            phi = nn.functional.normalize(module.phi, dim=0) * module.scaler
        else:
            x_in = x
            phi = module.phi

        logits = torch.einsum("bmd,dnp->bmnp", x_in, phi)

        # Match original: training-only jitter. (Steering is typically at eval
        # time; this branch usually won't fire.)
        if module.training:
            logits = jitter_noise(logits, module.noise_scale)

        if padding_mask is not None:
            mask = prepare_padding_mask(padding_mask)
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)

        dispatch_weights = nn.functional.softmax(logits, dim=1, dtype=torch.float32)
        combine_weights = softmax(logits, dim=(2, 3))

        # ---- Steering intervention on combine_weights ----
        B, S, E, P = combine_weights.shape
        s = scales.to(device=combine_weights.device, dtype=combine_weights.dtype)
        s_expanded = s.view(1, 1, E, 1)  # broadcast over batch, seq, slots

        scaled = combine_weights * s_expanded
        denom = scaled.sum(dim=(2, 3), keepdim=True).clamp_min(torch.finfo(scaled.dtype).tiny)
        steered = scaled / denom

        if position_mask is not None:
            # position_mask: (B, S) bool. Keep original where False, steered where True.
            pm = position_mask.to(device=combine_weights.device)
            if pm.dtype != torch.bool:
                pm = pm.bool()
            if pm.shape != (B, S):
                raise ValueError(
                    f"position_mask shape {tuple(pm.shape)} does not match (batch, seq_len) "
                    f"= ({B}, {S})"
                )
            pm_expanded = pm.view(B, S, 1, 1)
            combine_weights = torch.where(pm_expanded, steered, combine_weights)
        else:
            combine_weights = steered
        # ---- End intervention ----

        # Expose for any registered analyzer hooks (match original attribute
        # semantics, though in steering we typically don't re-analyze).
        module.dispatch_weights_ = dispatch_weights
        module.combine_weights_ = combine_weights

        Xs = torch.einsum("bmd,bmnp->bnpd", x_in, dispatch_weights)
        expert_outputs = module.experts(Xs)
        combined_outputs = torch.einsum(
            "bnpd,bmnp->bmd",
            expert_outputs,
            combine_weights,
        )
        return combined_outputs.to(x.dtype)

    return steered_forward


@contextlib.contextmanager
def apply_steering(
    model: nn.Module,
    config: SteeringConfig,
    position_mask: Optional[torch.Tensor] = None,
):
    """
    Context manager that temporarily replaces SoftRouter.forward on each
    targeted layer with a steered version. On exit, originals are restored
    even if an exception is raised inside the block.
    """
    saved = []  # (module, original_bound_forward)

    try:
        for name, module in model.named_modules():
            if not isinstance(module, SoftRouter):
                continue
            scales = config.scales_for_layer(name, module.num_experts)
            if scales is None:
                continue

            # Bind the replacement forward to this specific module.
            original_forward = module.forward
            steered = _make_steered_forward(
                original_forward, module, scales, position_mask,
            )
            saved.append((module, original_forward))
            module.forward = steered

        if not saved:
            import warnings
            warnings.warn(
                "apply_steering: no SoftRouter layers matched the config. "
                f"Config layers: {list(config.layer_expert_alphas.keys())}. "
                "Check layer names via [n for n, m in model.named_modules() if isinstance(m, SoftRouter)].",
                RuntimeWarning,
                stacklevel=2,
            )

        yield

    finally:
        for module, original_forward in saved:
            module.forward = original_forward


# ----------------------------------------------------------------------------
# Evaluation helper: measure whether steering shifted MLM predictions in the
# expected direction.
# ----------------------------------------------------------------------------

AA_PROPERTIES = {
    "hydrophobic": set("AILMFWVP"),
    "polar": set("STNQ"),
    "positive": set("RHK"),
    "negative": set("DE"),
    "aromatic": set("FWY"),
    "small": set("GAS"),
    "cysteine": set("C"),
}


def property_prob_mass(
    logits_at_masked: torch.Tensor,  # (N_masked, vocab_size)
    id_to_aa: Dict[int, str],
    property_group: str,
    ignore_ids: Optional[set] = None,
) -> torch.Tensor:
    """
    Given MLM logits at masked positions, return the total probability mass
    assigned to amino acids in the given property group, per position.
    Returns a (N_masked,) tensor.
    """
    ignore_ids = ignore_ids or set()
    group_aas = AA_PROPERTIES[property_group]
    vocab_size = logits_at_masked.shape[-1]

    group_mask = torch.zeros(vocab_size, dtype=torch.bool, device=logits_at_masked.device)
    for tid, aa in id_to_aa.items():
        if tid in ignore_ids:
            continue
        if tid >= vocab_size:
            continue
        if aa in group_aas:
            group_mask[tid] = True

    probs = torch.softmax(logits_at_masked.float(), dim=-1)
    return probs[:, group_mask].sum(dim=-1)


def evaluate_steering_shift(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    masked_positions: torch.Tensor,  # (B, S) bool; True at masked positions
    lm_head,  # callable: last_hidden_states -> logits
    id_to_aa: Dict[int, str],
    target_property: str,
    steering_config: SteeringConfig,
    ignore_ids: Optional[set] = None,
    get_hidden_states=None,
):
    """
    Run the model once without steering and once with steering, measure the
    per-masked-position probability mass on the target property before and
    after, and return the shift.

    Args:
        model: the full Ares model.
        input_ids, attention_mask: batch inputs.
        masked_positions: (B, S) bool mask of positions to evaluate at.
        lm_head: callable mapping last hidden states to logits over the vocab.
        id_to_aa: token_id -> amino acid letter mapping.
        target_property: property group name whose predicted mass we track.
        steering_config: SteeringConfig describing the intervention.
        ignore_ids: token ids to exclude from property mass (pad, cls, eos).
        get_hidden_states: callable(model, input_ids, attention_mask) -> last hidden
            states of shape (B, S, D). If None, assumes model(...) returns them.

    Returns:
        dict with keys:
            baseline_mass: (N_masked,) tensor of pre-steering property mass
            steered_mass:  (N_masked,) tensor of post-steering property mass
            shift:         (N_masked,) tensor of (steered - baseline)
            mean_shift:    scalar mean shift
            baseline_mean: scalar mean baseline mass
            steered_mean:  scalar mean steered mass
    """
    model.eval()

    if get_hidden_states is None:
        def get_hidden_states(m, ids, mask):
            out = m(ids, attention_mask=mask)
            # Assume output is last hidden states; adapt if your model returns a tuple/dict
            return out

    # Indices of masked positions in the flattened (B*S,) axis.
    with torch.no_grad():
        # Baseline forward
        hidden_base = get_hidden_states(model, input_ids, attention_mask)
        logits_base = lm_head(hidden_base)  # (B, S, V)

        # Steered forward
        with apply_steering(model, steering_config, position_mask=masked_positions):
            hidden_steer = get_hidden_states(model, input_ids, attention_mask)
            logits_steer = lm_head(hidden_steer)  # (B, S, V)

    flat_mask = masked_positions.reshape(-1)
    logits_base_m = logits_base.reshape(-1, logits_base.shape[-1])[flat_mask]
    logits_steer_m = logits_steer.reshape(-1, logits_steer.shape[-1])[flat_mask]

    baseline_mass = property_prob_mass(logits_base_m, id_to_aa, target_property, ignore_ids)
    steered_mass = property_prob_mass(logits_steer_m, id_to_aa, target_property, ignore_ids)
    shift = steered_mass - baseline_mass

    return {
        "baseline_mass": baseline_mass.detach().cpu(),
        "steered_mass": steered_mass.detach().cpu(),
        "shift": shift.detach().cpu(),
        "mean_shift": float(shift.mean().item()),
        "baseline_mean": float(baseline_mass.mean().item()),
        "steered_mean": float(steered_mass.mean().item()),
    }