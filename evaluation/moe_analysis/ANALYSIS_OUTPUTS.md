# MoE Analysis Outputs

This document explains every JSON / pickle artifact produced by
`soft_moe_analysis.py` and `expert_choice_analysis.py`, what each field
means, and how to interpret the numbers when reading them in a paper or a
plot. It also covers what the analyses **don't** measure and the standard
caveats reviewers will look for.

The two analysis pipelines produce structurally similar outputs but with
**different "weight types"** — read the [Weight types](#weight-types)
section before interpreting any specialization number.

---

## Quick reference

| File | Soft MoE | Expert Choice | Description |
|---|---|---|---|
| `amino_acid_preferences.json` | ✓ | ✓ | Per `(layer, expert, AA)` summary statistics |
| `property_preferences.json` | ✓ | ✓ | Per `(layer, expert, property_group)` statistics |
| `positional_preferences.json` | ✓ | ✓ | Per `(layer, expert, position_bin)` statistics |
| `dispatch_heatmaps.pkl` | ✓ | – | Per-sequence routing matrices for visualization |
| `routing_heatmaps.pkl` | – | ✓ | Per-sequence routing matrices for visualization |
| `token_coverage.json` | – | ✓ | Drop rate / over-subscription stats |
| `expert_co_occurrence.json` | – | ✓ | Pairwise expert co-selection matrix |
| `expert_knockout.json` | optional | optional | Causal importance per `(layer, expert)` |

---

## Weight types

The key thing to understand before interpreting any number is **which
distribution** is being measured. The same metric (mean / baseline /
ratio) on each weight type means a different thing:

### Soft MoE

- **`dispatch`** — softmax over **sequence positions**, per
  `(expert, slot)`. After summing over slots, `dispatch[b, s, e]` is the
  *cumulative attention mass* that expert `e`'s slots place on token
  `(b, s)`. **Does not** sum to 1 over experts or tokens. Reflects
  **where experts look** — attention allocation.
- **`combine`** — softmax over `(experts, slots)`, per **token**.
  After summing over slots, `combine[b, s, e]` is the *fraction of token
  `(b, s)`'s output* coming from expert `e`. **Does** sum to 1 over
  experts per token. Reflects **functional contribution** to the layer
  output.

### Expert Choice

- **`selection`** — binary 0/1 indicator. `selection[b, s, e] = 1` iff
  expert `e` picked token `(b, s)` into one of its top-k slots. Treats
  every selection equally; ignores router confidence.
- **`weighted`** — `weighted[b, s, e] = router_softmax_prob` if selected,
  else 0. Couples selection rate with router confidence; most directly
  proportional to functional contribution. Analogous to SoftMoE
  `combine`.

> **The dispatch / combine and selection / weighted ratios are not
> directly comparable.** Always state which weight type you're plotting.
> When comparing soft MoE to expert choice, the closest pair is
> `combine` ↔ `weighted` (functional contribution).

---

## Common per-class entry schema

Every leaf in `amino_acid_preferences.json`, `property_preferences.json`,
and `positional_preferences.json` has the same six fields:

```json
{
  "mean_weight": 0.184,
  "baseline": 0.173,
  "specialization_ratio": 1.064,
  "log_ratio": 0.062,
  "count": 25194,
  "low_support": false
}
```

| Field | Meaning |
|---|---|
| `mean_weight` | Mean weight (of the chosen weight type) given to tokens **of this class** by **this expert**. |
| `baseline` | Mean weight that **this expert** gives to **any valid token**. Per-expert, **not** per-class. |
| `specialization_ratio` | `mean_weight / baseline`. Equal to 1 means the expert treats this class no differently than its average token. > 1 means preference; < 1 means avoidance. |
| `log_ratio` | `log(mean_weight) - log(baseline)`. Symmetric around 0 (above-baseline and below-baseline are equally visible). `null` when either side is non-positive. **Prefer this in plots.** |
| `count` | Number of valid tokens of this class observed across the dataset. Drives statistical significance. |
| `low_support` | `true` iff `count < min_count` (configurable via `compute(min_count=...)`, default 50). Use this to filter unreliable entries in plots. |

### Worked example

> Layer `layers.10.ff`, expert 0, weight type `dispatch`, group `cysteine`:
>
> ```json
> "mean_weight": 0.183, "baseline": 0.173, "specialization_ratio": 1.059,
> "log_ratio": 0.057, "count": 25194, "low_support": false
> ```
>
> **Reading**: at layer 10, expert 0 places about **6 % more dispatch
> mass on cysteine residues than on its average token**. The effect is
> small in absolute terms (`log_ratio ≈ 0.06` ≈ 6 %), and supported by
> 25 k cysteine observations, so it is real but modest. To check whether
> this is genuine specialization rather than incidental, cross-reference
> with the per-AA result for `C` (it should also be elevated) and
> ideally with the knockout analysis (does removing this expert hurt
> reconstruction of cysteine residues?).

---

## File-by-file

### `amino_acid_preferences.json`

```
{
  "_metadata": {
    "weight_descriptions": {dispatch / combine descriptions},
    "min_count": 50
  },
  "<weight_type>": {
    "<layer_name>": {
      "<expert_idx>": {
        "<AA>": { mean_weight, baseline, specialization_ratio,
                  log_ratio, count, low_support }
      }
    }
  }
}
```

- `<weight_type>` is `"dispatch"` or `"combine"` (soft MoE) /
  `"selection"` or `"weighted"` (expert choice).
- `<AA>` is a single-letter amino-acid code. The output may include
  ambiguity codes (`B`, `Z`, `J`, `X`) and special tokens
  (`<mask>`, `<unk>`) depending on tokenizer; **filter these in plots**.
  Use the canonical 20: `ACDEFGHIKLMNPQRSTVWY`.
- `_metadata.min_count` is the threshold the analyzer used for
  `low_support`.

**Use this file for**: identifying the dominant residue(s) each expert
prefers / avoids. Cross-check property-level claims here.

---

### `property_preferences.json`

Same shape as `amino_acid_preferences.json`, but the inner key is a
**property group** (`hydrophobic`, `polar_uncharged`, `positive`,
`negative`, `charged`, `aromatic`, `small`, `cysteine`, `proline`).

Property groups **intentionally overlap** — an `F` residue counts
toward both `hydrophobic` and `aromatic`. The analyzer handles overlap
correctly (each token contributes additively to every group it belongs
to), but **interpretation requires care**: a high signal on two
correlated groups (e.g. hydrophobic + aromatic) is often driven by a
shared subset (F, W). Always cross-check against
`amino_acid_preferences.json`.

The `_metadata` block lists the exact group definitions used:

```json
"_metadata": {
  "property_groups": {
    "hydrophobic": ["A", "I", "L", "M", "F", "W", "V"],
    "polar_uncharged": ["S", "T", "N", "Q", "Y"],
    ...
  },
  "overlap_note": "Property groups intentionally overlap..."
}
```

**Use this file for**: high-level interpretable summary in the paper
("layer 10 expert 0 is moderately cysteine-preferring"). Always pair
with the per-AA breakdown for at least the most-specialized experts.

---

### `positional_preferences.json`

Same shape, with bin labels like `"0-20%"`, `"20-40%"`, ..., `"80-100%"`
of the **valid sequence length**. Bins are computed per-sequence after
masking padding, so sequences of any length contribute meaningfully.

**Use this file for**: detecting experts that specialize in N-terminal
or C-terminal regions, signal peptides, etc. A diagonal pattern (each
bin highest in a different expert) suggests positional partitioning;
flat ratios suggest position-agnostic experts.

---

### `dispatch_heatmaps.pkl` (Soft MoE) / `routing_heatmaps.pkl` (Expert Choice)

Pickled dict, structured as:

```
{
  "<weight_type>": {
    "<layer_name>": [
      { "weights": np.ndarray (seq_len, num_experts),
        "sequence": ["M", "K", "T", ...],
        "length":   seq_len },
      ...  # up to max_sequences entries
    ]
  }
}
```

`weights[s, e]` is the value (after slot-summation, or selection /
weighted indicator for EC) for token `s` and expert `e`. These are raw
per-sequence matrices — not aggregated — for qualitative inspection
and figure generation.

**Use this for**: example heatmaps in the paper showing how routing
varies along a single sequence. Pick 1–3 representative sequences with
visually distinct routing patterns.

---

### `token_coverage.json` (Expert Choice only)

```
{
  "<layer_name>": {
    "drop_rate":              float,   // fraction of valid tokens picked by 0 experts
    "mean_experts_per_token": float,   // average # experts that picked a valid token
    "coverage_histogram":     {0: int, 1: int, ..., num_experts: int},
    "coverage_fractions":     {0: float, ..., num_experts: float}
  }
}
```

| Field | Meaning |
|---|---|
| `drop_rate` | Fraction of valid (non-padding) tokens that **no** expert selected. Higher means the layer is bottlenecked at top-k capacity. |
| `mean_experts_per_token` | Total selections ÷ total valid tokens = average over-subscription per token. With capacity factor `c` and `E` experts, the algebraic max is `c` (ideal balanced load). |
| `coverage_histogram` | Raw counts of "how many experts picked this token", over `{0, 1, …, E}`. Sums to total valid tokens. |
| `coverage_fractions` | `coverage_histogram` normalized by total valid tokens. |

**Use this for**: routing-behaviour diagnostics. A high `drop_rate`
indicates capacity is too tight (or the router is collapsing onto a
subset of tokens). A high tail in the histogram (lots of tokens picked
by many experts) indicates redundant routing.

This is **routing analysis**, not specialization analysis — keep these
in the appendix unless your paper is specifically about routing.

---

### `expert_co_occurrence.json` (Expert Choice only)

```
{
  "<layer_name>": {
    "co_occurrence_counts":   E×E float matrix (as nested list),
    "selections_per_expert":  list of length E,  // = diagonal of co_occurrence_counts
    "jaccard":                E×E float matrix,
    "total_valid_tokens":     int
  }
}
```

| Field | Meaning |
|---|---|
| `co_occurrence_counts[i, j]` | Number of valid tokens picked by both expert `i` and expert `j`. The diagonal is the per-expert selection count. |
| `selections_per_expert[i]` | Number of valid tokens picked by expert `i`. Equals top-k × (number of forward passes) when expert `i` is never starved by padding. |
| `jaccard[i, j]` | `co_occurrence[i,j] / (selections[i] + selections[j] - co_occurrence[i,j])`. Symmetric, in `[0, 1]`. 0 = experts pick disjoint tokens, 1 = experts always pick the same tokens. |
| `total_valid_tokens` | Total valid tokens observed across the dataset for this layer. |

**Use this for**: diagnosing redundancy / specialization between
experts. High off-diagonal Jaccard = redundancy (two experts selecting
similar tokens, possibly doing similar work). Low off-diagonal Jaccard
= clean partitioning of token space.

---

### `expert_knockout.json` (optional; only if you ran `ExpertKnockoutAnalyzer`)

```
{
  "<layer_name>": {
    "baseline_loss":              float,
    "delta_per_expert":           {expert_idx: float, ...},
    "relative_delta_per_expert":  {expert_idx: float, ...}
  }
}
```

| Field | Meaning |
|---|---|
| `baseline_loss` | `forward_fn(model, batch).item()` averaged over the dataset, with no expert ablated. |
| `delta_per_expert[e]` | `mean_loss_with_expert_e_removed − baseline_loss`. **Positive = removing expert `e` hurts.** Magnitude is in raw loss units. |
| `relative_delta_per_expert[e]` | `delta_per_expert[e] / baseline_loss`. Convenient unit-free measure: 0.05 means "removing expert `e` increases loss by 5 %". |

The ablation **renormalizes** the remaining experts' combine weights so
each token still has a properly normalized distribution — this is the
correct counterfactual for "what if expert `e` didn't exist?" rather
than "what if expert `e`'s weights were corrupted?".

**Use this for**: causal evidence of specialization. A correlational
specialization signal (high `log_ratio` on hydrophobic) becomes a real
claim only when paired with a non-trivial `relative_delta` — i.e.,
removing the expert measurably hurts the model's behavior.

---

## Cross-cutting caveats

1. **Always read the weight type.** The same number means different
   things on `dispatch` vs `combine` (or `selection` vs `weighted`).

2. **Filter `low_support` in plots.** AAs with fewer than the
   `min_count` tokens are statistically unreliable. The flag is
   already in the JSON; just skip those entries.

3. **Property groups overlap.** A high signal on two correlated groups
   (hydrophobic + aromatic) is often a single-residue signal in
   disguise. Cross-check the per-AA file before claiming a "property"
   specialization.

4. **Specialization ≠ importance.** A high `log_ratio` shows the expert
   *prefers* a class, not that the model needs the expert for that
   class. Pair with `expert_knockout.json` for causal claims.

5. **The baseline is per-expert.** `baseline = mean weight expert e
   gives to any valid token`. It is **not** the average across all
   experts. Comparisons across experts at the same `(layer, class)`
   should use `mean_weight`, not `specialization_ratio` (which is
   already normalized away).
