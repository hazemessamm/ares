# Ares

**Sparse Mixture-of-Experts protein language models, trained on TPU.**

[![Paper](https://img.shields.io/badge/OpenReview-gq0R7xiPjg-8c1b13)](https://openreview.net/forum?id=gq0R7xiPjg)
[![Venue](https://img.shields.io/badge/GenBio%20%40%20ICML%202026-Spotlight-f5a623)](https://openreview.net/forum?id=gq0R7xiPjg)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HazemLab-ffcc4d)](https://huggingface.co/HazemLab)

> Spotlight at the **GenBio Workshop, ICML 2026**. [OpenReview](https://openreview.net/forum?id=gq0R7xiPjg)

Ares is a masked protein language model that replaces the dense feed-forward stack with
sparse Mixture-of-Experts layers. The ~4B-parameter models here are pretrained on UniRef50
and evaluated on ProteinGym plus seven downstream property-prediction benchmarks.

The repository implements **two routing strategies** and the tooling to compare them,
including the analysis pipeline used to characterize where each one succeeds and fails.

---

## What's here

- A complete MoE protein LM: encoder, both routers, tokenizer, optimizers, packing pipeline.
- TPU/XLA SPMD training with sequence packing, block-diagonal masking, and resumable
  checkpointing to cloud storage.
- Downstream evaluation across ProteinGym and seven benchmark tasks.
- **MoE interpretability tooling**: expert specialization, causal knockout, routing
  heatmaps, and steering experiments.

## Architecture

| Component | Choice |
|---|---|
| Position encoding | Rotary (RoPE) |
| Normalization | RMSNorm, pre-norm |
| Attention | Grouped-query (16 heads / 8 KV heads) |
| Feed-forward | Gated SiLU (SwiGLU) |
| MoE placement | Dense for the first *N* layers, MoE after (consecutive or interleaved) |
| Objective | MLM with scheduled masking + mutation noising |

**Routing strategies**

- **Soft routing** (`moe_type: soft_router`): each expert holds a fixed set of learned
  slots; every token contributes to every slot through a softmax dispatch, and outputs are
  recombined per token. No token dropping, no load-balancing loss. Optional L2
  normalization on the routing projection (`moe_normalize`).
- **Expert choice** (`moe_type: expert_choice`): each expert selects its top-*k* tokens up
  to a capacity factor. Load is balanced by construction, but tokens can be dropped or
  over-subscribed.

> **Note:** the current expert-choice implementation is not fully faithful to the original
> formulation: its routing softmax normalizes over tokens rather than over experts. This
> will be addressed in a future revision.

**Sequence packing**: multiple sequences share a batch row with block-diagonal attention
masking and per-sequence position IDs, so no compute is spent on padding. Correctness
against unpacked inference is asserted in
[`tests/test_packing_correctness.py`](tests/test_packing_correctness.py).

## Installation

```bash
git clone https://github.com/hazemessamm/ares.git
cd ares

pip install -e .                # library only
pip install -e ".[training]"    # + hydra, wandb, gcsfs, fsspec
pip install -e ".[evaluation]"  # + pandas, scipy, scikit-learn, matplotlib
pip install -e ".[full]"        # everything, including torch-xla
```

## Quickstart

```python
from ares.models import Ares, AresConfig
from ares.tokenization import AresProteinTokenizer

tokenizer = AresProteinTokenizer()
config = AresConfig(
    embed_dim=1024,
    num_layers=20,
    moe_type="soft_router",   # or "expert_choice"
    moe_after_num_layers=10,  # dense below this depth, MoE above
    num_experts=32,
    moe_num_slots=64,
)
model = Ares(config)

batch = tokenizer(["MKTAYIAKQRQISFVKSHFSRQ"], return_tensors="pt")
outputs = model(**batch)
print(outputs.logits.shape)
```

### Pretrained checkpoints

All five checkpoints are on the Hub at
[**huggingface.co/HazemLab**](https://huggingface.co/HazemLab):

| Checkpoint | Routing | MoE placement | Steps | ProteinGym |
|---|---|---|---|---|
| [`ares-softmoe-4b-consecutive-150K`](https://huggingface.co/HazemLab/ares-softmoe-4b-consecutive-150K) | Soft | Consecutive | 150,000 | **0.341** |
| [`ares-softmoe-4b-l2-consecutive-225K`](https://huggingface.co/HazemLab/ares-softmoe-4b-l2-consecutive-225K) | Soft + L2 | Consecutive | 225,000 | **0.341** |
| [`ares-softmoe-4b-l2-consecutive-150K`](https://huggingface.co/HazemLab/ares-softmoe-4b-l2-consecutive-150K) | Soft + L2 | Consecutive | 150,000 | **0.319** |
| [`ares-expert-choice-4b-interleaved-150K`](https://huggingface.co/HazemLab/ares-expert-choice-4b-interleaved-150K) | Expert choice | Interleaved | 150,000 | **0.126** |
| [`ares-ec-moe-4b-86k`](https://huggingface.co/HazemLab/ares-ec-moe-4b-86k) | Expert choice | Consecutive | 86,000 | not evaluated |

ProteinGym is the Fisher-*z* aggregated Spearman over 217 DMS substitution assays; per-assay
breakdowns for every row are in
[`evaluation/proteingym_results/`](evaluation/proteingym_results/). Start with
`ares-softmoe-4b-consecutive-150K` unless you have a reason not to: it and the 225K L2 run
are effectively tied at the top, and it is the simpler of the two.

### Loading a pretrained model from the Hub

Load checkpoints with `Ares.from_pretrained` and build the tokenizer directly.
No `trust_remote_code` is needed:

```python
import torch
from ares import Ares, AresProteinTokenizer

model = Ares.from_pretrained(
    "HazemLab/ares-softmoe-4b-consecutive-150K",
    dtype=torch.bfloat16,
).eval()
tokenizer = AresProteinTokenizer()

batch = tokenizer(["MKTAYIAKQRQISFVKSHFSRQ"], return_tensors="pt")
with torch.no_grad():
    outputs = model(**batch)

print(outputs.logits.shape)          # (batch, length, vocab)
print(outputs.hidden_states[0].shape)  # (batch, length, embed_dim)
```

`AresProteinTokenizer` builds its vocabulary in code, so it needs no download
and is identical across every checkpoint. Weights are stored in float32
(~17 GB); pass `dtype=torch.bfloat16` unless you specifically need float32.

The `ares` package must be installed. Checkpoints carry an `auto_map`, but the
bundled modules import from `ares`, so `trust_remote_code=True` is not a
substitute for installing it.

Only the core dependencies are needed for inference; the training and data
extras are not imported on the model path.

## Training

Two Hydra-driven entrypoints, one per routing strategy:

```bash
python train_soft_moe.py                                        # config/config_soft_moe_v6e.yaml
python train_soft_moe.py --config-name=config_soft_moe_v6e_l2   # L2-normalized routing
python train_expert_choice.py                                   # config/config_expert_choice_moe.yaml
```

Any config value can be overridden inline:

```bash
python train_soft_moe.py model.num_experts=16 training.per_device_train_batch_size=8
```

Before your first run, set `training.checkpoint_dir`. The committed configs use a
`gs://your-bucket/...` placeholder. Training targets TPU via PyTorch/XLA with SPMD
sharding; `training.distributed: true` enables multi-host checkpointing.

### Config variants

| Config | Variant |
|---|---|
| `config_soft_moe_v6e.yaml` | Baseline soft-MoE, 20 layers, 32 experts |
| `config_soft_moe_v6e_l2.yaml` | L2-normalized routing (`moe_normalize: true`) |
| `config_soft_moe_v6e_interleaved.yaml` | MoE layers interleaved with dense |
| `config_soft_moe_v6e_v2.yaml` | Scaled to 30 layers |
| `config_expert_choice_moe.yaml` | Expert-choice routing |

## Evaluation

Downstream benchmarks, configured through
[`evaluation/configs/config.yaml`](evaluation/configs/config.yaml):

```bash
python evaluation/proteingym_eval.py    # zero-shot DMS substitutions
python evaluation/gb1.py                # epistasis / fitness
python evaluation/flouroscence.py       # fluorescence regression
python evaluation/stability.py          # stability regression
python evaluation/remote_homology.py    # fold-level classification
python evaluation/ssp3.py               # secondary structure (3-state)
python evaluation/ssp8.py               # secondary structure (8-state)
python evaluation/localization.py       # subcellular localization
```

`*_embed.py` variants run the frozen-embedding version of a task.

## MoE analysis

[`evaluation/moe_analysis/`](evaluation/moe_analysis/) contains the interpretability
pipeline. Read [`ANALYSIS_OUTPUTS.md`](evaluation/moe_analysis/ANALYSIS_OUTPUTS.md) first;
it documents every artifact, the metric definitions, and the caveats.

| Script | Purpose |
|---|---|
| `soft_moe_analysis.py` / `expert_choice_analysis.py` | Expert specialization by amino acid, biochemical property, position |
| `l7_knockout.py` | Causal expert-ablation importance |
| `layer_anomaly_diagnostics.py` | Per-layer routing anomaly detection |
| `compare_l2_vs_no_l2.py` | L2-normalized vs unnormalized routing |
| `steering.py` / `steering_experiment.py` | Expert-steering interventions |
| `visualize_*.py` | Routing heatmaps and specialization plots |

> **Interpreting specialization:** soft-MoE `dispatch` weights measure *where experts look*;
> `combine` weights measure *functional contribution* to the output. They normalize over
> different axes and are not directly comparable. `ANALYSIS_OUTPUTS.md` explains why.

## Repository layout

```
ares/
  models/         encoder, attention, soft + expert-choice routers, RoPE, config
  optimizers/     AdamW, Adafactor, SOAP
  pipelines/      dataset, sequence packing, checkpointing, XLA sharding, logging
  preprocessing/  noising, truncation, masking schedules
  tokenization/   protein tokenizer and vocabulary
  eval/           downstream heads, pooling, metrics
config/           Hydra training configs
evaluation/       benchmarks, ProteinGym results, MoE analysis
tests/            unit and packing-correctness tests
```

## Tests

```bash
pip install -e ".[dev,data]"
pytest tests/
```

All 253 tests pass. The suite covers the model and encoder layers, both
routers, rotary embeddings, masking schedules, sequence packing, and the
`NaNObserver` debugging hook.

## Contributing

Issues and pull requests are welcome.

The model code in [`ares/models/`](ares/models/) is device-agnostic — only the training
pipeline is TPU-specific. **A GPU training path is the single most useful thing someone
could add.** The XLA coupling is confined to
[`ares/pipelines/checkpoint.py`](ares/pipelines/checkpoint.py),
[`xla_sharding.py`](ares/pipelines/xla_sharding.py),
[`metrics.py`](ares/pipelines/metrics.py),
[`utils.py`](ares/pipelines/utils.py), and the two `train_*.py` entrypoints; a CUDA/FSDP
path would slot in beside the `xla` extra in `pyproject.toml` without touching the model.
`ares.pipelines` already resolves its imports lazily, so a `torch-xla`-free install works
today. Please open an issue first so we can agree on the interface.

Other things that would help: additional downstream evaluations, a faithful expert-choice
router (see the note under [Architecture](#architecture)), and fixes to the MoE analysis
tooling.

Keep `pytest tests/` green — in particular
[`tests/test_packing_correctness.py`](tests/test_packing_correctness.py), which asserts
that packed training matches unpacked inference.

## Citation

```bibtex
# Not the final version, will add the final version soon.

@inproceedings{alsamkary2026ares,
  title     = {Ares: Loss-Free Mixture-of-Experts Routing for Bidirectional Protein Encoders},
  author    = {Alsamkary, Hazem},
  booktitle = {ICML 2026 Workshop on Generative AI and Biology (GenBio)},
  year      = {2026},
  note      = {Spotlight},
  url       = {https://openreview.net/forum?id=gq0R7xiPjg}
}
```

## License

MIT. See [LICENSE](LICENSE).

Affiliation: Proteinea