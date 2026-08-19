import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ares.models.model import Ares
from ares.models.soft_router import SoftRouter
from ares.tokenization.protein_tokenizer import AresProteinTokenizer


class UniRef30(Dataset):
    def __init__(self, max_examples: Optional[int] = None, split: str = "train"):
        self.data = load_dataset(
            "hazemessam/sprot",
            data_files={split: split + ".parquet"},
            streaming=False,
        )[split]
        self.max_examples = max_examples

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        return len(self.data)

    def __getitem__(self, idx):
        return {"sequence": self.data[idx]["sequence"]}


@dataclass
class Collator:
    tokenizer: AresProteinTokenizer

    def __call__(self, batch):
        grouped = defaultdict(list)
        for example in batch:
            for key, value in example.items():
                grouped[key].append(value)
        return self.tokenizer(
            grouped["sequence"],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )


def quantile_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "mean": float(values.mean()),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0:
        return 0.0
    entropy = -(probabilities * np.log(probabilities)).sum()
    return float(entropy / np.log(len(probabilities)))


def convert_to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(key): convert_to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_jsonable(value) for value in obj]
    return obj


def assess_layer_health(
    combine_mass_entropy_norm: float,
    top1_entropy_norm: float,
    top1_max_share: float,
    slot_top_weight_p95: float,
) -> str:
    if (
        combine_mass_entropy_norm < 0.92
        or top1_entropy_norm < 0.80
        or top1_max_share > 0.20
        or slot_top_weight_p95 > 0.95
    ):
        return "bad"
    if (
        combine_mass_entropy_norm < 0.97
        or top1_entropy_norm < 0.90
        or top1_max_share > 0.12
        or slot_top_weight_p95 > 0.70
    ):
        return "warning"
    return "healthy"


def summarize_layer_health(layer_summaries: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    counts = {"healthy": 0, "warning": 0, "bad": 0}
    worst_layers = []

    for layer_name, stats in layer_summaries.items():
        rating = assess_layer_health(
            combine_mass_entropy_norm=stats["combine_mass_entropy_norm"],
            top1_entropy_norm=stats["top1_entropy_norm"],
            top1_max_share=stats["top1_max_share"],
            slot_top_weight_p95=stats["slot_top_weight_p95"],
        )
        counts[rating] += 1
        worst_layers.append(
            {
                "layer": layer_name,
                "rating": rating,
                "combine_mass_entropy_norm": stats["combine_mass_entropy_norm"],
                "top1_entropy_norm": stats["top1_entropy_norm"],
                "top1_max_share": stats["top1_max_share"],
                "slot_top_weight_p95": stats["slot_top_weight_p95"],
            }
        )

    severity_rank = {"bad": 0, "warning": 1, "healthy": 2}
    worst_layers.sort(
        key=lambda entry: (
            severity_rank[entry["rating"]],
            entry["combine_mass_entropy_norm"],
            entry["top1_entropy_norm"],
            -entry["top1_max_share"],
            -entry["slot_top_weight_p95"],
        )
    )
    return {
        "counts": counts,
        "worst_layers": worst_layers[:5],
    }


def analyze_checkpoint(
    checkpoint: str,
    device: str,
    split: str,
    max_examples: int,
    batch_size: int,
) -> Dict[str, Any]:
    model = Ares.from_pretrained(checkpoint).to(device)
    tokenizer = AresProteinTokenizer()
    dataset = UniRef30(max_examples=max_examples, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=Collator(tokenizer))
    model.eval()

    dispatch_all_values = []
    dispatch_valid_values = []
    layer_dispatch_stats = {}
    layer_expert_mass = {}
    layer_top1_counts = {}
    layer_token_entropy = {}
    layer_slot_top = {}
    layer_dispatch_token_mass = {}
    num_batches = 0
    num_valid_tokens = 0

    for batch in tqdm(dataloader, desc="Analyzing Soft-MoE"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        valid_mask = attention_mask.bool().cpu()

        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)

        num_batches += 1
        num_valid_tokens += int(valid_mask.sum().item())

        for name, module in model.named_modules():
            if not isinstance(module, SoftRouter):
                continue

            dispatch = module.dispatch_weights_.float().cpu()
            combine = module.combine_weights_.float().cpu()
            valid_dispatch = dispatch[valid_mask[:, :, None, None].expand_as(dispatch)]
            dispatch_values = dispatch.flatten().numpy()
            valid_values = valid_dispatch.numpy()

            dispatch_all_values.append(dispatch_values)
            dispatch_valid_values.append(valid_values)

            per_slot_sums = dispatch.sum(dim=1).numpy()
            token_total_mass = dispatch.sum(dim=(2, 3))[valid_mask].numpy()
            slot_top_weights = dispatch.amax(dim=1).numpy().reshape(-1)

            if name not in layer_dispatch_stats:
                layer_dispatch_stats[name] = {
                    "shape": list(dispatch.shape),
                    "all_values": [],
                    "valid_values": [],
                    "slot_top_weights": [],
                    "token_total_mass": [],
                    "seq_softmax_sum_min": [],
                    "seq_softmax_sum_max": [],
                }

            layer_dispatch_stats[name]["all_values"].append(dispatch_values)
            layer_dispatch_stats[name]["valid_values"].append(valid_values)
            layer_dispatch_stats[name]["slot_top_weights"].append(slot_top_weights)
            layer_dispatch_stats[name]["token_total_mass"].append(token_total_mass)
            layer_dispatch_stats[name]["seq_softmax_sum_min"].append(float(per_slot_sums.min()))
            layer_dispatch_stats[name]["seq_softmax_sum_max"].append(float(per_slot_sums.max()))

            expert_probs = combine.sum(dim=-1)
            valid_expert_probs = expert_probs[valid_mask]
            if valid_expert_probs.numel() == 0:
                continue

            mass = valid_expert_probs.sum(dim=0).numpy()
            token_top1 = valid_expert_probs.argmax(dim=-1).numpy()
            counts = np.bincount(token_top1, minlength=expert_probs.shape[-1])
            token_entropy = -(
                valid_expert_probs.numpy()
                * np.log(np.clip(valid_expert_probs.numpy(), 1e-12, 1.0))
            ).sum(axis=-1) / np.log(expert_probs.shape[-1])

            layer_expert_mass.setdefault(name, []).append(mass)
            layer_top1_counts.setdefault(name, []).append(counts)
            layer_token_entropy.setdefault(name, []).append(token_entropy)
            layer_slot_top.setdefault(name, []).append(slot_top_weights)
            layer_dispatch_token_mass.setdefault(name, []).append(token_total_mass)

    global_all = np.concatenate(dispatch_all_values)
    global_valid = np.concatenate(dispatch_valid_values)

    dispatch_summary = {
        "all_entries": quantile_stats(global_all),
        "valid_entries": quantile_stats(global_valid),
    }

    per_layer_dispatch = {}
    for name, stats in sorted(layer_dispatch_stats.items()):
        all_values = np.concatenate(stats["all_values"])
        valid_values = np.concatenate(stats["valid_values"])
        slot_top_weights = np.concatenate(stats["slot_top_weights"])
        token_total_mass = np.concatenate(stats["token_total_mass"])

        per_layer_dispatch[name] = {
            "shape": stats["shape"],
            "all_entries": quantile_stats(all_values),
            "valid_entries": quantile_stats(valid_values),
            "seq_softmax_sum_min": float(min(stats["seq_softmax_sum_min"])),
            "seq_softmax_sum_max": float(max(stats["seq_softmax_sum_max"])),
            "slot_top_weight_mean": float(slot_top_weights.mean()),
            "slot_top_weight_p95": float(np.quantile(slot_top_weights, 0.95)),
            "slot_top_weight_max": float(slot_top_weights.max()),
            "dispatch_token_mass_mean": float(token_total_mass.mean()),
            "dispatch_token_mass_std": float(token_total_mass.std()),
        }

    expert_balance = {}
    for name in sorted(layer_expert_mass.keys()):
        mass = np.sum(layer_expert_mass[name], axis=0)
        mass_probs = mass / mass.sum()
        top1_counts = np.sum(layer_top1_counts[name], axis=0)
        top1_probs = top1_counts / top1_counts.sum()
        token_entropy = np.concatenate(layer_token_entropy[name])
        slot_top_weights = np.concatenate(layer_slot_top[name])
        token_total_mass = np.concatenate(layer_dispatch_token_mass[name])

        expert_balance[name] = {
            "combine_mass_entropy_norm": normalized_entropy(mass_probs),
            "combine_mass_cv": float(mass.std() / mass.mean()),
            "combine_mass_min_share": float(mass_probs.min()),
            "combine_mass_max_share": float(mass_probs.max()),
            "combine_mass_top5_shares": [float(x) for x in sorted(mass_probs, reverse=True)[:5]],
            "top1_entropy_norm": normalized_entropy(top1_probs),
            "top1_cv": float(top1_counts.std() / top1_counts.mean()),
            "top1_min_share": float(top1_probs.min()),
            "top1_max_share": float(top1_probs.max()),
            "token_expert_entropy_mean_norm": float(token_entropy.mean()),
            "token_expert_entropy_p05_norm": float(np.quantile(token_entropy, 0.05)),
            "token_expert_entropy_p50_norm": float(np.quantile(token_entropy, 0.50)),
            "token_expert_entropy_p95_norm": float(np.quantile(token_entropy, 0.95)),
            "slot_top_weight_mean": float(slot_top_weights.mean()),
            "slot_top_weight_p95": float(np.quantile(slot_top_weights, 0.95)),
            "slot_top_weight_max": float(slot_top_weights.max()),
            "dispatch_token_mass_mean": float(token_total_mass.mean()),
            "dispatch_token_mass_std": float(token_total_mass.std()),
            "health_flag": assess_layer_health(
                combine_mass_entropy_norm=normalized_entropy(mass_probs),
                top1_entropy_norm=normalized_entropy(top1_probs),
                top1_max_share=float(top1_probs.max()),
                slot_top_weight_p95=float(np.quantile(slot_top_weights, 0.95)),
            ),
        }

    return {
        "config": {
            "checkpoint": checkpoint,
            "device": device,
            "split": split,
            "max_examples": max_examples,
            "batch_size": batch_size,
            "moe_type": model.config.moe_type,
            "moe_normalize": model.config.moe_normalize,
            "moe_after_num_layers": model.config.moe_after_num_layers,
            "num_layers": model.config.num_layers,
            "num_experts": model.config.num_experts,
            "moe_num_slots": model.config.moe_num_slots,
        },
        "run_summary": {
            "num_batches": num_batches,
            "num_valid_tokens": num_valid_tokens,
            "num_router_layers": len(expert_balance),
        },
        "dispatch_distribution": dispatch_summary,
        "per_layer_dispatch": per_layer_dispatch,
        "expert_balance": expert_balance,
        "health_summary": summarize_layer_health(expert_balance),
    }


def print_human_summary(results: Dict[str, Any]) -> None:
    dispatch = results["dispatch_distribution"]["valid_entries"]
    print("\n=== Global Valid Dispatch Distribution ===")
    print(
        "min={min:.3e} p01={p01:.3e} p05={p05:.3e} median={median:.3e} "
        "mean={mean:.3e} p95={p95:.3e} p99={p99:.3e} max={max:.3e}".format(**dispatch)
    )

    print("\n=== Layer Health Summary ===")
    counts = results["health_summary"]["counts"]
    print(
        f"healthy={counts['healthy']} warning={counts['warning']} bad={counts['bad']}"
    )

    print("\n=== Worst Layers ===")
    for entry in results["health_summary"]["worst_layers"]:
        print(
            f"{entry['layer']}: {entry['rating']} | "
            f"combine_entropy={entry['combine_mass_entropy_norm']:.3f} | "
            f"top1_entropy={entry['top1_entropy_norm']:.3f} | "
            f"top1_max={entry['top1_max_share']:.3f} | "
            f"slot_top_p95={entry['slot_top_weight_p95']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft-MoE checkpoint diagnostics")
    parser.add_argument(
        "--checkpoint",
        default="HazemLab/ares-softmoe-4b-consecutive",
        help="HF checkpoint or local checkpoint path",
    )
    parser.add_argument("--device", default="cuda:1", help="Torch device")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--max-examples", type=int, default=50, help="Number of sequences")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument(
        "--output",
        default="analysis_outputs/soft_moe_diagnostics.json",
        help="Path to write JSON diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = analyze_checkpoint(
        checkpoint=args.checkpoint,
        device=args.device,
        split=args.split,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(convert_to_jsonable(results), handle, indent=2)

    print_human_summary(results)
    print(f"\nSaved detailed diagnostics to {output_path}")


if __name__ == "__main__":
    main()
