"""
Packing correctness test: verify that packed and unpacked forward passes
produce equivalent per-sequence logits.

Usage:
    cd tests && python test_packing_correctness.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import torch
import torch.nn.functional as F

from ares.tokenization.protein_tokenizer import AresProteinTokenizer
from ares.tokenization.constants import AA_VOCAB
from ares.models.config import AresConfig
from ares.models.model import Ares
from ares.pipelines.sequence_packing import heap_pack_sequences
from ares.pipelines.debugging import NaNObserver

AMINO_ACIDS = [ch for ch, idx in AA_VOCAB.items() if 5 <= idx <= 24]
MAX_SEQ_LEN = 64
NUM_SEQUENCES = 8
SEED = 42
ATOL = 1e-4
COS_SIM_THRESHOLD = 0.999


def generate_random_sequences(n, min_len=10, max_len=50):
    rng = random.Random(SEED)
    return [
        "".join(rng.choices(AMINO_ACIDS, k=rng.randint(min_len, max_len)))
        for _ in range(n)
    ]


def tokenize_sequences(sequences, tokenizer):
    """Tokenize each sequence individually, returning list of 1-D token tensors."""
    token_lists = []
    for seq in sequences:
        enc = tokenizer(
            seq,
            return_tensors=None,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        token_lists.append(torch.tensor(enc["input_ids"], dtype=torch.long))
    return token_lists


def build_unpacked_batch(token_lists, max_seq_len, pad_token_id):
    """Pad each sequence to max_seq_len independently -> [N, max_seq_len]."""
    batch_ids = []
    batch_mask = []
    batch_pos = []

    for tokens in token_lists:
        seq_len = len(tokens)
        pad_len = max_seq_len - seq_len

        ids = F.pad(tokens, (0, pad_len), value=pad_token_id)
        mask = torch.cat(
            [
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )
        pos = torch.cat(
            [
                torch.arange(seq_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ]
        )

        batch_ids.append(ids)
        batch_mask.append(mask)
        batch_pos.append(pos)

    return {
        "input_ids": torch.stack(batch_ids),
        "attention_mask": torch.stack(batch_mask),
        "position_ids": torch.stack(batch_pos),
    }


def build_packed_batch(token_lists, bins, max_seq_len, pad_token_id):
    """Pack sequences into bins -> [num_bins, max_seq_len]."""
    all_ids = []
    all_mask = []
    all_pos = []
    all_seq_ids = []

    for bin_indices in bins:
        ids = []
        seq_ids = []
        pos = []

        for local_seq_idx, global_idx in enumerate(bin_indices):
            tokens = token_lists[global_idx]
            seq_len = len(tokens)
            ids.extend(tokens.tolist())
            seq_ids.extend([local_seq_idx] * seq_len)
            pos.extend(range(seq_len))

        real_len = len(ids)
        pad_len = max_seq_len - real_len

        ids.extend([pad_token_id] * pad_len)
        seq_ids.extend([local_seq_idx] * pad_len)
        pos.extend([0] * pad_len)
        mask = [1] * real_len + [0] * pad_len

        all_ids.append(torch.tensor(ids, dtype=torch.long))
        all_mask.append(torch.tensor(mask, dtype=torch.long))
        all_pos.append(torch.tensor(pos, dtype=torch.long))
        all_seq_ids.append(torch.tensor(seq_ids, dtype=torch.long))

    return {
        "input_ids": torch.stack(all_ids),
        "attention_mask": torch.stack(all_mask),
        "position_ids": torch.stack(all_pos),
        "sequence_ids": torch.stack(all_seq_ids),
    }


def print_bins(bins, lengths, max_seq_len):
    print(f"\n{'─'*60}")
    print("Bin layout:")
    print(f"{'─'*60}")
    for i, bin_indices in enumerate(bins):
        bin_lengths = [lengths[j] for j in bin_indices]
        total = sum(bin_lengths)
        remaining = max_seq_len - total
        print(
            f"  Bin {i}: indices={bin_indices} "
            f"lengths={bin_lengths} "
            f"total={total}/{max_seq_len} remaining={remaining}"
        )
    print(f"{'─'*60}\n")


def extract_packed_logits(
    packed_logits, packed_seq_ids, packed_attn_mask, bins, token_lists
):
    """Extract per-sequence logits from packed output, keyed by global index."""
    result = {}
    for bin_idx, bin_indices in enumerate(bins):
        for local_seq_idx, global_idx in enumerate(bin_indices):
            seq_mask = (packed_seq_ids[bin_idx] == local_seq_idx) & (
                packed_attn_mask[bin_idx] == 1
            )
            result[global_idx] = packed_logits[bin_idx][seq_mask]
    return result


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    print(f"{'='*60}")
    print("  Packing Correctness Test")
    print(f"{'='*60}")

    tokenizer = AresProteinTokenizer()
    pad_token_id = tokenizer.pad_token_id

    # 1. Generate and tokenize
    sequences = generate_random_sequences(NUM_SEQUENCES)
    token_lists = tokenize_sequences(sequences, tokenizer)
    lengths = [len(t) for t in token_lists]

    print(f"\n[1] Generated {NUM_SEQUENCES} sequences:")
    for i, (seq, tlen) in enumerate(zip(sequences, lengths)):
        print(
            f"    seq {i}: {len(seq)} residues -> {tlen} tokens (with CLS+EOS)"
        )

    # 2. Pack with heap
    bins = heap_pack_sequences(lengths, MAX_SEQ_LEN)
    print_bins(bins, lengths, MAX_SEQ_LEN)

    # 3. Build batches
    unpacked = build_unpacked_batch(token_lists, MAX_SEQ_LEN, pad_token_id)
    packed = build_packed_batch(token_lists, bins, MAX_SEQ_LEN, pad_token_id)

    print(
        f"[3] Unpacked batch: input_ids {tuple(unpacked['input_ids'].shape)}"
    )
    print(f"    Packed batch:   input_ids {tuple(packed['input_ids'].shape)}")

    # 4. Build model
    config = AresConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ff_dim=256,
        moe_type=None,
        moe_after_num_layers=None,
        num_experts=1,
        pad_token_id=pad_token_id,
    )
    model = Ares(config)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n[4] Model: {n_params:,} params (dense, {config.num_layers} layers)\n"
    )

    # 5. Forward pass
    with torch.no_grad():
        with NaNObserver(
            nan_only=True, ignore_functions=["masked_fill_", "add"]
        ):
            print(
                unpacked["input_ids"].shape,
                unpacked["attention_mask"].shape,
                unpacked["position_ids"].shape,
            )
            unpacked_out = model(
                input_ids=unpacked["input_ids"],
                attention_mask=unpacked["attention_mask"],
                position_ids=unpacked["position_ids"],
            )
        with NaNObserver(
            nan_only=True, ignore_functions=["masked_fill_", "add"]
        ):
            print(
                packed["input_ids"].shape,
                packed["attention_mask"].shape,
                packed["position_ids"].shape,
                packed["sequence_ids"].shape,
            )
            packed_out = model(
                input_ids=packed["input_ids"],
                attention_mask=packed["attention_mask"],
                position_ids=packed["position_ids"],
                sequence_ids=packed["sequence_ids"],
            )

    # 6. Compare per-sequence logits
    packed_per_seq = extract_packed_logits(
        packed_out.logits,
        packed["sequence_ids"],
        packed["attention_mask"],
        bins,
        token_lists,
    )

    print(f"{'─'*60}")
    print("Per-sequence comparison (packed vs unpacked):")
    print(f"{'─'*60}")

    all_passed = True
    for i in range(NUM_SEQUENCES):
        seq_len = lengths[i]
        unpacked_logits = unpacked_out.logits[i, :seq_len]
        packed_logits = packed_per_seq[i]

        assert unpacked_logits.shape == packed_logits.shape, (
            f"Shape mismatch for seq {i}: "
            f"unpacked={unpacked_logits.shape} packed={packed_logits.shape}"
        )

        max_diff = (unpacked_logits - packed_logits).abs().max().item()
        cos_sim = F.cosine_similarity(
            unpacked_logits.flatten().unsqueeze(0),
            packed_logits.flatten().unsqueeze(0),
        ).item()

        passed = max_diff < ATOL and cos_sim > COS_SIM_THRESHOLD
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"\n  Sequence {i} (len={seq_len}):")
        print(f"    max |diff|  = {max_diff:.2e}")
        print(f"    cosine sim  = {cos_sim:.6f}")
        print(f"    {status}")

    print(f"\n{'='*60}")
    if all_passed:
        print("  All sequences PASSED")
    else:
        print("  Some sequences FAILED")
    print(f"{'='*60}\n")

    assert all_passed, "Packing correctness test failed"


if __name__ == "__main__":
    main()
