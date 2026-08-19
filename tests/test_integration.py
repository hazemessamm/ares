import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import random
import torch

from ares.tokenization.protein_tokenizer import AresProteinTokenizer
from ares.tokenization.constants import AA_VOCAB
from ares.preprocessing.truncation import random_truncation
from ares.preprocessing.utils import MLMProbabilitySampler
from ares.preprocessing.noising import SequenceCorruptor
from ares.models.config import AresConfig
from ares.models.model import Ares

AMINO_ACIDS = [ch for ch, idx in AA_VOCAB.items() if idx >= 5 and idx <= 24]
MAX_SEQ_LEN = 64
BATCH_SIZE = 4


def generate_random_sequences(n: int, min_len: int = 30, max_len: int = 200):
    return [
        "".join(
            random.choices(AMINO_ACIDS, k=random.randint(min_len, max_len))
        )
        for _ in range(n)
    ]


def run(moe_type: str):
    print(f"\n{'='*60}")
    print(f"  Running integration test with moe_type = '{moe_type}'")
    print(f"{'='*60}")

    tokenizer = AresProteinTokenizer()
    vocab_size = tokenizer.vocab_size

    sampler = MLMProbabilitySampler(
        mlm_probs=[0.15, 0.25],
        masking_probs=[0.80, 0.80],
        mutation_probs=[0.10, 0.10],
    )
    corruptor = SequenceCorruptor(
        tokenizer=tokenizer, mlm_probability_sampler=sampler
    )

    # 1. Generate random protein sequences
    sequences = generate_random_sequences(BATCH_SIZE)
    print(
        f"\n[1] Generated {BATCH_SIZE} random sequences (lengths: {[len(s) for s in sequences]})"
    )

    # 2. Truncate
    truncated = []
    for seq in sequences:
        trunc_seq, start, end, was_truncated = random_truncation(
            seq, MAX_SEQ_LEN
        )
        truncated.append(trunc_seq)
        if was_truncated:
            print(
                f"    Truncated {len(seq)} -> {len(trunc_seq)}  (slice [{start}:{end}])"
            )
    print(
        f"[2] Truncated to max_len={MAX_SEQ_LEN} (lengths: {[len(s) for s in truncated]})"
    )

    # 3. Tokenize
    encoded = tokenizer(
        truncated,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN + 2,  # +2 for <cls> and <eos>
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    print(
        f"[3] Tokenized -> input_ids shape: {tuple(input_ids.shape)}, "
        f"attention_mask shape: {tuple(attention_mask.shape)}"
    )

    # 4. Corrupt (MLM)
    corrupted_ids, labels = corruptor(input_ids)
    n_masked = (labels != -100).sum().item()
    print(
        f"[4] Corrupted -> {n_masked} tokens selected for MLM out of {input_ids.numel()}"
    )

    # 5. Build model
    config = AresConfig(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        ff_dim=256,
        moe_type=moe_type,
        moe_after_num_layers=1,
        num_experts=4,
        expert_capacity_factor=2.0,
        moe_num_slots=8,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = Ares(config)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[5] Built Ares model ({n_params:,} params, {config.num_layers} layers, "
        f"MoE after layer {config.moe_after_num_layers})"
    )

    # 6. Forward pass
    with torch.no_grad():
        output = model(
            input_ids=corrupted_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    loss = output.loss.item()
    perplexity = math.exp(loss)

    print(f"[6] Forward pass complete")
    print(f"    logits shape : {tuple(output.logits.shape)}")
    print(f"    loss         : {loss:.4f}")
    print(f"    perplexity   : {perplexity:.2f}")
    print(
        f"    logits range : [{output.logits.min().item():.4f}, {output.logits.max().item():.4f}]"
    )

    # Sanity check: for a random init the perplexity should be in the
    # neighbourhood of vocab_size (uniform predictions). Anything below 1
    # is impossible and anything above 2x vocab_size signals a problem.
    assert 1.0 < perplexity < 2 * vocab_size, (
        f"Perplexity {perplexity:.2f} is outside the expected range "
        f"(1, {2 * vocab_size}) for a randomly initialised model"
    )
    print(
        f"    perplexity check PASSED (expected ~{vocab_size}, got {perplexity:.2f})"
    )


if __name__ == "__main__":
    run("soft_router")
    run("expert_choice")
    print(f"\n{'='*60}")
    print("  All integration tests passed!")
    print(f"{'='*60}")
