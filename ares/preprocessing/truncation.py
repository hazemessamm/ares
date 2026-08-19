from __future__ import annotations
import random


def random_truncation(
    sequence: str,
    max_length: int,
    rng: random.Random | None = None,
):
    if len(sequence) <= max_length:
        return sequence, 0, len(sequence), False
    truncation_rng = rng or random
    start_idx = truncation_rng.randint(0, len(sequence) - max_length)
    end_idx = start_idx + max_length
    return sequence[start_idx:end_idx], start_idx, end_idx, True
