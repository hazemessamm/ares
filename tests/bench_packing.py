"""
Benchmark: greedy_pack_sequences (FFD, O(n*m)) vs heap_pack_sequences (WFD, O(n log m))

Usage:
    python tests/bench_packing.py
"""

import heapq
import time
import random
import math
import sys
import os
from typing import List, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Inline the two functions to avoid pulling in torch/datasets via the ares package.


def greedy_pack_sequences(
    lengths: List[int],
    max_seq_len: int,
) -> List[List[int]]:
    indexed_lengths = sorted(
        enumerate(lengths), key=lambda x: x[1], reverse=True
    )
    bins: List[List[int]] = []
    bin_remaining: List[int] = []
    for orig_idx, seq_len in indexed_lengths:
        placed = False
        for bin_idx in range(len(bins)):
            if bin_remaining[bin_idx] >= seq_len:
                bins[bin_idx].append(orig_idx)
                bin_remaining[bin_idx] -= seq_len
                placed = True
                break
        if not placed:
            bins.append([orig_idx])
            bin_remaining.append(max_seq_len - seq_len)
    return bins


def heap_pack_sequences(
    lengths: List[int],
    max_seq_len: int,
) -> List[List[int]]:
    indexed_lengths = sorted(
        enumerate(lengths), key=lambda x: x[1], reverse=True
    )
    bins: List[List[int]] = []
    heap: List[tuple] = []
    for orig_idx, seq_len in indexed_lengths:
        if heap and -heap[0][0] >= seq_len:
            neg_remaining, bin_idx = heapq.heappop(heap)
            bins[bin_idx].append(orig_idx)
            new_remaining = -neg_remaining - seq_len
            heapq.heappush(heap, (-new_remaining, bin_idx))
        else:
            bin_idx = len(bins)
            bins.append([orig_idx])
            heapq.heappush(heap, (-(max_seq_len - seq_len), bin_idx))
    return bins


MAX_SEQ_LEN = 1024
SCALES = [10_000, 100_000, 1_000_000]
RUNS_PER_SCALE = 3
SEED = 42


def generate_protein_lengths(n: int, seed: int = SEED) -> List[int]:
    """Log-normal distribution mimicking UniRef50 protein lengths (+2 for special tokens)."""
    rng = random.Random(seed)
    lengths = []
    for _ in range(n):
        raw = int(math.exp(rng.gauss(mu=4.5, sigma=1.2)))
        length = max(3, min(raw + 2, MAX_SEQ_LEN))
        lengths.append(length)
    return lengths


def validate_packing(
    bins: List[List[int]], lengths: List[int], max_seq_len: int
) -> None:
    seen = set()
    for bin_indices in bins:
        total = sum(lengths[i] for i in bin_indices)
        assert total <= max_seq_len, f"Bin overflow: {total} > {max_seq_len}"
        for idx in bin_indices:
            assert idx not in seen, f"Duplicate index: {idx}"
            seen.add(idx)

    assert seen == set(
        range(len(lengths))
    ), f"Missing indices: {set(range(len(lengths))) - seen}"


def benchmark_fn(
    fn: Callable, lengths: List[int], max_seq_len: int, runs: int
) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn(lengths, max_seq_len)
        times.append(time.perf_counter() - start)
    return min(times)


def packing_efficiency(bins: List[List[int]], lengths: List[int]) -> float:
    total_tokens = sum(lengths)
    total_capacity = len(bins) * MAX_SEQ_LEN
    return total_tokens / total_capacity * 100


def compare_co_occurrence(
    bins_a: List[List[int]], bins_b: List[List[int]], n: int
) -> dict:
    """
    Compare whether the same indices end up in the same bins across two packings.

    Builds a set of co-occurring index pairs for each packing, then computes
    the Jaccard similarity between them.

    Returns:
        dict with pair counts for each packing, intersection/union sizes,
        and Jaccard similarity (1.0 = identical groupings, 0.0 = completely different).
    """

    def pair_set(bins):
        pairs = set()
        for b in bins:
            idxs = sorted(b)
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    pairs.add((idxs[i], idxs[j]))
        return pairs

    pairs_a = pair_set(bins_a)
    pairs_b = pair_set(bins_b)
    intersection = len(pairs_a & pairs_b)
    union = len(pairs_a | pairs_b)
    jaccard = intersection / union if union > 0 else 1.0

    return {
        "pairs_ffd": len(pairs_a),
        "pairs_wfd": len(pairs_b),
        "intersection": intersection,
        "union": union,
        "jaccard": jaccard,
    }


def main():
    print(f"{'='*70}")
    print(f"Bin Packing Benchmark: FFD (linear) vs WFD (heap)")
    print(f"max_seq_len={MAX_SEQ_LEN}, runs_per_scale={RUNS_PER_SCALE}")
    print(f"{'='*70}\n")

    for n in SCALES:
        lengths = generate_protein_lengths(n)

        mean_len = sum(lengths) / len(lengths)
        print(f"n={n:>10,}  (mean_len={mean_len:.0f})")

        ffd_bins = greedy_pack_sequences(lengths, MAX_SEQ_LEN)
        validate_packing(ffd_bins, lengths, MAX_SEQ_LEN)

        wfd_bins = heap_pack_sequences(lengths, MAX_SEQ_LEN)
        validate_packing(wfd_bins, lengths, MAX_SEQ_LEN)

        ffd_eff = packing_efficiency(ffd_bins, lengths)
        wfd_eff = packing_efficiency(wfd_bins, lengths)

        co = compare_co_occurrence(ffd_bins, wfd_bins, n)

        ffd_time = benchmark_fn(
            greedy_pack_sequences, lengths, MAX_SEQ_LEN, RUNS_PER_SCALE
        )
        wfd_time = benchmark_fn(
            heap_pack_sequences, lengths, MAX_SEQ_LEN, RUNS_PER_SCALE
        )

        speedup = ffd_time / wfd_time if wfd_time > 0 else float("inf")
        bin_diff = (len(wfd_bins) - len(ffd_bins)) / len(ffd_bins) * 100

        print(
            f"  FFD (linear): {len(ffd_bins):>8,} bins, "
            f"{ffd_time:>8.3f}s, efficiency={ffd_eff:.1f}%"
        )
        print(
            f"  WFD (heap):   {len(wfd_bins):>8,} bins, "
            f"{wfd_time:>8.3f}s, efficiency={wfd_eff:.1f}%"
        )
        print(f"  speedup: {speedup:.1f}x, " f"bin diff: {bin_diff:+.2f}%")
        print(
            f"  co-occurrence: jaccard={co['jaccard']:.3f} "
            f"(shared={co['intersection']:,}/{co['union']:,} pairs)"
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
