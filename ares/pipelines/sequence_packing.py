import os
import heapq
import pickle
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import torch
import datasets
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizer
from ares.preprocessing.noising import SequenceCorruptor
from ares.preprocessing.truncation import random_truncation
from ares.models.packing import build_packing_attention_bias  # noqa: F401

logger = logging.getLogger(__name__)


@dataclass
class PackedCollator:
    """Simple collator that just stacks tensors. All masking and packing
    logic is handled in the dataset."""

    def __call__(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return {
            key: torch.stack([x[key] for x in batch])
            for key in batch[0].keys()
        }


def greedy_pack_sequences(
    lengths: List[int],
    max_seq_len: int,
) -> List[List[int]]:
    """
    Greedy first-fit-decreasing bin packing.

    Args:
        lengths: Sequence lengths (including special tokens)
        max_seq_len: Maximum packed length

    Returns:
        List of bins, each bin is a list of indices into the dataset
    """
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


def _heap_pack_chunk(
    args: Tuple[np.ndarray, np.ndarray, int],
) -> List[List[int]]:
    """Pack a chunk of sorted indices using the shared lengths array."""
    chunk_indices, lengths_arr, max_seq_len = args
    bins: List[List[int]] = []
    heap: List[tuple] = []

    for orig_idx in chunk_indices:
        seq_len = int(lengths_arr[orig_idx])
        if heap and -heap[0][0] >= seq_len:
            neg_remaining, bin_idx = heapq.heappop(heap)
            bins[bin_idx].append(int(orig_idx))
            new_remaining = -neg_remaining - seq_len
            heapq.heappush(heap, (-new_remaining, bin_idx))
        else:
            bin_idx = len(bins)
            bins.append([int(orig_idx)])
            heapq.heappush(heap, (-(max_seq_len - seq_len), bin_idx))

    return bins


def heap_pack_sequences(
    lengths: List[int],
    max_seq_len: int,
    num_workers: int = None,
) -> List[List[int]]:
    """
    Worst-fit-decreasing bin packing using a max-heap.

    Sorts with torch.argsort (parallel), then splits work across processes.
    Each worker gets an interleaved slice of the sorted order so it
    sees a balanced mix of lengths.

    Args:
        lengths: Sequence lengths (including special tokens)
        max_seq_len: Maximum packed length
        num_workers: Parallel workers (default: os.cpu_count())

    Returns:
        List of bins, each bin is a list of indices into the dataset
    """
    if num_workers is None:
        num_workers = os.cpu_count() // 2 or 10

    lengths_arr = np.array(lengths, dtype=np.int64)

    cache_path = Path.home() / ".cache" / "ares" / "sorted_indices.pt"
    if cache_path.exists():
        logger.info(f"Loading cached sorted indices from {cache_path}")
        sorted_indices = torch.load(cache_path, weights_only=True).numpy()
    else:
        sorted_indices = torch.argsort(
            torch.from_numpy(lengths_arr), descending=True
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sorted_indices, cache_path)
        logger.info(f"Cached sorted indices to {cache_path}")
        sorted_indices = sorted_indices.numpy()

    if num_workers <= 1:
        return _heap_pack_chunk((sorted_indices, lengths_arr, max_seq_len))

    # Interleave across workers via numpy slicing
    chunks = [sorted_indices[i::num_workers] for i in range(num_workers)]

    with Pool(num_workers) as pool:
        results = pool.map(
            _heap_pack_chunk,
            [(chunk, lengths_arr, max_seq_len) for chunk in chunks],
        )

    all_bins: List[List[int]] = []
    for bins in results:
        all_bins.extend(bins)

    return all_bins


def _shuffle_and_skip_bins(
    bin_indices: List[int],
    skip_bins: int,
) -> List[int]:
    """Apply resume skipping within the epoch's shuffled bin order."""
    shuffled_bin_indices = list(bin_indices)
    random.shuffle(shuffled_bin_indices)

    if skip_bins <= 0:
        return shuffled_bin_indices
    if skip_bins >= len(shuffled_bin_indices):
        return []
    return shuffled_bin_indices[skip_bins:]


def _truncate_for_packing(
    sequence,
    max_length: int,
):
    """Mirror HFDataset truncation without inventing boundary tokens."""
    truncated, start_idx, end_idx, _ = random_truncation(sequence, max_length)

    add_cls = start_idx == 0
    add_eos = end_idx == len(sequence)
    num_special = int(add_cls) + int(add_eos)

    target_aa_len = max_length - num_special
    if len(truncated) > target_aa_len:
        excess = len(truncated) - target_aa_len
        if add_eos and not add_cls:
            truncated = truncated[excess:]
            start_idx += excess
        else:
            truncated = truncated[:target_aa_len]
            end_idx = start_idx + target_aa_len

    if add_cls:
        truncated = ["<cls>"] + truncated
    if add_eos:
        truncated = truncated + ["<eos>"]

    return truncated


class PackedUniRef50Dataset(IterableDataset):
    SCHEDULER_LOG_EVERY = 10000

    def __init__(
        self,
        repo_id: str,
        column_name: str,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        seed: int = 42,
        noise_fn: SequenceCorruptor = None,
        skip_examples: int = 0,
        use_precomputed_bins: bool = False,
        bins_repo_id: str = "hazemessam/bins-dataset",
    ):
        self.tokenizer = tokenizer
        self.noise_fn = noise_fn
        self.max_length = max_length
        self.column_name = column_name
        self.seed = seed
        self.skip_examples = skip_examples
        self._skipped = False

        self.data = load_dataset(
            repo_id,
            split=split,
            streaming=False,
            verification_mode=datasets.VerificationMode.NO_CHECKS,
        )

        if use_precomputed_bins:
            logger.info(f"Downloading precomputed bins from {bins_repo_id}")
            local_dir = snapshot_download(bins_repo_id, repo_type="dataset")
            bins_path = Path(local_dir) / "packed_bins.pkl"
            with open(bins_path, "rb") as f:
                self.bins = pickle.load(f)
        else:
            cache_dir = Path.home() / ".cache" / "ares"
            bins_cache = cache_dir / "packed_bins.pkl"

            if bins_cache.exists():
                logger.info(f"Loading cached bins from {bins_cache}")
                with open(bins_cache, "rb") as f:
                    self.bins = pickle.load(f)
            else:
                _ml = max_length
                _col = column_name
                self.data = self.data.map(
                    lambda batch: {
                        "_len": [min(len(s) + 2, _ml) for s in batch[_col]]
                    },
                    batched=True,
                    batch_size=10_000,
                    num_proc=10,
                    keep_in_memory=True,
                )
                lengths = self.data["_len"]
                logger.info("Packing sequences...")
                self.bins = heap_pack_sequences(
                    lengths, max_length, num_workers=30
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                with open(bins_cache, "wb") as f:
                    pickle.dump(self.bins, f)
                logger.info(f"Cached bins to {bins_cache}")

        logger.info(
            f"Dataset initialized: {len(self.data)} sequences → "
            f"{len(self.bins)} packed bins "
            f"(factor: {len(self.data) / len(self.bins):.2f}x, "
            f"max_length={max_length})"
        )

    @property
    def num_packed_examples(self) -> int:
        return len(self.bins)

    def _pack_bin(self, pack_indices):
        input_ids = []
        labels = []
        sequence_ids = []
        position_ids = []

        for seq_idx, global_idx in enumerate(pack_indices):
            sequence = self.data[global_idx][self.column_name]
            if isinstance(sequence, str):
                sequence = list(sequence)

            remaining = self.max_length - len(input_ids)
            if remaining <= 2:
                break

            sequence = _truncate_for_packing(sequence, remaining)

            encoding = self.tokenizer(
                sequence,
                return_tensors=None,
                padding=False,
                truncation=False,
                is_split_into_words=True,
                add_special_tokens=False,
            )
            tokens = torch.tensor(encoding["input_ids"], dtype=torch.long)

            masked_tokens, seq_labels = self.noise_fn(tokens)

            seq_labels = seq_labels.masked_fill(
                masked_tokens == self.tokenizer.pad_token_id, -100
            )

            seq_len = len(masked_tokens)
            input_ids.extend(masked_tokens.tolist())
            labels.extend(seq_labels.tolist())
            sequence_ids.extend([seq_idx] * seq_len)
            position_ids.extend(range(seq_len))

        real_len = len(input_ids)
        pad_len = self.max_length - real_len
        pad_token_id = self.tokenizer.pad_token_id or 0

        input_ids.extend([pad_token_id] * pad_len)
        labels.extend([-100] * pad_len)
        sequence_ids.extend([seq_idx] * pad_len)
        position_ids.extend([0] * pad_len)
        padding_mask = [1] * real_len + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "sequence_ids": torch.tensor(sequence_ids, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padding_mask, dtype=torch.long),
        }

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else "main"
        scheduler = getattr(
            getattr(self.noise_fn, "mlm_probability_sampler", None),
            "scheduler",
            None,
        )
        bins_seen = 0

        if worker_info is not None:
            seed = self.seed + worker_info.seed
            random.seed(seed)
            torch.manual_seed(seed)
        else:
            seed = self.seed

        # Shard bins across workers
        total_bins = len(self.bins)
        if worker_info is not None:
            per_worker = total_bins // worker_info.num_workers
            start = worker_info.id * per_worker
            end = (
                start + per_worker
                if worker_info.id < worker_info.num_workers - 1
                else total_bins
            )
            worker_bin_indices = list(range(start, end))
            logger.info(
                f"Worker {worker_info.id}/{worker_info.num_workers} "
                f"({len(worker_bin_indices)} bins)"
            )
        else:
            worker_bin_indices = list(range(total_bins))

        # Resume skipping must apply within the epoch's shuffled order.
        per_worker_skip = self.skip_examples
        if worker_info is not None:
            per_worker_skip = self.skip_examples // worker_info.num_workers

        worker_total_bins = len(worker_bin_indices)
        worker_bin_indices = _shuffle_and_skip_bins(
            worker_bin_indices,
            per_worker_skip if not self._skipped else 0,
        )

        if per_worker_skip > 0 and not self._skipped:
            logger.info(
                f"Skipped {min(per_worker_skip, worker_total_bins)} bins, "
                f"{len(worker_bin_indices)} remaining"
            )
            self._skipped = True

        for bin_idx in worker_bin_indices:
            if (
                scheduler is not None
                and bins_seen % self.SCHEDULER_LOG_EVERY == 0
            ):
                logger.info(
                    f"Worker {worker_id} sees mlm_scheduler_step="
                    f"{scheduler.current_step.get()} "
                    f"after {bins_seen} packed bins"
                )
            yield self._pack_bin(self.bins[bin_idx])
            bins_seen += 1
