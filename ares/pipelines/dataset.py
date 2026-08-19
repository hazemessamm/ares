import hashlib
import logging
from torch.utils.data import IterableDataset, get_worker_info

from ares.preprocessing.truncation import random_truncation
from ares.tokenization import AresProteinTokenizer
from collections import defaultdict
import random

import torch
from datasets import load_dataset

from ares.preprocessing.noising import SequenceCorruptor

logger = logging.getLogger(__name__)


def group_batch(batch):
    output = defaultdict(list)
    for example in batch:
        for k, v in example.items():
            output[k].append(v)
    return output


class AresCollator:
    def __call__(self, batch):
        batch = group_batch(batch)
        for k, v in batch.items():
            batch[k] = torch.cat(v, dim=0)
        return dict(batch)


def add_boundary_tokens(
    truncated_sequence: list[str],
    add_cls: bool,
    add_eos: bool,
) -> list[str]:
    """Prepend <cls> and/or append <eos> based on fragment position."""
    if add_cls:
        truncated_sequence = ["<cls>"] + truncated_sequence
    if add_eos:
        truncated_sequence = truncated_sequence + ["<eos>"]
    return truncated_sequence


class HFDataset(IterableDataset):
    SCHEDULER_LOG_EVERY = 10000

    def __init__(
        self,
        repo_id: str,
        column_name: str,
        max_length: int,
        noise_fn: SequenceCorruptor,
        tokenizer: AresProteinTokenizer,
        split: str = "train",
        seed: int = 42,
        skip_examples: int = 0,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.noise_fn = noise_fn
        self.tokenizer = tokenizer
        self.split = split
        self.max_length = max_length
        self.column_name = column_name
        self.seed = seed
        self.epoch_index = 0
        self.skip_examples = skip_examples
        self._skipped = False
        self.special_token_ids = list(tokenizer.all_special_ids)
        self.hf_ds = load_dataset(repo_id, streaming=False, split=split)

    def _seed_from_components(self, *components: int) -> int:
        seed_material = ":".join(str(component) for component in components)
        return int.from_bytes(
            hashlib.blake2b(
                seed_material.encode("utf-8"),
                digest_size=8,
            ).digest(),
            byteorder="big",
        )

    def _get_worker_shuffle_seed(self, worker_info) -> int:
        worker_id = worker_info.id if worker_info is not None else 0
        return self._seed_from_components(
            self.seed,
            self.epoch_index,
            worker_id,
        )

    def _get_per_worker_skip(self, worker_info) -> int:
        if worker_info is None:
            return self.skip_examples

        base_skip = self.skip_examples // worker_info.num_workers
        remainder = self.skip_examples % worker_info.num_workers
        return base_skip + int(worker_info.id < remainder)

    def _get_example_seed(self, worker_info, example_position: int) -> int:
        worker_id = worker_info.id if worker_info is not None else 0
        return self._seed_from_components(
            self.seed,
            self.epoch_index,
            worker_id,
            example_position,
        )

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else "main"

        if worker_info is None:
            iterable_ds = self.hf_ds
        else:
            num_shards = worker_info.num_workers
            index = worker_info.id
            logger.info(
                f"Worker {worker_info.id}/{worker_info.num_workers} "
                f"(shard {index}/{num_shards})"
            )
            seed = self.seed + worker_info.id
            iterable_ds = self.hf_ds.shard(
                num_shards=num_shards,
                index=index,
            )

        per_worker_skip = self._get_per_worker_skip(worker_info)
        shuffle_seed = self._get_worker_shuffle_seed(worker_info)
        shuffled_ds = iterable_ds.shuffle(seed=shuffle_seed)
        scheduler = getattr(
            getattr(self.noise_fn, "mlm_probability_sampler", None),
            "scheduler",
            None,
        )
        examples_seen = 0

        if per_worker_skip > 0 and not self._skipped:
            total = len(shuffled_ds)
            if per_worker_skip < total:
                shuffled_ds = shuffled_ds.select(range(per_worker_skip, total))
                logger.info(
                    f"Skipped {per_worker_skip} examples, "
                    f"{len(shuffled_ds)} remaining"
                )
            self._skipped = True

        for example_position, example in enumerate(
            shuffled_ds,
            start=per_worker_skip,
        ):
            if (
                scheduler is not None
                and examples_seen % self.SCHEDULER_LOG_EVERY == 0
            ):
                logger.info(
                    f"Worker {worker_id} sees mlm_scheduler_step="
                    f"{scheduler.current_step.get()} "
                    f"after {examples_seen} examples"
                )
            sequence = example[self.column_name]

            if isinstance(sequence, str):
                sequence = list(sequence)

            if len(sequence) < 1:
                continue

            # Truncate using the full max_length budget, then make room
            # for boundary tokens based on where the fragment sits.
            example_seed = self._get_example_seed(
                worker_info,
                example_position,
            )
            example_rng = random.Random(example_seed)
            example_torch_generator = torch.Generator(device="cpu")
            example_torch_generator.manual_seed(example_seed)
            truncated, start_idx, end_idx, _ = random_truncation(
                sequence,
                self.max_length,
                rng=example_rng,
            )

            add_cls = start_idx == 0
            add_eos = end_idx == len(sequence)
            num_special = int(add_cls) + int(add_eos)

            target_aa_len = self.max_length - num_special
            if len(truncated) > target_aa_len:
                excess = len(truncated) - target_aa_len
                if add_eos and not add_cls:
                    truncated = truncated[excess:]
                    start_idx += excess
                else:
                    truncated = truncated[:target_aa_len]
                    end_idx = start_idx + target_aa_len

            truncated = add_boundary_tokens(truncated, add_cls, add_eos)

            encoded_inputs = self.tokenizer(
                truncated,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                is_split_into_words=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

            input_ids = encoded_inputs["input_ids"]
            attention_mask = encoded_inputs["attention_mask"]

            input_ids, labels = self.noise_fn(
                input_ids,
                excluded_ids=self.special_token_ids,
                rng=example_rng,
                torch_generator=example_torch_generator,
            )

            labels = labels.masked_fill(
                input_ids == self.tokenizer.pad_token_id,
                -100,
            )

            yield {
                "input_ids": input_ids.squeeze(0),
                "attention_mask": attention_mask.squeeze(0),
                "labels": labels.squeeze(0),
            }
            examples_seen += 1
