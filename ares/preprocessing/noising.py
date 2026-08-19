from __future__ import annotations

import typing

import torch
from transformers import PreTrainedTokenizer

from ares.preprocessing.utils import MLMProbabilitySampler


class SequenceCorruptor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mlm_probability_sampler: MLMProbabilitySampler,
    ):
        self.tokenizer = tokenizer
        self.special_token_ids = list(tokenizer.all_special_ids)
        self.aa_token_ids = torch.tensor(
            [
                _id
                for _id in self.tokenizer.get_vocab().values()
                if _id not in self.special_token_ids
            ],
            dtype=torch.long,
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.cls_token_id
        self.mlm_probability_sampler = mlm_probability_sampler

    def __call__(
        self,
        sequences: torch.LongTensor,
        excluded_ids: typing.List[int] | None = None,
        rng=None,
        torch_generator: torch.Generator | None = None,
    ) -> typing.Tuple[torch.LongTensor, torch.LongTensor]:
        # Sample probabilities
        (
            mlm_prob,
            masking_prob,
            mutation_prob,
        ) = self.mlm_probability_sampler.sample(rng=rng)

        if excluded_ids is None:
            excluded_ids = [
                self.pad_token_id,
                self.eos_token_id,
                self.bos_token_id,
            ]

        device = sequences.device
        sequences = sequences.clone()
        labels = sequences.clone()

        # 1. Create MLM Mask (avoiding torch.isin for speed)
        # Check against individual IDs is faster for small lists on TPU
        is_excluded = torch.zeros_like(sequences, dtype=torch.bool)
        for _id in excluded_ids:
            is_excluded |= sequences == _id

        # Sample which tokens to participate in MLM
        masked_indices = (
            torch.rand(
                sequences.shape,
                device=device,
                generator=torch_generator,
            )
            < mlm_prob
        ) & ~is_excluded

        # Set non-MLM positions to -100
        labels[~masked_indices] = -100

        # 2. Decide sub-actions for the masked_indices
        random_roll = torch.rand(
            sequences.shape,
            device=device,
            generator=torch_generator,
        )

        # Action: Replace with [MASK] (usually 80%)
        indices_to_mask = masked_indices & (random_roll < masking_prob)
        sequences[indices_to_mask] = self.mask_token_id

        # Action: Replace with Random (usually 10%)
        # Logic: If roll is between masking_prob and
        # (masking_prob + mutation_prob)
        # Only mutate tokens that were selected for MLM but NOT already masked
        indices_to_mutate = (
            masked_indices
            & ~indices_to_mask
            & (random_roll >= masking_prob)
            & (random_roll < (masking_prob + mutation_prob))
        )

        if indices_to_mutate.any():
            aa_token_ids = self.aa_token_ids.to(device)
            # Generate random indices for all positions,
            # then select only mutated ones
            random_indices = torch.randint(
                0,
                len(aa_token_ids),
                sequences.shape,
                device=device,
                generator=torch_generator,
            )
            sequences[indices_to_mutate] = aa_token_ids[
                random_indices[indices_to_mutate]
            ]
        # Action: Keep same (the remaining 10%) -
        # no code needed, labels already set
        return sequences, labels
