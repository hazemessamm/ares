from __future__ import annotations

import numpy as np
import random
import typing

if typing.TYPE_CHECKING:
    from ares.preprocessing.scheduling import Scheduler


def create_default_mlm_weights(
    lower: float,
    upper: float,
    increment: float,
) -> np.ndarray:
    num_steps = int(round((upper - lower) / increment)) + 1
    return np.linspace(lower, upper, num_steps)


class MLMProbabilitySampler:
    def __init__(
        self,
        mlm_probs: typing.List[float],
        masking_probs: typing.List[float],
        mutation_probs: typing.List[float],
        scheduler: Scheduler | None = None,
    ):
        assert len(mlm_probs) == len(masking_probs) == len(mutation_probs)
        self.mlm_probs = mlm_probs
        self.masking_probs = masking_probs
        self.mutation_probs = mutation_probs
        self.scheduler = scheduler
        self.prob_idx = None
        self.training = True

    def eval(self, idx: int):
        if idx < 0 or idx >= len(self.mlm_probs):
            raise ValueError(
                f"idx must be between 0 and {len(self.mlm_probs) - 1}"
            )
        self.prob_idx = idx
        self.training = False

    def train(self):
        self.training = True
        self.prob_idx = None

    def sample_from_scheduler(self, rng: random.Random | None = None):
        sampler_rng = rng or random
        weights = self.scheduler.sample()
        weights = [max(0.0, w) for w in weights]
        if sum(weights) > 0:
            idx = sampler_rng.choices(
                range(len(self.mlm_probs)), weights=weights
            )[0]
        else:
            idx = sampler_rng.randrange(len(self.mlm_probs))
        return idx

    def sample_randomly(self, rng: random.Random | None = None):
        sampler_rng = rng or random
        idx = sampler_rng.randrange(len(self.mlm_probs))
        return idx

    def sample_from_prob_idx(self):
        return self.prob_idx

    def sample(self, rng: random.Random | None = None):
        if self.scheduler is not None and self.training:
            idx = self.sample_from_scheduler(rng=rng)
        elif self.prob_idx is not None and not self.training:
            idx = self.sample_from_prob_idx()
        else:
            idx = self.sample_randomly(rng=rng)

        return (
            self.mlm_probs[idx],
            self.masking_probs[idx],
            self.mutation_probs[idx],
        )
