from __future__ import annotations

from typing import List

import multiprocessing as mp
import random


class Scheduler:
    def step(self):
        raise NotImplementedError

    def sample(self) -> List[float]:
        raise NotImplementedError


class SharedCounter:
    def __init__(self, initial_value: int, final_value: int):
        self.value = mp.Value("i", initial_value)
        self.final_value = final_value

    def step(self):
        with self.value.get_lock():
            current_value = self.value.value
            new_value = min(current_value + 1, self.final_value)
            self.value.value = new_value

    def get(self):
        return self.value.value


class LinearScheduler(Scheduler):
    """
    Linearly interpolates between initial and final weights for
    MLM probabilities over a specified number of training steps.

    This allows for a curriculum learning approach where the distribution of
    masking probabilities can change during training.
    """

    def __init__(
        self,
        initial_weights: List[float],
        final_weights: List[float],
        warmup_steps: int,
    ):
        if len(initial_weights) != len(final_weights):
            raise ValueError(
                "Initial and final weights must have the same length."
            )

        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.warmup_steps = warmup_steps
        self.current_step = SharedCounter(0, warmup_steps)

    def step(self):
        """Advance the scheduler by one step."""
        self.current_step.step()

    def sample(self) -> List[float]:
        """Get the interpolated weights for the current step."""
        progress = self.current_step.get() / self.warmup_steps
        progress = min(progress, 1.0)

        return [
            iw + progress * (fw - iw)
            for iw, fw in zip(self.initial_weights, self.final_weights)
        ]


class StagedLinearScheduler(Scheduler):
    """
    Staged curriculum scheduler that introduces masking difficulties
    sequentially. Each stage introduces a new difficulty level, ramping
    its weight linearly from 0 to equal weight with existing difficulties.
    """

    def __init__(
        self,
        difficulties: List[float],
        warmup_steps: int,
    ):
        if not difficulties:
            raise ValueError("difficulties must not be empty.")
        if warmup_steps < len(difficulties):
            raise ValueError(
                "warmup_steps must be at least the number of difficulties."
            )

        # Preserve the caller's curriculum order rather than re-sorting it.
        self.difficulties = list(difficulties)
        self.num_stages = len(difficulties)
        self.warmup_steps = warmup_steps
        self.current_step = SharedCounter(0, warmup_steps)

    def step(self):
        self.current_step.step()

    def get_weights(self) -> List[float]:
        current_step = min(self.current_step.get(), self.warmup_steps)

        weights = []
        for i, _ in enumerate(self.difficulties):
            stage_start = (i * self.warmup_steps) // self.num_stages
            stage_end = ((i + 1) * self.warmup_steps) // self.num_stages

            if current_step >= stage_end:
                weights.append(1.0)
            elif current_step <= stage_start:
                weights.append(0.0)
            else:
                stage_progress = (current_step - stage_start) / (
                    stage_end - stage_start
                )
                weights.append(stage_progress)

        total = sum(weights)
        if total == 0:
            weights[0] = 1.0
            total = 1.0
        return [w / total for w in weights]

    def sample(self) -> List[float]:
        """Return the staged weights for all masking difficulties."""
        return self.get_weights()

    def sample_difficulty(self) -> float:
        """Sample a single masking difficulty based on the staged weights."""
        weights = self.get_weights()
        return random.choices(self.difficulties, weights=weights, k=1)[0]


class EMAScheduler(Scheduler):
    """
    Updates weights using a bias-corrected exponential moving average.
    This provides a smooth, exponential transition from initial to final
    weights. The update rule is equivalent to an exponential interpolation:
    w_t = beta^t * w_0 + (1 - beta^t) * w_final
    Where t is the effective step count.
    """

    def __init__(
        self,
        initial_weights: List[float],
        final_weights: List[float],
        beta: float = 0.999,
        multiplier: float = 1.0,
    ):
        if len(initial_weights) != len(final_weights):
            raise ValueError(
                "Initial and final weights must have the same length."
            )
        if not (0.0 < beta < 1.0):
            raise ValueError("Beta must be between 0 and 1.")

        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.beta = beta
        self.current_step = SharedCounter(0, 2**31 - 1)
        self.multiplier = multiplier

    def step(self):
        """Advance the scheduler by one step."""
        self.current_step.step()

    def sample(self) -> List[float]:
        """Get the interpolated weights for the current step."""
        effective_step = self.current_step.get() * self.multiplier
        decay_factor = self.beta**effective_step
        return [
            decay_factor * iw + (1 - decay_factor) * fw
            for iw, fw in zip(self.initial_weights, self.final_weights)
        ]
