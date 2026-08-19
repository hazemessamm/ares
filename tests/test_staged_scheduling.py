from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

SCHEDULING_PATH = (
    Path(__file__).resolve().parents[1]
    / "ares"
    / "preprocessing"
    / "scheduling.py"
)
SCHEDULING_SPEC = spec_from_file_location("staged_scheduling", SCHEDULING_PATH)
scheduling = module_from_spec(SCHEDULING_SPEC)
assert SCHEDULING_SPEC is not None and SCHEDULING_SPEC.loader is not None
SCHEDULING_SPEC.loader.exec_module(scheduling)
StagedLinearScheduler = scheduling.StagedLinearScheduler


class TestStagedLinearScheduler:
    def test_starts_with_only_first_difficulty_active(self):
        scheduler = StagedLinearScheduler(
            difficulties=[0.15, 0.3, 0.45],
            warmup_steps=9,
        )

        assert scheduler.sample() == [1.0, 0.0, 0.0]

    def test_last_stage_reaches_equal_weights_at_warmup(self):
        scheduler = StagedLinearScheduler(
            difficulties=[0.15, 0.3, 0.45],
            warmup_steps=10,
        )

        for _ in range(10):
            scheduler.step()

        assert scheduler.sample() == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_last_stage_progress_does_not_wrap(self):
        scheduler = StagedLinearScheduler(
            difficulties=[0.15, 0.3, 0.45],
            warmup_steps=10,
        )

        for _ in range(8):
            scheduler.step()
        weights_at_step_8 = scheduler.sample()

        scheduler.step()
        weights_at_step_9 = scheduler.sample()

        assert weights_at_step_9[2] > weights_at_step_8[2]

    def test_preserves_curriculum_order(self):
        scheduler = StagedLinearScheduler(
            difficulties=[0.45, 0.15, 0.3],
            warmup_steps=9,
        )

        assert scheduler.difficulties == [0.45, 0.15, 0.3]

    def test_requires_non_empty_difficulties(self):
        with pytest.raises(ValueError, match="must not be empty"):
            StagedLinearScheduler(difficulties=[], warmup_steps=1)

    def test_requires_at_least_one_step_per_difficulty(self):
        with pytest.raises(
            ValueError,
            match="at least the number of difficulties",
        ):
            StagedLinearScheduler(
                difficulties=[0.15, 0.3, 0.45],
                warmup_steps=2,
            )
