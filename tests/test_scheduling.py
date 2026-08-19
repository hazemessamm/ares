import pytest
from ares.preprocessing.scheduling import LinearScheduler, EMAScheduler


class TestLinearScheduler:
    """Interpolates initial -> final weights over ``warmup_steps``.

    ``step()`` advances the counter and returns None; read the weights with
    ``sample()``. The step counter clamps at ``warmup_steps``.
    """

    def test_initial_weights(self):
        sched = LinearScheduler(
            initial_weights=[0.0, 0.0],
            final_weights=[1.0, 1.0],
            warmup_steps=10,
        )
        assert sched.sample() == [0.0, 0.0]

    def test_final_weights_after_warmup(self):
        sched = LinearScheduler(
            initial_weights=[0.0, 0.0],
            final_weights=[1.0, 1.0],
            warmup_steps=10,
        )
        for _ in range(10):
            sched.step()
        assert sched.sample() == pytest.approx([1.0, 1.0])

    def test_midpoint(self):
        sched = LinearScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            warmup_steps=10,
        )
        for _ in range(5):
            sched.step()
        assert sched.sample()[0] == pytest.approx(0.5)

    def test_monotonic_interpolation(self):
        sched = LinearScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            warmup_steps=10,
        )
        prev = sched.sample()[0]
        for _ in range(10):
            sched.step()
            current = sched.sample()[0]
            assert current >= prev
            prev = current

    def test_clamps_past_warmup_steps(self):
        sched = LinearScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            warmup_steps=5,
        )
        for _ in range(20):
            sched.step()
        assert sched.sample()[0] == pytest.approx(1.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            LinearScheduler(
                initial_weights=[0.0],
                final_weights=[1.0, 2.0],
                warmup_steps=10,
            )

    def test_step_returns_none(self):
        """step() only advances the counter; weights come from sample()."""
        sched = LinearScheduler(
            initial_weights=[0.0],
            final_weights=[1.0],
            warmup_steps=10,
        )
        assert sched.step() is None
        assert sched.sample()[0] == pytest.approx(0.1)


class TestEMAScheduler:
    def test_initial_state(self):
        sched = EMAScheduler(
            initial_weights=[1.0, 1.0],
            final_weights=[0.0, 0.0],
        )
        weights = sched.sample()
        assert weights == [1.0, 1.0]

    def test_converges_to_final(self):
        sched = EMAScheduler(
            initial_weights=[1.0],
            final_weights=[0.0],
            beta=0.9,
        )
        for _ in range(1000):
            sched.step()
        weights = sched.sample()
        assert abs(weights[0]) < 1e-6

    def test_monotonic_transition(self):
        sched = EMAScheduler(
            initial_weights=[1.0],
            final_weights=[0.0],
            beta=0.99,
        )
        prev = sched.sample()[0]
        for _ in range(50):
            sched.step()
            w = sched.sample()
            assert w[0] <= prev
            prev = w[0]

    def test_multiplier_affects_speed(self):
        sched_slow = EMAScheduler(
            initial_weights=[1.0],
            final_weights=[0.0],
            beta=0.99,
            multiplier=0.5,
        )
        sched_fast = EMAScheduler(
            initial_weights=[1.0],
            final_weights=[0.0],
            beta=0.99,
            multiplier=2.0,
        )
        for _ in range(10):
            sched_slow.step()
            sched_fast.step()
        assert sched_fast.sample()[0] < sched_slow.sample()[0]

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            EMAScheduler(
                initial_weights=[0.0],
                final_weights=[1.0, 2.0],
            )

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError):
            EMAScheduler(
                initial_weights=[0.0],
                final_weights=[1.0],
                beta=1.0,
            )
        with pytest.raises(ValueError):
            EMAScheduler(
                initial_weights=[0.0],
                final_weights=[1.0],
                beta=0.0,
            )
