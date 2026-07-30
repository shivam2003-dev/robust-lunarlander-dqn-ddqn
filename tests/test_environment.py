"""Fast specification tests for the stochastic action-failure wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from robust_lunarlander.envs import StochasticActionFailureWrapper, is_safe_landing
from robust_lunarlander.verification import ScriptedTerminalEnvironment


def safe_observation() -> np.ndarray:
    """Provide one terminal state that satisfies all safe-landing predicates."""

    return np.array([0.0, 0.0, 0.02, -0.03, 0.04, 0.0, 1.0, 1.0], dtype=np.float32)


@pytest.mark.parametrize(
    ("index", "value"),
    [(2, 0.10), (2, -0.10), (3, 0.10), (4, -0.10), (6, 0.0), (7, 0.0)],
)
def test_safe_landing_uses_strict_thresholds(index: int, value: float) -> None:
    """Reject threshold equality and incomplete leg contact."""

    observation = safe_observation()
    observation[index] = value
    assert not is_safe_landing(observation, terminated=True, truncated=False)


def test_safe_landing_requires_terminal_non_truncated_transition() -> None:
    """Reject non-terminal and truncated states even when kinematics are safe."""

    observation = safe_observation()
    assert is_safe_landing(observation, terminated=True, truncated=False)
    assert not is_safe_landing(observation, terminated=False, truncated=False)
    assert not is_safe_landing(observation, terminated=True, truncated=True)


def test_misfired_thruster_is_still_penalized_and_hidden() -> None:
    """Charge attempted fuel, execute action zero, and preserve the base info object."""

    base = ScriptedTerminalEnvironment(
        safe_observation(),
        base_reward=7.0,
        terminated=False,
    )
    action_space = base.action_space
    observation_space = base.observation_space
    wrapped = StochasticActionFailureWrapper(base, failure_probability=1.0)
    wrapped.reset(seed=148)
    _, reward, terminated, truncated, info = wrapped.step(2)

    assert base.received_action == 0
    assert reward == pytest.approx(6.7)
    assert not terminated
    assert not truncated
    assert info is base.info_object
    assert set(info) == {"source"}
    assert wrapped.action_space is action_space
    assert wrapped.observation_space is observation_space
    assert wrapped.verification_counters == {
        "attempted_thruster_actions": 1,
        "misfired_thruster_actions": 1,
        "executed_thruster_actions": 0,
        "applied_fuel_penalties": 1,
        "safe_landing_bonuses": 0,
    }


def test_successful_thruster_receives_penalty_and_safe_bonus() -> None:
    """Apply the selected-action penalty and terminal safe-landing bonus together."""

    base = ScriptedTerminalEnvironment(safe_observation(), base_reward=7.0)
    wrapped = StochasticActionFailureWrapper(base, failure_probability=0.0)
    wrapped.reset(seed=148)
    _, reward, _, _, _ = wrapped.step(3)

    assert base.received_action == 3
    assert reward == pytest.approx(56.7)
    assert wrapped.verification_counters["attempted_thruster_actions"] == 1
    assert wrapped.verification_counters["executed_thruster_actions"] == 1
    assert wrapped.verification_counters["safe_landing_bonuses"] == 1


def test_do_nothing_draws_no_fuel_penalty() -> None:
    """Leave base reward unchanged for action zero when no safe bonus applies."""

    base = ScriptedTerminalEnvironment(
        safe_observation(),
        base_reward=7.0,
        terminated=False,
    )
    wrapped = StochasticActionFailureWrapper(base)
    wrapped.reset(seed=148)
    _, reward, _, _, _ = wrapped.step(0)
    assert reward == pytest.approx(7.0)
    assert wrapped.verification_counters["applied_fuel_penalties"] == 0


def test_safe_terminated_landing_adds_exactly_fifty() -> None:
    """Add exactly 50 to the base reward when every safe predicate holds."""

    base = ScriptedTerminalEnvironment(safe_observation(), base_reward=7.0)
    wrapped = StochasticActionFailureWrapper(base)
    wrapped.reset(seed=148)
    _, reward, _, _, _ = wrapped.step(0)
    assert reward == pytest.approx(57.0)


@pytest.mark.parametrize(
    ("index", "value", "terminated", "truncated"),
    [
        (2, 0.10, True, False),
        (3, 0.10, True, False),
        (4, -0.10, True, False),
        (6, 0.0, True, False),
        (7, 0.0, True, False),
        (2, 0.02, True, True),
    ],
)
def test_unsafe_or_truncated_transition_gets_no_bonus(
    index: int,
    value: float,
    terminated: bool,
    truncated: bool,
) -> None:
    """Reject every requested missing-contact, velocity, angle, and truncation case."""

    observation = safe_observation()
    observation[index] = value
    base = ScriptedTerminalEnvironment(
        observation,
        base_reward=7.0,
        terminated=terminated,
        truncated=truncated,
    )
    wrapped = StochasticActionFailureWrapper(base)
    wrapped.reset(seed=148)
    _, reward, _, _, _ = wrapped.step(0)
    assert reward == pytest.approx(7.0)
    assert wrapped.verification_counters["safe_landing_bonuses"] == 0


def test_invalid_action_is_rejected_before_base_step() -> None:
    """Fail explicitly for actions outside the unchanged Discrete(4) space."""

    base = ScriptedTerminalEnvironment(safe_observation())
    wrapped = StochasticActionFailureWrapper(base)
    wrapped.reset(seed=148)
    with pytest.raises(ValueError, match="Invalid action"):
        wrapped.step(4)
    assert base.received_action is None
