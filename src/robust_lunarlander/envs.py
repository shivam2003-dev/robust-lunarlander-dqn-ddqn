"""Environment construction and the assignment-specified action-failure wrapper."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np


def is_safe_landing(
    observation: np.ndarray,
    terminated: bool,
    truncated: bool,
) -> bool:
    """Return True only when every assignment safe-landing condition is met."""

    observation = np.asarray(observation)
    return bool(
        terminated
        and not truncated
        and observation[6] == 1
        and observation[7] == 1
        and abs(float(observation[2])) < 0.10
        and abs(float(observation[3])) < 0.10
        and abs(float(observation[4])) < 0.10
    )


class StochasticActionFailureWrapper(gym.Wrapper):
    """Apply hidden thruster failures, attempted-fuel cost, and safe-landing bonus.

    The selected and executed actions are retained only in private verification
    counters. The returned observation, termination flags, truncation flag, and
    info object come directly from the base environment. In particular, no failure
    indicator is leaked to the agent.

    A private RNG is used for actuator failures so drawing the failure event does
    not consume random numbers from LunarLander's own transition generator.
    """

    def __init__(
        self,
        env: gym.Env,
        failure_probability: float = 0.15,
        attempted_thruster_penalty: float = 0.3,
        safe_landing_bonus: float = 50.0,
    ) -> None:
        """Initialize the wrapper while preserving the base action and observation spaces."""

        super().__init__(env)
        if not 0.0 <= failure_probability <= 1.0:
            raise ValueError("failure_probability must lie in [0, 1].")
        if attempted_thruster_penalty < 0.0:
            raise ValueError("attempted_thruster_penalty must be non-negative.")
        self.failure_probability = float(failure_probability)
        self.attempted_thruster_penalty = float(attempted_thruster_penalty)
        self.safe_landing_bonus = float(safe_landing_bonus)
        self._failure_rng = np.random.default_rng()
        self._verification_counters = {
            "attempted_thruster_actions": 0,
            "misfired_thruster_actions": 0,
            "executed_thruster_actions": 0,
            "applied_fuel_penalties": 0,
            "safe_landing_bonuses": 0,
        }

    @property
    def verification_counters(self) -> dict[str, int]:
        """Return a defensive copy of private counters for external verification."""

        return deepcopy(self._verification_counters)

    def clear_verification_counters(self) -> None:
        """Reset private counters without changing any agent-visible state."""

        for key in self._verification_counters:
            self._verification_counters[key] = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset LunarLander and independently seed the hidden failure process."""

        if seed is not None:
            # A distinct SeedSequence component avoids coupling failure draws to
            # the base environment even though both originate from one run seed.
            failure_seed = np.random.SeedSequence([int(seed), 0xFA11])
            self._failure_rng = np.random.default_rng(failure_seed)
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one assignment-compliant transition in the specified order."""

        selected_action = int(action)
        if not self.action_space.contains(selected_action):
            raise ValueError(f"Invalid action {selected_action!r} for {self.action_space}.")

        # Only a selected thruster command can fail. Action zero is never replaced.
        executed_action = selected_action
        if selected_action in (1, 2, 3):
            self._verification_counters["attempted_thruster_actions"] += 1
            if self._failure_rng.random() < self.failure_probability:
                executed_action = 0
                self._verification_counters["misfired_thruster_actions"] += 1
            else:
                self._verification_counters["executed_thruster_actions"] += 1

        # Delegate the final action to the unchanged LunarLander transition logic.
        observation, base_reward, terminated, truncated, info = self.env.step(executed_action)

        fuel_cost = self.attempted_thruster_penalty if selected_action != 0 else 0.0
        self._verification_counters["applied_fuel_penalties"] += int(selected_action != 0)
        landing_bonus = (
            self.safe_landing_bonus if is_safe_landing(observation, terminated, truncated) else 0.0
        )
        self._verification_counters["safe_landing_bonuses"] += int(landing_bonus != 0.0)
        modified_reward = float(base_reward) - fuel_cost + landing_bonus

        # Return the base info object exactly; diagnostics remain private.
        return observation, modified_reward, terminated, truncated, info


def make_environment(
    *,
    modified: bool,
    failure_probability: float = 0.15,
    attempted_thruster_penalty: float = 0.3,
    safe_landing_bonus: float = 50.0,
    render_mode: str | None = None,
) -> gym.Env:
    """Create either original LunarLander-v3 or the specified modified variant."""

    base_environment = gym.make("LunarLander-v3", render_mode=render_mode)
    if not modified:
        return base_environment
    return StochasticActionFailureWrapper(
        base_environment,
        failure_probability=failure_probability,
        attempted_thruster_penalty=attempted_thruster_penalty,
        safe_landing_bonus=safe_landing_bonus,
    )
