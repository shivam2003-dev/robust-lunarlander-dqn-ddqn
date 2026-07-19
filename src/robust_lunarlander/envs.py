"""Environment construction and the assignment-specified action-failure wrapper."""

from __future__ import annotations

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

    The selected action is retained only within step. The returned observation,
    termination flags, truncation flag, and info object come directly from the base
    environment. In particular, no failure indicator is leaked to the agent.
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

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one assignment-compliant transition in the specified order."""

        selected_action = int(action)
        if not self.action_space.contains(selected_action):
            raise ValueError(f"Invalid action {selected_action!r} for {self.action_space}.")

        executed_action = selected_action
        if selected_action in (1, 2, 3) and self.np_random.random() < self.failure_probability:
            executed_action = 0

        observation, base_reward, terminated, truncated, info = self.env.step(executed_action)

        fuel_cost = self.attempted_thruster_penalty if selected_action != 0 else 0.0
        landing_bonus = (
            self.safe_landing_bonus if is_safe_landing(observation, terminated, truncated) else 0.0
        )
        modified_reward = float(base_reward) - fuel_cost + landing_bonus

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
