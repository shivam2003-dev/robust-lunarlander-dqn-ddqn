"""Experimental and boundary verification of the modified environment."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .envs import StochasticActionFailureWrapper, is_safe_landing


class ActionRewardSpy(gym.Wrapper):
    """Record base-environment inputs and outputs outside the agent-facing wrapper."""

    def __init__(self, env: gym.Env) -> None:
        """Initialize an empty last-transition record."""

        super().__init__(env)
        self.last: dict[str, Any] | None = None

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Forward the action and retain the unmodified response for verification."""

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last = {
            "executed_action": int(action),
            "observation": np.asarray(observation).copy(),
            "base_reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }
        return observation, reward, terminated, truncated, info


class ScriptedTerminalEnvironment(gym.Env):
    """Return one specified terminal transition for controlled reward tests."""

    metadata: dict[str, Any] = {}

    def __init__(
        self,
        observation: np.ndarray,
        *,
        base_reward: float = 7.0,
        terminated: bool = True,
        truncated: bool = False,
    ) -> None:
        """Store the exact response that step must return."""

        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.response_observation = np.asarray(observation, dtype=np.float32)
        self.base_reward = float(base_reward)
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.received_action: int | None = None
        self.info_object = {"source": "scripted-base"}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset Gymnasium's random generator and return a neutral observation."""

        super().reset(seed=seed)
        del options
        return np.zeros(8, dtype=np.float32), {}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Return the configured response while recording the executed action."""

        self.received_action = int(action)
        return (
            self.response_observation.copy(),
            self.base_reward,
            self.terminated,
            self.truncated,
            self.info_object,
        )


@dataclass(frozen=True)
class BoundaryCase:
    """Describe one controlled safe-landing or fuel-penalty check."""

    name: str
    observation: list[float]
    terminated: bool
    truncated: bool
    selected_action: int
    failure_probability: float
    expected_bonus: float
    expected_executed_action: int


def _safe_observation() -> np.ndarray:
    """Return an observation satisfying all five state thresholds."""

    return np.array([0.0, 0.0, 0.02, -0.03, 0.04, 0.0, 1.0, 1.0], dtype=np.float32)


def controlled_boundary_cases() -> list[BoundaryCase]:
    """Enumerate positive, negative, boundary, truncation, and misfire cases."""

    safe = _safe_observation()
    cases: list[BoundaryCase] = [
        BoundaryCase("all_conditions_true", safe.tolist(), True, False, 0, 0.0, 50.0, 0),
        BoundaryCase("attempted_thruster_success", safe.tolist(), True, False, 2, 0.0, 50.0, 2),
        BoundaryCase("attempted_thruster_misfire", safe.tolist(), True, False, 2, 1.0, 50.0, 0),
    ]
    mutations = [
        ("not_terminated", 2, 0.02, False, False),
        ("episode_truncated", 2, 0.02, True, True),
        ("left_leg_absent", 6, 0.0, True, False),
        ("right_leg_absent", 7, 0.0, True, False),
        ("excess_horizontal_velocity", 2, 0.11, True, False),
        ("excess_vertical_velocity", 3, -0.11, True, False),
        ("excess_orientation_angle", 4, 0.11, True, False),
    ]
    for name, index, value, terminated, truncated in mutations:
        observation = safe.copy()
        observation[index] = value
        cases.append(
            BoundaryCase(
                name,
                observation.tolist(),
                terminated,
                truncated,
                0,
                0.0,
                0.0,
                0,
            )
        )
    return cases


def run_controlled_boundary_verification() -> pd.DataFrame:
    """Execute every controlled case and prove exact reward/action semantics."""

    rows: list[dict[str, Any]] = []
    for case in controlled_boundary_cases():
        base = ScriptedTerminalEnvironment(
            np.asarray(case.observation, dtype=np.float32),
            terminated=case.terminated,
            truncated=case.truncated,
        )
        wrapped = StochasticActionFailureWrapper(
            base,
            failure_probability=case.failure_probability,
        )
        wrapped.reset(seed=148)
        observation, reward, terminated, truncated, returned_info = wrapped.step(
            case.selected_action
        )
        counters = wrapped.verification_counters
        expected_penalty = 0.3 if case.selected_action != 0 else 0.0
        expected_reward = base.base_reward - expected_penalty + case.expected_bonus
        rows.append(
            {
                "case": case.name,
                "selected_action": case.selected_action,
                "executed_action": base.received_action,
                "expected_executed_action": case.expected_executed_action,
                "safe_landing_detected": is_safe_landing(observation, terminated, truncated),
                "expected_bonus": case.expected_bonus,
                "observed_reward": float(reward),
                "expected_reward": expected_reward,
                "fuel_penalty": expected_penalty,
                "counter_attempted_actions": counters["attempted_thruster_actions"],
                "counter_executed_actions": counters["executed_thruster_actions"],
                "counter_misfires": counters["misfired_thruster_actions"],
                "info_identity_preserved": returned_info is base.info_object,
                "passed": bool(
                    base.received_action == case.expected_executed_action
                    and math.isclose(float(reward), expected_reward, abs_tol=1e-7)
                    and returned_info is base.info_object
                    and counters["applied_fuel_penalties"] == int(case.selected_action != 0)
                    and counters["safe_landing_bonuses"] == int(case.expected_bonus != 0.0)
                ),
            }
        )
        wrapped.close()
    return pd.DataFrame(rows)


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return a 95 percent Wilson score interval for a binomial proportion."""

    if trials == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return centre - radius, centre + radius


def run_random_policy_verification(
    *,
    episodes: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run many random LunarLander episodes and compare selected/executed actions."""

    spy = ActionRewardSpy(gym.make("LunarLander-v3"))
    environment = StochasticActionFailureWrapper(spy)
    action_rng = np.random.default_rng(seed + 10_000)
    records: list[dict[str, Any]] = []
    total_steps = 0
    attempted_thrusters = 0
    misfires = 0
    penalty_checks = 0
    penalty_mismatches = 0
    info_identity_mismatches = 0
    unexpected_action_replacements = 0
    observed_safe_landings = 0

    for episode in range(1, episodes + 1):
        observation, _ = environment.reset(seed=seed + episode - 1)
        del observation
        terminated = truncated = False
        episode_step = 0
        while not (terminated or truncated):
            selected_action = int(action_rng.integers(environment.action_space.n))
            observation, reward, terminated, truncated, info = environment.step(selected_action)
            if spy.last is None:
                raise RuntimeError("The verification spy did not record a base transition.")

            executed_action = int(spy.last["executed_action"])
            safe = is_safe_landing(observation, terminated, truncated)
            expected_penalty = 0.3 if selected_action != 0 else 0.0
            expected_bonus = 50.0 if safe else 0.0
            expected_reward = float(spy.last["base_reward"]) - expected_penalty + expected_bonus
            penalty_ok = math.isclose(float(reward), expected_reward, abs_tol=1e-6)

            if selected_action != 0:
                attempted_thrusters += 1
                penalty_checks += 1
                misfires += int(executed_action == 0)
                penalty_mismatches += int(not penalty_ok)
                unexpected_action_replacements += int(executed_action not in (0, selected_action))
            else:
                unexpected_action_replacements += int(executed_action != 0)
            info_identity_mismatches += int(info is not spy.last["info"])
            observed_safe_landings += int(safe)
            total_steps += 1
            episode_step += 1

            if len(records) < 500:
                records.append(
                    {
                        "episode": episode,
                        "step": episode_step,
                        "selected_action": selected_action,
                        "executed_action": executed_action,
                        "misfire": selected_action != 0 and executed_action == 0,
                        "base_reward": float(spy.last["base_reward"]),
                        "modified_reward": float(reward),
                        "expected_fuel_penalty": expected_penalty,
                        "safe_landing_bonus": expected_bonus,
                        "reward_check_passed": penalty_ok,
                    }
                )

    internal_counters = environment.verification_counters
    environment.close()
    interval_low, interval_high = _wilson_interval(misfires, attempted_thrusters)
    observed_rate = misfires / attempted_thrusters
    summary = {
        "episodes": episodes,
        "total_steps": total_steps,
        "attempted_thruster_actions": attempted_thrusters,
        "misfired_thruster_actions": misfires,
        "observed_misfire_rate": observed_rate,
        "target_misfire_rate": 0.15,
        "misfire_rate_absolute_error": abs(observed_rate - 0.15),
        "misfire_rate_wilson_95_low": interval_low,
        "misfire_rate_wilson_95_high": interval_high,
        "target_inside_wilson_interval": interval_low <= 0.15 <= interval_high,
        "fuel_penalty_checks": penalty_checks,
        "fuel_penalty_count": internal_counters["applied_fuel_penalties"],
        "fuel_penalty_count_matches_attempts": (
            internal_counters["applied_fuel_penalties"] == attempted_thrusters
        ),
        "fuel_penalty_mismatches": penalty_mismatches,
        "internal_attempt_count_matches": (
            internal_counters["attempted_thruster_actions"] == attempted_thrusters
        ),
        "internal_misfire_count_matches": (
            internal_counters["misfired_thruster_actions"] == misfires
        ),
        "executed_thruster_actions": internal_counters["executed_thruster_actions"],
        "safe_landing_bonus_count": internal_counters["safe_landing_bonuses"],
        "safe_landing_bonus_count_matches": (
            internal_counters["safe_landing_bonuses"] == observed_safe_landings
        ),
        "info_identity_mismatches": info_identity_mismatches,
        "unexpected_action_replacements": unexpected_action_replacements,
        "safe_landings_observed_under_random_policy": observed_safe_landings,
        "passed": bool(
            interval_low <= 0.15 <= interval_high
            and penalty_mismatches == 0
            and internal_counters["applied_fuel_penalties"] == attempted_thrusters
            and internal_counters["attempted_thruster_actions"] == attempted_thrusters
            and internal_counters["misfired_thruster_actions"] == misfires
            and internal_counters["safe_landing_bonuses"] == observed_safe_landings
            and info_identity_mismatches == 0
            and unexpected_action_replacements == 0
        ),
    }
    return summary, pd.DataFrame(records)


def save_verification_artifacts(
    *,
    output_dir: Path,
    episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Run all checks and write machine-readable evidence files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_results = run_controlled_boundary_verification()
    random_summary, random_samples = run_random_policy_verification(
        episodes=episodes,
        seed=seed,
    )
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": seed,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "gymnasium_version": gym.__version__,
        "random_policy": random_summary,
        "controlled_boundary_cases": {
            "count": int(len(boundary_results)),
            "passed": int(boundary_results["passed"].sum()),
            "all_passed": bool(boundary_results["passed"].all()),
        },
        "overall_passed": bool(random_summary["passed"] and boundary_results["passed"].all()),
    }
    (output_dir / "wrapper_verification.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    boundary_results.to_csv(output_dir / "controlled_boundary_cases.csv", index=False)
    random_samples.to_csv(output_dir / "random_policy_transition_sample.csv", index=False)
    return summary


def main() -> None:
    """Parse command-line options and produce the complete verification bundle."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=148)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/verification"),
    )
    args = parser.parse_args()
    summary = save_verification_artifacts(
        output_dir=args.output_dir,
        episodes=args.episodes,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    if not summary["overall_passed"]:
        raise SystemExit("Wrapper verification failed.")


if __name__ == "__main__":
    main()
