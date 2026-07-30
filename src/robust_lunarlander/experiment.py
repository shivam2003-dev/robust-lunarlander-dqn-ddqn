"""Reproducible training, evaluation, and evidence generation for all four agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .agent import Algorithm, ValueAgent
from .config import TrainingConfig, linear_epsilon
from .envs import StochasticActionFailureWrapper, is_safe_landing, make_environment
from .plotting import create_all_plots, summarize_experiments

EXPERIMENTS: tuple[tuple[Algorithm, bool], ...] = (
    ("dqn", False),
    ("ddqn", False),
    ("dqn", True),
    ("ddqn", True),
)


def experiment_name(algorithm: Algorithm, modified: bool) -> str:
    """Return a stable artifact name for one algorithm/environment pairing."""

    environment_name = "modified" if modified else "original"
    return f"{algorithm}_{environment_name}"


def set_reproducible_seeds(seed: int, environment: gym.Env | None = None) -> None:
    """Seed all supported RNGs and request deterministic PyTorch behavior.

    Box2D and some GPU kernels can still vary across platforms, so provenance is
    recorded and exact cross-machine bitwise equality is not promised.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if environment is not None:
        environment.action_space.seed(seed)
        environment.reset(seed=seed)


def collect_fixed_validation_states(
    config: TrainingConfig,
    destination: Path,
) -> np.ndarray:
    """Collect one deterministic random-policy state set shared by all agents."""

    if destination.exists():
        states = np.load(destination)["states"]
        expected_shape = (config.validation_state_count, 8)
        if states.shape != expected_shape:
            raise ValueError(
                f"Existing validation set has shape {states.shape}, expected {expected_shape}."
            )
        return states.astype(np.float32, copy=False)

    environment = make_environment(modified=False)
    set_reproducible_seeds(config.seed, environment)
    action_rng = np.random.default_rng(config.seed + 20_000)
    states: list[np.ndarray] = []
    episode = 0
    while len(states) < config.validation_state_count:
        observation, _ = environment.reset(seed=config.seed + episode)
        terminated = truncated = False
        while not (terminated or truncated) and len(states) < config.validation_state_count:
            states.append(np.asarray(observation, dtype=np.float32).copy())
            action = int(action_rng.integers(environment.action_space.n))
            observation, _, terminated, truncated, _ = environment.step(action)
        episode += 1
    environment.close()

    validation_states = np.stack(states)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, states=validation_states)
    return validation_states


def validation_set_sha256(validation_states: np.ndarray) -> str:
    """Hash the exact validation states to prove that all runs use one fixed set."""

    return hashlib.sha256(validation_states.tobytes()).hexdigest()


def train_agent(
    *,
    algorithm: Algorithm,
    modified: bool,
    config: TrainingConfig,
    validation_states: np.ndarray,
    force: bool = False,
) -> pd.DataFrame:
    """Train one agent and write per-episode metrics plus its final checkpoint."""

    name = experiment_name(algorithm, modified)
    metrics_dir = config.output_dir / "metrics"
    checkpoints_dir = config.output_dir / "checkpoints"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{name}.csv"
    checkpoint_path = checkpoints_dir / f"{name}.pt"

    required_columns = {
        "episode",
        "episode_reward",
        "average_predicted_q",
        "successful_safe_landing",
        "moving_safe_landing_rate_100",
        "attempted_thruster_activations",
        "executed_thruster_activations",
        "average_attempted_thruster_activations_per_episode",
        "episode_steps",
        "epsilon",
        "training_loss",
        "environment_type",
        "algorithm",
        "random_seed",
    }
    if metrics_path.exists() and checkpoint_path.exists() and not force:
        metrics = pd.read_csv(metrics_path)
        if len(metrics) == config.episodes and required_columns.issubset(metrics.columns):
            print(f"[{name}] Reusing complete artifacts ({config.episodes} episodes).", flush=True)
            return metrics
        print(f"[{name}] Existing artifacts use an incomplete schema; retraining.", flush=True)

    environment = make_environment(
        modified=modified,
        failure_probability=config.failure_probability,
        attempted_thruster_penalty=config.attempted_thruster_penalty,
        safe_landing_bonus=config.safe_landing_bonus,
    )
    set_reproducible_seeds(config.seed, environment)
    observation_size = int(np.prod(environment.observation_space.shape))
    action_count = int(environment.action_space.n)
    agent = ValueAgent(observation_size, action_count, algorithm, config)

    rows: list[dict[str, Any]] = []
    global_step = 0
    run_started = time.perf_counter()
    progress_dir = config.output_dir / "logs"
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / f"{name}.log"
    progress_path.write_text(
        "episode,total_reward,average_q,safe_rate_100,attempted,executed,"
        "steps,epsilon,training_loss\n",
        encoding="utf-8",
    )

    for episode in range(1, config.episodes + 1):
        episode_started = time.perf_counter()
        observation, _ = environment.reset(seed=config.seed + episode - 1)
        counters_before = (
            environment.verification_counters
            if isinstance(environment, StochasticActionFailureWrapper)
            else None
        )
        terminated = truncated = False
        episode_reward = 0.0
        attempted_thrusters = 0
        episode_losses: list[float] = []
        last_observation = np.asarray(observation, dtype=np.float32)
        episode_steps = 0

        for _ in range(config.max_steps_per_episode):
            episode_steps += 1
            epsilon = linear_epsilon(global_step, config)
            action = agent.select_action(observation, epsilon)
            attempted_thrusters += int(action != 0)
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            episode_done = terminated or truncated

            agent.replay.add(
                np.asarray(observation, dtype=np.float32),
                action,
                float(reward),
                np.asarray(next_observation, dtype=np.float32),
                bool(terminated),
                bool(truncated),
            )
            if (
                global_step >= config.learning_starts
                and global_step % config.train_frequency == 0
                and len(agent.replay) >= config.batch_size
            ):
                episode_losses.append(agent.update())

            episode_reward += float(reward)
            observation = next_observation
            last_observation = np.asarray(next_observation, dtype=np.float32)
            global_step += 1
            if episode_done:
                break

        successful_landing = is_safe_landing(last_observation, terminated, truncated)
        previous_successes = [int(row["successful_safe_landing"]) for row in rows[-99:]]
        moving_success_rate = float(np.mean([*previous_successes, int(successful_landing)]))
        average_q = agent.average_max_q(validation_states)
        if isinstance(environment, StochasticActionFailureWrapper):
            counters_after = environment.verification_counters
            assert counters_before is not None
            executed_thrusters = (
                counters_after["executed_thruster_actions"]
                - counters_before["executed_thruster_actions"]
            )
        else:
            executed_thrusters = attempted_thrusters
        average_attempts = (
            sum(float(row["attempted_thruster_activations"]) for row in rows) + attempted_thrusters
        ) / episode
        training_loss = float(np.mean(episode_losses)) if episode_losses else np.nan
        episode_seconds = time.perf_counter() - episode_started
        rows.append(
            {
                "experiment": name,
                "algorithm": algorithm.upper(),
                "environment_type": "Modified" if modified else "Original",
                "episode": episode,
                "episode_reward": episode_reward,
                "average_predicted_q": average_q,
                "successful_safe_landing": int(successful_landing),
                "moving_safe_landing_rate_100": moving_success_rate,
                "attempted_thruster_activations": attempted_thrusters,
                "executed_thruster_activations": executed_thrusters,
                "average_attempted_thruster_activations_per_episode": average_attempts,
                "episode_steps": episode_steps,
                "epsilon": linear_epsilon(global_step, config),
                "training_loss": training_loss,
                "global_step": global_step,
                "episode_seconds": episode_seconds,
                "random_seed": config.seed,
                "episode_seed": config.seed + episode - 1,
            }
        )

        loss_text = "nan" if np.isnan(training_loss) else f"{training_loss:.6f}"
        progress_line = (
            f"[{name}] episode {episode:4d}/{config.episodes} "
            f"reward={episode_reward:9.2f} q={average_q:8.2f} "
            f"safe100={moving_success_rate:6.1%} "
            f"attempted={attempted_thrusters:4d} executed={executed_thrusters:4d} "
            f"steps={episode_steps:4d} epsilon={linear_epsilon(global_step, config):.4f} "
            f"loss={loss_text}"
        )
        print(progress_line, flush=True)
        with progress_path.open("a", encoding="utf-8") as progress_file:
            progress_file.write(
                f"{episode},{episode_reward:.10f},{average_q:.10f},"
                f"{moving_success_rate:.10f},{attempted_thrusters},"
                f"{executed_thrusters},{episode_steps},"
                f"{linear_epsilon(global_step, config):.10f},{loss_text}\n"
            )

    environment.close()
    agent.save(str(checkpoint_path))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    elapsed = time.perf_counter() - run_started
    print(f"[{name}] completed in {elapsed / 60.0:.2f} minutes.", flush=True)
    return metrics


def evaluate_agent(
    *,
    algorithm: Algorithm,
    modified: bool,
    config: TrainingConfig,
) -> pd.DataFrame:
    """Evaluate one trained agent greedily over a shared deterministic seed list."""

    name = experiment_name(algorithm, modified)
    checkpoint_path = config.output_dir / "checkpoints" / f"{name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    environment = make_environment(
        modified=modified,
        failure_probability=config.failure_probability,
        attempted_thruster_penalty=config.attempted_thruster_penalty,
        safe_landing_bonus=config.safe_landing_bonus,
    )
    set_reproducible_seeds(config.seed, environment)
    observation_size = int(np.prod(environment.observation_space.shape))
    action_count = int(environment.action_space.n)
    agent = ValueAgent(observation_size, action_count, algorithm, config)
    agent.load(str(checkpoint_path))
    rows: list[dict[str, Any]] = []

    for episode in range(1, config.evaluation_episodes + 1):
        observation, _ = environment.reset(seed=config.seed + 100_000 + episode - 1)
        counters_before = (
            environment.verification_counters
            if isinstance(environment, StochasticActionFailureWrapper)
            else None
        )
        terminated = truncated = False
        episode_reward = 0.0
        attempted_thrusters = 0
        last_observation = np.asarray(observation, dtype=np.float32)
        episode_steps = 0

        for _ in range(config.max_steps_per_episode):
            episode_steps += 1
            action = agent.select_action(observation, epsilon=0.0)
            attempted_thrusters += int(action != 0)
            observation, reward, terminated, truncated, _ = environment.step(action)
            episode_reward += float(reward)
            last_observation = np.asarray(observation, dtype=np.float32)
            if terminated or truncated:
                break

        if isinstance(environment, StochasticActionFailureWrapper):
            counters_after = environment.verification_counters
            assert counters_before is not None
            executed_thrusters = (
                counters_after["executed_thruster_actions"]
                - counters_before["executed_thruster_actions"]
            )
        else:
            executed_thrusters = attempted_thrusters
        rows.append(
            {
                "experiment": name,
                "algorithm": algorithm.upper(),
                "environment_type": "Modified" if modified else "Original",
                "evaluation_episode": episode,
                "reward": episode_reward,
                "successful_safe_landing": int(
                    is_safe_landing(last_observation, terminated, truncated)
                ),
                "attempted_thruster_activations": attempted_thrusters,
                "executed_thruster_activations": executed_thrusters,
                "episode_steps": episode_steps,
                "random_seed": config.seed,
                "episode_seed": config.seed + 100_000 + episode - 1,
            }
        )

    environment.close()
    return pd.DataFrame(rows)


def write_provenance(
    config: TrainingConfig,
    validation_states: np.ndarray,
) -> None:
    """Record exact configuration, library versions, hardware, and wall-clock time."""

    try:
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_worktree_dirty = bool(git_status.strip())
    except (OSError, subprocess.CalledProcessError):
        git_worktree_dirty = None

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "training_config.json").write_text(
        json.dumps(config.as_serializable_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    cuda_available = torch.cuda.is_available()
    provenance = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "gymnasium": gym.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seaborn": sns.__version__,
        "torch": torch.__version__,
        "device": config.device,
        "cuda_available": cuda_available,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "deterministic_algorithms_requested": torch.are_deterministic_algorithms_enabled(),
        "git_commit": git_commit,
        "git_worktree_dirty_at_execution": git_worktree_dirty,
        "environment_id": "LunarLander-v3",
        "validation_set_shape": list(validation_states.shape),
        "validation_set_sha256": validation_set_sha256(validation_states),
    }
    (config.output_dir / "system_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )


def run_complete_study(config: TrainingConfig, *, force: bool = False) -> dict[str, Any]:
    """Run the four fair experiments, greedy evaluations, plots, and summaries."""

    validation_path = config.output_dir / "validation" / "fixed_validation_states.npz"
    validation_states = collect_fixed_validation_states(config, validation_path)
    set_reproducible_seeds(config.seed)
    write_provenance(config, validation_states)

    metrics: dict[str, pd.DataFrame] = {}
    for algorithm, modified in EXPERIMENTS:
        name = experiment_name(algorithm, modified)
        metrics[name] = train_agent(
            algorithm=algorithm,
            modified=modified,
            config=config,
            validation_states=validation_states,
            force=force,
        )

    evaluations = pd.concat(
        [
            evaluate_agent(algorithm=algorithm, modified=modified, config=config)
            for algorithm, modified in EXPERIMENTS
        ],
        ignore_index=True,
    )
    evaluations.to_csv(config.output_dir / "evaluation_episodes.csv", index=False)
    evaluation_summary = (
        evaluations.groupby(
            ["experiment", "algorithm", "environment_type"],
            as_index=False,
        )
        .agg(
            mean_reward=("reward", "mean"),
            reward_std=("reward", "std"),
            safe_landing_rate=("successful_safe_landing", "mean"),
            mean_attempted_thruster_activations=("attempted_thruster_activations", "mean"),
            mean_executed_thruster_activations=("executed_thruster_activations", "mean"),
            mean_episode_steps=("episode_steps", "mean"),
        )
        .sort_values("experiment")
    )
    evaluation_summary.to_csv(config.output_dir / "evaluation_summary.csv", index=False)

    create_all_plots(metrics, config.output_dir / "plots")
    summary = summarize_experiments(metrics, evaluation_summary)
    final_comparison = pd.DataFrame(summary["final_comparison"])
    final_comparison.to_csv(config.output_dir / "final_comparison.csv", index=False)
    (config.output_dir / "study_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> None:
    """Parse command-line options and launch the complete assignment experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=148)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = replace(
        TrainingConfig(),
        episodes=args.episodes,
        evaluation_episodes=args.evaluation_episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )
    run_complete_study(config, force=args.force)


if __name__ == "__main__":
    main()
