"""Central configuration for fair and reproducible DQN/DDQN experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """Store every experimental choice shared by all four training runs."""

    seed: int = 148
    episodes: int = 800
    max_steps_per_episode: int = 1_000
    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    learning_starts: int = 1_000
    train_frequency: int = 1
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000
    hidden_sizes: tuple[int, int] = (128, 128)
    gradient_clip_norm: float = 10.0
    validation_state_count: int = 512
    evaluation_episodes: int = 100
    failure_probability: float = 0.15
    attempted_thruster_penalty: float = 0.3
    safe_landing_bonus: float = 50.0
    device: str = "cpu"
    output_dir: Path = Path("artifacts")

    def as_serializable_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation suitable for provenance files."""

        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        values["hidden_sizes"] = list(self.hidden_sizes)
        return values


def linear_epsilon(step: int, config: TrainingConfig) -> float:
    """Linearly anneal epsilon from its start value to its final value."""

    progress = min(max(step, 0) / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
