"""A single agent implementation whose target calculation selects DQN or DDQN."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig
from .network import QNetwork
from .replay import ReplayBatch, ReplayBuffer

Algorithm = Literal["dqn", "ddqn"]


class ValueAgent:
    """Train DQN or DDQN with an otherwise identical architecture and optimizer."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        algorithm: Algorithm,
        config: TrainingConfig,
    ) -> None:
        """Create online/target networks, Adam optimizer, replay memory, and RNG."""

        if algorithm not in ("dqn", "ddqn"):
            raise ValueError("algorithm must be either 'dqn' or 'ddqn'.")
        self.algorithm = algorithm
        self.action_count = action_count
        self.config = config
        self.device = torch.device(config.device)
        self.rng = np.random.default_rng(config.seed)

        self.online_network = QNetwork(
            observation_size,
            action_count,
            config.hidden_sizes,
        ).to(self.device)
        self.target_network = QNetwork(
            observation_size,
            action_count,
            config.hidden_sizes,
        ).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
        )
        self.loss_function = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(
            config.replay_capacity,
            observation_size,
            config.seed + 1,
        )
        self.update_count = 0

    def select_action(self, observation: np.ndarray, epsilon: float) -> int:
        """Choose a random action with probability epsilon, otherwise act greedily."""

        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        with torch.no_grad():
            state = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            return int(self.online_network(state).argmax(dim=1).item())

    def _bootstrap_values(self, batch: ReplayBatch) -> torch.Tensor:
        """Compute the only algorithm-specific term: the next-state target value."""

        with torch.no_grad():
            if self.algorithm == "dqn":
                return self.target_network(batch.next_observations).max(dim=1, keepdim=True).values

            online_actions = self.online_network(batch.next_observations).argmax(
                dim=1,
                keepdim=True,
            )
            return self.target_network(batch.next_observations).gather(1, online_actions)

    def _target_values(self, batch: ReplayBatch) -> torch.Tensor:
        """Build TD targets, masking terminals but bootstrapping truncations."""

        bootstrap_values = self._bootstrap_values(batch)
        # A true MDP terminal state has no future return. Time-limit truncation is
        # stored separately and still bootstraps because the underlying state is
        # not terminal; the episode ended only because of an external step limit.
        return batch.rewards + self.config.gamma * (1.0 - batch.terminated) * bootstrap_values

    def update(self) -> float:
        """Perform one replay-based temporal-difference optimization step."""

        batch = self.replay.sample(self.config.batch_size, self.device)
        predicted_values = self.online_network(batch.observations).gather(1, batch.actions)
        target_values = self._target_values(batch)

        loss = self.loss_function(predicted_values, target_values)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            self.config.gradient_clip_norm,
        )
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return float(loss.item())

    def average_max_q(self, validation_states: np.ndarray) -> float:
        """Measure mean max action value on one unchanged validation-state set."""

        with torch.no_grad():
            states = torch.as_tensor(
                validation_states,
                dtype=torch.float32,
                device=self.device,
            )
            values = self.online_network(states).max(dim=1).values
        return float(values.mean().item())

    def save(self, path: str) -> None:
        """Persist trained weights and the experimental configuration."""

        torch.save(
            {
                "algorithm": self.algorithm,
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.as_serializable_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore online and target weights from a saved checkpoint."""

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["algorithm"] != self.algorithm:
            raise ValueError("Checkpoint algorithm does not match this agent.")
        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
