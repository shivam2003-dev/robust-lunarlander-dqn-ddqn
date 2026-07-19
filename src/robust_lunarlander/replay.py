"""Preallocated experience replay used identically by DQN and DDQN."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class ReplayBatch(NamedTuple):
    """Hold one tensor batch sampled uniformly from replay memory."""

    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Store transitions in a fixed-size circular buffer."""

    def __init__(
        self,
        capacity: int,
        observation_size: int,
        seed: int,
    ) -> None:
        """Allocate storage once and initialize an isolated sampling generator."""

        self.capacity = int(capacity)
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Insert one transition, overwriting the oldest transition when full."""

        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        """Sample transitions uniformly and move the resulting tensors to a device."""

        if self.size < batch_size:
            raise ValueError("Cannot sample more transitions than the buffer currently stores.")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return ReplayBatch(
            torch.as_tensor(self.observations[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device).unsqueeze(1),
            torch.as_tensor(self.rewards[indices], device=device).unsqueeze(1),
            torch.as_tensor(self.next_observations[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        """Return the number of valid transitions currently stored."""

        return self.size
