"""Tests proving that DQN and DDQN differ only in target selection."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from robust_lunarlander.agent import ValueAgent
from robust_lunarlander.config import TrainingConfig, linear_epsilon
from robust_lunarlander.replay import ReplayBatch, ReplayBuffer


class ConstantNetwork(nn.Module):
    """Return one fixed row of action values for every input state."""

    def __init__(self, values: list[float]) -> None:
        """Store the action-value row as a non-trainable buffer."""

        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Repeat the fixed values for the input batch size."""

        return self.values.unsqueeze(0).repeat(observations.shape[0], 1)


def make_batch() -> ReplayBatch:
    """Construct a two-transition tensor batch for target calculations."""

    return ReplayBatch(
        observations=torch.zeros((2, 8)),
        actions=torch.zeros((2, 1), dtype=torch.int64),
        rewards=torch.zeros((2, 1)),
        next_observations=torch.zeros((2, 8)),
        dones=torch.zeros((2, 1)),
    )


def test_dqn_uses_maximum_target_network_value() -> None:
    """Confirm the DQN target performs max selection in the target network."""

    agent = ValueAgent(8, 4, "dqn", TrainingConfig())
    agent.online_network = ConstantNetwork([0.0, 9.0, 0.0, 0.0])
    agent.target_network = ConstantNetwork([1.0, 2.0, 7.0, 3.0])
    values = agent._bootstrap_values(make_batch())
    assert values.tolist() == [[7.0], [7.0]]


def test_ddqn_selects_online_action_and_evaluates_it_with_target_network() -> None:
    """Confirm DDQN decouples next-action selection from target evaluation."""

    agent = ValueAgent(8, 4, "ddqn", TrainingConfig())
    agent.online_network = ConstantNetwork([0.0, 9.0, 0.0, 0.0])
    agent.target_network = ConstantNetwork([1.0, 2.0, 7.0, 3.0])
    values = agent._bootstrap_values(make_batch())
    assert values.tolist() == [[2.0], [2.0]]


def test_linear_epsilon_is_bounded_and_reaches_final_value() -> None:
    """Verify the exploration schedule starts, interpolates, and then clamps."""

    config = TrainingConfig(epsilon_decay_steps=100)
    assert linear_epsilon(0, config) == pytest.approx(1.0)
    assert linear_epsilon(50, config) == pytest.approx(0.505)
    assert linear_epsilon(100, config) == pytest.approx(0.01)
    assert linear_epsilon(1_000, config) == pytest.approx(0.01)


def test_replay_buffer_is_seeded_and_returns_expected_shapes() -> None:
    """Check deterministic sampling, capacity wraparound, and tensor geometry."""

    first = ReplayBuffer(capacity=4, observation_size=8, seed=148)
    second = ReplayBuffer(capacity=4, observation_size=8, seed=148)
    for index in range(6):
        observation = np.full(8, index, dtype=np.float32)
        next_observation = observation + 1
        transition = (observation, index % 4, float(index), next_observation, bool(index % 2))
        first.add(*transition)
        second.add(*transition)

    assert len(first) == 4
    batch_one = first.sample(3, torch.device("cpu"))
    batch_two = second.sample(3, torch.device("cpu"))
    assert batch_one.observations.shape == (3, 8)
    assert batch_one.actions.shape == (3, 1)
    assert torch.equal(batch_one.observations, batch_two.observations)
    assert torch.equal(batch_one.actions, batch_two.actions)
