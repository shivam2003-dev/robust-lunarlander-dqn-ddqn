"""Neural-network definition shared without alteration by DQN and DDQN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class QNetwork(nn.Module):
    """Map an eight-dimensional LunarLander state to four action values."""

    def __init__(
        self,
        observation_size: int,
        action_count: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ) -> None:
        """Build a multilayer perceptron with ReLU hidden activations."""

        super().__init__()
        dimensions = [observation_size, *hidden_sizes, action_count]
        layers: list[nn.Module] = []
        for input_size, output_size in zip(dimensions[:-2], dimensions[1:-1], strict=True):
            layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return one predicted Q-value per discrete action."""

        return self.model(observations)
