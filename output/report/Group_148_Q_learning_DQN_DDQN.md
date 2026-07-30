---
title: "Robust Reinforcement Learning under Stochastic Action Failure"
subtitle: "Experiential Learning - Assignment 2 | Deep Reinforcement Learning (S2-25_AIMLCZG512)"
author: "Group 148"
date: "2026-07-31"
papersize: a4
fontsize: 10pt
geometry: "top=20mm,bottom=20mm,left=18mm,right=18mm"
mainfont: "Arial"
monofont: "Menlo"
colorlinks: true
linkcolor: "MidnightBlue"
urlcolor: "MidnightBlue"
---

# Group contribution declaration {.unnumbered}

**MANDATORY BEFORE SUBMISSION:** These contribution percentages are declared but must be confirmed by all five group members.

| Group member | BITS ID | Contribution (%) |
| --- | --- | --- |
| Shivam Kumar | 2025AA05312 | 20 |
| Nunna Swahan Bapaji | 2025AA05569 | 20 |
| Garimella Nikhitha | 2025AA05512 | 20 |
| S H Aishwarya | 2025AA05594 | 20 |
| Omkareshwar Vaijanath Telee | 2025AB05010 | 20 |

**Total contribution: 100%**

| Submission field | Value |
|---|---|
| Course | Deep Reinforcement Learning (S2-25_AIMLCZG512) |
| Assignment | Experiential Learning - Assignment 2 |
| Group | 148 |
| Recorded study execution | 2026-07-30T23:58:07+05:30 |
| Recorded execution environment | macOS-26.5.2-arm64-arm-64bit-Mach-O; Python 3.13.12; Gymnasium 1.2.3; PyTorch 2.13.0; device cpu |
| Report build date | 2026-07-31 |
| Intended final PDF filename | `Group148_Q_learning_DQN_DDQN.pdf` |

**Filename note:** Check the exact final filename against the latest instructor
guidance before upload. The build currently follows the requested pattern
`Group148_Q_learning_DQN_DDQN.pdf`.

\tableofcontents

\newpage

# Executive summary

This report implements the specified hidden 15% actuator-failure wrapper and
compares DQN and Double DQN in a controlled 2 x 2 experiment. The four agents use
the same seed, network, initialization procedure, optimizer, replay settings,
exploration schedule, target-update cadence, duration, and fixed validation-state
set. The wrapper is the only environment difference; the target calculation is
the only DQN-versus-DDQN difference.

Across 250 random-policy verification episodes,
2,402 of
16,460 attempted thruster actions
misfired (14.593%). The target
15% lies in the Wilson 95% interval [14.062% to 15.141%]. The internal
fuel-penalty count equals the attempted-action count, and no returned `info`
object was changed.

For the recorded single-seed study, DDQN on the modified environment achieved
greedy mean reward -225.82 and
safe-landing rate 3%.
DQN achieved 48.02 and
44%. These values
are evidence for this execution only, not a claim of statistical significance.

# 1. Introduction

LunarLander-v3 has an eight-dimensional continuous observation: horizontal and
vertical position, horizontal and vertical velocity, angle, angular velocity,
and two leg-contact indicators. Its four discrete actions are:

| Action | Meaning |
|---:|---|
| 0 | Do nothing |
| 1 | Fire left orientation engine |
| 2 | Fire main engine |
| 3 | Fire right orientation engine |

Actuator failures are realistic in robotics because commands can be lost or
partially ineffective through mechanical, electrical, communication, or control
faults. Here, the agent selects a thruster action but may unknowingly execute
no-op. This hidden replacement creates transition uncertainty: similar observed
state-action pairs can lead to different physical next states.

The reward charges 0.3 for the **selected** thruster action even when it
misfires. It can therefore encourage a conservative policy, but only if the
learner discovers that avoiding unnecessary attempts improves long-run return.
DQN can overestimate action values because a maximization over noisy estimates
uses the same target network for selection and evaluation. DDQN reduces this
bias by selecting the next action with the online network and evaluating that
action with the target network.

# 2. Reproducibility and environment inspection

## 2.1 BITS virtual-lab installation

The environment is explicitly LunarLander-v3; no fallback version is used.
On a Debian/Ubuntu-style virtual lab, install Box2D build support and this
repository as follows:

~~~{.bash}
sudo apt-get update
sudo apt-get install -y swig build-essential python3-dev
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); print(env.observation_space, env.action_space); env.close()"
~~~

If `sudo` is unavailable, ask the lab administrator to install `swig` and the
compiler packages; do not silently switch LunarLander versions.

## 2.2 Seeding and provenance

One function seeds Python `random`, NumPy, PyTorch CPU, all CUDA devices,
Gymnasium reset, and the action space. It also requests deterministic PyTorch
algorithms and disables cuDNN benchmarking where available. Box2D,
floating-point kernels, drivers, and hardware can still produce cross-platform
nondeterminism, so versions and hardware are recorded.

| Parameter | Shared value |
| --- | --- |
| Global seed | 148 |
| Training episodes per agent | 800 |
| Maximum steps per episode | 1000 |
| Evaluation episodes per agent | 100 |
| Network | 8-128-128-4, ReLU |
| Trainable parameters | 18,180 |
| Initialization | PyTorch Linear default; same seed before every agent |
| Optimizer / loss | Adam / Smooth L1 |
| Learning rate | 0.0005 |
| Discount factor | 0.99 |
| Replay capacity / batch | 100,000 / 64 |
| Learning warm-up | 1,000 steps |
| Target update | Hard copy every 500 updates |
| Epsilon schedule | 1.00 to 0.01 over 100,000 steps |
| Fixed validation states | 512 |
| Failure probability | 0.15 |
| Attempted-thruster penalty | 0.3 |
| Safe-landing bonus | 50.0 |
| Device | cpu |

Recorded provenance:

- Python: 3.13.12
- Gymnasium: 1.2.3
- PyTorch: 2.13.0
- NumPy / pandas / Seaborn: 2.4.4 / 3.0.2 / 0.13.2
- Platform / logical CPUs: macOS-26.5.2-arm64-arm-64bit-Mach-O / 14
- CUDA available / device: False / None
- Device used: cpu
- Global random seed: 148
- Git commit at execution: `3eb6b9bf0e382fc67c02976cdb55ed8dde1dffa7`
- Git worktree dirty at execution: True

# 3. Custom environment and verification - 2.5 marks

For selected action $a$, the wrapper samples a private RNG only when
$a \in \{1,2,3\}$. With probability 0.15 it executes action 0; otherwise it
executes $a$. A private RNG avoids consuming the base LunarLander RNG and hence
keeps ordinary transition randomness isolated. Observation/action spaces,
termination, and truncation are delegated unchanged.

The modified reward is

$$R = R_{base} - 0.3\,\mathbf{1}(a \ne 0) + B,$$

where $B=50$ only for a terminated, non-truncated transition with both legs in
contact and absolute horizontal velocity, vertical velocity, and angle each
strictly below 0.10. Private counters are externally readable for verification
but are never inserted into the observation or returned `info`.

## 3.1 Deterministic mock-environment tests

| Case | Selected | Executed | Bonus | Penalty | Info unchanged | Passed |
| --- | --- | --- | --- | --- | --- | --- |
| Safe terminal | 0 | 0 | 50.0 | 0.0 | True | True |
| Firing + safe | 2 | 2 | 50.0 | 0.3 | True | True |
| Misfire + safe | 2 | 0 | 50.0 | 0.3 | True | True |
| Not terminated | 0 | 0 | 0.0 | 0.0 | True | True |
| Truncated | 0 | 0 | 0.0 | 0.0 | True | True |
| Left leg absent | 0 | 0 | 0.0 | 0.0 | True | True |
| Right leg absent | 0 | 0 | 0.0 | 0.0 | True | True |
| Excess x velocity | 0 | 0 | 0.0 | 0.0 | True | True |
| Excess y velocity | 0 | 0 | 0.0 | 0.0 | True | True |
| Excess angle | 0 | 0 | 0.0 | 0.0 | True | True |

The automated unit suite additionally checks action 0, successful firing,
misfiring, strict threshold equality, both leg contacts, truncation, unchanged
spaces, invalid actions, and exact `info` identity.

## 3.2 Random-policy evidence

| Measure | Recorded result |
|---|---:|
| Episodes | 250 |
| Environment steps | 21,912 |
| Attempted thruster actions | 16,460 |
| Misfires | 2,402 |
| Observed misfire rate | 14.593% |
| Absolute difference from 0.15 | 0.004070 |
| Wilson 95% interval | 14.062% to 15.141% |
| Fuel-penalty count | 16,460 |
| Fuel-penalty count equals attempts | True |
| Fuel-penalty mismatches | 0 |
| Returned-info identity mismatches | 0 |
| Random-policy safe bonuses observed | 12 |

Successful-fire and forced-misfire mock cases both show the 0.3 selected-action
penalty. Deterministic landing-boundary cases are authoritative for the +50
bonus because random policies do not guarantee useful landing coverage.

# 4. Replay, network, DQN, and DDQN - 8 marks

## 4.1 Replay and terminal masking

The replay buffer stores state, action, reward, next state, `terminated`, and
`truncated` separately. The target masks only true terminal states:

$$y = r + \gamma(1-\text{terminated})\,Q_{bootstrap}.$$

A time-limit truncation still bootstraps because the underlying MDP state is not
terminal; the episode ended due to an external horizon. This choice is explicit
and unit-tested.

## 4.2 Shared Q-network

All experiments use the same 8-128-128-4 multilayer perceptron, ReLU
activations, PyTorch `Linear` initialization after the same global seed, Adam,
and Smooth L1 loss. It has **18,180 trainable parameters**.

## 4.3 The only algorithmic branch

For DQN:

$$y_{DQN} = r + \gamma(1-t)\max_a Q_{target}(s',a).$$

For DDQN:

$$a^* = \arg\max_a Q_{online}(s',a), \qquad
y_{DDQN} = r + \gamma(1-t)Q_{target}(s',a^*).$$

The online/target architectures, optimizer, replay, epsilon policy, discount,
warm-up, batch size, learning rate, target updates, seed, and training duration
are otherwise identical.

## 4.4 Shared epsilon-greedy schedule

| Checkpoint | Environment step | Epsilon |
| --- | --- | --- |
| Start | 0 | 1.000 |
| Half decay | 50000 | 0.505 |
| End of decay | 100000 | 0.010 |
| After decay | 200000 | 0.010 |

![Shared epsilon-greedy schedule](artifacts/plots/epsilon_schedule.png)

# 5. Fixed validation states, metrics, and controlled runs

The validation states were collected once from the original environment with a
reproducible random policy and never resampled during training. Reusing an
identical set prevents state-distribution drift from being confused with
algorithmic differences in predicted Q-values.

- Shape: (512, 8)
- SHA-256: `b765a8c76d9ffa0e93207cc98d3bda2bf16adca80d38c8871380e832e25c7f8a`

Every episode records reward, fixed-set average maximum Q-value, strict safe
landing, moving safe-landing rate over up to 100 episodes, attempted and
executed thruster actions, cumulative average attempts per episode, steps,
epsilon, mean training loss, environment type, algorithm, global seed, episode
seed, global step, and measured duration.

Four fresh agents and replay buffers are trained:

1. DQN - Original Environment
2. DDQN - Original Environment
3. DQN - Modified Environment
4. DDQN - Modified Environment

Each episode prints one compact progress line and writes one log record. The
full study is parameterized, so a short smoke run and the full run use the same
code path.

# 6. Performance evaluation - 2 marks

![All four required training metrics](artifacts/plots/four_metric_overview.png)

The figure contains the four required comparisons: episode reward, fixed-set
average predicted Q-value, 100-episode moving safe-landing rate, and attempted
thruster activations per episode. Raw reward and activation traces are light;
moderate moving means make trends legible without concealing instability.
Colours and line styles are consistent across panels.

## 6.1 Required final comparison table

The two panels below form one comparison table while remaining legible on A4.

**Panel A - performance and value estimates**

| Algorithm | Environment | Mean reward final 100 | Reward SD final 100 | Best MA(100) reward | Final fixed-set Q | Safe rate final 100 |
| --- | --- | --- | --- | --- | --- | --- |
| DQN | Original | 150.52 | 113.67 | 154.40 | 49.12 | 36% |
| DDQN | Original | 218.24 | 98.01 | 226.34 | 80.51 | 71% |
| DQN | Modified | -172.73 | 192.67 | -56.87 | 48.56 | 19% |
| DDQN | Modified | -150.42 | 200.69 | -54.05 | -6.72 | 25% |

**Panel B - action use, successes, and duration**

| Algorithm | Environment | Mean attempted final 100 | Mean executed final 100 | Safe landings total | Training seconds |
| --- | --- | --- | --- | --- | --- |
| DQN | Original | 409.80 | 409.80 | 116 | 166.63 |
| DDQN | Original | 304.11 | 304.11 | 166 | 137.97 |
| DQN | Modified | 655.06 | 557.94 | 86 | 89.09 |
| DDQN | Modified | 595.74 | 507.37 | 83 | 98.72 |

Mean reward, reward SD, safe rate, and both action-use means use the final 100
training episodes. The best moving average, final Q-value, total successes, and
duration use the scopes stated in their labels. Executed actions equal attempts
in the original environment; modified runs report actual non-misfired executions.

## 6.2 Greedy evaluation on shared seeds

**Panel A - reward and strict landing performance**

| Algorithm | Environment | Mean reward | Reward SD | Safe landing rate |
| --- | --- | --- | --- | --- |
| DDQN | Modified | -225.82 | 102.17 | 3% |
| DDQN | Original | 169.05 | 136.75 | 61% |
| DQN | Modified | 48.02 | 186.73 | 44% |
| DQN | Original | 67.15 | 95.71 | 24% |

**Panel B - action use and episode length**

| Algorithm | Environment | Mean attempted | Mean executed | Mean steps |
| --- | --- | --- | --- | --- |
| DDQN | Modified | 730.92 | 622.07 | 972.34 |
| DDQN | Original | 250.72 | 250.72 | 360.51 |
| DQN | Modified | 423.85 | 359.97 | 615.37 |
| DQN | Original | 138.07 | 138.07 | 188.34 |

This evaluation uses 100 shared episode seeds per experiment and epsilon 0.
It is supplementary to the assignment's final-100 training summary.

# 7. Discussion - 2.5 marks

## 7.1 Does failure increase the DQN-DDQN Q-value difference?

For this execution, yes. The final-100 mean absolute Q-gap is
16.23 in the original
environment and 28.68 in the
modified environment, an increase of 12.46.
The fixed-state Q-value panel provides the trajectory; these numerical values
provide the final-window comparison. A single seed cannot establish a
population-level effect.

## 7.2 Why is temporal credit assignment harder?

The same selected thruster action can yield an impulse or a no-op while the
agent receives no replacement indicator. Replay therefore contains
higher-variance outcomes for apparently similar state-action inputs. The
attempted-action cost is charged in either case, separating a certain immediate
cost from an uncertain physical effect and making delayed crash or landing
credit harder to assign.

## 7.3 Does the 0.3 penalty encourage a conservative strategy?

Not in the attempted-action metric for this execution. Greedy DDQN attempts
730.92
actions in the modified environment versus
250.72
in the original. Greedy DQN attempts
423.85
versus 138.07.
Both modified policies therefore attempt more, not fewer, thruster actions. The
longer modified episodes and need to compensate for no-op commands may outweigh
the 0.3 cost. The penalty alone did not produce a demonstrably conservative
strategy.

## 7.4 Which algorithm performs better under failure?

The answer is metric-dependent. In the final 100 training episodes, DDQN has
the higher mean reward (-150.42
versus -172.73)
and safe-landing rate
(25%
versus 19%).
However, frozen-checkpoint greedy evaluation strongly favors DQN: mean reward
48.02 and safe-landing rate
44% versus
DDQN's -225.82 and
3%. Therefore
this run does not give an unambiguous DDQN advantage and is not cleanly
consistent with the theoretical expectation.

## 7.5 Limitation and improvement

The executed study uses one training seed, so between-run variance is not
estimated. The strongest improvement is a preregistered paired multi-seed study
using the same four-run design, followed by confidence intervals for final
reward, safe-landing rate, Q-gap, and thruster use. A later failure-probability
sweep could test whether any DDQN advantage scales with uncertainty.

# 8. Conclusion

In the recorded single-seed experiment, hidden actuator failure increased the
DQN-DDQN fixed-state Q-value gap from
16.23 to
28.68. The selected-action fuel
penalty did not reduce attempted activations in the modified runs. DDQN was
slightly stronger in the final training window, while DQN was substantially
stronger in greedy evaluation, so no algorithm wins every metric. The main
limitation is the single seed; paired multi-seed replication is the next
improvement. This conclusion must be regenerated if the official virtual-lab
run changes the recorded outputs.

# References {.unnumbered}

1. Mnih, V. et al. (2015). Human-level control through deep reinforcement
   learning. *Nature*, 518, 529-533. [DOI record](https://doi.org/10.1038/nature14236).
2. van Hasselt, H., Guez, A., and Silver, D. (2016). Deep Reinforcement Learning
   with Double Q-Learning. *Proceedings of the AAAI Conference on Artificial
   Intelligence*, 30(1). [DOI record](https://doi.org/10.1609/aaai.v30i1.10295).
3. Farama Foundation. [Gymnasium LunarLander-v3 documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

\newpage

# Appendix A - Complete commented source

## src/robust_lunarlander/config.py

~~~{.python}
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
~~~


## src/robust_lunarlander/envs.py

~~~{.python}
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
~~~


## src/robust_lunarlander/network.py

~~~{.python}
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
~~~


## src/robust_lunarlander/replay.py

~~~{.python}
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
    terminated: torch.Tensor
    truncated: torch.Tensor


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
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.truncated = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Insert one transition with both Gymnasium ending flags preserved."""

        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.terminated[index] = float(terminated)
        self.truncated[index] = float(truncated)
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
            torch.as_tensor(self.terminated[indices], device=device).unsqueeze(1),
            torch.as_tensor(self.truncated[indices], device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        """Return the number of valid transitions currently stored."""

        return self.size
~~~


## src/robust_lunarlander/agent.py

~~~{.python}
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
~~~


## src/robust_lunarlander/verification.py

~~~{.python}
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
~~~


## src/robust_lunarlander/experiment.py

~~~{.python}
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
~~~


## src/robust_lunarlander/plotting.py

~~~{.python}
"""Publication-quality plots and evidence summaries for the four experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DISPLAY_ORDER = [
    "dqn_original",
    "ddqn_original",
    "dqn_modified",
    "ddqn_modified",
]
DISPLAY_LABELS = {
    "dqn_original": "DQN - Original",
    "ddqn_original": "DDQN - Original",
    "dqn_modified": "DQN - Modified",
    "ddqn_modified": "DDQN - Modified",
}
COLORS = {
    "dqn_original": "#2563EB",
    "ddqn_original": "#0F766E",
    "dqn_modified": "#DC2626",
    "ddqn_modified": "#D97706",
}
LINESTYLES = {
    "dqn_original": "-",
    "ddqn_original": "-",
    "dqn_modified": "--",
    "ddqn_modified": "--",
}


def _configure_style() -> None:
    """Apply one accessible visual system to all report figures."""

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "axes.edgecolor": "#475569",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": ":",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFD",
        }
    )


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    """Write sharp PNG and editable SVG copies of one figure."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def _plot_series(
    axis: plt.Axes,
    metrics: dict[str, pd.DataFrame],
    *,
    column: str,
    rolling_window: int | None,
    raw_alpha: float = 0.0,
) -> None:
    """Draw all four experiments with optional raw and smoothed traces."""

    for name in DISPLAY_ORDER:
        frame = metrics[name]
        color = COLORS[name]
        if raw_alpha > 0.0:
            axis.plot(
                frame["episode"],
                frame[column],
                color=color,
                alpha=raw_alpha,
                linewidth=0.55,
            )
        values = (
            frame[column].rolling(rolling_window, min_periods=1).mean()
            if rolling_window
            else frame[column]
        )
        axis.plot(
            frame["episode"],
            values,
            label=DISPLAY_LABELS[name],
            color=color,
            linestyle=LINESTYLES[name],
            linewidth=2.0,
        )


def create_all_plots(metrics: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Create each required plot and a compact four-panel overview."""

    _configure_style()
    specifications = [
        (
            "episode_reward",
            "episode_reward",
            "Episode reward vs. training episode",
            "Episode reward",
            50,
            0.15,
            "Thin lines show raw rewards; strong lines show a 50-episode moving mean.",
        ),
        (
            "average_predicted_q",
            "average_predicted_q",
            "Average predicted Q-value on fixed validation states",
            "Mean max predicted Q-value",
            20,
            0.0,
            "The same 512 validation states are used at every episode for all four agents.",
        ),
        (
            "moving_safe_landing_rate_100",
            "success_rate_100",
            "Safe-landing rate vs. training episode",
            "Safe-landing rate (previous 100 episodes)",
            None,
            0.0,
            "Success uses the assignment's terminal leg-contact, velocity, and angle criterion.",
        ),
        (
            "attempted_thruster_activations",
            "thruster_activations",
            "Average attempted thruster activations per episode",
            "Attempted thruster actions per episode",
            20,
            0.08,
            "Strong lines show a 20-episode moving mean of selected actions 1, 2, and 3.",
        ),
    ]

    for column, stem, title, ylabel, window, raw_alpha, caption in specifications:
        figure, axis = plt.subplots(figsize=(11.2, 6.2))
        _plot_series(
            axis,
            metrics,
            column=column,
            rolling_window=window,
            raw_alpha=raw_alpha,
        )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Training episode")
        axis.set_ylabel(ylabel)
        axis.legend(ncol=2, loc="best")
        figure.text(0.125, 0.015, caption, color="#475569", fontsize=9)
        figure.subplots_adjust(bottom=0.14)
        _save_figure(figure, output_dir, stem)

    overview, axes = plt.subplots(2, 2, figsize=(15.2, 10.0))
    for axis, (column, _, title, ylabel, window, raw_alpha, _) in zip(
        axes.flat,
        specifications,
        strict=True,
    ):
        _plot_series(
            axis,
            metrics,
            column=column,
            rolling_window=window,
            raw_alpha=min(raw_alpha, 0.10),
        )
        axis.set_title(title, loc="left", fontweight="bold", fontsize=12)
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel, fontsize=10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    overview.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.943),
        ncol=4,
        frameon=False,
    )
    overview.suptitle(
        "DQN vs. DDQN under stochastic actuator failure",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    overview.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.07,
        top=0.86,
        hspace=0.34,
        wspace=0.18,
    )
    _save_figure(overview, output_dir, "four_metric_overview")

    epsilon_figure, epsilon_axis = plt.subplots(figsize=(10.0, 4.8))
    reference = metrics[DISPLAY_ORDER[0]]
    epsilon_axis.plot(
        reference["global_step"],
        reference["epsilon"],
        color="#6D28D9",
        linewidth=2.0,
        label="Shared linear epsilon schedule",
    )
    epsilon_axis.set_title("Epsilon-greedy exploration schedule", loc="left", fontweight="bold")
    epsilon_axis.set_xlabel("Environment step")
    epsilon_axis.set_ylabel("Epsilon")
    epsilon_axis.legend(loc="best")
    epsilon_axis.set_ylim(0.0, 1.05)
    epsilon_figure.text(
        0.125,
        0.015,
        "The same start, end, and step-based decay are applied to all four agents.",
        color="#475569",
        fontsize=9,
    )
    epsilon_figure.subplots_adjust(bottom=0.18)
    _save_figure(epsilon_figure, output_dir, "epsilon_schedule")


def summarize_experiments(
    metrics: dict[str, pd.DataFrame],
    evaluation_summary: pd.DataFrame,
) -> dict[str, Any]:
    """Compute report-ready comparisons without inventing unsupported conclusions."""

    final_window: dict[str, dict[str, float]] = {}
    final_comparison: list[dict[str, Any]] = []
    for name in DISPLAY_ORDER:
        tail = metrics[name].tail(100)
        moving_window = min(100, len(metrics[name]))
        final_window[name] = {
            "mean_training_reward_last_100": float(tail["episode_reward"].mean()),
            "reward_std_last_100": float(tail["episode_reward"].std()),
            "best_100_episode_moving_average_reward": float(
                metrics[name]["episode_reward"]
                .rolling(moving_window, min_periods=moving_window)
                .mean()
                .max()
            ),
            "final_fixed_set_average_q": float(metrics[name]["average_predicted_q"].iloc[-1]),
            "safe_landing_rate_last_100": float(tail["successful_safe_landing"].mean()),
            "mean_predicted_q_last_100": float(tail["average_predicted_q"].mean()),
            "mean_attempted_thruster_activations_last_100": float(
                tail["attempted_thruster_activations"].mean()
            ),
            "mean_executed_thruster_activations_last_100": float(
                tail["executed_thruster_activations"].mean()
            ),
            "successful_safe_landings_total": int(metrics[name]["successful_safe_landing"].sum()),
            "training_duration_seconds": float(metrics[name]["episode_seconds"].sum()),
        }
        final_comparison.append(
            {
                "experiment": name,
                "algorithm": str(metrics[name]["algorithm"].iloc[0]),
                "environment": str(metrics[name]["environment_type"].iloc[0]),
                "mean_reward_final_100": final_window[name]["mean_training_reward_last_100"],
                "reward_std_final_100": final_window[name]["reward_std_last_100"],
                "best_moving_average_reward_100": final_window[name][
                    "best_100_episode_moving_average_reward"
                ],
                "final_fixed_set_average_q": final_window[name]["final_fixed_set_average_q"],
                "safe_landing_rate_final_100": final_window[name]["safe_landing_rate_last_100"],
                "mean_attempted_thrusters_final_100": final_window[name][
                    "mean_attempted_thruster_activations_last_100"
                ],
                "mean_executed_thrusters_final_100": final_window[name][
                    "mean_executed_thruster_activations_last_100"
                ],
                "successful_safe_landings_total": final_window[name][
                    "successful_safe_landings_total"
                ],
                "training_duration_seconds": final_window[name]["training_duration_seconds"],
            }
        )

    original_gap = float(
        (
            metrics["dqn_original"]["average_predicted_q"].tail(100).reset_index(drop=True)
            - metrics["ddqn_original"]["average_predicted_q"].tail(100).reset_index(drop=True)
        )
        .abs()
        .mean()
    )
    modified_gap = float(
        (
            metrics["dqn_modified"]["average_predicted_q"].tail(100).reset_index(drop=True)
            - metrics["ddqn_modified"]["average_predicted_q"].tail(100).reset_index(drop=True)
        )
        .abs()
        .mean()
    )
    evaluation_records = evaluation_summary.to_dict(orient="records")
    modified_evaluation = evaluation_summary[
        evaluation_summary["environment_type"] == "Modified"
    ].sort_values(["safe_landing_rate", "mean_reward"], ascending=False)
    best_modified = str(modified_evaluation.iloc[0]["algorithm"])

    return {
        "final_100_training_episodes": final_window,
        "q_value_gap": {
            "original_environment_mean_absolute_gap": original_gap,
            "modified_environment_mean_absolute_gap": modified_gap,
            "modified_minus_original": modified_gap - original_gap,
            "failure_increased_gap": modified_gap > original_gap,
        },
        "attempted_thruster_change_modified_minus_original": {
            "DQN": final_window["dqn_modified"]["mean_attempted_thruster_activations_last_100"]
            - final_window["dqn_original"]["mean_attempted_thruster_activations_last_100"],
            "DDQN": final_window["ddqn_modified"]["mean_attempted_thruster_activations_last_100"]
            - final_window["ddqn_original"]["mean_attempted_thruster_activations_last_100"],
        },
        "final_comparison": final_comparison,
        "greedy_evaluation": evaluation_records,
        "best_algorithm_under_modified_environment": best_modified,
    }
~~~


\newpage

# Appendix B - Per-iteration output evidence

Every training episode emits a compact console line and is persisted in both a
CSV ledger and a progress log. This appendix reproduces all
3,200 per-iteration records: 800 rows for each of the four
controlled experiments. Values follow the compact progress output convention:
reward, fixed-set average Q, moving safe-landing rate, attempted/executed
thrusters, steps, epsilon, and training loss. `nan` denotes the pre-warm-up
episodes where no gradient update was yet performed.

| Experiment | CSV rows | Log records | Episode range | Complete |
| --- | --- | --- | --- | --- |
| dqn_original | 800 | 800 | 1-800 | True |
| ddqn_original | 800 | 800 | 1-800 | True |
| dqn_modified | 800 | 800 | 1-800 | True |
| ddqn_modified | 800 | 800 | 1-800 | True |

<!-- ITERATION_TABLES_FOR_HTML -->

\scriptsize
\setlength{\tabcolsep}{1.8pt}
\renewcommand{\arraystretch}{0.86}
\begin{longtable}{rrrrrrrr}
\caption{Complete per-iteration training output - Dqn Original (800 episodes).}\\
\toprule
Episode & Reward & Avg. Q & Safe100 & Attempted/Executed & Steps & Epsilon & Loss \\
\midrule
\endfirsthead
\multicolumn{8}{c}{\small Continued: Dqn Original complete per-iteration output} \\
\toprule
Episode & Reward & Avg. Q & Safe100 & Attempted/Executed & Steps & Epsilon & Loss \\
\midrule
\endhead
\bottomrule
\endfoot
\bottomrule
\endlastfoot
1 & -132.79 & 0.24 & 0.0\% & 51/51 & 71 & 0.9993 & nan \\
2 & -436.57 & 0.24 & 0.0\% & 66/66 & 93 & 0.9984 & nan \\
3 & -112.58 & 0.24 & 0.0\% & 104/104 & 124 & 0.9971 & nan \\
4 & -74.45 & 0.24 & 0.0\% & 67/67 & 86 & 0.9963 & nan \\
5 & -141.56 & 0.24 & 0.0\% & 104/104 & 139 & 0.9949 & nan \\
6 & -168.37 & 0.24 & 0.0\% & 82/82 & 113 & 0.9938 & nan \\
7 & -170.35 & 0.24 & 0.0\% & 69/69 & 88 & 0.9929 & nan \\
8 & -264.71 & 0.24 & 0.0\% & 43/43 & 63 & 0.9923 & nan \\
9 & -262.58 & 0.24 & 0.0\% & 88/88 & 114 & 0.9912 & nan \\
10 & -256.16 & -0.01 & 0.0\% & 94/94 & 126 & 0.9899 & 2.111354 \\
11 & -57.57 & -0.18 & 9.1\% & 51/51 & 65 & 0.9893 & 2.197836 \\
12 & -174.41 & 0.17 & 8.3\% & 67/67 & 86 & 0.9884 & 1.900430 \\
13 & -304.37 & 0.38 & 7.7\% & 68/68 & 92 & 0.9875 & 2.101976 \\
14 & -285.82 & 0.69 & 7.1\% & 74/74 & 102 & 0.9865 & 1.573831 \\
15 & -245.45 & 0.75 & 6.7\% & 86/86 & 122 & 0.9853 & 1.507664 \\
16 & -108.64 & 0.92 & 6.2\% & 65/65 & 84 & 0.9845 & 1.839810 \\
17 & -214.15 & 0.90 & 5.9\% & 86/86 & 104 & 0.9834 & 1.404884 \\
18 & -182.99 & 0.97 & 5.6\% & 62/62 & 85 & 0.9826 & 1.330281 \\
19 & -78.73 & 0.82 & 10.5\% & 81/81 & 104 & 0.9816 & 1.395608 \\
20 & -76.50 & 1.19 & 10.0\% & 57/57 & 74 & 0.9808 & 1.516414 \\
21 & -187.82 & 1.13 & 9.5\% & 51/51 & 71 & 0.9801 & 1.515786 \\
22 & -378.98 & 1.45 & 9.1\% & 80/80 & 104 & 0.9791 & 1.434591 \\
23 & -350.93 & 1.19 & 8.7\% & 69/69 & 87 & 0.9782 & 1.317473 \\
24 & -26.12 & 1.51 & 8.3\% & 75/75 & 97 & 0.9773 & 1.059367 \\
25 & -79.35 & 1.37 & 8.0\% & 76/76 & 99 & 0.9763 & 1.418375 \\
26 & -174.05 & 1.71 & 7.7\% & 50/50 & 61 & 0.9757 & 1.130719 \\
27 & -95.26 & 1.75 & 7.4\% & 58/58 & 83 & 0.9749 & 1.399712 \\
28 & -173.95 & 1.61 & 7.1\% & 92/92 & 120 & 0.9737 & 1.178418 \\
29 & -102.62 & 1.75 & 6.9\% & 47/47 & 67 & 0.9730 & 0.985096 \\
30 & -240.45 & 1.83 & 6.7\% & 67/67 & 87 & 0.9722 & 1.205984 \\
31 & -164.48 & 1.86 & 6.5\% & 118/118 & 146 & 0.9707 & 1.207793 \\
32 & -89.86 & 2.06 & 6.2\% & 52/52 & 74 & 0.9700 & 1.239478 \\
33 & -200.89 & 2.37 & 6.1\% & 84/84 & 113 & 0.9689 & 1.086603 \\
34 & -20.86 & 2.36 & 5.9\% & 98/98 & 134 & 0.9675 & 1.228904 \\
35 & -251.55 & 2.49 & 5.7\% & 86/86 & 110 & 0.9665 & 1.167238 \\
36 & -335.83 & 2.66 & 5.6\% & 84/84 & 106 & 0.9654 & 1.139542 \\
37 & -106.60 & 2.87 & 5.4\% & 64/64 & 74 & 0.9647 & 1.179119 \\
38 & -112.47 & 3.09 & 5.3\% & 59/59 & 71 & 0.9640 & 1.299323 \\
39 & -88.76 & 2.89 & 5.1\% & 65/65 & 89 & 0.9631 & 1.138951 \\
40 & -161.58 & 2.79 & 5.0\% & 88/88 & 113 & 0.9620 & 1.123193 \\
41 & -195.98 & 2.85 & 4.9\% & 88/88 & 120 & 0.9608 & 1.115919 \\
42 & -187.76 & 3.72 & 4.8\% & 51/51 & 71 & 0.9601 & 1.253932 \\
43 & -239.04 & 3.67 & 4.7\% & 93/93 & 121 & 0.9589 & 1.294743 \\
44 & -103.19 & 3.33 & 4.5\% & 95/95 & 123 & 0.9577 & 1.101671 \\
45 & -92.13 & 4.00 & 4.4\% & 47/47 & 62 & 0.9571 & 1.253441 \\
46 & -234.35 & 3.67 & 4.3\% & 93/93 & 109 & 0.9560 & 1.171772 \\
47 & -117.76 & 4.71 & 4.3\% & 83/83 & 102 & 0.9550 & 1.327842 \\
48 & -208.50 & 4.37 & 4.2\% & 72/72 & 93 & 0.9540 & 1.232274 \\
49 & -133.23 & 4.49 & 4.1\% & 84/84 & 113 & 0.9529 & 1.210450 \\
50 & -287.21 & 4.50 & 4.0\% & 53/53 & 67 & 0.9523 & 1.417502 \\
51 & -16.45 & 4.31 & 3.9\% & 84/84 & 114 & 0.9511 & 1.245028 \\
52 & -71.91 & 5.40 & 3.8\% & 107/107 & 143 & 0.9497 & 1.329888 \\
53 & -163.40 & 5.03 & 3.8\% & 91/91 & 118 & 0.9485 & 1.228288 \\
54 & -105.90 & 5.17 & 3.7\% & 45/45 & 63 & 0.9479 & 1.079115 \\
55 & -81.13 & 5.31 & 3.6\% & 63/63 & 84 & 0.9471 & 1.186178 \\
56 & -483.11 & 5.22 & 3.6\% & 95/95 & 128 & 0.9458 & 1.180866 \\
57 & 11.33 & 6.19 & 3.5\% & 62/62 & 83 & 0.9450 & 1.452056 \\
58 & -98.10 & 6.02 & 3.4\% & 95/95 & 133 & 0.9437 & 1.216991 \\
59 & -66.84 & 6.01 & 3.4\% & 41/41 & 59 & 0.9431 & 1.334488 \\
60 & -108.26 & 5.94 & 3.3\% & 85/85 & 110 & 0.9420 & 1.317638 \\
61 & -212.80 & 6.19 & 3.3\% & 54/54 & 73 & 0.9413 & 1.312369 \\
62 & -124.19 & 7.90 & 3.2\% & 56/56 & 77 & 0.9405 & 1.325598 \\
63 & -245.88 & 7.00 & 3.2\% & 105/105 & 138 & 0.9392 & 1.350404 \\
64 & -204.32 & 6.85 & 3.1\% & 85/85 & 109 & 0.9381 & 1.379324 \\
65 & -208.55 & 7.07 & 3.1\% & 69/69 & 99 & 0.9371 & 1.309410 \\
66 & -122.98 & 6.97 & 3.0\% & 56/56 & 78 & 0.9363 & 1.123987 \\
67 & -111.92 & 8.43 & 3.0\% & 65/65 & 87 & 0.9355 & 1.279522 \\
68 & -215.86 & 7.95 & 2.9\% & 82/82 & 102 & 0.9345 & 1.667938 \\
69 & -108.49 & 7.98 & 2.9\% & 64/64 & 90 & 0.9336 & 1.294002 \\
70 & -140.15 & 8.30 & 2.9\% & 78/78 & 106 & 0.9325 & 1.422064 \\
71 & -98.17 & 7.97 & 2.8\% & 101/101 & 132 & 0.9312 & 1.435579 \\
72 & 20.05 & 9.07 & 2.8\% & 104/104 & 145 & 0.9298 & 1.454593 \\
73 & -88.93 & 9.31 & 2.7\% & 51/51 & 72 & 0.9291 & 1.419533 \\
74 & -138.76 & 9.05 & 2.7\% & 55/55 & 69 & 0.9284 & 1.328961 \\
75 & -70.51 & 9.08 & 2.7\% & 62/62 & 78 & 0.9276 & 1.352209 \\
76 & -215.37 & 8.86 & 2.6\% & 70/70 & 87 & 0.9267 & 1.441207 \\
77 & -83.07 & 9.00 & 2.6\% & 69/69 & 89 & 0.9259 & 1.502981 \\
78 & -107.71 & 10.47 & 2.6\% & 66/66 & 86 & 0.9250 & 1.541481 \\
79 & -105.43 & 10.15 & 2.5\% & 43/43 & 60 & 0.9244 & 1.642682 \\
80 & -93.15 & 10.16 & 2.5\% & 66/66 & 82 & 0.9236 & 1.241167 \\
81 & -53.54 & 10.37 & 2.5\% & 56/56 & 69 & 0.9229 & 1.419300 \\
82 & -119.15 & 10.36 & 2.4\% & 108/108 & 132 & 0.9216 & 1.413143 \\
83 & -119.06 & 11.74 & 2.4\% & 85/85 & 101 & 0.9206 & 1.517266 \\
84 & -244.76 & 11.25 & 2.4\% & 54/54 & 71 & 0.9199 & 1.601497 \\
85 & -114.49 & 11.42 & 2.4\% & 73/73 & 93 & 0.9190 & 1.296057 \\
86 & -73.93 & 11.38 & 2.3\% & 86/86 & 105 & 0.9180 & 1.425455 \\
87 & -122.68 & 11.23 & 2.3\% & 79/79 & 107 & 0.9169 & 1.469567 \\
88 & -366.05 & 11.69 & 2.3\% & 80/80 & 102 & 0.9159 & 1.549518 \\
89 & -79.22 & 12.47 & 2.2\% & 49/49 & 65 & 0.9152 & 1.514544 \\
90 & -137.95 & 12.31 & 2.2\% & 50/50 & 75 & 0.9145 & 1.328946 \\
91 & -111.61 & 12.27 & 2.2\% & 87/87 & 103 & 0.9135 & 1.298701 \\
92 & -288.01 & 12.43 & 2.2\% & 67/67 & 86 & 0.9126 & 1.636529 \\
93 & -115.04 & 12.44 & 2.2\% & 78/78 & 98 & 0.9117 & 1.522967 \\
94 & -77.74 & 13.92 & 2.1\% & 100/100 & 134 & 0.9103 & 1.486537 \\
95 & -71.35 & 13.73 & 2.1\% & 110/110 & 139 & 0.9090 & 1.618968 \\
96 & -81.41 & 13.70 & 2.1\% & 89/89 & 115 & 0.9078 & 1.404128 \\
97 & -66.37 & 13.92 & 2.1\% & 85/85 & 114 & 0.9067 & 1.474638 \\
98 & -108.43 & 13.63 & 2.0\% & 56/56 & 68 & 0.9060 & 1.487921 \\
99 & -135.54 & 14.99 & 2.0\% & 48/48 & 63 & 0.9054 & 1.619277 \\
100 & -116.09 & 14.80 & 2.0\% & 96/96 & 123 & 0.9042 & 1.370284 \\
101 & -47.07 & 15.07 & 2.0\% & 54/54 & 69 & 0.9035 & 1.577216 \\
102 & -97.14 & 14.50 & 2.0\% & 66/66 & 79 & 0.9027 & 1.288804 \\
103 & -359.71 & 14.72 & 2.0\% & 98/98 & 118 & 0.9015 & 1.437954 \\
104 & -59.01 & 16.21 & 2.0\% & 58/58 & 75 & 0.9008 & 1.700163 \\
105 & -201.64 & 16.22 & 2.0\% & 89/89 & 116 & 0.8997 & 1.490932 \\
106 & -232.70 & 16.16 & 2.0\% & 98/98 & 119 & 0.8985 & 1.324214 \\
107 & -51.81 & 16.38 & 2.0\% & 58/58 & 67 & 0.8978 & 1.428422 \\
108 & -224.17 & 16.16 & 2.0\% & 64/64 & 73 & 0.8971 & 1.473464 \\
109 & -192.16 & 16.82 & 2.0\% & 97/97 & 122 & 0.8959 & 1.578725 \\
110 & -340.38 & 16.85 & 2.0\% & 73/73 & 88 & 0.8950 & 1.880671 \\
111 & -148.51 & 16.85 & 1.0\% & 55/55 & 71 & 0.8943 & 1.578203 \\
112 & -61.08 & 16.93 & 2.0\% & 91/91 & 121 & 0.8931 & 1.489684 \\
113 & -265.68 & 16.76 & 2.0\% & 78/78 & 97 & 0.8921 & 1.575589 \\
114 & -144.69 & 18.06 & 2.0\% & 131/131 & 155 & 0.8906 & 1.627576 \\
115 & -153.86 & 17.74 & 2.0\% & 83/83 & 103 & 0.8896 & 1.487052 \\
116 & -59.56 & 18.40 & 2.0\% & 79/79 & 114 & 0.8885 & 1.758846 \\
117 & -85.36 & 17.77 & 2.0\% & 74/74 & 93 & 0.8875 & 1.488655 \\
118 & -52.79 & 18.42 & 2.0\% & 67/67 & 82 & 0.8867 & 1.398832 \\
119 & -99.25 & 19.10 & 1.0\% & 72/72 & 93 & 0.8858 & 1.440740 \\
120 & -117.89 & 19.56 & 1.0\% & 75/75 & 102 & 0.8848 & 1.521538 \\
121 & -75.03 & 19.38 & 1.0\% & 57/57 & 67 & 0.8841 & 1.195997 \\
122 & -13.20 & 19.69 & 1.0\% & 116/116 & 147 & 0.8827 & 1.575629 \\
123 & -100.36 & 19.69 & 1.0\% & 47/47 & 64 & 0.8821 & 1.611541 \\
124 & -98.80 & 19.63 & 1.0\% & 51/51 & 67 & 0.8814 & 1.786981 \\
125 & -81.80 & 20.35 & 1.0\% & 81/81 & 95 & 0.8804 & 1.530408 \\
126 & -75.22 & 20.09 & 2.0\% & 92/92 & 124 & 0.8792 & 1.613012 \\
127 & -70.44 & 20.30 & 2.0\% & 102/102 & 119 & 0.8780 & 1.569762 \\
128 & -89.57 & 19.91 & 2.0\% & 81/81 & 108 & 0.8770 & 1.465846 \\
129 & -85.75 & 20.69 & 2.0\% & 84/84 & 111 & 0.8759 & 1.558024 \\
130 & -72.32 & 20.87 & 3.0\% & 78/78 & 97 & 0.8749 & 1.826849 \\
131 & -146.71 & 20.61 & 3.0\% & 71/71 & 87 & 0.8741 & 1.722103 \\
132 & -124.50 & 20.80 & 3.0\% & 69/69 & 83 & 0.8732 & 1.524684 \\
133 & -117.62 & 20.99 & 3.0\% & 60/60 & 81 & 0.8724 & 1.516726 \\
134 & -69.80 & 20.80 & 4.0\% & 67/67 & 88 & 0.8716 & 1.400355 \\
135 & -73.01 & 21.62 & 4.0\% & 79/79 & 95 & 0.8706 & 1.694743 \\
136 & -57.95 & 21.61 & 4.0\% & 56/56 & 75 & 0.8699 & 1.833061 \\
137 & -64.27 & 21.93 & 4.0\% & 94/94 & 124 & 0.8686 & 1.502878 \\
138 & -85.23 & 21.79 & 4.0\% & 101/101 & 139 & 0.8673 & 1.478509 \\
139 & -61.53 & 21.58 & 4.0\% & 54/54 & 69 & 0.8666 & 1.267713 \\
140 & -208.89 & 22.61 & 4.0\% & 90/90 & 116 & 0.8654 & 1.626684 \\
141 & -53.10 & 22.71 & 4.0\% & 62/62 & 79 & 0.8647 & 1.584719 \\
142 & -154.71 & 22.62 & 4.0\% & 106/106 & 141 & 0.8633 & 1.491198 \\
143 & -106.15 & 22.89 & 4.0\% & 66/66 & 87 & 0.8624 & 1.371870 \\
144 & -67.55 & 22.66 & 4.0\% & 62/62 & 79 & 0.8616 & 1.619559 \\
145 & -61.37 & 23.79 & 4.0\% & 88/88 & 110 & 0.8605 & 1.450032 \\
146 & -84.68 & 23.60 & 4.0\% & 84/84 & 111 & 0.8594 & 1.635570 \\
147 & -273.23 & 23.42 & 4.0\% & 60/60 & 83 & 0.8586 & 1.267417 \\
148 & -53.19 & 23.66 & 4.0\% & 93/93 & 123 & 0.8574 & 1.750101 \\
149 & -61.36 & 23.39 & 4.0\% & 56/56 & 73 & 0.8567 & 1.442780 \\
150 & -64.59 & 24.09 & 4.0\% & 52/52 & 68 & 0.8560 & 1.293135 \\
151 & -101.49 & 24.24 & 4.0\% & 67/67 & 85 & 0.8552 & 1.323254 \\
152 & -89.19 & 24.47 & 4.0\% & 70/70 & 89 & 0.8543 & 1.591605 \\
153 & -84.92 & 24.32 & 4.0\% & 88/88 & 114 & 0.8531 & 1.457696 \\
154 & -79.96 & 24.31 & 4.0\% & 78/78 & 103 & 0.8521 & 1.524177 \\
155 & -129.26 & 25.11 & 4.0\% & 51/51 & 66 & 0.8515 & 1.656500 \\
156 & -120.89 & 25.09 & 4.0\% & 76/76 & 103 & 0.8505 & 1.548660 \\
157 & -40.73 & 25.38 & 4.0\% & 71/71 & 83 & 0.8496 & 1.442775 \\
158 & -57.12 & 25.09 & 4.0\% & 51/51 & 67 & 0.8490 & 1.550241 \\
159 & -58.15 & 25.14 & 4.0\% & 53/53 & 68 & 0.8483 & 1.584052 \\
160 & -64.84 & 25.33 & 4.0\% & 55/55 & 66 & 0.8476 & 1.096623 \\
161 & -95.97 & 25.06 & 4.0\% & 52/52 & 74 & 0.8469 & 1.311186 \\
162 & -123.13 & 26.29 & 4.0\% & 79/79 & 99 & 0.8459 & 1.487261 \\
163 & -81.78 & 26.53 & 4.0\% & 74/74 & 94 & 0.8450 & 1.628722 \\
164 & -68.70 & 26.17 & 4.0\% & 56/56 & 68 & 0.8443 & 1.708853 \\
165 & -118.86 & 26.42 & 4.0\% & 54/54 & 73 & 0.8436 & 1.485840 \\
166 & -85.77 & 26.24 & 4.0\% & 95/95 & 128 & 0.8423 & 1.315695 \\
167 & -86.04 & 26.25 & 4.0\% & 80/80 & 107 & 0.8413 & 1.854084 \\
168 & -143.64 & 26.35 & 4.0\% & 57/57 & 73 & 0.8406 & 1.381531 \\
169 & -90.46 & 26.81 & 4.0\% & 95/95 & 123 & 0.8393 & 1.628735 \\
170 & -100.95 & 26.31 & 4.0\% & 73/73 & 90 & 0.8384 & 1.255750 \\
171 & -237.26 & 26.78 & 4.0\% & 96/96 & 119 & 0.8373 & 1.478187 \\
172 & -78.27 & 26.72 & 4.0\% & 55/55 & 66 & 0.8366 & 1.343799 \\
173 & -89.39 & 26.90 & 4.0\% & 52/52 & 64 & 0.8360 & 1.377220 \\
174 & -71.24 & 26.73 & 5.0\% & 60/60 & 82 & 0.8352 & 1.483567 \\
175 & -123.16 & 26.76 & 5.0\% & 86/86 & 107 & 0.8341 & 1.891692 \\
176 & -113.16 & 26.92 & 5.0\% & 101/101 & 125 & 0.8329 & 1.489870 \\
177 & -92.12 & 26.53 & 5.0\% & 67/67 & 91 & 0.8320 & 1.175172 \\
178 & -85.90 & 26.29 & 5.0\% & 54/54 & 76 & 0.8312 & 1.958608 \\
179 & -89.97 & 26.53 & 5.0\% & 105/105 & 125 & 0.8300 & 1.342431 \\
180 & -349.17 & 26.70 & 5.0\% & 86/86 & 110 & 0.8289 & 1.586825 \\
181 & -85.20 & 26.71 & 5.0\% & 60/60 & 79 & 0.8281 & 1.707335 \\
182 & -85.18 & 26.39 & 5.0\% & 103/103 & 130 & 0.8268 & 1.544973 \\
183 & -61.03 & 26.76 & 5.0\% & 84/84 & 106 & 0.8258 & 1.973738 \\
184 & -87.31 & 27.16 & 5.0\% & 57/57 & 74 & 0.8250 & 1.638357 \\
185 & -106.14 & 26.47 & 5.0\% & 75/75 & 94 & 0.8241 & 1.564892 \\
186 & -90.28 & 27.58 & 5.0\% & 80/80 & 112 & 0.8230 & 1.636100 \\
187 & -138.45 & 27.07 & 5.0\% & 82/82 & 105 & 0.8220 & 1.774899 \\
188 & -128.95 & 27.11 & 5.0\% & 75/75 & 93 & 0.8210 & 1.820779 \\
189 & -22.25 & 26.98 & 5.0\% & 120/120 & 149 & 0.8196 & 1.833248 \\
190 & -105.43 & 26.88 & 5.0\% & 61/61 & 75 & 0.8188 & 1.361761 \\
191 & -107.00 & 26.55 & 5.0\% & 78/78 & 94 & 0.8179 & 1.471876 \\
192 & -90.21 & 27.45 & 5.0\% & 89/89 & 111 & 0.8168 & 1.780228 \\
193 & -59.57 & 27.37 & 5.0\% & 77/77 & 89 & 0.8159 & 1.653543 \\
194 & -121.87 & 26.84 & 5.0\% & 102/102 & 129 & 0.8146 & 1.655400 \\
195 & -82.33 & 27.27 & 5.0\% & 88/88 & 116 & 0.8135 & 1.572435 \\
196 & -105.63 & 26.55 & 5.0\% & 51/51 & 65 & 0.8128 & 1.487846 \\
197 & -100.08 & 25.93 & 5.0\% & 86/86 & 112 & 0.8117 & 1.613053 \\
198 & -243.21 & 26.68 & 5.0\% & 79/79 & 95 & 0.8108 & 1.649063 \\
199 & -67.37 & 27.04 & 5.0\% & 69/69 & 87 & 0.8099 & 1.890146 \\
200 & -112.97 & 26.55 & 5.0\% & 83/83 & 103 & 0.8089 & 1.764121 \\
201 & -117.23 & 26.09 & 5.0\% & 74/74 & 111 & 0.8078 & 1.965949 \\
202 & -99.90 & 26.17 & 5.0\% & 88/88 & 109 & 0.8067 & 1.608318 \\
203 & -80.65 & 26.30 & 5.0\% & 62/62 & 76 & 0.8060 & 1.439143 \\
204 & -86.44 & 26.62 & 5.0\% & 61/61 & 77 & 0.8052 & 1.302299 \\
205 & -129.39 & 26.29 & 5.0\% & 89/89 & 106 & 0.8042 & 1.368889 \\
206 & -42.54 & 26.45 & 5.0\% & 53/53 & 69 & 0.8035 & 1.671000 \\
207 & -68.61 & 26.19 & 5.0\% & 62/62 & 79 & 0.8027 & 1.401159 \\
208 & -48.26 & 26.28 & 5.0\% & 65/65 & 78 & 0.8019 & 1.657288 \\
209 & -63.76 & 26.25 & 5.0\% & 50/50 & 63 & 0.8013 & 1.516010 \\
210 & -60.19 & 26.30 & 5.0\% & 92/92 & 118 & 0.8001 & 1.479153 \\
211 & -53.20 & 26.61 & 5.0\% & 81/81 & 98 & 0.7992 & 1.931113 \\
212 & -55.43 & 26.83 & 4.0\% & 46/46 & 64 & 0.7985 & 1.668519 \\
213 & -107.57 & 26.38 & 4.0\% & 71/71 & 94 & 0.7976 & 1.464684 \\
214 & -105.66 & 26.72 & 4.0\% & 66/66 & 87 & 0.7967 & 1.659493 \\
215 & -36.75 & 26.77 & 5.0\% & 56/56 & 72 & 0.7960 & 1.719652 \\
216 & -46.76 & 26.55 & 5.0\% & 80/80 & 102 & 0.7950 & 1.523213 \\
217 & -142.11 & 27.21 & 5.0\% & 81/81 & 101 & 0.7940 & 1.818099 \\
218 & -80.40 & 26.57 & 5.0\% & 78/78 & 101 & 0.7930 & 1.528571 \\
219 & -99.02 & 27.77 & 5.0\% & 81/81 & 97 & 0.7921 & 1.644388 \\
220 & -122.06 & 26.21 & 5.0\% & 70/70 & 94 & 0.7911 & 1.672793 \\
221 & -62.69 & 26.50 & 5.0\% & 68/68 & 90 & 0.7902 & 1.575978 \\
222 & -68.41 & 26.89 & 5.0\% & 54/54 & 65 & 0.7896 & 1.603405 \\
223 & -53.54 & 26.81 & 5.0\% & 54/54 & 66 & 0.7889 & 1.638188 \\
224 & -112.81 & 26.54 & 5.0\% & 75/75 & 91 & 0.7880 & 1.470540 \\
225 & -81.33 & 25.12 & 5.0\% & 83/83 & 104 & 0.7870 & 1.600476 \\
226 & -74.59 & 26.24 & 4.0\% & 50/50 & 65 & 0.7864 & 1.469472 \\
227 & -70.70 & 25.83 & 4.0\% & 70/70 & 86 & 0.7855 & 1.570140 \\
228 & -19.11 & 25.88 & 5.0\% & 93/93 & 119 & 0.7843 & 1.594381 \\
229 & -52.25 & 26.48 & 6.0\% & 79/79 & 97 & 0.7834 & 1.633891 \\
230 & -140.85 & 26.41 & 5.0\% & 70/70 & 101 & 0.7824 & 1.589230 \\
231 & -78.44 & 25.49 & 5.0\% & 57/57 & 68 & 0.7817 & 1.695189 \\
232 & -75.26 & 25.12 & 5.0\% & 65/65 & 78 & 0.7809 & 1.739530 \\
233 & -58.07 & 25.39 & 5.0\% & 72/72 & 91 & 0.7800 & 1.908603 \\
234 & -96.96 & 25.46 & 4.0\% & 51/51 & 68 & 0.7794 & 1.421748 \\
235 & -119.88 & 25.40 & 4.0\% & 75/75 & 93 & 0.7784 & 1.367479 \\
236 & -72.52 & 25.36 & 4.0\% & 78/78 & 95 & 0.7775 & 1.617569 \\
237 & -79.88 & 25.01 & 4.0\% & 60/60 & 79 & 0.7767 & 1.515743 \\
238 & -103.16 & 24.80 & 4.0\% & 68/68 & 90 & 0.7758 & 1.565967 \\
239 & -174.37 & 25.05 & 4.0\% & 110/110 & 131 & 0.7745 & 1.745135 \\
240 & -83.51 & 24.68 & 4.0\% & 88/88 & 106 & 0.7735 & 1.512653 \\
241 & -85.07 & 24.59 & 4.0\% & 49/49 & 63 & 0.7729 & 1.448766 \\
242 & -113.21 & 24.76 & 4.0\% & 75/75 & 91 & 0.7720 & 1.718916 \\
243 & -61.35 & 24.78 & 4.0\% & 77/77 & 97 & 0.7710 & 1.691580 \\
244 & -102.81 & 24.70 & 4.0\% & 93/93 & 112 & 0.7699 & 1.699395 \\
245 & -49.67 & 24.75 & 5.0\% & 57/57 & 74 & 0.7692 & 1.664775 \\
246 & -171.00 & 24.83 & 5.0\% & 60/60 & 77 & 0.7684 & 1.416763 \\
247 & -78.11 & 24.50 & 5.0\% & 51/51 & 60 & 0.7678 & 1.671559 \\
248 & -97.86 & 24.62 & 5.0\% & 73/73 & 95 & 0.7669 & 1.820242 \\
249 & -163.88 & 24.03 & 5.0\% & 71/71 & 94 & 0.7659 & 1.586567 \\
250 & -62.89 & 24.44 & 6.0\% & 81/81 & 103 & 0.7649 & 1.501691 \\
251 & -77.98 & 25.02 & 6.0\% & 72/72 & 84 & 0.7641 & 1.897511 \\
252 & -109.88 & 24.10 & 6.0\% & 60/60 & 84 & 0.7632 & 1.469516 \\
253 & -59.02 & 24.25 & 6.0\% & 49/49 & 58 & 0.7627 & 1.446086 \\
254 & -20.95 & 23.97 & 6.0\% & 107/107 & 128 & 0.7614 & 1.804672 \\
255 & -83.36 & 23.91 & 6.0\% & 93/93 & 114 & 0.7603 & 1.786525 \\
256 & -32.87 & 24.19 & 7.0\% & 99/99 & 126 & 0.7590 & 1.809394 \\
257 & -162.55 & 24.22 & 7.0\% & 83/83 & 107 & 0.7580 & 1.610611 \\
258 & -100.35 & 23.76 & 7.0\% & 67/67 & 85 & 0.7571 & 1.629845 \\
259 & -66.24 & 24.03 & 7.0\% & 78/78 & 95 & 0.7562 & 1.776336 \\
260 & -82.45 & 23.58 & 7.0\% & 69/69 & 93 & 0.7553 & 1.506323 \\
261 & -64.78 & 23.69 & 7.0\% & 53/53 & 69 & 0.7546 & 1.383717 \\
262 & -46.31 & 23.72 & 7.0\% & 76/76 & 95 & 0.7536 & 1.513800 \\
263 & -40.74 & 23.71 & 7.0\% & 56/56 & 68 & 0.7530 & 1.520614 \\
264 & -111.66 & 23.58 & 7.0\% & 92/92 & 111 & 0.7519 & 1.705278 \\
265 & -90.61 & 23.46 & 7.0\% & 109/109 & 132 & 0.7506 & 1.613424 \\
266 & -76.08 & 23.02 & 7.0\% & 103/103 & 123 & 0.7493 & 1.514840 \\
267 & -99.76 & 22.84 & 7.0\% & 81/81 & 99 & 0.7484 & 1.698353 \\
268 & -36.80 & 23.48 & 8.0\% & 50/50 & 62 & 0.7477 & 1.926447 \\
269 & -115.30 & 22.34 & 8.0\% & 71/71 & 87 & 0.7469 & 1.682132 \\
270 & -205.81 & 22.62 & 8.0\% & 95/95 & 126 & 0.7456 & 1.358899 \\
271 & -75.95 & 22.94 & 9.0\% & 72/72 & 94 & 0.7447 & 1.550614 \\
272 & -44.38 & 22.35 & 10.0\% & 68/68 & 86 & 0.7439 & 1.583684 \\
273 & -49.87 & 22.63 & 10.0\% & 82/82 & 98 & 0.7429 & 1.378509 \\
274 & -79.99 & 22.42 & 9.0\% & 67/67 & 88 & 0.7420 & 1.840617 \\
275 & -85.38 & 22.49 & 9.0\% & 75/75 & 92 & 0.7411 & 1.769359 \\
276 & -50.50 & 22.20 & 9.0\% & 63/63 & 79 & 0.7403 & 1.159865 \\
277 & -26.16 & 21.96 & 10.0\% & 91/91 & 113 & 0.7392 & 1.496208 \\
278 & -87.29 & 22.52 & 10.0\% & 60/60 & 79 & 0.7384 & 1.763550 \\
279 & -116.31 & 21.72 & 10.0\% & 95/95 & 114 & 0.7373 & 1.702121 \\
280 & -50.01 & 21.68 & 10.0\% & 57/57 & 75 & 0.7366 & 1.669839 \\
281 & -77.81 & 21.72 & 10.0\% & 67/67 & 82 & 0.7357 & 1.885429 \\
282 & -240.73 & 21.69 & 10.0\% & 73/73 & 89 & 0.7349 & 1.551954 \\
283 & -125.46 & 22.14 & 10.0\% & 87/87 & 108 & 0.7338 & 1.378341 \\
284 & -44.15 & 21.07 & 10.0\% & 99/99 & 130 & 0.7325 & 1.753301 \\
285 & -89.33 & 21.53 & 10.0\% & 77/77 & 92 & 0.7316 & 1.560487 \\
286 & -81.95 & 21.58 & 10.0\% & 78/78 & 94 & 0.7307 & 1.980043 \\
287 & -33.56 & 21.75 & 10.0\% & 98/98 & 121 & 0.7295 & 1.645130 \\
288 & -82.51 & 21.70 & 10.0\% & 69/69 & 84 & 0.7286 & 1.686377 \\
289 & -105.22 & 21.94 & 10.0\% & 63/63 & 84 & 0.7278 & 1.840046 \\
290 & -40.99 & 20.33 & 10.0\% & 80/80 & 98 & 0.7268 & 1.785865 \\
291 & -19.14 & 20.96 & 11.0\% & 102/102 & 125 & 0.7256 & 1.576077 \\
292 & -222.88 & 20.33 & 11.0\% & 118/118 & 143 & 0.7242 & 1.512681 \\
293 & -64.72 & 20.62 & 11.0\% & 86/86 & 102 & 0.7232 & 1.608906 \\
294 & -108.12 & 20.76 & 11.0\% & 72/72 & 97 & 0.7222 & 1.628580 \\
295 & -134.69 & 20.59 & 11.0\% & 75/75 & 102 & 0.7212 & 1.738389 \\
296 & -64.54 & 20.49 & 11.0\% & 75/75 & 106 & 0.7201 & 1.780805 \\
297 & -124.59 & 20.16 & 11.0\% & 66/66 & 77 & 0.7194 & 1.993632 \\
298 & -281.73 & 20.99 & 11.0\% & 88/88 & 105 & 0.7183 & 1.648067 \\
299 & -80.71 & 19.93 & 11.0\% & 75/75 & 82 & 0.7175 & 2.074172 \\
300 & -35.27 & 19.43 & 12.0\% & 96/96 & 125 & 0.7163 & 1.417082 \\
301 & -32.82 & 20.17 & 12.0\% & 101/101 & 121 & 0.7151 & 2.076162 \\
302 & -55.22 & 20.18 & 12.0\% & 63/63 & 77 & 0.7143 & 1.831566 \\
303 & -50.48 & 20.06 & 13.0\% & 84/84 & 108 & 0.7133 & 1.952185 \\
304 & -49.00 & 19.45 & 13.0\% & 44/44 & 61 & 0.7127 & 2.066209 \\
305 & -143.29 & 19.10 & 13.0\% & 94/94 & 119 & 0.7115 & 1.735314 \\
306 & -68.87 & 19.14 & 13.0\% & 78/78 & 89 & 0.7106 & 1.659685 \\
307 & -28.66 & 19.19 & 13.0\% & 117/117 & 141 & 0.7092 & 1.707224 \\
308 & -119.57 & 19.22 & 13.0\% & 74/74 & 88 & 0.7083 & 1.851744 \\
309 & -44.52 & 18.02 & 14.0\% & 64/64 & 80 & 0.7075 & 1.936012 \\
310 & -122.64 & 18.77 & 14.0\% & 57/57 & 66 & 0.7069 & 1.756571 \\
311 & -62.81 & 18.36 & 14.0\% & 90/90 & 116 & 0.7057 & 1.791892 \\
312 & -84.96 & 17.93 & 14.0\% & 51/51 & 62 & 0.7051 & 1.770406 \\
313 & -57.51 & 18.11 & 14.0\% & 92/92 & 111 & 0.7040 & 1.537989 \\
314 & -57.37 & 18.08 & 14.0\% & 51/51 & 65 & 0.7034 & 1.687235 \\
315 & -78.11 & 18.00 & 13.0\% & 78/78 & 101 & 0.7024 & 1.836397 \\
316 & -69.08 & 18.18 & 13.0\% & 85/85 & 102 & 0.7014 & 1.861847 \\
317 & -61.93 & 17.53 & 13.0\% & 53/53 & 74 & 0.7006 & 1.800928 \\
318 & -50.22 & 17.85 & 14.0\% & 68/68 & 88 & 0.6998 & 1.455161 \\
319 & -126.45 & 18.24 & 14.0\% & 59/59 & 76 & 0.6990 & 1.668143 \\
320 & -59.63 & 18.04 & 15.0\% & 76/76 & 91 & 0.6981 & 1.678668 \\
321 & -53.40 & 17.32 & 15.0\% & 97/97 & 116 & 0.6970 & 1.746313 \\
322 & -46.70 & 17.01 & 15.0\% & 106/106 & 125 & 0.6957 & 1.801074 \\
323 & -58.41 & 17.50 & 15.0\% & 64/64 & 85 & 0.6949 & 1.871027 \\
324 & -24.62 & 16.58 & 15.0\% & 80/80 & 103 & 0.6939 & 1.579588 \\
325 & -11.01 & 16.78 & 15.0\% & 74/74 & 97 & 0.6929 & 1.591653 \\
326 & -72.28 & 16.75 & 15.0\% & 42/42 & 59 & 0.6923 & 1.531634 \\
327 & -48.81 & 17.12 & 16.0\% & 56/56 & 76 & 0.6916 & 1.387910 \\
328 & -116.32 & 17.44 & 15.0\% & 66/66 & 80 & 0.6908 & 1.773861 \\
329 & -18.86 & 17.19 & 14.0\% & 79/79 & 91 & 0.6899 & 1.872267 \\
330 & -41.07 & 17.17 & 14.0\% & 88/88 & 114 & 0.6888 & 1.865366 \\
331 & -42.60 & 16.93 & 14.0\% & 70/70 & 86 & 0.6879 & 1.754085 \\
332 & -82.95 & 16.28 & 14.0\% & 83/83 & 107 & 0.6868 & 1.603006 \\
333 & -16.44 & 16.69 & 14.0\% & 92/92 & 119 & 0.6857 & 1.799694 \\
334 & -46.29 & 16.81 & 14.0\% & 86/86 & 99 & 0.6847 & 1.799144 \\
335 & -201.85 & 16.88 & 14.0\% & 93/93 & 124 & 0.6835 & 1.972734 \\
336 & -22.46 & 15.80 & 14.0\% & 64/64 & 76 & 0.6827 & 1.535983 \\
337 & -110.99 & 16.66 & 14.0\% & 59/59 & 76 & 0.6820 & 1.759243 \\
338 & -61.40 & 15.89 & 14.0\% & 60/60 & 71 & 0.6812 & 1.931025 \\
339 & -122.31 & 16.34 & 14.0\% & 99/99 & 124 & 0.6800 & 1.741466 \\
340 & -98.47 & 15.84 & 14.0\% & 68/68 & 87 & 0.6792 & 1.690666 \\
341 & -62.02 & 16.15 & 14.0\% & 101/101 & 118 & 0.6780 & 1.672343 \\
342 & -97.70 & 16.08 & 14.0\% & 92/92 & 110 & 0.6769 & 1.869687 \\
343 & -25.27 & 15.61 & 15.0\% & 50/50 & 67 & 0.6762 & 1.834508 \\
344 & -98.17 & 15.58 & 15.0\% & 90/90 & 108 & 0.6752 & 1.687799 \\
345 & -83.80 & 16.12 & 14.0\% & 86/86 & 98 & 0.6742 & 2.044104 \\
346 & -106.66 & 15.25 & 14.0\% & 99/99 & 121 & 0.6730 & 1.888783 \\
347 & -73.39 & 15.82 & 14.0\% & 68/68 & 84 & 0.6722 & 1.757684 \\
348 & -9.87 & 15.22 & 14.0\% & 59/59 & 74 & 0.6714 & 1.783733 \\
349 & -54.74 & 14.97 & 14.0\% & 69/69 & 85 & 0.6706 & 1.955536 \\
350 & -47.46 & 14.87 & 13.0\% & 94/94 & 121 & 0.6694 & 1.989640 \\
351 & -86.46 & 15.16 & 13.0\% & 72/72 & 85 & 0.6686 & 1.896518 \\
352 & -43.45 & 15.14 & 13.0\% & 103/103 & 126 & 0.6673 & 1.891866 \\
353 & -62.31 & 15.14 & 14.0\% & 72/72 & 90 & 0.6664 & 1.926734 \\
354 & -41.41 & 14.94 & 14.0\% & 75/75 & 92 & 0.6655 & 1.972740 \\
355 & -56.19 & 14.65 & 14.0\% & 69/69 & 86 & 0.6647 & 1.859285 \\
356 & -29.58 & 14.60 & 13.0\% & 61/61 & 69 & 0.6640 & 1.935376 \\
357 & -86.69 & 14.42 & 13.0\% & 83/83 & 100 & 0.6630 & 1.754790 \\
358 & 16.64 & 14.16 & 14.0\% & 81/81 & 102 & 0.6620 & 1.905739 \\
359 & -72.64 & 14.47 & 14.0\% & 71/71 & 82 & 0.6612 & 1.889100 \\
360 & -47.72 & 15.00 & 14.0\% & 76/76 & 95 & 0.6602 & 1.564476 \\
361 & -30.67 & 14.67 & 14.0\% & 75/75 & 90 & 0.6593 & 1.759931 \\
362 & -32.85 & 15.33 & 15.0\% & 95/95 & 116 & 0.6582 & 1.738984 \\
363 & -109.04 & 14.54 & 15.0\% & 78/78 & 101 & 0.6572 & 1.747104 \\
364 & -60.96 & 14.67 & 15.0\% & 61/61 & 73 & 0.6565 & 1.769159 \\
365 & -36.60 & 14.91 & 15.0\% & 114/114 & 133 & 0.6551 & 1.621841 \\
366 & -50.91 & 15.00 & 15.0\% & 119/119 & 144 & 0.6537 & 1.537255 \\
367 & -98.73 & 14.60 & 15.0\% & 66/66 & 84 & 0.6529 & 1.754956 \\
368 & -174.26 & 15.07 & 14.0\% & 123/123 & 143 & 0.6515 & 1.600036 \\
369 & -125.69 & 14.20 & 14.0\% & 52/52 & 61 & 0.6509 & 1.744739 \\
370 & -58.38 & 14.04 & 14.0\% & 69/69 & 88 & 0.6500 & 1.956068 \\
371 & -89.24 & 14.40 & 13.0\% & 83/83 & 104 & 0.6490 & 1.681788 \\
372 & -9.69 & 14.58 & 12.0\% & 111/111 & 129 & 0.6477 & 1.884943 \\
373 & -51.27 & 14.10 & 12.0\% & 61/61 & 75 & 0.6469 & 1.660531 \\
374 & -39.50 & 14.37 & 13.0\% & 91/91 & 111 & 0.6458 & 1.971175 \\
375 & -38.58 & 14.41 & 13.0\% & 125/125 & 148 & 0.6444 & 1.926273 \\
376 & -63.76 & 13.40 & 13.0\% & 85/85 & 103 & 0.6434 & 1.853180 \\
377 & -52.76 & 13.43 & 12.0\% & 67/67 & 83 & 0.6425 & 1.974796 \\
378 & -66.53 & 12.23 & 12.0\% & 74/74 & 86 & 0.6417 & 1.657912 \\
379 & -112.87 & 13.47 & 12.0\% & 98/98 & 126 & 0.6404 & 1.860765 \\
380 & -66.60 & 13.38 & 12.0\% & 68/68 & 87 & 0.6396 & 1.601088 \\
381 & -62.88 & 13.46 & 12.0\% & 82/82 & 104 & 0.6386 & 1.715886 \\
382 & -43.05 & 13.49 & 12.0\% & 59/59 & 77 & 0.6378 & 2.087928 \\
383 & -58.24 & 13.76 & 12.0\% & 106/106 & 125 & 0.6366 & 1.657168 \\
384 & -110.16 & 13.41 & 12.0\% & 102/102 & 123 & 0.6353 & 1.602685 \\
385 & -48.22 & 13.57 & 12.0\% & 65/65 & 82 & 0.6345 & 1.992476 \\
386 & -86.67 & 13.52 & 12.0\% & 82/82 & 107 & 0.6335 & 1.923120 \\
387 & -70.20 & 12.60 & 12.0\% & 86/86 & 110 & 0.6324 & 1.752192 \\
388 & -84.18 & 13.42 & 12.0\% & 68/68 & 86 & 0.6315 & 1.909667 \\
389 & -61.04 & 13.77 & 12.0\% & 114/114 & 137 & 0.6302 & 1.505875 \\
390 & -63.23 & 13.20 & 12.0\% & 88/88 & 107 & 0.6291 & 1.565003 \\
391 & -78.04 & 12.72 & 11.0\% & 94/94 & 108 & 0.6280 & 1.953839 \\
392 & -13.37 & 12.80 & 11.0\% & 78/78 & 93 & 0.6271 & 1.757620 \\
393 & -54.91 & 12.44 & 11.0\% & 107/107 & 129 & 0.6258 & 2.053123 \\
394 & -47.02 & 13.15 & 11.0\% & 89/89 & 108 & 0.6248 & 1.960711 \\
395 & -77.94 & 13.20 & 11.0\% & 73/73 & 91 & 0.6239 & 1.822483 \\
396 & 11.38 & 12.52 & 12.0\% & 73/73 & 90 & 0.6230 & 2.152686 \\
397 & -204.73 & 12.54 & 12.0\% & 99/99 & 123 & 0.6218 & 1.961305 \\
398 & -43.94 & 12.43 & 12.0\% & 59/59 & 70 & 0.6211 & 1.716983 \\
399 & -75.56 & 11.86 & 12.0\% & 60/60 & 76 & 0.6203 & 1.984464 \\
400 & -73.87 & 12.57 & 11.0\% & 80/80 & 100 & 0.6193 & 1.786896 \\
401 & -53.73 & 11.33 & 11.0\% & 52/52 & 62 & 0.6187 & 1.920979 \\
402 & -89.37 & 11.89 & 11.0\% & 71/71 & 99 & 0.6177 & 1.943319 \\
403 & -55.76 & 12.51 & 10.0\% & 58/58 & 67 & 0.6171 & 1.704164 \\
404 & -28.99 & 11.65 & 11.0\% & 52/52 & 67 & 0.6164 & 1.675194 \\
405 & -17.39 & 12.27 & 12.0\% & 82/82 & 97 & 0.6154 & 1.991226 \\
406 & -46.81 & 11.98 & 12.0\% & 60/60 & 77 & 0.6147 & 1.472614 \\
407 & 8.53 & 12.36 & 13.0\% & 97/97 & 116 & 0.6135 & 1.723015 \\
408 & -167.15 & 12.75 & 13.0\% & 108/108 & 140 & 0.6121 & 2.057488 \\
409 & -135.32 & 12.03 & 12.0\% & 116/116 & 140 & 0.6108 & 1.896270 \\
410 & -42.92 & 12.49 & 12.0\% & 85/85 & 115 & 0.6096 & 1.895292 \\
411 & -14.33 & 11.21 & 13.0\% & 108/108 & 126 & 0.6084 & 1.891892 \\
412 & -21.99 & 11.50 & 13.0\% & 102/102 & 120 & 0.6072 & 1.749937 \\
413 & -24.66 & 12.10 & 14.0\% & 105/105 & 121 & 0.6060 & 1.985496 \\
414 & -50.44 & 11.71 & 14.0\% & 91/91 & 108 & 0.6049 & 2.170080 \\
415 & -22.54 & 11.82 & 14.0\% & 85/85 & 106 & 0.6039 & 1.908250 \\
416 & -20.56 & 11.58 & 14.0\% & 75/75 & 90 & 0.6030 & 1.747738 \\
417 & -13.57 & 11.77 & 14.0\% & 78/78 & 93 & 0.6021 & 2.044258 \\
418 & -62.50 & 12.42 & 13.0\% & 60/60 & 71 & 0.6014 & 1.760826 \\
419 & -40.11 & 12.06 & 13.0\% & 87/87 & 105 & 0.6003 & 1.740884 \\
420 & -52.61 & 11.36 & 12.0\% & 94/94 & 133 & 0.5990 & 2.121130 \\
421 & -60.55 & 11.18 & 12.0\% & 95/95 & 119 & 0.5978 & 1.796276 \\
422 & -24.93 & 11.32 & 12.0\% & 63/63 & 79 & 0.5970 & 1.776442 \\
423 & -51.15 & 11.32 & 12.0\% & 66/66 & 74 & 0.5963 & 1.855590 \\
424 & -49.65 & 11.21 & 12.0\% & 62/62 & 70 & 0.5956 & 1.642970 \\
425 & -33.89 & 10.91 & 12.0\% & 124/124 & 138 & 0.5942 & 1.988169 \\
426 & -74.03 & 11.43 & 12.0\% & 87/87 & 97 & 0.5933 & 1.879806 \\
427 & -40.83 & 11.70 & 11.0\% & 102/102 & 120 & 0.5921 & 1.832551 \\
428 & 3.22 & 12.53 & 11.0\% & 109/109 & 131 & 0.5908 & 1.968376 \\
429 & -7.45 & 12.05 & 11.0\% & 106/106 & 128 & 0.5895 & 1.903419 \\
430 & -41.22 & 12.23 & 12.0\% & 71/71 & 88 & 0.5887 & 1.692673 \\
431 & -27.13 & 12.34 & 12.0\% & 72/72 & 89 & 0.5878 & 2.015875 \\
432 & -33.22 & 11.59 & 12.0\% & 85/85 & 112 & 0.5867 & 1.738739 \\
433 & -1.32 & 12.24 & 13.0\% & 67/67 & 82 & 0.5859 & 1.749491 \\
434 & -63.08 & 11.73 & 13.0\% & 94/94 & 119 & 0.5847 & 1.767782 \\
435 & -28.35 & 12.47 & 13.0\% & 95/95 & 127 & 0.5834 & 1.717847 \\
436 & -37.12 & 12.88 & 13.0\% & 106/106 & 121 & 0.5822 & 1.806583 \\
437 & -5.40 & 12.30 & 14.0\% & 100/100 & 128 & 0.5810 & 1.738230 \\
438 & -23.57 & 12.74 & 15.0\% & 69/69 & 83 & 0.5801 & 2.036648 \\
439 & -67.88 & 12.97 & 15.0\% & 86/86 & 99 & 0.5792 & 1.909593 \\
440 & -82.79 & 13.00 & 15.0\% & 71/71 & 87 & 0.5783 & 1.883315 \\
441 & -58.96 & 12.65 & 15.0\% & 126/126 & 148 & 0.5768 & 2.060362 \\
442 & -96.91 & 12.09 & 15.0\% & 82/82 & 96 & 0.5759 & 1.958850 \\
443 & -33.46 & 12.31 & 14.0\% & 78/78 & 100 & 0.5749 & 1.747816 \\
444 & -27.79 & 14.60 & 14.0\% & 94/94 & 114 & 0.5738 & 1.743751 \\
445 & -56.47 & 13.96 & 14.0\% & 76/76 & 92 & 0.5729 & 1.867676 \\
446 & -21.51 & 14.32 & 14.0\% & 108/108 & 132 & 0.5715 & 1.577361 \\
447 & -50.34 & 13.12 & 14.0\% & 69/69 & 82 & 0.5707 & 1.837183 \\
448 & -95.64 & 13.37 & 14.0\% & 97/97 & 112 & 0.5696 & 1.896712 \\
449 & 15.58 & 13.90 & 14.0\% & 84/84 & 93 & 0.5687 & 1.688727 \\
450 & -59.74 & 14.68 & 14.0\% & 93/93 & 115 & 0.5676 & 1.852338 \\
451 & -16.20 & 14.61 & 14.0\% & 75/75 & 86 & 0.5667 & 1.832542 \\
452 & -24.66 & 13.92 & 14.0\% & 60/60 & 82 & 0.5659 & 1.458904 \\
453 & -53.72 & 14.30 & 13.0\% & 60/60 & 69 & 0.5652 & 1.911603 \\
454 & -40.48 & 14.18 & 13.0\% & 686/686 & 820 & 0.5571 & 1.765085 \\
455 & -11.46 & 14.35 & 13.0\% & 94/94 & 104 & 0.5561 & 1.925525 \\
456 & -30.66 & 14.47 & 13.0\% & 103/103 & 132 & 0.5548 & 1.682301 \\
457 & -46.26 & 14.52 & 13.0\% & 105/105 & 128 & 0.5535 & 1.651991 \\
458 & -29.42 & 14.91 & 12.0\% & 107/107 & 128 & 0.5522 & 2.076428 \\
459 & -4.96 & 15.37 & 12.0\% & 97/97 & 112 & 0.5511 & 1.889755 \\
460 & -40.23 & 14.91 & 12.0\% & 82/82 & 91 & 0.5502 & 1.543212 \\
461 & -43.10 & 15.74 & 12.0\% & 105/105 & 131 & 0.5489 & 1.822628 \\
462 & -0.72 & 15.77 & 12.0\% & 78/78 & 93 & 0.5480 & 1.697952 \\
463 & -45.55 & 14.72 & 12.0\% & 95/95 & 117 & 0.5468 & 1.966412 \\
464 & -3.59 & 15.70 & 12.0\% & 116/116 & 131 & 0.5456 & 1.839905 \\
465 & -53.20 & 15.65 & 12.0\% & 73/73 & 81 & 0.5447 & 1.730525 \\
466 & -18.94 & 16.18 & 12.0\% & 73/73 & 84 & 0.5439 & 1.848235 \\
467 & -27.30 & 16.32 & 12.0\% & 130/130 & 155 & 0.5424 & 2.062738 \\
468 & -42.78 & 15.68 & 12.0\% & 77/77 & 86 & 0.5415 & 1.766384 \\
469 & -16.10 & 16.09 & 12.0\% & 71/71 & 89 & 0.5406 & 2.058623 \\
470 & -4.16 & 16.71 & 12.0\% & 85/85 & 95 & 0.5397 & 1.724385 \\
471 & -74.00 & 15.95 & 12.0\% & 73/73 & 84 & 0.5389 & 1.812072 \\
472 & -19.45 & 15.20 & 12.0\% & 91/91 & 107 & 0.5378 & 1.964569 \\
473 & -2.17 & 16.25 & 12.0\% & 153/153 & 174 & 0.5361 & 1.687279 \\
474 & -24.43 & 15.91 & 11.0\% & 111/111 & 139 & 0.5347 & 1.689843 \\
475 & -46.93 & 15.51 & 11.0\% & 102/102 & 122 & 0.5335 & 1.960945 \\
476 & -21.84 & 16.23 & 11.0\% & 61/61 & 65 & 0.5329 & 1.738283 \\
477 & -39.74 & 16.27 & 11.0\% & 63/63 & 71 & 0.5322 & 2.141520 \\
478 & -43.63 & 16.47 & 11.0\% & 89/89 & 108 & 0.5311 & 1.820170 \\
479 & -11.34 & 16.34 & 11.0\% & 139/139 & 156 & 0.5296 & 1.700067 \\
480 & 27.12 & 16.30 & 11.0\% & 113/113 & 132 & 0.5282 & 1.595318 \\
481 & -87.07 & 17.40 & 11.0\% & 171/171 & 195 & 0.5263 & 2.034232 \\
482 & 11.78 & 17.06 & 12.0\% & 141/141 & 168 & 0.5247 & 1.964190 \\
483 & -30.21 & 16.55 & 12.0\% & 75/75 & 85 & 0.5238 & 1.859590 \\
484 & -57.24 & 16.94 & 12.0\% & 85/85 & 103 & 0.5228 & 1.948736 \\
485 & -97.87 & 16.94 & 12.0\% & 84/84 & 98 & 0.5218 & 2.120703 \\
486 & -70.79 & 17.20 & 12.0\% & 117/117 & 136 & 0.5205 & 1.666331 \\
487 & -12.58 & 17.38 & 12.0\% & 77/77 & 97 & 0.5195 & 2.035150 \\
488 & -14.42 & 18.53 & 12.0\% & 108/108 & 119 & 0.5183 & 2.015028 \\
489 & -127.19 & 18.13 & 12.0\% & 97/97 & 115 & 0.5172 & 1.751508 \\
490 & -198.64 & 18.20 & 12.0\% & 110/110 & 125 & 0.5160 & 1.770783 \\
491 & -80.30 & 18.29 & 12.0\% & 857/857 & 1000 & 0.5061 & 1.934576 \\
492 & 0.07 & 18.91 & 12.0\% & 117/117 & 144 & 0.5046 & 1.899997 \\
493 & -43.82 & 19.74 & 12.0\% & 92/92 & 101 & 0.5036 & 1.577335 \\
494 & -30.98 & 19.64 & 12.0\% & 89/89 & 102 & 0.5026 & 1.756455 \\
495 & -23.34 & 18.91 & 12.0\% & 113/113 & 133 & 0.5013 & 1.620740 \\
496 & -31.42 & 19.13 & 11.0\% & 428/428 & 497 & 0.4964 & 1.843104 \\
497 & 4.25 & 19.64 & 12.0\% & 92/92 & 116 & 0.4952 & 2.150203 \\
498 & 6.91 & 19.19 & 12.0\% & 125/125 & 136 & 0.4939 & 1.850569 \\
499 & -43.01 & 19.43 & 12.0\% & 125/125 & 158 & 0.4923 & 1.662099 \\
500 & -35.87 & 19.28 & 12.0\% & 116/116 & 136 & 0.4910 & 1.829379 \\
501 & -17.86 & 20.06 & 12.0\% & 78/78 & 85 & 0.4901 & 1.945464 \\
502 & -47.86 & 20.43 & 12.0\% & 99/99 & 119 & 0.4890 & 1.726128 \\
503 & -5.19 & 19.19 & 12.0\% & 107/107 & 120 & 0.4878 & 1.985880 \\
504 & -25.15 & 19.41 & 11.0\% & 122/122 & 156 & 0.4862 & 1.911523 \\
505 & -30.27 & 19.67 & 10.0\% & 91/91 & 110 & 0.4851 & 1.929749 \\
506 & -27.79 & 19.58 & 10.0\% & 85/85 & 96 & 0.4842 & 1.851226 \\
507 & 19.62 & 19.99 & 9.0\% & 96/96 & 116 & 0.4830 & 1.657479 \\
508 & -36.41 & 19.98 & 9.0\% & 83/83 & 92 & 0.4821 & 1.733831 \\
509 & -49.37 & 19.89 & 9.0\% & 101/101 & 130 & 0.4808 & 1.879646 \\
510 & 18.57 & 19.89 & 9.0\% & 139/139 & 154 & 0.4793 & 1.791346 \\
511 & -3.74 & 19.92 & 8.0\% & 81/81 & 88 & 0.4784 & 1.842928 \\
512 & -70.47 & 20.10 & 8.0\% & 101/101 & 122 & 0.4772 & 1.852956 \\
513 & -45.22 & 19.36 & 7.0\% & 78/78 & 89 & 0.4764 & 1.731670 \\
514 & 40.42 & 20.42 & 7.0\% & 159/159 & 183 & 0.4745 & 1.864583 \\
515 & -43.15 & 19.89 & 7.0\% & 129/129 & 147 & 0.4731 & 1.535419 \\
516 & -26.45 & 19.75 & 7.0\% & 115/115 & 132 & 0.4718 & 1.845210 \\
517 & 36.17 & 20.52 & 8.0\% & 95/95 & 109 & 0.4707 & 1.850332 \\
518 & -112.17 & 19.88 & 8.0\% & 320/320 & 369 & 0.4671 & 1.748151 \\
519 & -3.01 & 19.92 & 8.0\% & 241/241 & 279 & 0.4643 & 1.696921 \\
520 & 1.12 & 20.91 & 8.0\% & 90/90 & 107 & 0.4632 & 1.793097 \\
521 & -60.04 & 19.94 & 8.0\% & 79/79 & 89 & 0.4624 & 1.630981 \\
522 & -27.51 & 19.33 & 8.0\% & 112/112 & 123 & 0.4611 & 1.755271 \\
523 & -40.98 & 21.00 & 8.0\% & 77/77 & 88 & 0.4603 & 1.493924 \\
524 & -192.08 & 20.11 & 8.0\% & 874/874 & 1000 & 0.4504 & 1.738391 \\
525 & -9.49 & 19.74 & 8.0\% & 294/294 & 339 & 0.4470 & 1.818639 \\
526 & -68.02 & 19.01 & 8.0\% & 879/879 & 1000 & 0.4371 & 1.851008 \\
527 & 17.19 & 18.80 & 8.0\% & 146/146 & 171 & 0.4354 & 1.723157 \\
528 & -29.28 & 20.01 & 8.0\% & 100/100 & 114 & 0.4343 & 2.212268 \\
529 & -26.64 & 19.80 & 8.0\% & 115/115 & 133 & 0.4330 & 1.684892 \\
530 & -43.36 & 20.33 & 7.0\% & 665/665 & 742 & 0.4256 & 1.807052 \\
531 & -241.18 & 20.46 & 7.0\% & 439/439 & 506 & 0.4206 & 1.838210 \\
532 & -44.26 & 20.20 & 7.0\% & 858/858 & 1000 & 0.4107 & 1.873838 \\
533 & 30.60 & 20.33 & 6.0\% & 135/135 & 156 & 0.4092 & 1.795188 \\
534 & 31.00 & 20.62 & 7.0\% & 135/135 & 163 & 0.4076 & 1.873580 \\
535 & 37.61 & 20.63 & 7.0\% & 109/109 & 132 & 0.4062 & 1.756573 \\
536 & 23.94 & 21.65 & 7.0\% & 111/111 & 122 & 0.4050 & 2.112042 \\
537 & -4.18 & 20.65 & 6.0\% & 96/96 & 109 & 0.4040 & 1.791404 \\
538 & -41.79 & 20.60 & 5.0\% & 850/850 & 1000 & 0.3941 & 1.913913 \\
539 & -63.57 & 20.26 & 5.0\% & 76/76 & 86 & 0.3932 & 2.050056 \\
540 & -68.38 & 21.40 & 5.0\% & 881/881 & 1000 & 0.3833 & 2.006860 \\
541 & 27.34 & 20.87 & 5.0\% & 119/119 & 135 & 0.3820 & 1.857266 \\
542 & -25.95 & 22.00 & 5.0\% & 892/892 & 1000 & 0.3721 & 2.006811 \\
543 & 4.58 & 22.28 & 5.0\% & 128/128 & 139 & 0.3707 & 1.980499 \\
544 & -79.69 & 24.79 & 5.0\% & 889/889 & 1000 & 0.3608 & 2.092618 \\
545 & -15.14 & 24.87 & 5.0\% & 835/835 & 1000 & 0.3509 & 2.070472 \\
546 & -39.18 & 26.76 & 5.0\% & 850/850 & 1000 & 0.3410 & 2.094484 \\
547 & -43.14 & 27.60 & 5.0\% & 874/874 & 1000 & 0.3311 & 2.088716 \\
548 & -21.64 & 27.91 & 5.0\% & 842/842 & 1000 & 0.3212 & 2.098296 \\
549 & 34.33 & 28.60 & 5.0\% & 117/117 & 130 & 0.3199 & 1.890805 \\
550 & 31.46 & 28.23 & 5.0\% & 101/101 & 110 & 0.3188 & 2.015213 \\
551 & -74.31 & 28.82 & 5.0\% & 893/893 & 1000 & 0.3089 & 2.200392 \\
552 & -64.30 & 30.85 & 5.0\% & 846/846 & 1000 & 0.2990 & 2.069074 \\
553 & -39.09 & 32.34 & 5.0\% & 850/850 & 1000 & 0.2891 & 2.074233 \\
554 & -37.84 & 33.77 & 5.0\% & 808/808 & 1000 & 0.2792 & 2.089280 \\
555 & -29.06 & 35.87 & 5.0\% & 852/852 & 1000 & 0.2693 & 2.015395 \\
556 & -4.07 & 37.55 & 5.0\% & 799/799 & 1000 & 0.2594 & 2.140471 \\
557 & -41.68 & 37.87 & 5.0\% & 809/809 & 1000 & 0.2495 & 2.031187 \\
558 & -9.09 & 38.25 & 5.0\% & 844/844 & 1000 & 0.2396 & 2.135389 \\
559 & 17.49 & 38.58 & 5.0\% & 779/779 & 1000 & 0.2297 & 2.083182 \\
560 & -3.95 & 39.93 & 5.0\% & 793/793 & 1000 & 0.2198 & 2.003288 \\
561 & -158.23 & 39.87 & 5.0\% & 448/448 & 598 & 0.2139 & 1.971108 \\
562 & -193.70 & 40.21 & 4.0\% & 409/409 & 501 & 0.2089 & 1.998481 \\
563 & -43.89 & 41.63 & 4.0\% & 839/839 & 1000 & 0.1990 & 2.099272 \\
564 & -53.01 & 40.71 & 4.0\% & 846/846 & 1000 & 0.1891 & 2.101420 \\
565 & -57.79 & 40.84 & 4.0\% & 812/812 & 1000 & 0.1792 & 2.116747 \\
566 & -105.34 & 42.88 & 4.0\% & 855/855 & 1000 & 0.1693 & 2.149631 \\
567 & -39.22 & 42.73 & 4.0\% & 920/920 & 1000 & 0.1594 & 2.091721 \\
568 & -32.47 & 44.21 & 4.0\% & 867/867 & 1000 & 0.1495 & 2.102244 \\
569 & -49.98 & 42.55 & 4.0\% & 886/886 & 1000 & 0.1396 & 2.109497 \\
570 & -85.06 & 43.29 & 4.0\% & 869/869 & 1000 & 0.1297 & 1.811279 \\
571 & -59.22 & 43.36 & 4.0\% & 946/946 & 1000 & 0.1198 & 2.021152 \\
572 & -75.28 & 42.55 & 4.0\% & 902/902 & 1000 & 0.1099 & 1.953389 \\
573 & -106.91 & 44.34 & 4.0\% & 847/847 & 1000 & 0.1000 & 1.845580 \\
574 & -60.36 & 44.68 & 4.0\% & 849/849 & 1000 & 0.0901 & 1.843238 \\
575 & -27.23 & 46.78 & 4.0\% & 861/861 & 1000 & 0.0802 & 1.899847 \\
576 & -52.52 & 44.98 & 4.0\% & 815/815 & 1000 & 0.0703 & 1.925485 \\
577 & -30.54 & 44.85 & 4.0\% & 813/813 & 1000 & 0.0604 & 1.803491 \\
578 & -40.23 & 45.86 & 4.0\% & 873/873 & 1000 & 0.0505 & 1.900650 \\
579 & -98.40 & 44.59 & 4.0\% & 762/762 & 1000 & 0.0406 & 1.870649 \\
580 & -43.08 & 45.96 & 4.0\% & 722/722 & 1000 & 0.0307 & 1.837314 \\
581 & -41.48 & 47.21 & 4.0\% & 711/711 & 1000 & 0.0208 & 1.848014 \\
582 & -59.82 & 48.20 & 3.0\% & 737/737 & 1000 & 0.0109 & 1.831935 \\
583 & -42.93 & 47.22 & 3.0\% & 714/714 & 1000 & 0.0100 & 1.873329 \\
584 & -53.06 & 48.15 & 3.0\% & 758/758 & 1000 & 0.0100 & 1.687419 \\
585 & -17.26 & 48.20 & 3.0\% & 718/718 & 1000 & 0.0100 & 1.741891 \\
586 & -65.64 & 46.49 & 3.0\% & 756/756 & 1000 & 0.0100 & 1.650325 \\
587 & -67.98 & 48.25 & 3.0\% & 705/705 & 1000 & 0.0100 & 1.570983 \\
588 & -5.25 & 48.92 & 3.0\% & 745/745 & 1000 & 0.0100 & 1.669104 \\
589 & 23.75 & 48.38 & 3.0\% & 719/719 & 1000 & 0.0100 & 1.496219 \\
590 & -48.76 & 47.84 & 3.0\% & 716/716 & 1000 & 0.0100 & 1.515890 \\
591 & -6.34 & 46.44 & 3.0\% & 731/731 & 1000 & 0.0100 & 1.468482 \\
592 & -12.93 & 47.21 & 3.0\% & 741/741 & 1000 & 0.0100 & 1.509099 \\
593 & 25.70 & 48.36 & 3.0\% & 753/753 & 1000 & 0.0100 & 1.448036 \\
594 & -18.61 & 48.14 & 3.0\% & 764/764 & 1000 & 0.0100 & 1.343785 \\
595 & 45.61 & 46.72 & 3.0\% & 807/807 & 1000 & 0.0100 & 1.353234 \\
596 & -23.83 & 47.02 & 3.0\% & 791/791 & 1000 & 0.0100 & 1.348726 \\
597 & 9.31 & 46.17 & 2.0\% & 796/796 & 1000 & 0.0100 & 1.322595 \\
598 & 230.37 & 45.78 & 3.0\% & 231/231 & 321 & 0.0100 & 1.320313 \\
599 & -1.36 & 45.15 & 3.0\% & 879/879 & 1000 & 0.0100 & 1.226314 \\
600 & -1.97 & 45.37 & 3.0\% & 889/889 & 1000 & 0.0100 & 1.277864 \\
601 & 2.62 & 44.72 & 3.0\% & 787/787 & 1000 & 0.0100 & 1.296708 \\
602 & 11.54 & 45.17 & 3.0\% & 818/818 & 1000 & 0.0100 & 1.181735 \\
603 & 15.44 & 46.03 & 3.0\% & 822/822 & 1000 & 0.0100 & 1.121776 \\
604 & -12.93 & 44.21 & 3.0\% & 857/857 & 1000 & 0.0100 & 1.202125 \\
605 & -8.02 & 43.64 & 3.0\% & 885/885 & 1000 & 0.0100 & 1.187926 \\
606 & -2.67 & 43.49 & 3.0\% & 841/841 & 1000 & 0.0100 & 1.199419 \\
607 & -14.04 & 42.25 & 3.0\% & 776/776 & 1000 & 0.0100 & 1.223444 \\
608 & 7.30 & 41.94 & 3.0\% & 764/764 & 1000 & 0.0100 & 1.203857 \\
609 & -29.50 & 39.77 & 3.0\% & 751/751 & 1000 & 0.0100 & 1.145234 \\
610 & -49.97 & 40.23 & 3.0\% & 775/775 & 1000 & 0.0100 & 1.076230 \\
611 & 24.42 & 39.71 & 3.0\% & 782/782 & 1000 & 0.0100 & 1.049706 \\
612 & -33.04 & 40.06 & 3.0\% & 734/734 & 1000 & 0.0100 & 1.121843 \\
613 & -6.96 & 41.29 & 3.0\% & 752/752 & 1000 & 0.0100 & 1.090572 \\
614 & -38.46 & 43.03 & 3.0\% & 757/757 & 1000 & 0.0100 & 1.053851 \\
615 & -11.81 & 43.72 & 3.0\% & 746/746 & 1000 & 0.0100 & 1.011194 \\
616 & -28.98 & 43.26 & 3.0\% & 816/816 & 1000 & 0.0100 & 0.975086 \\
617 & 56.22 & 43.66 & 2.0\% & 786/786 & 1000 & 0.0100 & 1.029162 \\
618 & -22.75 & 44.78 & 2.0\% & 837/837 & 1000 & 0.0100 & 1.002989 \\
619 & -29.09 & 46.69 & 2.0\% & 845/845 & 1000 & 0.0100 & 0.914043 \\
620 & 224.62 & 47.28 & 3.0\% & 340/340 & 456 & 0.0100 & 0.896180 \\
621 & 213.54 & 46.76 & 4.0\% & 627/627 & 740 & 0.0100 & 0.941897 \\
622 & -7.05 & 47.98 & 4.0\% & 776/776 & 1000 & 0.0100 & 0.928557 \\
623 & 156.82 & 48.94 & 5.0\% & 722/722 & 918 & 0.0100 & 0.894135 \\
624 & 54.82 & 48.70 & 5.0\% & 190/190 & 230 & 0.0100 & 0.852496 \\
625 & 26.55 & 48.35 & 5.0\% & 846/846 & 1000 & 0.0100 & 0.906508 \\
626 & 10.01 & 49.60 & 5.0\% & 902/902 & 1000 & 0.0100 & 0.850474 \\
627 & 0.39 & 49.97 & 5.0\% & 849/849 & 1000 & 0.0100 & 0.820422 \\
628 & 215.89 & 51.56 & 6.0\% & 503/503 & 632 & 0.0100 & 0.900650 \\
629 & -34.40 & 51.49 & 6.0\% & 858/858 & 1000 & 0.0100 & 0.794264 \\
630 & -13.81 & 47.83 & 6.0\% & 827/827 & 1000 & 0.0100 & 0.765384 \\
631 & -11.66 & 52.72 & 6.0\% & 828/828 & 1000 & 0.0100 & 0.787692 \\
632 & 22.70 & 53.10 & 6.0\% & 843/843 & 1000 & 0.0100 & 0.769071 \\
633 & -21.62 & 52.36 & 6.0\% & 825/825 & 1000 & 0.0100 & 0.768657 \\
634 & -26.88 & 54.79 & 5.0\% & 834/834 & 1000 & 0.0100 & 0.718427 \\
635 & -13.62 & 53.15 & 5.0\% & 777/777 & 1000 & 0.0100 & 0.666539 \\
636 & -11.41 & 53.95 & 5.0\% & 801/801 & 1000 & 0.0100 & 0.640526 \\
637 & 3.95 & 52.66 & 5.0\% & 796/796 & 1000 & 0.0100 & 0.622717 \\
638 & 1.99 & 53.38 & 5.0\% & 821/821 & 1000 & 0.0100 & 0.590218 \\
639 & -53.18 & 55.38 & 5.0\% & 779/779 & 1000 & 0.0100 & 0.580647 \\
640 & -34.87 & 56.82 & 5.0\% & 824/824 & 1000 & 0.0100 & 0.521042 \\
641 & 19.45 & 57.12 & 5.0\% & 856/856 & 1000 & 0.0100 & 0.572175 \\
642 & 224.35 & 59.45 & 6.0\% & 352/352 & 492 & 0.0100 & 0.590541 \\
643 & 277.86 & 57.24 & 7.0\% & 267/267 & 380 & 0.0100 & 0.525553 \\
644 & 235.28 & 58.30 & 8.0\% & 529/529 & 718 & 0.0100 & 0.491790 \\
645 & 229.44 & 59.62 & 9.0\% & 463/463 & 564 & 0.0100 & 0.480987 \\
646 & 232.50 & 58.55 & 10.0\% & 496/496 & 693 & 0.0100 & 0.477190 \\
647 & 201.71 & 59.46 & 11.0\% & 431/431 & 560 & 0.0100 & 0.459067 \\
648 & 229.53 & 59.04 & 12.0\% & 327/327 & 449 & 0.0100 & 0.476200 \\
649 & 231.84 & 59.23 & 13.0\% & 230/230 & 345 & 0.0100 & 0.448630 \\
650 & 297.13 & 60.33 & 14.0\% & 276/276 & 365 & 0.0100 & 0.471710 \\
651 & 232.35 & 60.43 & 15.0\% & 276/276 & 398 & 0.0100 & 0.418784 \\
652 & 239.48 & 60.48 & 16.0\% & 304/304 & 399 & 0.0100 & 0.440203 \\
653 & 255.88 & 60.16 & 17.0\% & 462/462 & 605 & 0.0100 & 0.417100 \\
654 & 197.41 & 64.85 & 18.0\% & 594/594 & 826 & 0.0100 & 0.430906 \\
655 & 257.35 & 65.17 & 19.0\% & 419/419 & 589 & 0.0100 & 0.432282 \\
656 & 268.94 & 64.83 & 20.0\% & 345/345 & 461 & 0.0100 & 0.397127 \\
657 & 229.19 & 66.93 & 21.0\% & 317/317 & 460 & 0.0100 & 0.342343 \\
658 & 197.26 & 67.24 & 22.0\% & 668/668 & 908 & 0.0100 & 0.381648 \\
659 & 217.66 & 67.35 & 22.0\% & 447/447 & 521 & 0.0100 & 0.368301 \\
660 & 177.36 & 67.81 & 23.0\% & 741/741 & 976 & 0.0100 & 0.376456 \\
661 & 241.27 & 68.27 & 24.0\% & 381/381 & 488 & 0.0100 & 0.365892 \\
662 & -19.27 & 69.78 & 24.0\% & 870/870 & 1000 & 0.0100 & 0.388007 \\
663 & 246.76 & 68.95 & 25.0\% & 712/712 & 961 & 0.0100 & 0.364343 \\
664 & 213.61 & 68.13 & 26.0\% & 535/535 & 654 & 0.0100 & 0.371458 \\
665 & 0.67 & 69.28 & 26.0\% & 922/922 & 1000 & 0.0100 & 0.342503 \\
666 & 9.22 & 67.09 & 26.0\% & 879/879 & 1000 & 0.0100 & 0.324959 \\
667 & 228.01 & 67.73 & 27.0\% & 571/571 & 705 & 0.0100 & 0.316321 \\
668 & 215.89 & 68.42 & 28.0\% & 373/373 & 433 & 0.0100 & 0.333555 \\
669 & 222.71 & 67.22 & 29.0\% & 370/370 & 445 & 0.0100 & 0.338031 \\
670 & 224.40 & 67.62 & 30.0\% & 295/295 & 373 & 0.0100 & 0.328643 \\
671 & 195.32 & 66.73 & 31.0\% & 625/625 & 734 & 0.0100 & 0.352222 \\
672 & 7.48 & 67.10 & 31.0\% & 850/850 & 1000 & 0.0100 & 0.329157 \\
673 & 5.30 & 66.82 & 31.0\% & 856/856 & 1000 & 0.0100 & 0.351434 \\
674 & -40.88 & 65.20 & 31.0\% & 893/893 & 1000 & 0.0100 & 0.320473 \\
675 & 22.52 & 64.37 & 31.0\% & 775/775 & 1000 & 0.0100 & 0.305741 \\
676 & 31.22 & 66.07 & 31.0\% & 825/825 & 1000 & 0.0100 & 0.321966 \\
677 & 208.39 & 66.35 & 32.0\% & 703/703 & 962 & 0.0100 & 0.305261 \\
678 & 20.13 & 66.36 & 32.0\% & 783/783 & 1000 & 0.0100 & 0.301755 \\
679 & 190.81 & 68.86 & 33.0\% & 709/709 & 993 & 0.0100 & 0.316662 \\
680 & 225.71 & 70.00 & 34.0\% & 480/480 & 669 & 0.0100 & 0.306582 \\
681 & 246.17 & 69.71 & 34.0\% & 379/379 & 545 & 0.0100 & 0.320402 \\
682 & -0.83 & 68.20 & 34.0\% & 811/811 & 1000 & 0.0100 & 0.339433 \\
683 & 141.00 & 67.50 & 34.0\% & 766/766 & 832 & 0.0100 & 0.335616 \\
684 & -34.15 & 67.86 & 34.0\% & 783/783 & 1000 & 0.0100 & 0.364299 \\
685 & 53.74 & 67.02 & 34.0\% & 782/782 & 1000 & 0.0100 & 0.352826 \\
686 & -95.72 & 71.04 & 34.0\% & 846/846 & 1000 & 0.0100 & 0.351653 \\
687 & 280.67 & 69.70 & 35.0\% & 415/415 & 588 & 0.0100 & 0.370868 \\
688 & -96.02 & 68.29 & 35.0\% & 826/826 & 1000 & 0.0100 & 0.368809 \\
689 & -111.93 & 67.19 & 35.0\% & 819/819 & 1000 & 0.0100 & 0.351183 \\
690 & -77.72 & 68.43 & 35.0\% & 811/811 & 1000 & 0.0100 & 0.372307 \\
691 & 26.39 & 69.94 & 35.0\% & 806/806 & 1000 & 0.0100 & 0.404472 \\
692 & -86.18 & 70.58 & 35.0\% & 858/858 & 1000 & 0.0100 & 0.409816 \\
693 & -2.94 & 69.44 & 35.0\% & 848/848 & 1000 & 0.0100 & 0.421000 \\
694 & 116.43 & 71.49 & 35.0\% & 664/664 & 1000 & 0.0100 & 0.437022 \\
695 & -39.57 & 69.98 & 35.0\% & 816/816 & 1000 & 0.0100 & 0.383624 \\
696 & 255.35 & 70.19 & 36.0\% & 273/273 & 355 & 0.0100 & 0.373685 \\
697 & 1.09 & 72.50 & 36.0\% & 748/748 & 1000 & 0.0100 & 0.384038 \\
698 & 111.27 & 71.65 & 35.0\% & 610/610 & 1000 & 0.0100 & 0.416129 \\
699 & -27.01 & 70.16 & 35.0\% & 758/758 & 1000 & 0.0100 & 0.430640 \\
700 & 218.93 & 68.88 & 35.0\% & 684/684 & 791 & 0.0100 & 0.412118 \\
701 & 235.12 & 70.41 & 35.0\% & 722/722 & 895 & 0.0100 & 0.397845 \\
702 & 141.67 & 67.75 & 35.0\% & 606/606 & 1000 & 0.0100 & 0.415722 \\
703 & 276.60 & 67.99 & 36.0\% & 170/170 & 245 & 0.0100 & 0.414278 \\
704 & 60.26 & 68.93 & 36.0\% & 712/712 & 1000 & 0.0100 & 0.367725 \\
705 & 222.13 & 70.63 & 37.0\% & 537/537 & 765 & 0.0100 & 0.429750 \\
706 & 280.98 & 71.06 & 38.0\% & 377/377 & 591 & 0.0100 & 0.396536 \\
707 & 216.29 & 71.52 & 39.0\% & 408/408 & 533 & 0.0100 & 0.400818 \\
708 & 142.54 & 71.75 & 39.0\% & 812/812 & 1000 & 0.0100 & 0.401424 \\
709 & 259.24 & 72.33 & 40.0\% & 327/327 & 457 & 0.0100 & 0.410128 \\
710 & 139.30 & 73.87 & 40.0\% & 678/678 & 1000 & 0.0100 & 0.408130 \\
711 & 61.45 & 72.05 & 40.0\% & 768/768 & 1000 & 0.0100 & 0.406834 \\
712 & 213.33 & 74.22 & 41.0\% & 491/491 & 637 & 0.0100 & 0.420285 \\
713 & 163.39 & 74.01 & 41.0\% & 615/615 & 1000 & 0.0100 & 0.419113 \\
714 & 95.24 & 76.35 & 41.0\% & 584/584 & 1000 & 0.0100 & 0.407543 \\
715 & 234.99 & 76.16 & 42.0\% & 341/341 & 527 & 0.0100 & 0.415019 \\
716 & 171.18 & 75.22 & 43.0\% & 680/680 & 845 & 0.0100 & 0.369036 \\
717 & 232.67 & 77.69 & 44.0\% & 278/278 & 401 & 0.0100 & 0.378550 \\
718 & 55.23 & 76.91 & 44.0\% & 71/71 & 84 & 0.0100 & 0.368392 \\
719 & 81.04 & 77.85 & 44.0\% & 638/638 & 1000 & 0.0100 & 0.384007 \\
720 & 209.54 & 81.02 & 44.0\% & 612/612 & 781 & 0.0100 & 0.392039 \\
721 & -1.54 & 80.65 & 43.0\% & 71/71 & 82 & 0.0100 & 0.388405 \\
722 & 80.70 & 81.14 & 43.0\% & 688/688 & 1000 & 0.0100 & 0.365559 \\
723 & 4.03 & 81.35 & 42.0\% & 66/66 & 77 & 0.0100 & 0.308220 \\
724 & 0.30 & 81.20 & 42.0\% & 77/77 & 97 & 0.0100 & 0.363285 \\
725 & 176.75 & 82.50 & 42.0\% & 573/573 & 1000 & 0.0100 & 0.362866 \\
726 & 250.69 & 83.78 & 43.0\% & 174/174 & 248 & 0.0100 & 0.355469 \\
727 & 252.82 & 82.54 & 44.0\% & 469/469 & 605 & 0.0100 & 0.370243 \\
728 & -92.59 & 83.47 & 43.0\% & 101/101 & 125 & 0.0100 & 0.367341 \\
729 & -17.77 & 83.32 & 43.0\% & 899/899 & 1000 & 0.0100 & 0.351885 \\
730 & 122.29 & 83.27 & 43.0\% & 574/574 & 1000 & 0.0100 & 0.343333 \\
731 & 223.17 & 84.82 & 44.0\% & 430/430 & 572 & 0.0100 & 0.383031 \\
732 & 254.26 & 83.78 & 45.0\% & 182/182 & 293 & 0.0100 & 0.352220 \\
733 & 303.29 & 83.04 & 46.0\% & 192/192 & 309 & 0.0100 & 0.376986 \\
734 & 153.98 & 82.16 & 46.0\% & 535/535 & 1000 & 0.0100 & 0.347208 \\
735 & 268.43 & 83.49 & 47.0\% & 443/443 & 786 & 0.0100 & 0.361721 \\
736 & -10.10 & 84.26 & 47.0\% & 69/69 & 85 & 0.0100 & 0.382618 \\
737 & 253.26 & 83.80 & 48.0\% & 253/253 & 371 & 0.0100 & 0.412595 \\
738 & 153.58 & 82.88 & 48.0\% & 558/558 & 1000 & 0.0100 & 0.335898 \\
739 & 254.02 & 81.81 & 49.0\% & 554/554 & 935 & 0.0100 & 0.352583 \\
740 & 137.12 & 80.73 & 49.0\% & 630/630 & 1000 & 0.0100 & 0.365429 \\
741 & 127.86 & 79.30 & 49.0\% & 738/738 & 1000 & 0.0100 & 0.344047 \\
742 & 167.20 & 82.06 & 48.0\% & 653/653 & 1000 & 0.0100 & 0.369556 \\
743 & 21.51 & 82.22 & 47.0\% & 896/896 & 1000 & 0.0100 & 0.345998 \\
744 & 217.74 & 80.34 & 47.0\% & 688/688 & 980 & 0.0100 & 0.359434 \\
745 & 211.55 & 81.02 & 47.0\% & 391/391 & 611 & 0.0100 & 0.383272 \\
746 & 97.00 & 79.36 & 46.0\% & 689/689 & 1000 & 0.0100 & 0.379160 \\
747 & 240.44 & 79.24 & 46.0\% & 394/394 & 725 & 0.0100 & 0.390116 \\
748 & 285.06 & 77.09 & 46.0\% & 329/329 & 480 & 0.0100 & 0.374088 \\
749 & 244.52 & 77.26 & 46.0\% & 199/199 & 271 & 0.0100 & 0.367763 \\
750 & -18.89 & 77.61 & 45.0\% & 122/122 & 146 & 0.0100 & 0.383477 \\
751 & -274.52 & 77.36 & 44.0\% & 229/229 & 245 & 0.0100 & 0.395319 \\
752 & 74.10 & 75.84 & 43.0\% & 727/727 & 1000 & 0.0100 & 0.405882 \\
753 & 267.41 & 74.18 & 43.0\% & 428/428 & 528 & 0.0100 & 0.397290 \\
754 & 143.68 & 70.91 & 42.0\% & 760/760 & 1000 & 0.0100 & 0.428368 \\
755 & 175.89 & 69.81 & 41.0\% & 640/640 & 1000 & 0.0100 & 0.405076 \\
756 & 255.09 & 69.97 & 40.0\% & 565/565 & 771 & 0.0100 & 0.427481 \\
757 & 290.80 & 69.57 & 40.0\% & 283/283 & 392 & 0.0100 & 0.429627 \\
758 & 292.26 & 68.16 & 40.0\% & 430/430 & 799 & 0.0100 & 0.393375 \\
759 & 177.56 & 67.73 & 40.0\% & 509/509 & 1000 & 0.0100 & 0.362362 \\
760 & 251.79 & 65.28 & 39.0\% & 648/648 & 869 & 0.0100 & 0.410304 \\
761 & 262.77 & 65.93 & 38.0\% & 289/289 & 306 & 0.0100 & 0.407441 \\
762 & 293.93 & 65.35 & 39.0\% & 336/336 & 649 & 0.0100 & 0.360781 \\
763 & 241.62 & 64.02 & 39.0\% & 415/415 & 872 & 0.0100 & 0.400244 \\
764 & 244.83 & 62.40 & 39.0\% & 347/347 & 736 & 0.0100 & 0.372195 \\
765 & 127.36 & 59.77 & 39.0\% & 511/511 & 1000 & 0.0100 & 0.385182 \\
766 & 64.45 & 56.89 & 39.0\% & 673/673 & 1000 & 0.0100 & 0.381052 \\
767 & 178.00 & 56.77 & 38.0\% & 702/702 & 1000 & 0.0100 & 0.378778 \\
768 & 163.90 & 54.96 & 37.0\% & 572/572 & 1000 & 0.0100 & 0.365789 \\
769 & 121.69 & 54.72 & 36.0\% & 612/612 & 1000 & 0.0100 & 0.410970 \\
770 & 235.36 & 53.94 & 36.0\% & 372/372 & 618 & 0.0100 & 0.387432 \\
771 & 187.09 & 54.79 & 35.0\% & 554/554 & 1000 & 0.0100 & 0.365574 \\
772 & 121.94 & 52.67 & 35.0\% & 536/536 & 1000 & 0.0100 & 0.358968 \\
773 & 154.46 & 51.21 & 35.0\% & 536/536 & 1000 & 0.0100 & 0.396574 \\
774 & 285.76 & 52.42 & 35.0\% & 272/272 & 535 & 0.0100 & 0.424074 \\
775 & 289.85 & 54.24 & 36.0\% & 335/335 & 586 & 0.0100 & 0.347595 \\
776 & 274.78 & 56.53 & 36.0\% & 242/242 & 430 & 0.0100 & 0.413514 \\
777 & 290.31 & 56.36 & 35.0\% & 174/174 & 256 & 0.0100 & 0.364317 \\
778 & 157.66 & 55.21 & 35.0\% & 792/792 & 1000 & 0.0100 & 0.384473 \\
779 & 243.20 & 56.73 & 35.0\% & 250/250 & 473 & 0.0100 & 0.417663 \\
780 & 201.75 & 55.39 & 34.0\% & 849/849 & 921 & 0.0100 & 0.370066 \\
781 & 17.33 & 54.72 & 35.0\% & 99/99 & 117 & 0.0100 & 0.439942 \\
782 & 52.10 & 56.70 & 36.0\% & 134/134 & 152 & 0.0100 & 0.375428 \\
783 & 251.39 & 53.65 & 36.0\% & 450/450 & 613 & 0.0100 & 0.421215 \\
784 & 260.40 & 54.37 & 36.0\% & 196/196 & 356 & 0.0100 & 0.362519 \\
785 & 29.57 & 52.79 & 36.0\% & 96/96 & 125 & 0.0100 & 0.430293 \\
786 & 43.62 & 53.80 & 36.0\% & 68/68 & 104 & 0.0100 & 0.296919 \\
787 & -8.20 & 53.08 & 35.0\% & 58/58 & 77 & 0.0100 & 0.335160 \\
788 & 12.64 & 53.40 & 35.0\% & 149/149 & 193 & 0.0100 & 0.352882 \\
789 & 52.42 & 52.22 & 35.0\% & 99/99 & 115 & 0.0100 & 0.418390 \\
790 & 284.04 & 52.28 & 36.0\% & 318/318 & 520 & 0.0100 & 0.384871 \\
791 & 18.53 & 51.50 & 36.0\% & 118/118 & 136 & 0.0100 & 0.423196 \\
792 & -68.19 & 51.49 & 36.0\% & 55/55 & 74 & 0.0100 & 0.489325 \\
793 & -11.65 & 53.06 & 36.0\% & 47/47 & 64 & 0.0100 & 0.459804 \\
794 & -3.86 & 52.89 & 36.0\% & 77/77 & 110 & 0.0100 & 0.384739 \\
795 & 302.35 & 47.14 & 37.0\% & 266/266 & 397 & 0.0100 & 0.388793 \\
796 & -22.98 & 49.35 & 36.0\% & 62/62 & 83 & 0.0100 & 0.319818 \\
797 & -17.40 & 50.33 & 36.0\% & 66/66 & 97 & 0.0100 & 0.453751 \\
798 & 21.90 & 50.42 & 36.0\% & 89/89 & 101 & 0.0100 & 0.446597 \\
799 & 148.52 & 47.93 & 36.0\% & 797/797 & 836 & 0.0100 & 0.409340 \\
800 & 41.60 & 49.12 & 36.0\% & 84/84 & 91 & 0.0100 & 0.430006 \\
\end{longtable}
\normalsize

\clearpage

\scriptsize
\setlength{\tabcolsep}{1.8pt}
\renewcommand{\arraystretch}{0.86}
\begin{longtable}{rrrrrrrr}
\caption{Complete per-iteration training output - Ddqn Original (800 episodes).}\\
\toprule
Episode & Reward & Avg. Q & Safe100 & Attempted/Executed & Steps & Epsilon & Loss \\
\midrule
\endfirsthead
\multicolumn{8}{c}{\small Continued: Ddqn Original complete per-iteration output} \\
\toprule
Episode & Reward & Avg. Q & Safe100 & Attempted/Executed & Steps & Epsilon & Loss \\
\midrule
\endhead
\bottomrule
\endfoot
\bottomrule
\endlastfoot
1 & -132.79 & 0.24 & 0.0\% & 51/51 & 71 & 0.9993 & nan \\
2 & -436.57 & 0.24 & 0.0\% & 66/66 & 93 & 0.9984 & nan \\
3 & -112.58 & 0.24 & 0.0\% & 104/104 & 124 & 0.9971 & nan \\
4 & -74.45 & 0.24 & 0.0\% & 67/67 & 86 & 0.9963 & nan \\
5 & -141.56 & 0.24 & 0.0\% & 104/104 & 139 & 0.9949 & nan \\
6 & -168.37 & 0.24 & 0.0\% & 82/82 & 113 & 0.9938 & nan \\
7 & -170.35 & 0.24 & 0.0\% & 69/69 & 88 & 0.9929 & nan \\
8 & -264.71 & 0.24 & 0.0\% & 43/43 & 63 & 0.9923 & nan \\
9 & -262.58 & 0.24 & 0.0\% & 88/88 & 114 & 0.9912 & nan \\
10 & -256.16 & -0.01 & 0.0\% & 94/94 & 126 & 0.9899 & 2.120335 \\
11 & -57.57 & -0.34 & 9.1\% & 51/51 & 65 & 0.9893 & 2.219106 \\
12 & -174.41 & -0.08 & 8.3\% & 67/67 & 86 & 0.9884 & 1.914462 \\
13 & -304.37 & 0.10 & 7.7\% & 68/68 & 92 & 0.9875 & 2.113536 \\
14 & -285.82 & 0.40 & 7.1\% & 74/74 & 102 & 0.9865 & 1.577444 \\
15 & -245.45 & 0.48 & 6.7\% & 86/86 & 122 & 0.9853 & 1.506126 \\
16 & -108.64 & 0.66 & 6.2\% & 65/65 & 84 & 0.9845 & 1.834367 \\
17 & -214.15 & 0.59 & 5.9\% & 86/86 & 104 & 0.9834 & 1.396790 \\
18 & -182.99 & 0.72 & 5.6\% & 62/62 & 85 & 0.9826 & 1.323052 \\
19 & -78.73 & 0.56 & 10.5\% & 81/81 & 104 & 0.9816 & 1.397083 \\
20 & -76.50 & 0.88 & 10.0\% & 57/57 & 74 & 0.9808 & 1.512228 \\
21 & -187.82 & 0.88 & 9.5\% & 51/51 & 71 & 0.9801 & 1.510924 \\
22 & -378.98 & 1.17 & 9.1\% & 80/80 & 104 & 0.9791 & 1.423495 \\
23 & -350.93 & 0.90 & 8.7\% & 69/69 & 87 & 0.9782 & 1.305524 \\
24 & -26.12 & 1.28 & 8.3\% & 75/75 & 97 & 0.9773 & 1.054310 \\
25 & -79.35 & 1.12 & 8.0\% & 76/76 & 99 & 0.9763 & 1.407919 \\
26 & -174.05 & 1.43 & 7.7\% & 50/50 & 61 & 0.9757 & 1.117375 \\
27 & -95.26 & 1.55 & 7.4\% & 58/58 & 83 & 0.9749 & 1.390071 \\
28 & -173.95 & 1.36 & 7.1\% & 92/92 & 120 & 0.9737 & 1.169645 \\
29 & -102.62 & 1.55 & 6.9\% & 47/47 & 67 & 0.9730 & 0.972897 \\
30 & -240.45 & 1.57 & 6.7\% & 67/67 & 87 & 0.9722 & 1.196482 \\
31 & -164.48 & 1.62 & 6.5\% & 118/118 & 146 & 0.9707 & 1.193377 \\
32 & -89.86 & 1.81 & 6.2\% & 52/52 & 74 & 0.9700 & 1.233512 \\
33 & -200.89 & 2.05 & 6.1\% & 84/84 & 113 & 0.9689 & 1.086943 \\
34 & -20.86 & 2.06 & 5.9\% & 98/98 & 134 & 0.9675 & 1.227838 \\
35 & -251.55 & 2.22 & 5.7\% & 86/86 & 110 & 0.9665 & 1.165912 \\
36 & -335.83 & 2.35 & 5.6\% & 84/84 & 106 & 0.9654 & 1.134788 \\
37 & -106.60 & 2.67 & 5.4\% & 64/64 & 74 & 0.9647 & 1.179391 \\
38 & -112.47 & 2.86 & 5.3\% & 59/59 & 71 & 0.9640 & 1.299894 \\
39 & -88.76 & 2.68 & 5.1\% & 65/65 & 89 & 0.9631 & 1.139534 \\
40 & -161.58 & 2.57 & 5.0\% & 88/88 & 113 & 0.9620 & 1.121565 \\
41 & -195.98 & 2.53 & 4.9\% & 88/88 & 120 & 0.9608 & 1.115067 \\
42 & -187.76 & 3.46 & 4.8\% & 51/51 & 71 & 0.9601 & 1.260285 \\
43 & -166.05 & 3.44 & 4.7\% & 98/98 & 127 & 0.9588 & 1.306968 \\
44 & -137.57 & 3.44 & 4.5\% & 98/98 & 126 & 0.9576 & 1.069580 \\
45 & -80.07 & 3.37 & 4.4\% & 48/48 & 62 & 0.9570 & 1.175972 \\
46 & -168.19 & 3.27 & 4.3\% & 93/93 & 110 & 0.9559 & 1.162469 \\
47 & -114.55 & 4.23 & 4.3\% & 86/86 & 107 & 0.9548 & 1.319227 \\
48 & -217.08 & 4.17 & 4.2\% & 73/73 & 94 & 0.9539 & 1.136204 \\
49 & -34.15 & 4.21 & 4.1\% & 91/91 & 119 & 0.9527 & 1.359850 \\
50 & -233.19 & 4.24 & 4.0\% & 56/56 & 70 & 0.9520 & 1.247217 \\
51 & 28.87 & 3.84 & 3.9\% & 94/94 & 130 & 0.9507 & 1.252429 \\
52 & -271.32 & 5.02 & 3.8\% & 83/83 & 111 & 0.9496 & 1.514646 \\
53 & -162.56 & 4.97 & 3.8\% & 97/97 & 132 & 0.9483 & 1.252136 \\
54 & -84.39 & 4.74 & 3.7\% & 52/52 & 65 & 0.9477 & 1.245617 \\
55 & -196.19 & 4.88 & 3.6\% & 65/65 & 88 & 0.9468 & 1.228242 \\
56 & -341.33 & 4.93 & 3.6\% & 85/85 & 114 & 0.9457 & 1.161186 \\
57 & -164.49 & 5.93 & 3.5\% & 47/47 & 64 & 0.9450 & 1.687367 \\
58 & -116.90 & 5.61 & 3.4\% & 95/95 & 133 & 0.9437 & 1.253310 \\
59 & -79.30 & 5.75 & 3.4\% & 40/40 & 58 & 0.9432 & 1.391669 \\
60 & -126.07 & 5.95 & 3.3\% & 77/77 & 99 & 0.9422 & 1.305260 \\
61 & -299.60 & 5.96 & 3.3\% & 53/53 & 71 & 0.9415 & 1.438606 \\
62 & -133.62 & 5.76 & 3.2\% & 56/56 & 76 & 0.9407 & 1.329350 \\
63 & -186.20 & 6.80 & 3.2\% & 82/82 & 113 & 0.9396 & 1.323877 \\
64 & -130.17 & 6.80 & 3.1\% & 101/101 & 128 & 0.9383 & 1.441404 \\
65 & -423.23 & 7.07 & 3.1\% & 76/76 & 108 & 0.9373 & 1.366945 \\
66 & -91.18 & 6.65 & 3.0\% & 59/59 & 81 & 0.9365 & 1.127322 \\
67 & -105.63 & 8.81 & 3.0\% & 69/69 & 92 & 0.9356 & 1.242958 \\
68 & -95.78 & 8.13 & 2.9\% & 87/87 & 109 & 0.9345 & 1.609852 \\
69 & -106.21 & 8.19 & 2.9\% & 65/65 & 93 & 0.9336 & 1.446459 \\
70 & -122.57 & 8.16 & 2.9\% & 74/74 & 101 & 0.9326 & 1.382510 \\
71 & -93.36 & 8.02 & 2.8\% & 107/107 & 138 & 0.9312 & 1.441290 \\
72 & -127.52 & 9.83 & 2.8\% & 77/77 & 103 & 0.9302 & 1.613293 \\
73 & -127.56 & 9.66 & 2.7\% & 45/45 & 72 & 0.9295 & 1.455809 \\
74 & -81.81 & 9.60 & 2.7\% & 61/61 & 78 & 0.9287 & 1.429500 \\
75 & -47.10 & 9.68 & 2.7\% & 61/61 & 80 & 0.9279 & 1.476662 \\
76 & -118.27 & 9.36 & 2.6\% & 77/77 & 90 & 0.9270 & 1.550372 \\
77 & -107.49 & 9.35 & 2.6\% & 64/64 & 83 & 0.9262 & 1.701725 \\
78 & -136.67 & 10.49 & 2.6\% & 68/68 & 92 & 0.9253 & 1.688008 \\
79 & -37.15 & 10.54 & 2.5\% & 57/57 & 74 & 0.9245 & 1.706836 \\
80 & -96.75 & 10.66 & 2.5\% & 68/68 & 87 & 0.9237 & 1.462011 \\
81 & -79.60 & 10.81 & 2.5\% & 55/55 & 68 & 0.9230 & 1.640296 \\
82 & -116.14 & 10.59 & 2.4\% & 111/111 & 136 & 0.9217 & 1.422630 \\
83 & -126.90 & 12.03 & 2.4\% & 81/81 & 96 & 0.9207 & 1.646084 \\
84 & -227.55 & 11.83 & 2.4\% & 59/59 & 73 & 0.9200 & 1.706392 \\
85 & -112.42 & 11.61 & 2.4\% & 69/69 & 89 & 0.9191 & 1.448884 \\
86 & -131.70 & 11.82 & 2.3\% & 70/70 & 93 & 0.9182 & 1.679928 \\
87 & -289.43 & 11.79 & 2.3\% & 81/81 & 98 & 0.9172 & 1.430375 \\
88 & -223.72 & 11.50 & 2.3\% & 48/48 & 65 & 0.9166 & 1.395078 \\
89 & -94.22 & 12.18 & 2.2\% & 55/55 & 75 & 0.9158 & 1.606973 \\
90 & -149.36 & 12.72 & 2.2\% & 52/52 & 72 & 0.9151 & 1.489153 \\
91 & -202.71 & 12.54 & 2.2\% & 75/75 & 103 & 0.9141 & 1.410939 \\
92 & -213.81 & 12.77 & 2.2\% & 75/75 & 84 & 0.9133 & 1.626074 \\
93 & -228.83 & 12.52 & 2.2\% & 99/99 & 129 & 0.9120 & 1.504622 \\
94 & -74.44 & 14.73 & 2.1\% & 83/83 & 117 & 0.9108 & 1.445100 \\
95 & -98.05 & 13.76 & 2.1\% & 94/94 & 114 & 0.9097 & 1.537599 \\
96 & -266.10 & 13.37 & 2.1\% & 103/103 & 135 & 0.9084 & 1.551501 \\
97 & -53.59 & 13.82 & 3.1\% & 93/93 & 119 & 0.9072 & 1.438574 \\
98 & -74.49 & 13.61 & 3.1\% & 57/57 & 77 & 0.9064 & 1.508911 \\
99 & -75.28 & 14.53 & 3.0\% & 61/61 & 73 & 0.9057 & 1.825171 \\
100 & -152.66 & 14.82 & 3.0\% & 90/90 & 114 & 0.9046 & 1.534334 \\
101 & -70.13 & 15.19 & 3.0\% & 50/50 & 67 & 0.9039 & 1.581905 \\
102 & -67.12 & 15.14 & 3.0\% & 92/92 & 111 & 0.9028 & 1.797954 \\
103 & -157.19 & 14.91 & 3.0\% & 85/85 & 104 & 0.9018 & 1.535515 \\
104 & -43.62 & 14.56 & 3.0\% & 52/52 & 69 & 0.9011 & 1.657230 \\
105 & -194.52 & 15.37 & 3.0\% & 91/91 & 115 & 0.9000 & 1.561399 \\
106 & -176.32 & 15.80 & 3.0\% & 98/98 & 118 & 0.8988 & 1.472515 \\
107 & -103.44 & 15.49 & 3.0\% & 51/51 & 61 & 0.8982 & 1.533862 \\
108 & -57.73 & 15.39 & 3.0\% & 87/87 & 100 & 0.8972 & 1.625300 \\
109 & -308.21 & 15.75 & 3.0\% & 92/92 & 115 & 0.8961 & 1.445679 \\
110 & -324.36 & 16.61 & 3.0\% & 79/79 & 96 & 0.8951 & 1.843895 \\
111 & 24.92 & 16.94 & 2.0\% & 56/56 & 72 & 0.8944 & 1.461619 \\
112 & -72.49 & 16.97 & 2.0\% & 83/83 & 112 & 0.8933 & 1.531529 \\
113 & 13.36 & 16.65 & 2.0\% & 77/77 & 99 & 0.8923 & 1.557509 \\
114 & -66.01 & 17.83 & 2.0\% & 130/130 & 156 & 0.8908 & 1.766659 \\
115 & -404.98 & 17.89 & 2.0\% & 96/96 & 115 & 0.8896 & 1.536547 \\
116 & -52.11 & 18.25 & 3.0\% & 93/93 & 127 & 0.8884 & 1.633969 \\
117 & -81.20 & 17.78 & 3.0\% & 82/82 & 102 & 0.8873 & 1.653379 \\
118 & -37.31 & 17.72 & 3.0\% & 55/55 & 71 & 0.8866 & 1.771166 \\
119 & -137.31 & 18.53 & 2.0\% & 66/66 & 85 & 0.8858 & 1.614253 \\
120 & -126.77 & 18.69 & 2.0\% & 78/78 & 106 & 0.8848 & 1.401425 \\
121 & -46.26 & 18.49 & 2.0\% & 56/56 & 65 & 0.8841 & 1.445665 \\
122 & -27.02 & 18.16 & 2.0\% & 108/108 & 140 & 0.8827 & 1.784458 \\
123 & -90.16 & 18.53 & 2.0\% & 52/52 & 69 & 0.8820 & 1.558937 \\
124 & -92.72 & 18.74 & 2.0\% & 51/51 & 67 & 0.8814 & 1.737782 \\
125 & -86.18 & 19.56 & 2.0\% & 81/81 & 95 & 0.8804 & 1.734063 \\
126 & -99.02 & 19.46 & 2.0\% & 95/95 & 129 & 0.8792 & 1.630488 \\
127 & -69.69 & 19.73 & 2.0\% & 121/121 & 142 & 0.8778 & 1.599967 \\
128 & -76.96 & 19.40 & 2.0\% & 74/74 & 101 & 0.8768 & 1.553383 \\
129 & -81.73 & 20.37 & 2.0\% & 101/101 & 129 & 0.8755 & 1.790738 \\
130 & -96.66 & 20.16 & 3.0\% & 71/71 & 89 & 0.8746 & 1.744820 \\
131 & -95.86 & 20.35 & 3.0\% & 66/66 & 87 & 0.8737 & 1.453757 \\
132 & -73.10 & 20.15 & 3.0\% & 75/75 & 91 & 0.8728 & 1.656667 \\
133 & -164.32 & 19.96 & 3.0\% & 63/63 & 85 & 0.8720 & 1.541116 \\
134 & -101.50 & 20.24 & 3.0\% & 69/69 & 90 & 0.8711 & 1.603781 \\
135 & -152.84 & 20.25 & 3.0\% & 82/82 & 106 & 0.8701 & 1.791160 \\
136 & -113.91 & 20.25 & 3.0\% & 57/57 & 72 & 0.8693 & 1.600377 \\
137 & -226.59 & 20.53 & 3.0\% & 82/82 & 106 & 0.8683 & 1.199955 \\
138 & -164.72 & 20.29 & 3.0\% & 94/94 & 130 & 0.8670 & 1.681583 \\
139 & -75.91 & 21.12 & 3.0\% & 59/59 & 73 & 0.8663 & 1.615417 \\
140 & -56.98 & 21.00 & 3.0\% & 92/92 & 121 & 0.8651 & 1.542354 \\
141 & -86.85 & 20.88 & 3.0\% & 62/62 & 82 & 0.8643 & 1.500009 \\
142 & -165.51 & 20.80 & 3.0\% & 108/108 & 140 & 0.8629 & 1.421913 \\
143 & -113.13 & 21.11 & 3.0\% & 68/68 & 89 & 0.8620 & 1.668391 \\
144 & -99.97 & 21.31 & 3.0\% & 58/58 & 72 & 0.8613 & 1.823670 \\
145 & -73.07 & 22.00 & 3.0\% & 94/94 & 116 & 0.8601 & 1.417464 \\
146 & -75.84 & 21.87 & 3.0\% & 67/67 & 98 & 0.8592 & 1.449709 \\
147 & -313.59 & 22.12 & 3.0\% & 64/64 & 79 & 0.8584 & 1.605624 \\
148 & -64.80 & 21.91 & 3.0\% & 87/87 & 120 & 0.8572 & 2.028254 \\
149 & -96.17 & 21.72 & 3.0\% & 49/49 & 70 & 0.8565 & 1.528067 \\
150 & -62.36 & 22.42 & 3.0\% & 59/59 & 68 & 0.8558 & 1.169085 \\
151 & -95.48 & 22.26 & 3.0\% & 71/71 & 89 & 0.8550 & 1.404538 \\
152 & -24.26 & 22.35 & 3.0\% & 66/66 & 87 & 0.8541 & 1.536002 \\
153 & -12.40 & 22.49 & 3.0\% & 106/106 & 139 & 0.8527 & 1.558118 \\
154 & -175.13 & 22.33 & 3.0\% & 67/67 & 87 & 0.8519 & 1.534283 \\
155 & 50.72 & 23.23 & 3.0\% & 68/68 & 83 & 0.8510 & 1.629656 \\
156 & -264.98 & 22.92 & 3.0\% & 75/75 & 93 & 0.8501 & 1.780934 \\
157 & -55.40 & 23.01 & 3.0\% & 58/58 & 70 & 0.8494 & 1.368846 \\
158 & -42.40 & 22.43 & 3.0\% & 47/47 & 65 & 0.8488 & 1.524035 \\
159 & -78.32 & 23.01 & 3.0\% & 57/57 & 72 & 0.8481 & 1.214622 \\
160 & -63.18 & 22.24 & 3.0\% & 50/50 & 64 & 0.8474 & 1.381111 \\
161 & -77.00 & 22.76 & 3.0\% & 57/57 & 77 & 0.8467 & 1.485037 \\
162 & -98.11 & 22.67 & 3.0\% & 78/78 & 101 & 0.8457 & 1.707239 \\
163 & -111.30 & 23.03 & 3.0\% & 65/65 & 82 & 0.8449 & 1.611874 \\
164 & -99.05 & 22.70 & 3.0\% & 52/52 & 65 & 0.8442 & 1.507015 \\
165 & -117.03 & 22.94 & 3.0\% & 59/59 & 81 & 0.8434 & 1.931908 \\
166 & -85.42 & 22.84 & 3.0\% & 84/84 & 115 & 0.8423 & 1.333633 \\
167 & -84.31 & 23.39 & 4.0\% & 81/81 & 106 & 0.8412 & 1.840733 \\
168 & -171.72 & 23.74 & 4.0\% & 55/55 & 71 & 0.8405 & 1.533223 \\
169 & -80.41 & 23.28 & 4.0\% & 94/94 & 124 & 0.8393 & 1.667692 \\
170 & -80.03 & 23.23 & 4.0\% & 80/80 & 97 & 0.8383 & 1.626951 \\
171 & -127.49 & 22.80 & 4.0\% & 85/85 & 106 & 0.8373 & 1.520922 \\
172 & -91.73 & 22.99 & 4.0\% & 55/55 & 68 & 0.8366 & 1.379708 \\
173 & -94.09 & 23.17 & 4.0\% & 53/53 & 65 & 0.8360 & 1.793131 \\
174 & -87.93 & 22.94 & 4.0\% & 59/59 & 81 & 0.8352 & 1.329110 \\
175 & -103.03 & 22.73 & 4.0\% & 85/85 & 105 & 0.8341 & 2.096388 \\
176 & -60.17 & 23.45 & 4.0\% & 92/92 & 112 & 0.8330 & 1.330064 \\
177 & -101.94 & 23.18 & 4.0\% & 72/72 & 95 & 0.8321 & 1.361179 \\
178 & -114.34 & 23.17 & 4.0\% & 48/48 & 70 & 0.8314 & 1.915216 \\
179 & -38.74 & 23.62 & 4.0\% & 99/99 & 124 & 0.8302 & 1.448774 \\
180 & -220.51 & 23.62 & 4.0\% & 85/85 & 104 & 0.8291 & 1.347604 \\
181 & -100.50 & 22.91 & 4.0\% & 59/59 & 78 & 0.8284 & 1.630677 \\
182 & -240.57 & 23.66 & 4.0\% & 85/85 & 108 & 0.8273 & 1.944989 \\
183 & -59.38 & 23.49 & 4.0\% & 86/86 & 115 & 0.8261 & 1.889959 \\
184 & -83.01 & 23.21 & 4.0\% & 68/68 & 80 & 0.8254 & 1.609631 \\
185 & -67.44 & 23.22 & 4.0\% & 69/69 & 88 & 0.8245 & 1.627768 \\
186 & -68.60 & 23.26 & 4.0\% & 85/85 & 113 & 0.8234 & 1.229601 \\
187 & -341.30 & 23.16 & 4.0\% & 79/79 & 99 & 0.8224 & 1.588216 \\
188 & -57.41 & 23.94 & 4.0\% & 79/79 & 104 & 0.8214 & 1.921886 \\
189 & -118.86 & 23.32 & 4.0\% & 112/112 & 132 & 0.8200 & 1.794255 \\
190 & -86.33 & 23.25 & 4.0\% & 60/60 & 73 & 0.8193 & 1.838684 \\
191 & -139.63 & 23.11 & 4.0\% & 66/66 & 85 & 0.8185 & 1.642867 \\
192 & -136.28 & 23.31 & 4.0\% & 98/98 & 117 & 0.8173 & 1.587177 \\
193 & -77.88 & 23.50 & 4.0\% & 63/63 & 77 & 0.8166 & 1.877965 \\
194 & -109.58 & 23.50 & 4.0\% & 110/110 & 134 & 0.8152 & 1.514314 \\
195 & -82.74 & 23.35 & 4.0\% & 84/84 & 112 & 0.8141 & 1.677510 \\
196 & -68.88 & 23.52 & 4.0\% & 53/53 & 70 & 0.8134 & 1.638373 \\
197 & -87.80 & 23.57 & 3.0\% & 68/68 & 88 & 0.8126 & 1.703030 \\
198 & -55.56 & 22.43 & 3.0\% & 92/92 & 108 & 0.8115 & 1.688982 \\
199 & -90.59 & 23.26 & 3.0\% & 57/57 & 71 & 0.8108 & 1.817013 \\
200 & -128.54 & 22.85 & 3.0\% & 104/104 & 131 & 0.8095 & 1.963644 \\
201 & -74.98 & 22.75 & 3.0\% & 88/88 & 120 & 0.8083 & 1.608513 \\
202 & -88.61 & 22.66 & 3.0\% & 66/66 & 89 & 0.8074 & 1.615185 \\
203 & -89.73 & 22.40 & 3.0\% & 63/63 & 79 & 0.8066 & 1.945853 \\
204 & -94.13 & 22.68 & 3.0\% & 69/69 & 87 & 0.8058 & 1.464193 \\
205 & -97.33 & 22.90 & 3.0\% & 100/100 & 121 & 0.8046 & 1.530377 \\
206 & -40.22 & 22.44 & 3.0\% & 55/55 & 65 & 0.8039 & 1.459274 \\
207 & -96.79 & 22.58 & 3.0\% & 75/75 & 97 & 0.8030 & 1.827271 \\
208 & -114.24 & 22.98 & 3.0\% & 65/65 & 78 & 0.8022 & 1.905567 \\
209 & -62.96 & 22.53 & 3.0\% & 48/48 & 66 & 0.8016 & 1.530909 \\
210 & -108.10 & 22.47 & 3.0\% & 91/91 & 113 & 0.8004 & 1.350212 \\
211 & -54.28 & 22.23 & 3.0\% & 77/77 & 95 & 0.7995 & 1.466489 \\
212 & -59.19 & 22.49 & 3.0\% & 59/59 & 68 & 0.7988 & 1.937802 \\
213 & -93.78 & 22.24 & 3.0\% & 73/73 & 102 & 0.7978 & 1.360372 \\
214 & -123.44 & 22.05 & 3.0\% & 68/68 & 89 & 0.7969 & 1.419657 \\
215 & -60.30 & 22.24 & 3.0\% & 56/56 & 70 & 0.7962 & 2.418931 \\
216 & -55.24 & 21.63 & 2.0\% & 66/66 & 85 & 0.7954 & 1.573498 \\
217 & -88.85 & 22.36 & 2.0\% & 82/82 & 104 & 0.7944 & 1.711153 \\
218 & -85.46 & 22.44 & 2.0\% & 70/70 & 92 & 0.7935 & 1.418986 \\
219 & -102.79 & 21.78 & 2.0\% & 69/69 & 87 & 0.7926 & 1.606614 \\
220 & -76.43 & 21.89 & 2.0\% & 71/71 & 90 & 0.7917 & 1.648639 \\
221 & -130.51 & 22.22 & 2.0\% & 66/66 & 84 & 0.7909 & 1.918015 \\
222 & -38.50 & 22.55 & 2.0\% & 50/50 & 68 & 0.7902 & 1.825389 \\
223 & -92.98 & 22.27 & 2.0\% & 51/51 & 61 & 0.7896 & 1.463219 \\
224 & -85.47 & 21.99 & 2.0\% & 92/92 & 111 & 0.7885 & 1.358225 \\
225 & -87.43 & 22.31 & 2.0\% & 92/92 & 114 & 0.7874 & 1.574465 \\
226 & -98.99 & 21.64 & 2.0\% & 48/48 & 64 & 0.7867 & 1.453475 \\
227 & -74.06 & 21.60 & 2.0\% & 54/54 & 74 & 0.7860 & 1.560851 \\
228 & -76.55 & 21.36 & 2.0\% & 103/103 & 128 & 0.7847 & 1.942909 \\
229 & -90.28 & 20.97 & 2.0\% & 79/79 & 99 & 0.7838 & 1.477824 \\
230 & -173.46 & 21.75 & 1.0\% & 83/83 & 107 & 0.7827 & 1.569286 \\
231 & -82.20 & 21.11 & 1.0\% & 51/51 & 66 & 0.7820 & 1.548526 \\
232 & -49.50 & 21.21 & 1.0\% & 66/66 & 79 & 0.7813 & 1.632938 \\
233 & -41.15 & 21.30 & 1.0\% & 66/66 & 84 & 0.7804 & 1.885037 \\
234 & -71.65 & 21.15 & 1.0\% & 64/64 & 82 & 0.7796 & 1.572154 \\
235 & -118.99 & 21.12 & 1.0\% & 65/65 & 87 & 0.7788 & 1.676862 \\
236 & -123.46 & 21.06 & 1.0\% & 69/69 & 84 & 0.7779 & 1.493541 \\
237 & -90.38 & 20.56 & 1.0\% & 66/66 & 79 & 0.7771 & 1.538128 \\
238 & -79.19 & 20.90 & 1.0\% & 76/76 & 99 & 0.7762 & 1.840692 \\
239 & -192.68 & 20.40 & 1.0\% & 111/111 & 136 & 0.7748 & 1.693084 \\
240 & -75.47 & 20.06 & 1.0\% & 90/90 & 108 & 0.7737 & 1.715704 \\
241 & 6.89 & 20.41 & 1.0\% & 59/59 & 71 & 0.7730 & 1.643630 \\
242 & -115.88 & 19.34 & 1.0\% & 80/80 & 97 & 0.7721 & 1.745693 \\
243 & -90.31 & 19.89 & 1.0\% & 77/77 & 98 & 0.7711 & 1.586654 \\
244 & -97.39 & 19.99 & 1.0\% & 89/89 & 106 & 0.7701 & 1.742659 \\
245 & -48.02 & 19.77 & 2.0\% & 56/56 & 70 & 0.7694 & 2.027872 \\
246 & -108.10 & 19.46 & 2.0\% & 61/61 & 80 & 0.7686 & 1.780995 \\
247 & -79.19 & 19.56 & 2.0\% & 52/52 & 61 & 0.7680 & 1.770342 \\
248 & -47.41 & 19.59 & 2.0\% & 76/76 & 101 & 0.7670 & 1.832935 \\
249 & -92.95 & 19.28 & 2.0\% & 69/69 & 88 & 0.7661 & 1.691975 \\
250 & -101.24 & 18.80 & 2.0\% & 80/80 & 102 & 0.7651 & 1.646006 \\
251 & -56.05 & 20.03 & 2.0\% & 63/63 & 80 & 0.7643 & 2.199724 \\
252 & -140.06 & 18.66 & 2.0\% & 70/70 & 86 & 0.7634 & 1.938795 \\
253 & -88.24 & 19.41 & 2.0\% & 47/47 & 56 & 0.7629 & 1.424933 \\
254 & -80.40 & 17.91 & 2.0\% & 96/96 & 120 & 0.7617 & 2.191430 \\
255 & -8.69 & 17.66 & 2.0\% & 113/113 & 135 & 0.7604 & 1.628138 \\
256 & -16.71 & 18.00 & 3.0\% & 97/97 & 125 & 0.7591 & 1.810280 \\
257 & -122.36 & 18.30 & 3.0\% & 83/83 & 111 & 0.7580 & 1.982012 \\
258 & -128.90 & 17.04 & 3.0\% & 68/68 & 83 & 0.7572 & 1.611440 \\
259 & -30.31 & 17.09 & 3.0\% & 77/77 & 101 & 0.7562 & 1.914634 \\
260 & -107.14 & 17.53 & 3.0\% & 73/73 & 93 & 0.7553 & 1.842728 \\
261 & -45.17 & 16.66 & 4.0\% & 56/56 & 72 & 0.7546 & 1.218604 \\
262 & -47.41 & 16.72 & 4.0\% & 77/77 & 97 & 0.7536 & 1.739137 \\
263 & -32.66 & 16.80 & 4.0\% & 59/59 & 71 & 0.7529 & 1.948145 \\
264 & -195.18 & 16.19 & 4.0\% & 91/91 & 109 & 0.7518 & 1.685366 \\
265 & -62.56 & 15.76 & 4.0\% & 110/110 & 131 & 0.7505 & 1.996075 \\
266 & -58.36 & 16.44 & 4.0\% & 110/110 & 129 & 0.7493 & 2.084319 \\
267 & -201.88 & 15.95 & 3.0\% & 84/84 & 103 & 0.7482 & 1.955176 \\
268 & -16.97 & 15.94 & 4.0\% & 56/56 & 70 & 0.7476 & 1.549837 \\
269 & -101.00 & 15.61 & 4.0\% & 64/64 & 84 & 0.7467 & 1.876105 \\
270 & -186.46 & 15.01 & 4.0\% & 110/110 & 145 & 0.7453 & 1.695637 \\
271 & -76.39 & 15.82 & 4.0\% & 70/70 & 96 & 0.7443 & 1.910067 \\
272 & -64.06 & 15.55 & 4.0\% & 80/80 & 94 & 0.7434 & 1.909203 \\
273 & -75.00 & 14.66 & 4.0\% & 90/90 & 114 & 0.7423 & 1.925705 \\
274 & -126.89 & 15.36 & 4.0\% & 82/82 & 100 & 0.7413 & 2.129779 \\
275 & -67.96 & 15.22 & 4.0\% & 63/63 & 78 & 0.7405 & 1.629749 \\
276 & -62.30 & 15.09 & 4.0\% & 72/72 & 85 & 0.7397 & 1.814668 \\
277 & -84.02 & 15.04 & 4.0\% & 82/82 & 102 & 0.7387 & 1.587045 \\
278 & -108.53 & 14.85 & 4.0\% & 62/62 & 74 & 0.7379 & 2.117667 \\
279 & -35.39 & 13.71 & 4.0\% & 100/100 & 122 & 0.7367 & 1.931841 \\
280 & -60.98 & 14.16 & 4.0\% & 54/54 & 66 & 0.7361 & 1.745285 \\
281 & -78.80 & 14.39 & 4.0\% & 73/73 & 92 & 0.7352 & 1.854308 \\
282 & -80.80 & 13.99 & 4.0\% & 79/79 & 98 & 0.7342 & 1.829802 \\
283 & -79.48 & 14.25 & 4.0\% & 92/92 & 109 & 0.7331 & 2.006370 \\
284 & -43.66 & 13.03 & 4.0\% & 102/102 & 128 & 0.7318 & 1.859217 \\
285 & -80.84 & 14.10 & 4.0\% & 68/68 & 84 & 0.7310 & 1.822400 \\
286 & -111.47 & 13.50 & 4.0\% & 73/73 & 93 & 0.7301 & 2.224558 \\
287 & -58.56 & 13.73 & 4.0\% & 115/115 & 139 & 0.7287 & 1.798636 \\
288 & -64.27 & 13.78 & 4.0\% & 53/53 & 72 & 0.7280 & 1.425495 \\
289 & -72.93 & 12.58 & 4.0\% & 66/66 & 82 & 0.7272 & 1.561913 \\
290 & -58.16 & 12.01 & 4.0\% & 80/80 & 99 & 0.7262 & 2.114531 \\
291 & -73.43 & 12.46 & 4.0\% & 103/103 & 122 & 0.7250 & 1.829760 \\
292 & -74.10 & 11.98 & 4.0\% & 70/70 & 84 & 0.7242 & 1.571418 \\
293 & -76.66 & 12.45 & 4.0\% & 83/83 & 101 & 0.7232 & 1.897919 \\
294 & -92.52 & 11.13 & 4.0\% & 74/74 & 101 & 0.7222 & 1.707758 \\
295 & -142.41 & 11.85 & 4.0\% & 75/75 & 98 & 0.7212 & 1.615216 \\
296 & -60.58 & 11.02 & 4.0\% & 85/85 & 113 & 0.7201 & 2.167725 \\
297 & -71.40 & 11.98 & 4.0\% & 69/69 & 82 & 0.7193 & 1.847643 \\
298 & -109.24 & 11.51 & 4.0\% & 69/69 & 85 & 0.7184 & 1.830173 \\
299 & -64.14 & 10.80 & 4.0\% & 77/77 & 87 & 0.7176 & 1.904852 \\
300 & -56.66 & 11.07 & 4.0\% & 88/88 & 113 & 0.7164 & 1.869629 \\
301 & -171.07 & 10.73 & 4.0\% & 85/85 & 100 & 0.7155 & 1.664337 \\
302 & -112.99 & 10.79 & 4.0\% & 54/54 & 66 & 0.7148 & 2.081488 \\
303 & -13.65 & 10.12 & 4.0\% & 102/102 & 125 & 0.7136 & 2.099709 \\
304 & -45.82 & 10.92 & 4.0\% & 47/47 & 66 & 0.7129 & 2.100694 \\
305 & -57.84 & 9.91 & 4.0\% & 106/106 & 122 & 0.7117 & 2.326991 \\
306 & -87.39 & 10.69 & 4.0\% & 82/82 & 97 & 0.7107 & 2.321657 \\
307 & -5.04 & 10.19 & 4.0\% & 115/115 & 136 & 0.7094 & 2.053267 \\
308 & -106.27 & 10.18 & 4.0\% & 73/73 & 90 & 0.7085 & 2.246442 \\
309 & -44.65 & 9.33 & 5.0\% & 65/65 & 79 & 0.7077 & 2.178892 \\
310 & -146.25 & 9.64 & 5.0\% & 62/62 & 73 & 0.7070 & 2.014849 \\
311 & -62.57 & 9.71 & 5.0\% & 89/89 & 120 & 0.7058 & 2.219280 \\
312 & -16.65 & 9.74 & 5.0\% & 57/57 & 67 & 0.7051 & 1.895181 \\
313 & -72.29 & 9.56 & 5.0\% & 119/119 & 138 & 0.7038 & 2.014676 \\
314 & -53.83 & 8.85 & 5.0\% & 54/54 & 75 & 0.7030 & 1.774247 \\
315 & -53.47 & 9.08 & 6.0\% & 82/82 & 105 & 0.7020 & 2.369001 \\
316 & -49.49 & 9.19 & 6.0\% & 108/108 & 134 & 0.7007 & 2.077648 \\
317 & -59.33 & 9.07 & 6.0\% & 63/63 & 79 & 0.6999 & 1.643977 \\
318 & -130.84 & 9.66 & 6.0\% & 74/74 & 94 & 0.6990 & 1.792806 \\
319 & -88.75 & 9.74 & 6.0\% & 65/65 & 79 & 0.6982 & 2.115097 \\
320 & -95.47 & 8.36 & 6.0\% & 72/72 & 87 & 0.6973 & 2.214764 \\
321 & -118.42 & 8.14 & 6.0\% & 86/86 & 100 & 0.6963 & 1.821083 \\
322 & -86.43 & 7.97 & 6.0\% & 73/73 & 91 & 0.6954 & 2.149022 \\
323 & -61.43 & 8.58 & 6.0\% & 75/75 & 97 & 0.6945 & 2.064251 \\
324 & -6.13 & 8.72 & 6.0\% & 111/111 & 136 & 0.6931 & 2.000525 \\
325 & -105.06 & 8.00 & 6.0\% & 51/51 & 73 & 0.6924 & 1.858976 \\
326 & -61.25 & 8.07 & 6.0\% & 55/55 & 73 & 0.6917 & 2.083863 \\
327 & -80.20 & 8.79 & 6.0\% & 67/67 & 80 & 0.6909 & 2.067063 \\
328 & -13.05 & 8.72 & 6.0\% & 89/89 & 109 & 0.6898 & 2.133909 \\
329 & -19.79 & 8.52 & 6.0\% & 67/67 & 84 & 0.6890 & 1.900431 \\
330 & -39.29 & 7.40 & 7.0\% & 91/91 & 113 & 0.6879 & 1.811281 \\
331 & -70.07 & 7.46 & 7.0\% & 62/62 & 79 & 0.6871 & 1.612279 \\
332 & -23.44 & 7.25 & 7.0\% & 97/97 & 127 & 0.6858 & 2.061373 \\
333 & -49.86 & 6.86 & 7.0\% & 82/82 & 100 & 0.6848 & 2.448787 \\
334 & -9.82 & 7.63 & 8.0\% & 83/83 & 105 & 0.6838 & 2.120857 \\
335 & -57.15 & 7.69 & 8.0\% & 88/88 & 111 & 0.6827 & 1.868202 \\
336 & -50.63 & 6.93 & 8.0\% & 65/65 & 75 & 0.6819 & 1.942321 \\
337 & -46.28 & 7.13 & 8.0\% & 76/76 & 91 & 0.6810 & 1.983668 \\
338 & -68.98 & 7.27 & 8.0\% & 57/57 & 71 & 0.6803 & 2.177882 \\
339 & -60.69 & 6.83 & 8.0\% & 110/110 & 139 & 0.6790 & 1.939929 \\
340 & -75.14 & 7.07 & 8.0\% & 84/84 & 99 & 0.6780 & 2.198638 \\
341 & -66.00 & 7.32 & 8.0\% & 100/100 & 120 & 0.6768 & 1.961078 \\
342 & -107.65 & 6.57 & 8.0\% & 79/79 & 103 & 0.6758 & 2.374653 \\
343 & -26.84 & 6.95 & 8.0\% & 63/63 & 74 & 0.6750 & 1.851807 \\
344 & -121.83 & 6.99 & 8.0\% & 99/99 & 113 & 0.6739 & 1.835206 \\
345 & -63.09 & 5.97 & 7.0\% & 71/71 & 84 & 0.6731 & 2.068032 \\
346 & -59.87 & 6.66 & 7.0\% & 88/88 & 109 & 0.6720 & 2.356334 \\
347 & -39.35 & 7.06 & 7.0\% & 69/69 & 84 & 0.6712 & 2.428952 \\
348 & -66.57 & 6.37 & 7.0\% & 60/60 & 74 & 0.6704 & 2.440426 \\
349 & -62.17 & 6.46 & 7.0\% & 49/49 & 63 & 0.6698 & 2.015766 \\
350 & -36.60 & 6.72 & 7.0\% & 84/84 & 108 & 0.6688 & 2.320975 \\
351 & -117.73 & 6.83 & 7.0\% & 59/59 & 72 & 0.6680 & 2.276506 \\
352 & -4.27 & 6.27 & 8.0\% & 80/80 & 105 & 0.6670 & 2.214620 \\
353 & -95.30 & 6.99 & 8.0\% & 66/66 & 89 & 0.6661 & 1.993103 \\
354 & -25.01 & 6.95 & 8.0\% & 100/100 & 115 & 0.6650 & 2.191291 \\
355 & 13.25 & 6.56 & 8.0\% & 81/81 & 97 & 0.6640 & 1.943993 \\
356 & -61.11 & 6.23 & 7.0\% & 55/55 & 64 & 0.6634 & 2.580250 \\
357 & -54.08 & 6.91 & 7.0\% & 96/96 & 124 & 0.6622 & 2.086806 \\
358 & -5.32 & 6.39 & 8.0\% & 71/71 & 89 & 0.6613 & 1.856523 \\
359 & -79.98 & 6.08 & 8.0\% & 72/72 & 86 & 0.6604 & 2.185174 \\
360 & -28.27 & 6.46 & 8.0\% & 86/86 & 103 & 0.6594 & 2.004092 \\
361 & -9.16 & 6.66 & 7.0\% & 69/69 & 84 & 0.6586 & 2.342319 \\
362 & -56.36 & 6.21 & 7.0\% & 78/78 & 100 & 0.6576 & 2.195731 \\
363 & -53.97 & 5.50 & 7.0\% & 88/88 & 113 & 0.6565 & 1.743275 \\
364 & -51.87 & 5.10 & 7.0\% & 68/68 & 78 & 0.6557 & 1.681448 \\
365 & -48.79 & 5.26 & 7.0\% & 102/102 & 132 & 0.6544 & 2.412744 \\
366 & -61.93 & 4.88 & 7.0\% & 109/109 & 124 & 0.6532 & 2.202618 \\
367 & -44.25 & 5.02 & 7.0\% & 114/114 & 133 & 0.6518 & 2.222927 \\
368 & -112.75 & 5.05 & 6.0\% & 94/94 & 111 & 0.6507 & 2.015333 \\
369 & -18.72 & 5.92 & 6.0\% & 59/59 & 74 & 0.6500 & 2.508833 \\
370 & -39.76 & 5.10 & 6.0\% & 94/94 & 113 & 0.6489 & 2.485069 \\
371 & -39.89 & 5.31 & 7.0\% & 95/95 & 106 & 0.6478 & 1.848503 \\
372 & -56.18 & 4.40 & 8.0\% & 85/85 & 112 & 0.6467 & 2.331068 \\
373 & -6.97 & 4.39 & 8.0\% & 87/87 & 102 & 0.6457 & 2.007565 \\
374 & -132.37 & 5.41 & 8.0\% & 74/74 & 87 & 0.6449 & 1.761904 \\
375 & -29.39 & 5.02 & 8.0\% & 107/107 & 129 & 0.6436 & 2.109115 \\
376 & -107.56 & 4.90 & 8.0\% & 72/72 & 88 & 0.6427 & 2.242170 \\
377 & -52.50 & 5.02 & 8.0\% & 67/67 & 81 & 0.6419 & 2.057042 \\
378 & -190.88 & 4.87 & 8.0\% & 82/82 & 101 & 0.6409 & 2.231789 \\
379 & -107.38 & 4.26 & 8.0\% & 91/91 & 118 & 0.6397 & 2.007592 \\
380 & -74.29 & 4.92 & 8.0\% & 75/75 & 97 & 0.6388 & 2.034870 \\
381 & -36.53 & 4.94 & 8.0\% & 82/82 & 109 & 0.6377 & 2.200733 \\
382 & -72.88 & 4.63 & 8.0\% & 62/62 & 72 & 0.6370 & 1.857883 \\
383 & -33.85 & 5.80 & 8.0\% & 103/103 & 132 & 0.6357 & 2.039110 \\
384 & -87.97 & 4.63 & 8.0\% & 106/106 & 128 & 0.6344 & 2.187184 \\
385 & -83.57 & 5.95 & 8.0\% & 65/65 & 80 & 0.6336 & 2.452026 \\
386 & -19.39 & 4.98 & 8.0\% & 86/86 & 113 & 0.6325 & 2.045042 \\
387 & -59.90 & 5.37 & 8.0\% & 90/90 & 119 & 0.6313 & 2.061913 \\
388 & -46.12 & 5.63 & 8.0\% & 79/79 & 94 & 0.6304 & 1.938592 \\
389 & -98.79 & 4.58 & 8.0\% & 117/117 & 141 & 0.6290 & 1.893088 \\
390 & -86.80 & 6.08 & 8.0\% & 105/105 & 127 & 0.6278 & 2.271907 \\
391 & -71.58 & 4.64 & 8.0\% & 82/82 & 105 & 0.6267 & 2.489404 \\
392 & -49.57 & 5.76 & 8.0\% & 69/69 & 78 & 0.6259 & 2.148888 \\
393 & 31.25 & 5.25 & 8.0\% & 127/127 & 152 & 0.6244 & 1.902861 \\
394 & -98.62 & 5.09 & 8.0\% & 77/77 & 98 & 0.6235 & 2.108707 \\
395 & -38.57 & 4.71 & 8.0\% & 84/84 & 106 & 0.6224 & 2.065863 \\
396 & -62.62 & 5.16 & 8.0\% & 57/57 & 71 & 0.6217 & 2.391584 \\
397 & -15.84 & 5.42 & 8.0\% & 808/808 & 1000 & 0.6118 & 2.141872 \\
398 & -17.99 & 4.90 & 8.0\% & 87/87 & 101 & 0.6108 & 2.347038 \\
399 & -49.77 & 4.39 & 8.0\% & 56/56 & 79 & 0.6100 & 2.297859 \\
400 & -48.85 & 6.10 & 8.0\% & 96/96 & 111 & 0.6089 & 2.386700 \\
401 & -70.90 & 5.11 & 8.0\% & 48/48 & 59 & 0.6083 & 2.658266 \\
402 & -69.22 & 6.22 & 8.0\% & 84/84 & 114 & 0.6072 & 2.199284 \\
403 & -37.32 & 5.89 & 8.0\% & 75/75 & 82 & 0.6064 & 2.527638 \\
404 & 23.05 & 6.73 & 8.0\% & 100/100 & 114 & 0.6053 & 2.422647 \\
405 & -17.29 & 6.21 & 8.0\% & 60/60 & 78 & 0.6045 & 2.704886 \\
406 & -45.18 & 6.08 & 8.0\% & 71/71 & 85 & 0.6037 & 2.169572 \\
407 & -20.39 & 6.91 & 8.0\% & 85/85 & 102 & 0.6027 & 2.138558 \\
408 & -113.95 & 5.99 & 8.0\% & 128/128 & 158 & 0.6011 & 2.521910 \\
409 & -22.18 & 6.31 & 7.0\% & 137/137 & 170 & 0.5994 & 2.474915 \\
410 & -107.59 & 6.92 & 7.0\% & 81/81 & 106 & 0.5984 & 2.091849 \\
411 & -85.42 & 6.17 & 7.0\% & 85/85 & 106 & 0.5973 & 2.597854 \\
412 & -25.84 & 6.74 & 7.0\% & 109/109 & 120 & 0.5961 & 2.350578 \\
413 & -73.85 & 6.80 & 7.0\% & 94/94 & 117 & 0.5950 & 2.164478 \\
414 & -29.28 & 7.95 & 7.0\% & 101/101 & 116 & 0.5938 & 1.999313 \\
415 & -97.25 & 7.48 & 6.0\% & 72/72 & 80 & 0.5930 & 1.644205 \\
416 & -71.11 & 7.60 & 6.0\% & 69/69 & 85 & 0.5922 & 2.326913 \\
417 & -63.42 & 7.68 & 6.0\% & 72/72 & 88 & 0.5913 & 2.477731 \\
418 & -26.25 & 8.40 & 6.0\% & 71/71 & 87 & 0.5904 & 2.214476 \\
419 & -33.68 & 7.78 & 6.0\% & 112/112 & 137 & 0.5891 & 2.150022 \\
420 & -21.11 & 7.68 & 6.0\% & 102/102 & 134 & 0.5878 & 2.424036 \\
421 & -74.47 & 7.00 & 6.0\% & 82/82 & 108 & 0.5867 & 2.089931 \\
422 & -21.17 & 7.74 & 6.0\% & 54/54 & 68 & 0.5860 & 1.892065 \\
423 & 22.87 & 7.87 & 6.0\% & 100/100 & 111 & 0.5849 & 2.008498 \\
424 & -41.85 & 7.99 & 6.0\% & 55/55 & 71 & 0.5842 & 1.752210 \\
425 & -91.42 & 8.35 & 6.0\% & 694/694 & 809 & 0.5762 & 2.094000 \\
426 & -102.32 & 7.90 & 6.0\% & 65/65 & 75 & 0.5755 & 2.195495 \\
427 & 2.29 & 9.14 & 6.0\% & 109/109 & 137 & 0.5741 & 2.210074 \\
428 & 17.03 & 9.59 & 6.0\% & 128/128 & 161 & 0.5725 & 2.244067 \\
429 & -46.40 & 9.71 & 6.0\% & 71/71 & 91 & 0.5716 & 2.266600 \\
430 & -19.63 & 9.50 & 6.0\% & 71/71 & 89 & 0.5707 & 2.049492 \\
431 & -57.21 & 9.60 & 6.0\% & 70/70 & 76 & 0.5700 & 2.355873 \\
432 & 13.55 & 10.44 & 6.0\% & 140/140 & 165 & 0.5684 & 2.148068 \\
433 & -54.14 & 10.43 & 6.0\% & 66/66 & 75 & 0.5676 & 2.616559 \\
434 & 20.49 & 9.62 & 5.0\% & 108/108 & 138 & 0.5662 & 2.014212 \\
435 & -200.49 & 9.82 & 5.0\% & 151/151 & 176 & 0.5645 & 2.032708 \\
436 & -5.57 & 10.21 & 5.0\% & 134/134 & 160 & 0.5629 & 2.489259 \\
437 & -58.44 & 10.68 & 5.0\% & 94/94 & 121 & 0.5617 & 2.230405 \\
438 & -35.48 & 10.18 & 5.0\% & 77/77 & 95 & 0.5608 & 1.824069 \\
439 & -97.00 & 10.40 & 5.0\% & 90/90 & 109 & 0.5597 & 1.928563 \\
440 & 10.52 & 11.04 & 5.0\% & 137/137 & 157 & 0.5581 & 1.852123 \\
441 & -62.32 & 11.14 & 5.0\% & 101/101 & 121 & 0.5569 & 2.072175 \\
442 & -0.98 & 11.31 & 5.0\% & 190/190 & 227 & 0.5547 & 1.988084 \\
443 & -74.80 & 11.72 & 5.0\% & 99/99 & 125 & 0.5535 & 1.981220 \\
444 & -83.15 & 11.74 & 5.0\% & 90/90 & 104 & 0.5524 & 2.189272 \\
445 & -13.21 & 12.31 & 5.0\% & 84/84 & 99 & 0.5515 & 2.012522 \\
446 & -24.29 & 12.46 & 5.0\% & 118/118 & 139 & 0.5501 & 2.053376 \\
447 & 1.84 & 12.50 & 5.0\% & 91/91 & 109 & 0.5490 & 1.963336 \\
448 & -79.27 & 11.45 & 5.0\% & 96/96 & 116 & 0.5478 & 1.996379 \\
449 & -4.97 & 11.71 & 5.0\% & 76/76 & 91 & 0.5469 & 2.264469 \\
450 & -28.58 & 12.60 & 5.0\% & 859/859 & 1000 & 0.5370 & 2.156582 \\
451 & 11.25 & 13.45 & 6.0\% & 88/88 & 102 & 0.5360 & 1.936949 \\
452 & 26.05 & 12.94 & 5.0\% & 107/107 & 126 & 0.5348 & 2.270718 \\
453 & -41.64 & 13.94 & 5.0\% & 69/69 & 79 & 0.5340 & 1.776997 \\
454 & 12.33 & 13.79 & 5.0\% & 281/281 & 339 & 0.5307 & 2.200667 \\
455 & -105.58 & 13.93 & 5.0\% & 94/94 & 104 & 0.5296 & 2.053877 \\
456 & -36.74 & 15.20 & 5.0\% & 102/102 & 131 & 0.5283 & 2.195576 \\
457 & 11.04 & 15.26 & 5.0\% & 105/105 & 129 & 0.5270 & 2.373856 \\
458 & -22.59 & 14.68 & 4.0\% & 92/92 & 126 & 0.5258 & 2.325152 \\
459 & -20.95 & 15.04 & 5.0\% & 96/96 & 112 & 0.5247 & 1.994342 \\
460 & -45.81 & 15.41 & 5.0\% & 93/93 & 107 & 0.5236 & 2.165225 \\
461 & -23.09 & 14.22 & 5.0\% & 96/96 & 114 & 0.5225 & 2.237493 \\
462 & -26.16 & 16.22 & 5.0\% & 66/66 & 78 & 0.5217 & 2.724532 \\
463 & -50.97 & 14.69 & 5.0\% & 82/82 & 102 & 0.5207 & 2.362777 \\
464 & 3.87 & 16.00 & 5.0\% & 78/78 & 102 & 0.5197 & 2.276784 \\
465 & -37.74 & 15.07 & 5.0\% & 69/69 & 77 & 0.5189 & 2.034264 \\
466 & -4.82 & 15.33 & 5.0\% & 86/86 & 98 & 0.5180 & 2.096598 \\
467 & -63.38 & 15.87 & 5.0\% & 93/93 & 114 & 0.5169 & 2.060322 \\
468 & -50.25 & 15.16 & 5.0\% & 80/80 & 91 & 0.5159 & 2.116940 \\
469 & 14.58 & 16.05 & 6.0\% & 94/94 & 109 & 0.5149 & 2.304878 \\
470 & -20.16 & 16.37 & 6.0\% & 77/77 & 89 & 0.5140 & 2.409199 \\
471 & -30.08 & 16.62 & 5.0\% & 94/94 & 110 & 0.5129 & 1.995865 \\
472 & 13.89 & 15.92 & 5.0\% & 77/77 & 92 & 0.5120 & 2.055146 \\
473 & 42.65 & 15.51 & 5.0\% & 128/128 & 171 & 0.5103 & 2.199956 \\
474 & 7.12 & 16.62 & 5.0\% & 134/134 & 158 & 0.5087 & 2.068596 \\
475 & 3.60 & 16.48 & 5.0\% & 104/104 & 142 & 0.5073 & 1.974928 \\
476 & -3.76 & 15.35 & 5.0\% & 83/83 & 96 & 0.5064 & 1.978720 \\
477 & 30.11 & 15.95 & 5.0\% & 69/69 & 81 & 0.5056 & 1.990688 \\
478 & 19.81 & 16.27 & 6.0\% & 95/95 & 117 & 0.5044 & 1.919618 \\
479 & -117.19 & 16.41 & 6.0\% & 124/124 & 138 & 0.5030 & 2.265060 \\
480 & 15.62 & 16.71 & 6.0\% & 101/101 & 124 & 0.5018 & 1.941855 \\
481 & -25.28 & 16.80 & 6.0\% & 64/64 & 76 & 0.5011 & 1.898428 \\
482 & 17.60 & 17.63 & 6.0\% & 120/120 & 150 & 0.4996 & 2.233117 \\
483 & -12.03 & 17.88 & 6.0\% & 94/94 & 110 & 0.4985 & 2.028969 \\
484 & 15.67 & 17.71 & 6.0\% & 107/107 & 126 & 0.4972 & 2.051550 \\
485 & -65.98 & 17.25 & 6.0\% & 66/66 & 81 & 0.4964 & 1.984363 \\
486 & -3.28 & 17.02 & 6.0\% & 91/91 & 115 & 0.4953 & 2.125995 \\
487 & -23.27 & 17.84 & 6.0\% & 142/142 & 165 & 0.4937 & 2.245756 \\
488 & -49.45 & 17.53 & 6.0\% & 100/100 & 118 & 0.4925 & 2.092334 \\
489 & -64.86 & 17.23 & 6.0\% & 128/128 & 154 & 0.4910 & 2.039883 \\
490 & -31.06 & 17.06 & 6.0\% & 102/102 & 114 & 0.4899 & 1.911560 \\
491 & -30.37 & 17.99 & 6.0\% & 93/93 & 105 & 0.4888 & 2.225257 \\
492 & 7.07 & 17.72 & 7.0\% & 98/98 & 126 & 0.4876 & 2.066245 \\
493 & -43.08 & 18.14 & 7.0\% & 93/93 & 107 & 0.4865 & 2.078609 \\
494 & -9.62 & 17.88 & 8.0\% & 102/102 & 122 & 0.4853 & 2.219338 \\
495 & -21.99 & 19.68 & 8.0\% & 118/118 & 137 & 0.4839 & 2.030998 \\
496 & -13.70 & 19.93 & 8.0\% & 91/91 & 111 & 0.4828 & 1.974329 \\
497 & -37.65 & 19.81 & 8.0\% & 102/102 & 117 & 0.4817 & 2.269247 \\
498 & -37.83 & 19.74 & 8.0\% & 99/99 & 123 & 0.4805 & 2.110061 \\
499 & -65.92 & 20.77 & 8.0\% & 119/119 & 141 & 0.4791 & 2.077336 \\
500 & -8.62 & 20.23 & 8.0\% & 107/107 & 127 & 0.4778 & 2.227235 \\
501 & -48.06 & 20.51 & 8.0\% & 63/63 & 77 & 0.4771 & 2.487910 \\
502 & 38.95 & 20.86 & 8.0\% & 118/118 & 144 & 0.4756 & 2.283982 \\
503 & -114.60 & 22.51 & 8.0\% & 771/771 & 891 & 0.4668 & 2.087320 \\
504 & 8.23 & 24.01 & 8.0\% & 874/874 & 1000 & 0.4569 & 2.113234 \\
505 & 26.40 & 23.70 & 8.0\% & 118/118 & 144 & 0.4555 & 1.906813 \\
506 & -50.34 & 24.68 & 8.0\% & 77/77 & 88 & 0.4546 & 2.321437 \\
507 & -16.42 & 23.59 & 8.0\% & 97/97 & 106 & 0.4536 & 1.910728 \\
508 & 10.13 & 25.73 & 8.0\% & 898/898 & 1000 & 0.4437 & 2.093337 \\
509 & 38.26 & 26.87 & 8.0\% & 111/111 & 137 & 0.4423 & 2.256134 \\
510 & 5.67 & 26.18 & 8.0\% & 107/107 & 120 & 0.4411 & 2.023076 \\
511 & 6.00 & 27.19 & 8.0\% & 70/70 & 87 & 0.4403 & 2.393104 \\
512 & -4.38 & 27.04 & 8.0\% & 127/127 & 149 & 0.4388 & 2.309848 \\
513 & 44.51 & 26.61 & 9.0\% & 114/114 & 127 & 0.4375 & 2.415775 \\
514 & 10.90 & 29.23 & 9.0\% & 893/893 & 1000 & 0.4276 & 2.156320 \\
515 & -162.42 & 30.79 & 9.0\% & 636/636 & 739 & 0.4203 & 2.239384 \\
516 & 27.71 & 30.72 & 10.0\% & 135/135 & 159 & 0.4187 & 2.291326 \\
517 & 45.34 & 31.02 & 10.0\% & 107/107 & 119 & 0.4176 & 1.817282 \\
518 & -46.63 & 30.68 & 10.0\% & 82/82 & 93 & 0.4166 & 2.107507 \\
519 & 20.80 & 31.43 & 10.0\% & 131/131 & 161 & 0.4150 & 2.255856 \\
520 & -32.67 & 32.82 & 10.0\% & 861/861 & 1000 & 0.4051 & 2.185491 \\
521 & 2.20 & 32.38 & 10.0\% & 84/84 & 95 & 0.4042 & 2.171878 \\
522 & -16.22 & 33.10 & 10.0\% & 871/871 & 1000 & 0.3943 & 2.224746 \\
523 & -60.50 & 33.44 & 10.0\% & 71/71 & 87 & 0.3934 & 2.170188 \\
524 & 27.40 & 33.36 & 11.0\% & 118/118 & 134 & 0.3921 & 2.384510 \\
525 & -39.38 & 34.43 & 11.0\% & 143/143 & 180 & 0.3903 & 2.069836 \\
526 & -59.15 & 35.41 & 11.0\% & 853/853 & 1000 & 0.3804 & 2.182550 \\
527 & 20.49 & 35.41 & 11.0\% & 145/145 & 163 & 0.3788 & 2.301114 \\
528 & 31.60 & 36.23 & 11.0\% & 861/861 & 1000 & 0.3689 & 2.178501 \\
529 & -81.61 & 35.91 & 11.0\% & 209/209 & 236 & 0.3666 & 2.134724 \\
530 & -0.14 & 36.17 & 10.0\% & 104/104 & 117 & 0.3654 & 2.115758 \\
531 & -49.79 & 35.90 & 10.0\% & 867/867 & 1000 & 0.3555 & 2.226328 \\
532 & -150.03 & 36.67 & 10.0\% & 214/214 & 262 & 0.3529 & 2.199675 \\
533 & -25.33 & 36.36 & 10.0\% & 898/898 & 1000 & 0.3430 & 2.163188 \\
534 & -5.61 & 37.76 & 10.0\% & 841/841 & 1000 & 0.3331 & 2.231871 \\
535 & 37.71 & 37.06 & 10.0\% & 85/85 & 97 & 0.3322 & 2.234358 \\
536 & -19.50 & 37.65 & 10.0\% & 897/897 & 1000 & 0.3223 & 2.229193 \\
537 & 3.17 & 37.46 & 10.0\% & 121/121 & 132 & 0.3210 & 1.913514 \\
538 & 23.78 & 37.05 & 10.0\% & 109/109 & 134 & 0.3196 & 2.288059 \\
539 & 52.86 & 37.46 & 10.0\% & 104/104 & 114 & 0.3185 & 1.970167 \\
540 & -10.31 & 36.99 & 10.0\% & 891/891 & 1000 & 0.3086 & 2.273201 \\
541 & 46.55 & 37.26 & 10.0\% & 116/116 & 132 & 0.3073 & 2.167633 \\
542 & -99.89 & 37.56 & 10.0\% & 145/145 & 163 & 0.3057 & 2.014391 \\
543 & 20.75 & 36.20 & 10.0\% & 144/144 & 169 & 0.3040 & 2.242050 \\
544 & -130.72 & 36.54 & 10.0\% & 617/617 & 657 & 0.2975 & 2.279354 \\
545 & -98.96 & 36.56 & 10.0\% & 292/292 & 322 & 0.2943 & 2.257443 \\
546 & -43.91 & 36.03 & 10.0\% & 494/494 & 536 & 0.2890 & 2.144823 \\
547 & 22.96 & 36.76 & 10.0\% & 822/822 & 1000 & 0.2791 & 2.268904 \\
548 & -6.92 & 36.57 & 10.0\% & 883/883 & 1000 & 0.2692 & 2.244905 \\
549 & -10.92 & 35.97 & 10.0\% & 869/869 & 1000 & 0.2593 & 2.169140 \\
550 & 45.43 & 33.72 & 10.0\% & 868/868 & 1000 & 0.2494 & 2.263227 \\
551 & -100.42 & 35.26 & 9.0\% & 342/342 & 385 & 0.2456 & 2.256447 \\
552 & -181.57 & 34.90 & 9.0\% & 626/626 & 688 & 0.2388 & 2.245526 \\
553 & -53.41 & 34.75 & 9.0\% & 867/867 & 1000 & 0.2289 & 2.213559 \\
554 & -34.84 & 34.29 & 9.0\% & 856/856 & 1000 & 0.2190 & 2.225482 \\
555 & 13.16 & 35.19 & 9.0\% & 854/854 & 1000 & 0.2091 & 2.117561 \\
556 & -1.31 & 34.06 & 9.0\% & 836/836 & 1000 & 0.1992 & 2.135824 \\
557 & -40.99 & 34.83 & 9.0\% & 799/799 & 1000 & 0.1893 & 2.177932 \\
558 & 26.23 & 35.90 & 9.0\% & 851/851 & 1000 & 0.1794 & 2.174556 \\
559 & 10.20 & 36.11 & 8.0\% & 834/834 & 1000 & 0.1695 & 2.159282 \\
560 & -6.65 & 36.45 & 8.0\% & 833/833 & 1000 & 0.1596 & 2.060927 \\
561 & 5.34 & 38.04 & 8.0\% & 842/842 & 1000 & 0.1497 & 2.053427 \\
562 & -31.43 & 38.88 & 8.0\% & 859/859 & 1000 & 0.1398 & 2.101515 \\
563 & -72.09 & 39.87 & 8.0\% & 829/829 & 1000 & 0.1299 & 2.029103 \\
564 & -50.12 & 39.05 & 8.0\% & 823/823 & 1000 & 0.1200 & 1.954713 \\
565 & -44.87 & 38.47 & 8.0\% & 865/865 & 1000 & 0.1101 & 2.220863 \\
566 & -8.51 & 39.99 & 8.0\% & 790/790 & 1000 & 0.1002 & 2.019323 \\
567 & -2.48 & 39.89 & 8.0\% & 843/843 & 1000 & 0.0903 & 1.910393 \\
568 & -6.61 & 40.47 & 8.0\% & 911/911 & 1000 & 0.0804 & 1.895835 \\
569 & -29.68 & 40.71 & 7.0\% & 867/867 & 1000 & 0.0705 & 1.907286 \\
570 & -53.29 & 41.67 & 7.0\% & 868/868 & 1000 & 0.0606 & 1.954839 \\
571 & -3.46 & 42.87 & 7.0\% & 868/868 & 1000 & 0.0507 & 1.937218 \\
572 & -20.08 & 41.58 & 6.0\% & 866/866 & 1000 & 0.0408 & 1.899397 \\
573 & -56.71 & 42.98 & 6.0\% & 868/868 & 1000 & 0.0309 & 1.849679 \\
574 & -10.37 & 42.22 & 6.0\% & 840/840 & 1000 & 0.0210 & 1.859757 \\
575 & 93.39 & 45.70 & 6.0\% & 797/797 & 1000 & 0.0111 & 1.854064 \\
576 & 46.06 & 44.65 & 6.0\% & 833/833 & 1000 & 0.0100 & 1.866808 \\
577 & 279.21 & 45.34 & 7.0\% & 364/364 & 467 & 0.0100 & 1.794767 \\
578 & 40.52 & 45.07 & 6.0\% & 823/823 & 1000 & 0.0100 & 1.919856 \\
579 & -22.69 & 46.19 & 6.0\% & 819/819 & 1000 & 0.0100 & 1.703514 \\
580 & 10.40 & 45.46 & 6.0\% & 829/829 & 1000 & 0.0100 & 1.702715 \\
581 & 215.74 & 46.02 & 7.0\% & 489/489 & 590 & 0.0100 & 1.656326 \\
582 & 31.96 & 46.76 & 7.0\% & 801/801 & 1000 & 0.0100 & 1.550960 \\
583 & 229.50 & 47.33 & 8.0\% & 320/320 & 452 & 0.0100 & 1.560385 \\
584 & 166.03 & 49.63 & 9.0\% & 749/749 & 920 & 0.0100 & 1.570260 \\
585 & 27.11 & 46.93 & 9.0\% & 821/821 & 1000 & 0.0100 & 1.480945 \\
586 & -49.96 & 49.36 & 9.0\% & 839/839 & 1000 & 0.0100 & 1.453697 \\
587 & -42.76 & 48.62 & 9.0\% & 848/848 & 1000 & 0.0100 & 1.436978 \\
588 & 5.90 & 48.20 & 9.0\% & 792/792 & 1000 & 0.0100 & 1.475060 \\
589 & -32.12 & 47.59 & 9.0\% & 689/689 & 1000 & 0.0100 & 1.402095 \\
590 & -36.29 & 47.38 & 9.0\% & 780/780 & 1000 & 0.0100 & 1.354233 \\
591 & -3.37 & 48.50 & 9.0\% & 786/786 & 1000 & 0.0100 & 1.354786 \\
592 & -16.08 & 50.67 & 8.0\% & 755/755 & 1000 & 0.0100 & 1.342594 \\
593 & 25.45 & 51.24 & 8.0\% & 793/793 & 1000 & 0.0100 & 1.283155 \\
594 & -24.48 & 51.61 & 7.0\% & 728/728 & 1000 & 0.0100 & 1.285044 \\
595 & 16.75 & 51.50 & 7.0\% & 872/872 & 1000 & 0.0100 & 1.255648 \\
596 & -18.57 & 51.41 & 7.0\% & 841/841 & 1000 & 0.0100 & 1.194250 \\
597 & 16.71 & 54.87 & 7.0\% & 806/806 & 1000 & 0.0100 & 1.240567 \\
598 & -49.56 & 53.35 & 7.0\% & 839/839 & 1000 & 0.0100 & 1.223573 \\
599 & 29.34 & 52.62 & 7.0\% & 790/790 & 1000 & 0.0100 & 1.166419 \\
600 & -0.18 & 51.99 & 7.0\% & 827/827 & 1000 & 0.0100 & 1.125839 \\
601 & -8.76 & 49.65 & 7.0\% & 828/828 & 1000 & 0.0100 & 1.070030 \\
602 & 2.89 & 49.41 & 7.0\% & 800/800 & 1000 & 0.0100 & 1.115206 \\
603 & -53.39 & 46.43 & 7.0\% & 797/797 & 1000 & 0.0100 & 1.072516 \\
604 & -24.98 & 47.51 & 7.0\% & 828/828 & 1000 & 0.0100 & 1.047292 \\
605 & -15.89 & 46.49 & 7.0\% & 855/855 & 1000 & 0.0100 & 1.029109 \\
606 & -3.55 & 45.93 & 7.0\% & 817/817 & 1000 & 0.0100 & 1.027574 \\
607 & -15.83 & 45.91 & 7.0\% & 786/786 & 1000 & 0.0100 & 0.972385 \\
608 & 13.01 & 45.24 & 7.0\% & 793/793 & 1000 & 0.0100 & 0.959675 \\
609 & -10.00 & 46.71 & 7.0\% & 806/806 & 1000 & 0.0100 & 0.973539 \\
610 & -44.97 & 47.25 & 7.0\% & 819/819 & 1000 & 0.0100 & 0.946316 \\
611 & 69.18 & 46.10 & 7.0\% & 800/800 & 1000 & 0.0100 & 0.887567 \\
612 & 43.50 & 45.89 & 7.0\% & 803/803 & 1000 & 0.0100 & 0.881577 \\
613 & -3.20 & 46.00 & 6.0\% & 837/837 & 1000 & 0.0100 & 0.880103 \\
614 & -49.07 & 45.57 & 6.0\% & 825/825 & 1000 & 0.0100 & 0.843194 \\
615 & 147.08 & 46.57 & 7.0\% & 799/799 & 987 & 0.0100 & 0.806115 \\
616 & 261.05 & 47.59 & 7.0\% & 282/282 & 361 & 0.0100 & 0.730023 \\
617 & 259.18 & 48.49 & 8.0\% & 248/248 & 349 & 0.0100 & 0.823380 \\
618 & 242.93 & 48.73 & 9.0\% & 154/154 & 219 & 0.0100 & 0.707065 \\
619 & 238.38 & 49.71 & 10.0\% & 326/326 & 463 & 0.0100 & 0.687388 \\
620 & -35.25 & 50.30 & 10.0\% & 843/843 & 1000 & 0.0100 & 0.743365 \\
621 & 10.59 & 48.91 & 10.0\% & 844/844 & 1000 & 0.0100 & 0.713110 \\
622 & 245.66 & 49.91 & 11.0\% & 315/315 & 451 & 0.0100 & 0.641609 \\
623 & 210.92 & 52.25 & 12.0\% & 685/685 & 851 & 0.0100 & 0.669727 \\
624 & 256.27 & 52.83 & 12.0\% & 359/359 & 459 & 0.0100 & 0.724296 \\
625 & 269.23 & 52.65 & 13.0\% & 392/392 & 490 & 0.0100 & 0.644377 \\
626 & 288.61 & 51.56 & 14.0\% & 491/491 & 698 & 0.0100 & 0.694258 \\
627 & 291.72 & 51.69 & 15.0\% & 174/174 & 228 & 0.0100 & 0.678116 \\
628 & 52.12 & 51.26 & 15.0\% & 823/823 & 1000 & 0.0100 & 0.623395 \\
629 & 224.09 & 52.31 & 16.0\% & 341/341 & 419 & 0.0100 & 0.591657 \\
630 & 275.54 & 52.14 & 17.0\% & 147/147 & 219 & 0.0100 & 0.636184 \\
631 & 11.78 & 52.03 & 17.0\% & 804/804 & 1000 & 0.0100 & 0.629307 \\
632 & 196.23 & 51.99 & 18.0\% & 703/703 & 955 & 0.0100 & 0.614482 \\
633 & 220.31 & 50.06 & 19.0\% & 427/427 & 565 & 0.0100 & 0.596930 \\
634 & 222.98 & 49.12 & 20.0\% & 301/301 & 429 & 0.0100 & 0.561561 \\
635 & 252.82 & 49.27 & 21.0\% & 220/220 & 299 & 0.0100 & 0.621492 \\
636 & 248.96 & 51.97 & 22.0\% & 351/351 & 561 & 0.0100 & 0.540351 \\
637 & 251.28 & 50.59 & 23.0\% & 370/370 & 490 & 0.0100 & 0.528677 \\
638 & 255.81 & 49.44 & 24.0\% & 284/284 & 391 & 0.0100 & 0.525225 \\
639 & 239.22 & 50.13 & 25.0\% & 150/150 & 223 & 0.0100 & 0.497624 \\
640 & 226.91 & 49.85 & 26.0\% & 489/489 & 741 & 0.0100 & 0.509268 \\
641 & 191.10 & 50.10 & 27.0\% & 533/533 & 746 & 0.0100 & 0.511242 \\
642 & 272.12 & 51.89 & 28.0\% & 127/127 & 206 & 0.0100 & 0.519933 \\
643 & 189.38 & 52.01 & 29.0\% & 624/624 & 767 & 0.0100 & 0.496559 \\
644 & 182.78 & 49.92 & 29.0\% & 618/618 & 1000 & 0.0100 & 0.458069 \\
645 & 67.51 & 48.43 & 29.0\% & 798/798 & 1000 & 0.0100 & 0.446498 \\
646 & 289.87 & 48.55 & 30.0\% & 252/252 & 395 & 0.0100 & 0.404672 \\
647 & 258.42 & 50.01 & 31.0\% & 210/210 & 420 & 0.0100 & 0.425631 \\
648 & 242.94 & 49.90 & 32.0\% & 336/336 & 448 & 0.0100 & 0.445050 \\
649 & 216.39 & 51.34 & 33.0\% & 345/345 & 428 & 0.0100 & 0.408972 \\
650 & 314.69 & 50.54 & 34.0\% & 200/200 & 250 & 0.0100 & 0.424440 \\
651 & 256.38 & 50.14 & 35.0\% & 236/236 & 384 & 0.0100 & 0.392141 \\
652 & 249.92 & 49.74 & 35.0\% & 337/337 & 558 & 0.0100 & 0.449590 \\
653 & 131.79 & 48.42 & 35.0\% & 765/765 & 1000 & 0.0100 & 0.404520 \\
654 & -59.68 & 51.97 & 35.0\% & 884/884 & 1000 & 0.0100 & 0.395341 \\
655 & 276.78 & 52.49 & 36.0\% & 321/321 & 409 & 0.0100 & 0.399780 \\
656 & 26.91 & 52.16 & 36.0\% & 772/772 & 1000 & 0.0100 & 0.414017 \\
657 & 178.39 & 57.54 & 37.0\% & 609/609 & 850 & 0.0100 & 0.391655 \\
658 & -30.14 & 58.60 & 37.0\% & 749/749 & 1000 & 0.0100 & 0.371682 \\
659 & 280.48 & 59.09 & 38.0\% & 357/357 & 495 & 0.0100 & 0.330422 \\
660 & 145.46 & 59.99 & 39.0\% & 700/700 & 929 & 0.0100 & 0.370412 \\
661 & 14.89 & 59.61 & 39.0\% & 784/784 & 1000 & 0.0100 & 0.363763 \\
662 & -47.63 & 62.39 & 39.0\% & 739/739 & 1000 & 0.0100 & 0.339301 \\
663 & 188.89 & 64.66 & 39.0\% & 615/615 & 1000 & 0.0100 & 0.316752 \\
664 & 111.95 & 66.01 & 39.0\% & 667/667 & 1000 & 0.0100 & 0.338865 \\
665 & 254.99 & 63.95 & 40.0\% & 352/352 & 439 & 0.0100 & 0.348107 \\
666 & 132.93 & 66.21 & 40.0\% & 681/681 & 1000 & 0.0100 & 0.336823 \\
667 & 295.69 & 67.69 & 41.0\% & 380/380 & 608 & 0.0100 & 0.335736 \\
668 & 248.44 & 68.26 & 42.0\% & 324/324 & 616 & 0.0100 & 0.322013 \\
669 & 133.82 & 68.65 & 42.0\% & 610/610 & 1000 & 0.0100 & 0.318516 \\
670 & 222.13 & 70.49 & 43.0\% & 433/433 & 720 & 0.0100 & 0.334718 \\
671 & 279.93 & 70.49 & 44.0\% & 250/250 & 430 & 0.0100 & 0.291869 \\
672 & 239.14 & 71.36 & 45.0\% & 417/417 & 556 & 0.0100 & 0.313304 \\
673 & 253.23 & 71.77 & 46.0\% & 452/452 & 772 & 0.0100 & 0.306655 \\
674 & 98.42 & 72.78 & 46.0\% & 689/689 & 1000 & 0.0100 & 0.283330 \\
675 & 116.02 & 72.51 & 46.0\% & 718/718 & 1000 & 0.0100 & 0.298865 \\
676 & 256.70 & 73.46 & 47.0\% & 525/525 & 732 & 0.0100 & 0.285059 \\
677 & 245.13 & 74.11 & 47.0\% & 595/595 & 933 & 0.0100 & 0.291244 \\
678 & 244.62 & 74.26 & 48.0\% & 242/242 & 352 & 0.0100 & 0.308327 \\
679 & 268.58 & 75.58 & 49.0\% & 370/370 & 537 & 0.0100 & 0.311488 \\
680 & 278.73 & 75.77 & 50.0\% & 318/318 & 582 & 0.0100 & 0.297741 \\
681 & 169.61 & 75.72 & 49.0\% & 543/543 & 1000 & 0.0100 & 0.307440 \\
682 & 252.92 & 78.05 & 50.0\% & 348/348 & 579 & 0.0100 & 0.308810 \\
683 & 264.94 & 77.18 & 50.0\% & 370/370 & 527 & 0.0100 & 0.320620 \\
684 & 210.52 & 75.68 & 50.0\% & 367/367 & 526 & 0.0100 & 0.305796 \\
685 & -525.38 & 74.99 & 50.0\% & 90/90 & 98 & 0.0100 & 0.313191 \\
686 & 249.79 & 68.62 & 51.0\% & 566/566 & 997 & 0.0100 & 0.346325 \\
687 & -115.47 & 66.16 & 51.0\% & 378/378 & 563 & 0.0100 & 0.340552 \\
688 & 241.25 & 66.41 & 52.0\% & 243/243 & 406 & 0.0100 & 0.332790 \\
689 & 244.74 & 67.32 & 53.0\% & 333/333 & 578 & 0.0100 & 0.342254 \\
690 & 263.75 & 66.66 & 54.0\% & 265/265 & 460 & 0.0100 & 0.293044 \\
691 & 120.35 & 67.95 & 54.0\% & 611/611 & 1000 & 0.0100 & 0.338854 \\
692 & 251.80 & 66.57 & 55.0\% & 264/264 & 456 & 0.0100 & 0.323626 \\
693 & 271.22 & 66.59 & 56.0\% & 315/315 & 399 & 0.0100 & 0.338996 \\
694 & 250.10 & 68.35 & 57.0\% & 340/340 & 443 & 0.0100 & 0.327743 \\
695 & 249.51 & 65.81 & 58.0\% & 303/303 & 441 & 0.0100 & 0.344796 \\
696 & 230.55 & 66.76 & 59.0\% & 485/485 & 809 & 0.0100 & 0.366118 \\
697 & 237.70 & 67.31 & 60.0\% & 251/251 & 444 & 0.0100 & 0.344531 \\
698 & 237.26 & 66.58 & 61.0\% & 271/271 & 404 & 0.0100 & 0.363775 \\
699 & 241.79 & 66.19 & 62.0\% & 223/223 & 306 & 0.0100 & 0.335062 \\
700 & 69.98 & 63.61 & 62.0\% & 841/841 & 1000 & 0.0100 & 0.378916 \\
701 & 98.50 & 65.63 & 62.0\% & 757/757 & 1000 & 0.0100 & 0.401114 \\
702 & 193.37 & 65.85 & 63.0\% & 628/628 & 782 & 0.0100 & 0.350956 \\
703 & 276.74 & 64.24 & 64.0\% & 216/216 & 317 & 0.0100 & 0.331195 \\
704 & 262.51 & 65.60 & 65.0\% & 172/172 & 272 & 0.0100 & 0.360370 \\
705 & 262.89 & 65.55 & 66.0\% & 232/232 & 360 & 0.0100 & 0.346361 \\
706 & 264.71 & 67.21 & 67.0\% & 361/361 & 666 & 0.0100 & 0.378309 \\
707 & 224.08 & 67.37 & 68.0\% & 334/334 & 471 & 0.0100 & 0.395763 \\
708 & 26.02 & 67.68 & 68.0\% & 845/845 & 1000 & 0.0100 & 0.364597 \\
709 & 277.00 & 69.06 & 69.0\% & 271/271 & 448 & 0.0100 & 0.350848 \\
710 & 250.68 & 68.64 & 70.0\% & 429/429 & 643 & 0.0100 & 0.341454 \\
711 & 211.74 & 69.13 & 71.0\% & 279/279 & 412 & 0.0100 & 0.368164 \\
712 & 244.99 & 68.60 & 72.0\% & 311/311 & 529 & 0.0100 & 0.333746 \\
713 & 68.46 & 69.10 & 72.0\% & 751/751 & 1000 & 0.0100 & 0.308837 \\
714 & 237.29 & 68.46 & 73.0\% & 252/252 & 503 & 0.0100 & 0.326201 \\
715 & 262.07 & 68.81 & 73.0\% & 145/145 & 218 & 0.0100 & 0.384118 \\
716 & 272.43 & 68.97 & 73.0\% & 239/239 & 433 & 0.0100 & 0.324413 \\
717 & 215.26 & 68.50 & 73.0\% & 424/424 & 600 & 0.0100 & 0.314506 \\
718 & 279.99 & 70.79 & 73.0\% & 296/296 & 507 & 0.0100 & 0.337828 \\
719 & 236.44 & 71.02 & 73.0\% & 276/276 & 465 & 0.0100 & 0.344704 \\
720 & 262.66 & 70.34 & 74.0\% & 239/239 & 357 & 0.0100 & 0.332580 \\
721 & 265.86 & 70.74 & 75.0\% & 308/308 & 471 & 0.0100 & 0.339727 \\
722 & 270.86 & 70.34 & 75.0\% & 121/121 & 232 & 0.0100 & 0.367570 \\
723 & 155.13 & 69.47 & 74.0\% & 495/495 & 1000 & 0.0100 & 0.354664 \\
724 & 174.64 & 69.53 & 73.0\% & 442/442 & 1000 & 0.0100 & 0.332997 \\
725 & 270.59 & 69.11 & 73.0\% & 329/329 & 639 & 0.0100 & 0.354424 \\
726 & 259.51 & 69.30 & 73.0\% & 100/100 & 180 & 0.0100 & 0.308646 \\
727 & 291.76 & 68.65 & 73.0\% & 212/212 & 292 & 0.0100 & 0.347418 \\
728 & 241.37 & 69.84 & 74.0\% & 342/342 & 593 & 0.0100 & 0.361483 \\
729 & 151.33 & 68.63 & 73.0\% & 617/617 & 1000 & 0.0100 & 0.357037 \\
730 & 249.77 & 71.64 & 73.0\% & 418/418 & 968 & 0.0100 & 0.341380 \\
731 & 262.46 & 71.64 & 74.0\% & 224/224 & 457 & 0.0100 & 0.334234 \\
732 & 264.95 & 69.40 & 74.0\% & 164/164 & 297 & 0.0100 & 0.382648 \\
733 & 291.73 & 68.42 & 74.0\% & 408/408 & 803 & 0.0100 & 0.351936 \\
734 & 285.98 & 69.73 & 74.0\% & 165/165 & 275 & 0.0100 & 0.404427 \\
735 & 286.80 & 69.63 & 74.0\% & 187/187 & 324 & 0.0100 & 0.349573 \\
736 & 271.30 & 73.29 & 74.0\% & 258/258 & 450 & 0.0100 & 0.344386 \\
737 & 257.75 & 72.72 & 74.0\% & 288/288 & 394 & 0.0100 & 0.385817 \\
738 & 285.14 & 71.55 & 74.0\% & 188/188 & 362 & 0.0100 & 0.409160 \\
739 & 54.23 & 71.87 & 73.0\% & 177/177 & 223 & 0.0100 & 0.354416 \\
740 & 259.58 & 74.50 & 73.0\% & 250/250 & 452 & 0.0100 & 0.389307 \\
741 & 56.92 & 73.99 & 72.0\% & 150/150 & 182 & 0.0100 & 0.426513 \\
742 & 304.18 & 74.29 & 72.0\% & 224/224 & 371 & 0.0100 & 0.366603 \\
743 & 272.01 & 76.48 & 72.0\% & 228/228 & 351 & 0.0100 & 0.353508 \\
744 & 284.34 & 75.41 & 73.0\% & 146/146 & 275 & 0.0100 & 0.379896 \\
745 & 36.80 & 76.76 & 73.0\% & 97/97 & 147 & 0.0100 & 0.458610 \\
746 & 268.41 & 74.57 & 73.0\% & 202/202 & 292 & 0.0100 & 0.419558 \\
747 & 267.58 & 74.33 & 73.0\% & 217/217 & 322 & 0.0100 & 0.319450 \\
748 & 258.64 & 76.34 & 73.0\% & 559/559 & 739 & 0.0100 & 0.431336 \\
749 & 120.02 & 78.63 & 72.0\% & 961/961 & 1000 & 0.0100 & 0.395125 \\
750 & 289.62 & 79.62 & 71.0\% & 305/305 & 487 & 0.0100 & 0.361823 \\
751 & 191.84 & 77.64 & 70.0\% & 543/543 & 1000 & 0.0100 & 0.377396 \\
752 & 257.07 & 77.53 & 71.0\% & 192/192 & 368 & 0.0100 & 0.341847 \\
753 & 304.27 & 78.50 & 72.0\% & 271/271 & 481 & 0.0100 & 0.352286 \\
754 & 134.27 & 82.00 & 72.0\% & 755/755 & 1000 & 0.0100 & 0.410839 \\
755 & 295.81 & 83.33 & 72.0\% & 197/197 & 332 & 0.0100 & 0.375907 \\
756 & 299.59 & 81.74 & 73.0\% & 164/164 & 274 & 0.0100 & 0.383154 \\
757 & 285.82 & 81.83 & 73.0\% & 273/273 & 333 & 0.0100 & 0.376212 \\
758 & 267.59 & 80.77 & 74.0\% & 383/383 & 555 & 0.0100 & 0.389588 \\
759 & 276.13 & 82.80 & 74.0\% & 266/266 & 367 & 0.0100 & 0.402467 \\
760 & 290.45 & 79.48 & 74.0\% & 117/117 & 199 & 0.0100 & 0.432836 \\
761 & 279.28 & 80.30 & 74.0\% & 333/333 & 364 & 0.0100 & 0.388077 \\
762 & 289.26 & 80.76 & 74.0\% & 439/439 & 493 & 0.0100 & 0.410146 \\
763 & 249.37 & 79.86 & 75.0\% & 265/265 & 476 & 0.0100 & 0.426374 \\
764 & 260.42 & 81.52 & 76.0\% & 226/226 & 400 & 0.0100 & 0.440993 \\
765 & 257.91 & 80.75 & 76.0\% & 205/205 & 293 & 0.0100 & 0.419717 \\
766 & 257.77 & 81.96 & 77.0\% & 252/252 & 413 & 0.0100 & 0.410287 \\
767 & 291.44 & 83.20 & 77.0\% & 255/255 & 422 & 0.0100 & 0.424103 \\
768 & 181.34 & 83.07 & 76.0\% & 498/498 & 1000 & 0.0100 & 0.430741 \\
769 & 29.81 & 84.84 & 77.0\% & 84/84 & 117 & 0.0100 & 0.499755 \\
770 & 269.73 & 84.44 & 77.0\% & 445/445 & 913 & 0.0100 & 0.456341 \\
771 & 31.26 & 84.67 & 76.0\% & 90/90 & 103 & 0.0100 & 0.345780 \\
772 & 245.98 & 85.58 & 76.0\% & 203/203 & 395 & 0.0100 & 0.433171 \\
773 & 265.51 & 84.47 & 76.0\% & 298/298 & 515 & 0.0100 & 0.436458 \\
774 & 241.79 & 85.18 & 76.0\% & 220/220 & 226 & 0.0100 & 0.432231 \\
775 & 289.57 & 84.37 & 77.0\% & 184/184 & 266 & 0.0100 & 0.479042 \\
776 & 64.46 & 84.32 & 76.0\% & 99/99 & 125 & 0.0100 & 0.423974 \\
777 & -100.08 & 84.73 & 75.0\% & 338/338 & 425 & 0.0100 & 0.403988 \\
778 & 281.38 & 83.15 & 75.0\% & 178/178 & 246 & 0.0100 & 0.412259 \\
779 & 252.33 & 84.16 & 75.0\% & 172/172 & 260 & 0.0100 & 0.451276 \\
780 & 270.02 & 83.63 & 75.0\% & 166/166 & 285 & 0.0100 & 0.462366 \\
781 & 237.57 & 84.43 & 76.0\% & 341/341 & 772 & 0.0100 & 0.449872 \\
782 & 25.51 & 83.21 & 75.0\% & 93/93 & 125 & 0.0100 & 0.406957 \\
783 & 289.54 & 83.54 & 75.0\% & 151/151 & 217 & 0.0100 & 0.432075 \\
784 & 275.98 & 84.43 & 75.0\% & 207/207 & 365 & 0.0100 & 0.423078 \\
785 & 79.10 & 82.19 & 75.0\% & 748/748 & 1000 & 0.0100 & 0.439670 \\
786 & 275.00 & 83.57 & 75.0\% & 226/226 & 499 & 0.0100 & 0.443625 \\
787 & 292.45 & 84.77 & 76.0\% & 368/368 & 813 & 0.0100 & 0.478075 \\
788 & -130.17 & 82.45 & 75.0\% & 720/720 & 1000 & 0.0100 & 0.428137 \\
789 & 268.88 & 82.74 & 75.0\% & 161/161 & 229 & 0.0100 & 0.372592 \\
790 & 290.75 & 82.81 & 75.0\% & 171/171 & 286 & 0.0100 & 0.416400 \\
791 & 266.63 & 81.88 & 76.0\% & 376/376 & 705 & 0.0100 & 0.423324 \\
792 & 302.80 & 83.01 & 76.0\% & 241/241 & 350 & 0.0100 & 0.435485 \\
793 & 302.94 & 81.78 & 75.0\% & 142/142 & 203 & 0.0100 & 0.494605 \\
794 & 236.87 & 80.57 & 74.0\% & 219/219 & 255 & 0.0100 & 0.488975 \\
795 & 239.12 & 82.76 & 74.0\% & 655/655 & 843 & 0.0100 & 0.405301 \\
796 & 48.49 & 81.60 & 73.0\% & 108/108 & 132 & 0.0100 & 0.571625 \\
797 & 254.51 & 80.64 & 73.0\% & 227/227 & 264 & 0.0100 & 0.463720 \\
798 & -56.47 & 80.28 & 72.0\% & 673/673 & 1000 & 0.0100 & 0.462566 \\
799 & 51.85 & 81.21 & 71.0\% & 142/142 & 167 & 0.0100 & 0.469429 \\
800 & 23.81 & 80.51 & 71.0\% & 92/92 & 100 & 0.0100 & 0.487393 \\
\end{longtable}
\normalsize

\clearpage

\scriptsize
\setlength{\tabcolsep}{1.8pt}
\renewcommand{\arraystretch}{0.86}
\begin{longtable}{rrrrrrrr}
\caption{Complete per-iteration training output - Dqn Modified (800 episodes).}\\
\toprule
Episode & Reward & Avg. Q & Safe100 & Attempted/Executed & Steps & Epsilon & Loss \\
\midrule
\endfirsthead
\multicolumn{8}{c}{\small Continued: Dqn Modified complete per-iteration output} \\
\toprule
Episode & Reward & Avg. Q & Safe100 & Attempted/Executed & Steps & Epsilon & Loss \\
\midrule
\endhead
\bottomrule
\endfoot
\bottomrule
\endlastfoot
1 & -147.06 & 0.24 & 0.0\% & 50/40 & 70 & 0.9993 & nan \\
2 & -359.38 & 0.24 & 0.0\% & 67/61 & 93 & 0.9984 & nan \\
3 & -158.89 & 0.24 & 0.0\% & 102/86 & 122 & 0.9972 & nan \\
4 & -141.98 & 0.24 & 0.0\% & 67/58 & 85 & 0.9963 & nan \\
5 & -173.75 & 0.24 & 0.0\% & 84/72 & 117 & 0.9952 & nan \\
6 & -329.77 & 0.24 & 0.0\% & 78/67 & 104 & 0.9941 & nan \\
7 & -327.33 & 0.24 & 0.0\% & 57/47 & 79 & 0.9934 & nan \\
8 & -179.77 & 0.24 & 0.0\% & 58/47 & 70 & 0.9927 & nan \\
9 & -351.41 & 0.24 & 0.0\% & 79/63 & 108 & 0.9916 & nan \\
10 & -298.78 & 0.24 & 0.0\% & 90/80 & 118 & 0.9904 & nan \\
11 & -78.47 & -0.36 & 0.0\% & 51/43 & 68 & 0.9898 & 2.378366 \\
12 & -159.12 & -0.58 & 0.0\% & 58/45 & 78 & 0.9890 & 2.183941 \\
13 & -116.79 & -0.43 & 0.0\% & 74/67 & 96 & 0.9880 & 2.109947 \\
14 & -90.14 & -0.15 & 0.0\% & 63/49 & 84 & 0.9872 & 1.890186 \\
15 & -168.22 & 0.23 & 0.0\% & 80/66 & 115 & 0.9861 & 1.631696 \\
16 & -243.62 & 0.20 & 0.0\% & 54/41 & 74 & 0.9853 & 1.630903 \\
17 & -280.66 & 0.14 & 0.0\% & 77/61 & 99 & 0.9844 & 1.782376 \\
18 & -175.90 & 0.30 & 0.0\% & 76/66 & 92 & 0.9834 & 1.357518 \\
19 & -160.71 & 0.18 & 0.0\% & 69/62 & 93 & 0.9825 & 1.550764 \\
20 & -33.68 & 0.12 & 5.0\% & 51/44 & 68 & 0.9819 & 1.540087 \\
21 & -159.27 & 0.14 & 4.8\% & 48/33 & 64 & 0.9812 & 1.444286 \\
22 & -153.45 & 0.46 & 4.5\% & 68/55 & 87 & 0.9804 & 1.372086 \\
23 & -175.76 & 0.17 & 4.3\% & 54/49 & 68 & 0.9797 & 1.640241 \\
24 & -132.43 & 0.25 & 4.2\% & 54/46 & 75 & 0.9789 & 1.242995 \\
25 & -120.53 & 0.37 & 4.0\% & 63/58 & 78 & 0.9782 & 1.178728 \\
26 & -93.94 & 0.27 & 3.8\% & 49/38 & 65 & 0.9775 & 1.211328 \\
27 & -295.67 & 0.43 & 3.7\% & 62/50 & 76 & 0.9768 & 1.148219 \\
28 & -355.29 & 0.39 & 3.6\% & 81/71 & 105 & 0.9757 & 1.296328 \\
29 & -108.34 & 0.75 & 3.4\% & 48/41 & 63 & 0.9751 & 2.050307 \\
30 & -192.05 & 0.45 & 3.3\% & 49/43 & 74 & 0.9744 & 1.610327 \\
31 & -336.50 & 1.01 & 3.2\% & 104/96 & 137 & 0.9730 & 1.229883 \\
32 & -135.93 & 0.77 & 3.1\% & 69/64 & 89 & 0.9721 & 1.376396 \\
33 & -98.78 & 0.63 & 3.0\% & 96/85 & 123 & 0.9709 & 1.291586 \\
34 & -253.02 & 0.40 & 2.9\% & 79/68 & 103 & 0.9699 & 1.638038 \\
35 & -152.95 & 0.51 & 2.9\% & 81/69 & 113 & 0.9688 & 1.408279 \\
36 & -207.13 & 0.66 & 2.8\% & 88/83 & 120 & 0.9676 & 1.354004 \\
37 & -597.02 & 0.50 & 2.7\% & 77/60 & 98 & 0.9666 & 1.301109 \\
38 & -185.11 & 0.50 & 2.6\% & 46/33 & 58 & 0.9661 & 1.548023 \\
39 & -124.51 & 1.23 & 2.6\% & 59/54 & 76 & 0.9653 & 1.571273 \\
40 & -182.35 & 0.96 & 2.5\% & 94/75 & 113 & 0.9642 & 1.588046 \\
41 & -101.52 & 0.71 & 2.4\% & 86/75 & 114 & 0.9631 & 1.373784 \\
42 & -104.34 & 0.80 & 2.4\% & 57/50 & 75 & 0.9623 & 1.223945 \\
43 & -194.28 & 0.89 & 2.3\% & 81/69 & 109 & 0.9612 & 1.665583 \\
44 & -134.75 & 0.81 & 2.3\% & 75/68 & 104 & 0.9602 & 1.311982 \\
45 & -129.05 & 0.87 & 2.2\% & 46/40 & 62 & 0.9596 & 1.577818 \\
46 & -138.97 & 0.98 & 2.2\% & 68/57 & 83 & 0.9588 & 1.610268 \\
47 & -170.58 & 0.90 & 2.1\% & 76/69 & 96 & 0.9578 & 1.378401 \\
48 & -152.79 & 1.21 & 2.1\% & 70/54 & 92 & 0.9569 & 1.474257 \\
49 & -193.73 & 1.21 & 2.0\% & 94/78 & 112 & 0.9558 & 1.290625 \\
50 & -96.76 & 1.11 & 2.0\% & 61/54 & 74 & 0.9551 & 1.377693 \\
51 & -171.03 & 1.43 & 2.0\% & 78/69 & 102 & 0.9541 & 1.497369 \\
52 & -223.71 & 0.98 & 1.9\% & 84/72 & 113 & 0.9529 & 1.477513 \\
53 & -360.34 & 1.03 & 1.9\% & 79/66 & 96 & 0.9520 & 1.345634 \\
54 & -138.15 & 1.33 & 1.9\% & 41/33 & 58 & 0.9514 & 1.390091 \\
55 & -33.19 & 1.57 & 3.6\% & 62/55 & 83 & 0.9506 & 1.574210 \\
56 & -163.62 & 1.04 & 3.6\% & 89/86 & 118 & 0.9494 & 1.532085 \\
57 & -111.22 & 1.00 & 3.5\% & 47/39 & 65 & 0.9488 & 1.452840 \\
58 & -51.10 & 1.18 & 5.2\% & 83/70 & 111 & 0.9477 & 1.566933 \\
59 & -336.09 & 1.37 & 5.1\% & 60/50 & 79 & 0.9469 & 1.480893 \\
60 & -371.88 & 1.28 & 5.0\% & 71/61 & 99 & 0.9459 & 1.343086 \\
61 & -116.42 & 1.75 & 4.9\% & 56/44 & 71 & 0.9452 & 1.757143 \\
62 & -95.35 & 1.60 & 4.8\% & 44/36 & 65 & 0.9446 & 1.680483 \\
63 & -277.99 & 1.86 & 4.8\% & 77/67 & 108 & 0.9435 & 1.714475 \\
64 & -290.81 & 1.63 & 4.7\% & 77/71 & 97 & 0.9425 & 1.589762 \\
65 & -203.93 & 1.68 & 4.6\% & 57/49 & 82 & 0.9417 & 1.496282 \\
66 & -153.73 & 1.66 & 4.5\% & 54/43 & 73 & 0.9410 & 1.655809 \\
67 & -263.87 & 2.30 & 4.5\% & 59/50 & 86 & 0.9402 & 1.773135 \\
68 & -277.42 & 1.89 & 4.4\% & 78/70 & 97 & 0.9392 & 1.536151 \\
69 & -159.74 & 2.00 & 4.3\% & 84/73 & 108 & 0.9381 & 1.571837 \\
70 & -302.09 & 1.96 & 4.3\% & 70/63 & 99 & 0.9371 & 1.592491 \\
71 & -21.41 & 2.19 & 4.2\% & 73/66 & 103 & 0.9361 & 1.558594 \\
72 & -46.53 & 2.41 & 5.6\% & 75/59 & 102 & 0.9351 & 1.847060 \\
73 & -213.67 & 2.68 & 5.5\% & 69/60 & 82 & 0.9343 & 1.756783 \\
74 & -102.19 & 2.20 & 5.4\% & 42/34 & 64 & 0.9337 & 1.832223 \\
75 & -110.53 & 2.36 & 5.3\% & 51/44 & 65 & 0.9330 & 1.860790 \\
76 & -162.99 & 2.58 & 5.3\% & 61/49 & 87 & 0.9322 & 1.976125 \\
77 & -188.42 & 2.62 & 5.2\% & 71/57 & 87 & 0.9313 & 1.537485 \\
78 & -128.26 & 3.20 & 5.1\% & 62/56 & 83 & 0.9305 & 1.933873 \\
79 & -96.99 & 3.08 & 5.1\% & 46/39 & 65 & 0.9298 & 2.002488 \\
80 & -119.17 & 3.04 & 5.0\% & 49/41 & 72 & 0.9291 & 1.687444 \\
81 & -85.53 & 2.88 & 4.9\% & 51/40 & 62 & 0.9285 & 1.919448 \\
82 & -112.73 & 3.18 & 4.9\% & 97/76 & 125 & 0.9273 & 1.910466 \\
83 & -163.70 & 3.13 & 4.8\% & 55/48 & 71 & 0.9266 & 1.714962 \\
84 & 4.12 & 2.95 & 4.8\% & 61/52 & 78 & 0.9258 & 1.964801 \\
85 & -93.72 & 3.61 & 4.7\% & 63/51 & 86 & 0.9249 & 2.072651 \\
86 & -139.00 & 3.44 & 4.7\% & 57/48 & 77 & 0.9242 & 1.913164 \\
87 & -167.14 & 3.58 & 4.6\% & 81/64 & 99 & 0.9232 & 2.073997 \\
88 & -115.04 & 3.48 & 4.5\% & 59/53 & 65 & 0.9226 & 1.993013 \\
89 & -113.09 & 4.10 & 4.5\% & 48/42 & 62 & 0.9219 & 2.352776 \\
90 & -312.91 & 3.69 & 4.4\% & 60/50 & 75 & 0.9212 & 1.715807 \\
91 & -293.90 & 3.79 & 4.4\% & 75/65 & 95 & 0.9203 & 1.961326 \\
92 & -133.27 & 4.19 & 4.3\% & 64/57 & 79 & 0.9195 & 1.755656 \\
93 & -113.86 & 4.42 & 4.3\% & 69/61 & 89 & 0.9186 & 2.478885 \\
94 & -68.20 & 3.87 & 4.3\% & 91/82 & 112 & 0.9175 & 2.078659 \\
95 & -131.16 & 4.00 & 4.2\% & 83/75 & 109 & 0.9164 & 1.932215 \\
96 & -129.84 & 5.22 & 4.2\% & 88/80 & 121 & 0.9152 & 2.351376 \\
97 & -98.31 & 4.58 & 4.1\% & 68/59 & 98 & 0.9142 & 1.944437 \\
98 & -111.69 & 5.19 & 4.1\% & 60/51 & 69 & 0.9136 & 2.082340 \\
99 & -86.88 & 4.90 & 4.0\% & 59/52 & 72 & 0.9129 & 1.942084 \\
100 & -163.44 & 4.96 & 4.0\% & 86/75 & 111 & 0.9118 & 1.828488 \\
101 & -51.32 & 4.79 & 4.0\% & 47/41 & 63 & 0.9111 & 2.425658 \\
102 & -114.04 & 5.14 & 4.0\% & 70/62 & 91 & 0.9102 & 1.969920 \\
103 & -222.30 & 5.19 & 4.0\% & 99/83 & 124 & 0.9090 & 1.948536 \\
104 & -93.57 & 5.28 & 4.0\% & 47/44 & 61 & 0.9084 & 1.799621 \\
105 & -163.92 & 5.38 & 4.0\% & 101/87 & 126 & 0.9071 & 2.047658 \\
106 & -197.71 & 5.11 & 4.0\% & 79/68 & 106 & 0.9061 & 2.087507 \\
107 & -92.71 & 5.60 & 4.0\% & 53/46 & 69 & 0.9054 & 2.108172 \\
108 & -90.17 & 5.70 & 4.0\% & 73/64 & 87 & 0.9046 & 2.133686 \\
109 & -80.04 & 5.57 & 4.0\% & 89/70 & 116 & 0.9034 & 1.907772 \\
110 & -119.08 & 5.50 & 4.0\% & 77/63 & 93 & 0.9025 & 1.746923 \\
111 & -90.69 & 5.72 & 4.0\% & 53/48 & 64 & 0.9019 & 2.234248 \\
112 & -122.88 & 5.90 & 4.0\% & 106/85 & 132 & 0.9005 & 1.972350 \\
113 & -117.97 & 6.17 & 4.0\% & 52/46 & 69 & 0.8999 & 1.964930 \\
114 & -189.60 & 6.20 & 4.0\% & 110/95 & 135 & 0.8985 & 1.901980 \\
115 & -103.38 & 5.88 & 4.0\% & 85/71 & 100 & 0.8975 & 1.963505 \\
116 & -161.77 & 5.95 & 4.0\% & 96/84 & 117 & 0.8964 & 2.004211 \\
117 & -105.84 & 6.40 & 4.0\% & 68/60 & 85 & 0.8955 & 1.971135 \\
118 & -22.83 & 6.27 & 5.0\% & 57/48 & 73 & 0.8948 & 2.431992 \\
119 & -208.90 & 6.39 & 5.0\% & 64/58 & 81 & 0.8940 & 1.884941 \\
120 & -123.52 & 6.32 & 4.0\% & 70/59 & 94 & 0.8931 & 2.069558 \\
121 & -137.95 & 6.20 & 4.0\% & 49/42 & 60 & 0.8925 & 1.947342 \\
122 & -106.61 & 6.70 & 4.0\% & 91/73 & 109 & 0.8914 & 1.761995 \\
123 & -109.22 & 6.83 & 4.0\% & 54/49 & 66 & 0.8908 & 2.351755 \\
124 & -119.28 & 6.91 & 4.0\% & 60/49 & 71 & 0.8901 & 1.659916 \\
125 & -115.19 & 6.68 & 4.0\% & 72/59 & 90 & 0.8892 & 1.718640 \\
126 & -121.10 & 7.17 & 4.0\% & 85/72 & 112 & 0.8881 & 1.674529 \\
127 & -153.13 & 7.04 & 4.0\% & 75/60 & 100 & 0.8871 & 1.682289 \\
128 & -132.46 & 6.76 & 4.0\% & 87/72 & 108 & 0.8860 & 2.223072 \\
129 & -153.25 & 6.66 & 4.0\% & 70/60 & 95 & 0.8851 & 1.637152 \\
130 & -147.94 & 7.23 & 4.0\% & 71/58 & 85 & 0.8842 & 1.821706 \\
131 & -132.57 & 6.81 & 4.0\% & 79/74 & 97 & 0.8832 & 1.514281 \\
132 & -183.28 & 7.34 & 4.0\% & 65/58 & 86 & 0.8824 & 2.043517 \\
133 & -105.22 & 6.99 & 4.0\% & 55/49 & 75 & 0.8817 & 1.850414 \\
134 & -126.42 & 7.88 & 4.0\% & 76/62 & 92 & 0.8807 & 1.713837 \\
135 & -153.82 & 7.46 & 4.0\% & 68/59 & 92 & 0.8798 & 1.727949 \\
136 & -99.62 & 8.00 & 4.0\% & 55/43 & 69 & 0.8792 & 1.523598 \\
137 & -147.77 & 8.05 & 4.0\% & 124/105 & 147 & 0.8777 & 1.751040 \\
138 & -153.68 & 7.59 & 4.0\% & 75/66 & 101 & 0.8767 & 1.526639 \\
139 & -101.65 & 7.95 & 4.0\% & 49/43 & 65 & 0.8761 & 1.884556 \\
140 & -79.92 & 7.96 & 4.0\% & 84/69 & 114 & 0.8749 & 1.713260 \\
141 & -102.57 & 8.03 & 4.0\% & 63/51 & 74 & 0.8742 & 1.446591 \\
142 & -115.54 & 7.85 & 4.0\% & 110/91 & 138 & 0.8728 & 1.840938 \\
143 & -243.08 & 8.18 & 4.0\% & 59/50 & 81 & 0.8720 & 1.465967 \\
144 & -113.72 & 8.97 & 4.0\% & 63/55 & 82 & 0.8712 & 1.808377 \\
145 & -184.22 & 8.85 & 4.0\% & 90/75 & 120 & 0.8700 & 1.625598 \\
146 & -138.19 & 8.98 & 4.0\% & 66/54 & 84 & 0.8692 & 2.102639 \\
147 & -125.32 & 8.84 & 4.0\% & 60/49 & 76 & 0.8684 & 1.800810 \\
148 & -102.88 & 8.81 & 4.0\% & 84/74 & 120 & 0.8673 & 1.588386 \\
149 & -94.46 & 8.59 & 4.0\% & 49/41 & 64 & 0.8666 & 1.611297 \\
150 & -63.85 & 9.46 & 4.0\% & 60/54 & 78 & 0.8658 & 1.657727 \\
151 & -90.26 & 8.62 & 4.0\% & 63/49 & 79 & 0.8651 & 1.500743 \\
152 & -122.11 & 8.75 & 4.0\% & 60/47 & 80 & 0.8643 & 1.949000 \\
153 & -190.20 & 9.03 & 4.0\% & 90/72 & 121 & 0.8631 & 2.283022 \\
154 & -166.08 & 8.96 & 4.0\% & 73/60 & 100 & 0.8621 & 1.467445 \\
155 & -75.74 & 9.02 & 3.0\% & 56/49 & 70 & 0.8614 & 1.868852 \\
156 & -119.97 & 9.07 & 3.0\% & 67/60 & 86 & 0.8605 & 2.017039 \\
157 & -94.55 & 9.03 & 3.0\% & 55/45 & 67 & 0.8599 & 1.559056 \\
158 & -76.66 & 9.05 & 2.0\% & 39/34 & 62 & 0.8593 & 2.128641 \\
159 & -108.50 & 9.12 & 2.0\% & 57/44 & 72 & 0.8585 & 1.759357 \\
160 & -95.90 & 9.12 & 2.0\% & 46/41 & 61 & 0.8579 & 1.307643 \\
161 & -106.12 & 9.20 & 2.0\% & 56/52 & 75 & 0.8572 & 1.611251 \\
162 & -170.42 & 8.76 & 2.0\% & 71/65 & 98 & 0.8562 & 1.751270 \\
163 & -34.03 & 9.01 & 3.0\% & 79/73 & 93 & 0.8553 & 1.826123 \\
164 & -84.22 & 8.87 & 3.0\% & 54/44 & 68 & 0.8546 & 1.247353 \\
165 & -123.47 & 9.72 & 3.0\% & 61/56 & 80 & 0.8538 & 2.128737 \\
166 & -58.24 & 8.79 & 3.0\% & 98/83 & 128 & 0.8526 & 1.789866 \\
167 & -121.79 & 9.00 & 3.0\% & 70/60 & 90 & 0.8517 & 1.494567 \\
168 & -122.01 & 8.70 & 3.0\% & 57/48 & 73 & 0.8510 & 1.991922 \\
169 & -104.64 & 9.37 & 3.0\% & 100/83 & 117 & 0.8498 & 1.910807 \\
170 & -125.75 & 9.15 & 3.0\% & 64/50 & 90 & 0.8489 & 1.907880 \\
171 & -145.90 & 9.29 & 3.0\% & 74/64 & 99 & 0.8479 & 1.961952 \\
172 & -85.86 & 8.82 & 2.0\% & 49/43 & 65 & 0.8473 & 1.677778 \\
173 & -31.91 & 9.37 & 3.0\% & 48/41 & 68 & 0.8466 & 1.619263 \\
174 & -111.72 & 9.36 & 3.0\% & 64/60 & 84 & 0.8458 & 1.880220 \\
175 & -148.14 & 8.94 & 3.0\% & 83/75 & 106 & 0.8447 & 1.697668 \\
176 & -137.17 & 9.46 & 3.0\% & 65/59 & 83 & 0.8439 & 1.531075 \\
177 & -116.47 & 9.44 & 3.0\% & 64/58 & 89 & 0.8430 & 1.486618 \\
178 & -110.02 & 9.32 & 3.0\% & 55/45 & 69 & 0.8424 & 1.354746 \\
179 & -122.10 & 9.57 & 3.0\% & 69/51 & 93 & 0.8414 & 1.695863 \\
180 & -11.22 & 9.71 & 3.0\% & 80/65 & 106 & 0.8404 & 1.592606 \\
181 & -154.61 & 8.97 & 3.0\% & 61/54 & 82 & 0.8396 & 2.032694 \\
182 & -156.98 & 8.89 & 3.0\% & 86/73 & 107 & 0.8385 & 1.768439 \\
183 & -152.01 & 9.32 & 3.0\% & 64/56 & 86 & 0.8377 & 1.517448 \\
184 & -93.24 & 9.14 & 3.0\% & 56/46 & 77 & 0.8369 & 1.173172 \\
185 & -159.42 & 9.31 & 3.0\% & 64/55 & 77 & 0.8361 & 1.889340 \\
186 & -120.71 & 9.66 & 3.0\% & 80/72 & 111 & 0.8350 & 1.745212 \\
187 & -146.76 & 9.25 & 3.0\% & 83/71 & 99 & 0.8341 & 1.534703 \\
188 & -101.46 & 9.27 & 3.0\% & 79/63 & 102 & 0.8330 & 1.859385 \\
189 & -159.11 & 9.10 & 3.0\% & 68/67 & 96 & 0.8321 & 1.473881 \\
190 & -111.21 & 9.07 & 3.0\% & 53/46 & 76 & 0.8313 & 1.586508 \\
191 & -111.18 & 9.22 & 3.0\% & 61/57 & 80 & 0.8306 & 1.963924 \\
192 & -119.71 & 9.16 & 3.0\% & 94/82 & 116 & 0.8294 & 1.499935 \\
193 & -90.34 & 9.18 & 3.0\% & 58/48 & 75 & 0.8287 & 1.548728 \\
194 & -103.55 & 9.10 & 3.0\% & 83/71 & 108 & 0.8276 & 1.358421 \\
195 & -128.32 & 9.37 & 3.0\% & 67/56 & 92 & 0.8267 & 1.922253 \\
196 & -73.14 & 8.88 & 3.0\% & 57/49 & 72 & 0.8260 & 1.826622 \\
197 & -149.45 & 9.01 & 3.0\% & 69/62 & 88 & 0.8251 & 2.183778 \\
198 & -58.66 & 9.24 & 4.0\% & 63/53 & 81 & 0.8243 & 1.589156 \\
199 & -80.12 & 8.50 & 4.0\% & 46/36 & 67 & 0.8236 & 1.555174 \\
200 & -165.50 & 8.55 & 4.0\% & 81/65 & 106 & 0.8226 & 1.583445 \\
201 & -97.28 & 9.16 & 4.0\% & 87/79 & 115 & 0.8214 & 1.769124 \\
202 & -114.52 & 8.81 & 4.0\% & 95/74 & 110 & 0.8204 & 1.617505 \\
203 & -176.70 & 9.11 & 4.0\% & 47/37 & 62 & 0.8197 & 2.073084 \\
204 & -137.61 & 9.21 & 4.0\% & 70/64 & 85 & 0.8189 & 1.502258 \\
205 & -163.05 & 9.11 & 4.0\% & 93/79 & 115 & 0.8178 & 1.594409 \\
206 & -88.05 & 8.55 & 4.0\% & 52/43 & 63 & 0.8171 & 1.775252 \\
207 & -115.82 & 8.58 & 4.0\% & 74/64 & 90 & 0.8162 & 2.004647 \\
208 & -127.80 & 8.13 & 4.0\% & 66/56 & 78 & 0.8155 & 1.382540 \\
209 & -67.95 & 8.37 & 4.0\% & 50/40 & 66 & 0.8148 & 1.730040 \\
210 & -165.89 & 8.65 & 4.0\% & 89/78 & 119 & 0.8136 & 1.627494 \\
211 & -186.22 & 8.72 & 4.0\% & 60/52 & 81 & 0.8128 & 1.465239 \\
212 & -72.34 & 8.27 & 4.0\% & 52/47 & 67 & 0.8122 & 1.839680 \\
213 & -130.03 & 8.18 & 4.0\% & 80/72 & 104 & 0.8111 & 1.542305 \\
214 & -94.07 & 8.43 & 4.0\% & 85/74 & 104 & 0.8101 & 1.500368 \\
215 & -102.18 & 8.48 & 4.0\% & 53/46 & 66 & 0.8095 & 1.417212 \\
216 & -111.15 & 8.56 & 4.0\% & 55/48 & 69 & 0.8088 & 2.249330 \\
217 & -139.74 & 8.20 & 4.0\% & 58/49 & 95 & 0.8078 & 1.811956 \\
218 & -159.60 & 8.07 & 3.0\% & 84/73 & 105 & 0.8068 & 2.146051 \\
219 & -148.50 & 7.74 & 3.0\% & 67/56 & 93 & 0.8059 & 1.868340 \\
220 & -114.07 & 7.77 & 3.0\% & 66/50 & 86 & 0.8050 & 2.381290 \\
221 & -143.26 & 8.17 & 3.0\% & 71/56 & 86 & 0.8042 & 2.022571 \\
222 & -92.65 & 7.42 & 3.0\% & 47/39 & 66 & 0.8035 & 2.189210 \\
223 & -88.56 & 8.05 & 3.0\% & 49/42 & 61 & 0.8029 & 1.868906 \\
224 & -188.20 & 8.19 & 3.0\% & 66/53 & 85 & 0.8021 & 1.522647 \\
225 & -38.81 & 6.98 & 4.0\% & 67/60 & 86 & 0.8012 & 2.334329 \\
226 & -68.71 & 6.57 & 4.0\% & 52/43 & 66 & 0.8006 & 1.842790 \\
227 & -116.89 & 6.79 & 4.0\% & 53/44 & 67 & 0.7999 & 2.205243 \\
228 & -14.31 & 7.11 & 5.0\% & 82/70 & 115 & 0.7988 & 1.685510 \\
229 & -162.94 & 6.83 & 5.0\% & 66/57 & 96 & 0.7978 & 1.620507 \\
230 & -124.62 & 6.08 & 5.0\% & 67/56 & 101 & 0.7968 & 1.914845 \\
231 & -73.59 & 6.08 & 5.0\% & 56/48 & 69 & 0.7961 & 1.837267 \\
232 & -71.27 & 6.02 & 5.0\% & 61/48 & 77 & 0.7954 & 1.587263 \\
233 & -123.51 & 5.87 & 5.0\% & 50/44 & 77 & 0.7946 & 2.400733 \\
234 & -145.55 & 5.46 & 5.0\% & 57/49 & 67 & 0.7940 & 1.631781 \\
235 & -40.10 & 6.41 & 5.0\% & 62/53 & 85 & 0.7931 & 1.755935 \\
236 & -108.79 & 5.53 & 5.0\% & 72/64 & 92 & 0.7922 & 1.789167 \\
237 & -22.37 & 4.69 & 5.0\% & 54/47 & 72 & 0.7915 & 1.762317 \\
238 & -141.71 & 4.75 & 5.0\% & 72/64 & 97 & 0.7905 & 1.774301 \\
239 & -177.79 & 4.60 & 5.0\% & 99/84 & 131 & 0.7892 & 2.245826 \\
240 & -116.77 & 5.07 & 5.0\% & 86/73 & 112 & 0.7881 & 1.759543 \\
241 & -139.09 & 5.23 & 5.0\% & 58/50 & 75 & 0.7874 & 1.974718 \\
242 & -156.19 & 4.79 & 5.0\% & 60/52 & 83 & 0.7866 & 2.203706 \\
243 & -111.15 & 4.53 & 5.0\% & 74/63 & 97 & 0.7856 & 1.737398 \\
244 & -129.56 & 4.77 & 5.0\% & 80/66 & 103 & 0.7846 & 1.667024 \\
245 & -88.05 & 4.52 & 5.0\% & 52/45 & 76 & 0.7838 & 1.921561 \\
246 & -131.30 & 4.72 & 5.0\% & 64/50 & 84 & 0.7830 & 1.928909 \\
247 & -65.73 & 4.94 & 5.0\% & 40/33 & 60 & 0.7824 & 1.929502 \\
248 & -44.37 & 4.72 & 6.0\% & 72/63 & 94 & 0.7815 & 1.634598 \\
249 & -91.99 & 4.57 & 6.0\% & 61/50 & 85 & 0.7806 & 2.350092 \\
250 & -110.12 & 4.46 & 6.0\% & 75/63 & 107 & 0.7796 & 1.916054 \\
251 & -100.00 & 4.09 & 6.0\% & 49/39 & 66 & 0.7789 & 1.688823 \\
252 & -109.86 & 4.29 & 6.0\% & 77/66 & 97 & 0.7780 & 1.922388 \\
253 & -74.52 & 4.30 & 6.0\% & 51/45 & 61 & 0.7773 & 1.733022 \\
254 & -129.60 & 3.76 & 6.0\% & 86/72 & 115 & 0.7762 & 1.824614 \\
255 & -109.38 & 3.82 & 6.0\% & 70/63 & 95 & 0.7753 & 2.196716 \\
256 & -117.08 & 3.98 & 6.0\% & 80/74 & 102 & 0.7743 & 1.960264 \\
257 & -107.96 & 3.90 & 6.0\% & 67/55 & 87 & 0.7734 & 1.602095 \\
258 & -106.20 & 3.48 & 6.0\% & 55/52 & 73 & 0.7727 & 1.613638 \\
259 & -102.48 & 2.81 & 6.0\% & 65/60 & 85 & 0.7718 & 2.117504 \\
260 & -126.12 & 2.54 & 6.0\% & 75/56 & 102 & 0.7708 & 1.860785 \\
261 & -18.85 & 3.12 & 7.0\% & 56/50 & 69 & 0.7701 & 2.449884 \\
262 & -17.80 & 2.76 & 7.0\% & 62/53 & 85 & 0.7693 & 2.192970 \\
263 & -89.56 & 2.93 & 6.0\% & 50/43 & 61 & 0.7687 & 1.810186 \\
264 & -133.07 & 2.98 & 6.0\% & 63/53 & 84 & 0.7679 & 1.967795 \\
265 & -120.67 & 1.70 & 6.0\% & 72/57 & 95 & 0.7669 & 1.894046 \\
266 & -119.88 & 1.95 & 6.0\% & 80/65 & 109 & 0.7658 & 2.158310 \\
267 & -125.49 & 1.53 & 6.0\% & 70/60 & 87 & 0.7650 & 1.904715 \\
268 & -55.43 & 1.60 & 6.0\% & 53/46 & 67 & 0.7643 & 1.534912 \\
269 & -177.00 & 1.68 & 6.0\% & 68/62 & 85 & 0.7635 & 1.870111 \\
270 & -159.18 & 1.75 & 6.0\% & 94/80 & 126 & 0.7622 & 1.998596 \\
271 & -80.82 & 1.60 & 6.0\% & 93/75 & 114 & 0.7611 & 2.030462 \\
272 & -24.10 & 1.25 & 7.0\% & 57/51 & 75 & 0.7604 & 2.230635 \\
273 & -60.46 & 1.70 & 6.0\% & 70/65 & 98 & 0.7594 & 1.773542 \\
274 & -131.39 & 2.50 & 6.0\% & 78/72 & 94 & 0.7585 & 1.848874 \\
275 & -111.52 & 1.45 & 6.0\% & 55/50 & 78 & 0.7577 & 2.040278 \\
276 & -84.92 & 0.72 & 6.0\% & 52/43 & 69 & 0.7570 & 2.631087 \\
277 & -105.47 & 1.22 & 6.0\% & 88/73 & 119 & 0.7558 & 2.017855 \\
278 & -137.65 & 1.04 & 6.0\% & 58/49 & 78 & 0.7551 & 2.205694 \\
279 & -119.34 & 0.92 & 6.0\% & 87/73 & 113 & 0.7539 & 1.793077 \\
280 & -66.02 & 1.05 & 6.0\% & 58/52 & 72 & 0.7532 & 1.842956 \\
281 & -105.22 & 0.20 & 6.0\% & 61/56 & 79 & 0.7524 & 2.058246 \\
282 & -139.41 & 0.10 & 6.0\% & 63/51 & 81 & 0.7516 & 2.045332 \\
283 & -87.72 & 0.02 & 6.0\% & 82/69 & 109 & 0.7506 & 1.928713 \\
284 & -101.40 & 0.67 & 6.0\% & 90/77 & 116 & 0.7494 & 2.215926 \\
285 & -138.52 & 0.38 & 6.0\% & 59/49 & 80 & 0.7486 & 1.896461 \\
286 & -95.65 & 0.90 & 6.0\% & 61/58 & 75 & 0.7479 & 1.993398 \\
287 & -112.71 & -0.18 & 6.0\% & 56/47 & 87 & 0.7470 & 1.888194 \\
288 & -103.00 & -0.94 & 6.0\% & 54/42 & 73 & 0.7463 & 1.573785 \\
289 & -120.32 & 0.15 & 6.0\% & 62/48 & 86 & 0.7454 & 1.891533 \\
290 & -126.25 & -0.41 & 6.0\% & 55/49 & 77 & 0.7447 & 2.479251 \\
291 & -13.08 & 0.93 & 7.0\% & 89/74 & 119 & 0.7435 & 2.656491 \\
292 & -113.16 & 0.39 & 7.0\% & 57/47 & 77 & 0.7427 & 1.971083 \\
293 & -93.72 & -0.33 & 7.0\% & 74/67 & 95 & 0.7418 & 1.938459 \\
294 & -119.89 & -1.27 & 7.0\% & 71/59 & 102 & 0.7408 & 2.330393 \\
295 & -114.21 & -0.93 & 7.0\% & 61/51 & 80 & 0.7400 & 1.868612 \\
296 & -108.76 & -0.58 & 7.0\% & 73/63 & 94 & 0.7391 & 1.942222 \\
297 & -51.20 & -0.62 & 8.0\% & 66/55 & 88 & 0.7382 & 2.041443 \\
298 & -124.58 & -1.72 & 7.0\% & 61/43 & 78 & 0.7374 & 2.053473 \\
299 & -11.04 & -1.63 & 8.0\% & 56/45 & 70 & 0.7367 & 2.421329 \\
300 & -35.79 & -0.83 & 9.0\% & 87/73 & 112 & 0.7356 & 1.743524 \\
301 & -110.55 & -1.19 & 9.0\% & 80/75 & 104 & 0.7346 & 1.950846 \\
302 & -66.47 & -1.51 & 9.0\% & 66/59 & 83 & 0.7338 & 1.593293 \\
303 & -124.75 & -0.83 & 9.0\% & 66/53 & 86 & 0.7329 & 1.952725 \\
304 & -42.50 & -1.80 & 9.0\% & 48/40 & 64 & 0.7323 & 2.324720 \\
305 & -73.25 & -1.98 & 9.0\% & 73/64 & 99 & 0.7313 & 1.984730 \\
306 & -248.07 & -1.98 & 9.0\% & 92/79 & 115 & 0.7302 & 2.832942 \\
307 & -96.53 & -1.99 & 9.0\% & 92/81 & 130 & 0.7289 & 2.044392 \\
308 & -36.70 & -1.52 & 10.0\% & 61/48 & 88 & 0.7280 & 2.262197 \\
309 & -88.76 & -1.73 & 10.0\% & 50/45 & 71 & 0.7273 & 2.093510 \\
310 & -50.70 & -1.33 & 10.0\% & 74/63 & 89 & 0.7264 & 2.096354 \\
311 & -123.85 & -2.55 & 10.0\% & 89/77 & 107 & 0.7254 & 1.736894 \\
312 & -38.11 & -1.84 & 10.0\% & 65/53 & 76 & 0.7246 & 1.941114 \\
313 & -116.30 & -1.71 & 10.0\% & 80/71 & 103 & 0.7236 & 2.918505 \\
314 & -90.97 & -2.74 & 10.0\% & 60/50 & 71 & 0.7229 & 2.083903 \\
315 & -121.05 & -2.32 & 10.0\% & 51/42 & 77 & 0.7221 & 2.342954 \\
316 & -124.94 & -2.80 & 10.0\% & 72/64 & 101 & 0.7211 & 2.059118 \\
317 & -55.99 & -1.98 & 10.0\% & 54/43 & 72 & 0.7204 & 2.246438 \\
318 & -125.34 & -2.49 & 10.0\% & 53/44 & 77 & 0.7197 & 1.751741 \\
319 & -98.16 & -2.20 & 10.0\% & 62/53 & 78 & 0.7189 & 2.057590 \\
320 & -132.00 & -2.81 & 10.0\% & 83/70 & 96 & 0.7179 & 2.252908 \\
321 & -92.88 & -3.04 & 10.0\% & 79/69 & 109 & 0.7169 & 2.130932 \\
322 & -132.75 & -2.05 & 10.0\% & 77/69 & 101 & 0.7159 & 2.241798 \\
323 & -121.12 & -1.73 & 10.0\% & 64/56 & 82 & 0.7150 & 1.683468 \\
324 & -7.26 & -1.95 & 11.0\% & 88/75 & 113 & 0.7139 & 2.232975 \\
325 & -82.36 & -2.49 & 10.0\% & 58/53 & 76 & 0.7132 & 2.157781 \\
326 & -89.20 & -3.31 & 10.0\% & 44/40 & 63 & 0.7125 & 2.154182 \\
327 & -0.19 & -2.76 & 11.0\% & 85/67 & 97 & 0.7116 & 2.209197 \\
328 & -106.25 & -2.98 & 10.0\% & 66/60 & 83 & 0.7108 & 2.423050 \\
329 & -76.71 & -2.51 & 10.0\% & 61/52 & 76 & 0.7100 & 1.915567 \\
330 & -37.94 & -2.25 & 11.0\% & 76/67 & 99 & 0.7090 & 2.347589 \\
331 & -128.54 & -2.70 & 11.0\% & 65/53 & 78 & 0.7083 & 1.784946 \\
332 & -126.33 & -3.57 & 11.0\% & 74/63 & 94 & 0.7073 & 1.743576 \\
333 & -104.77 & -3.69 & 11.0\% & 73/65 & 86 & 0.7065 & 2.464583 \\
334 & -115.35 & -3.45 & 11.0\% & 60/50 & 82 & 0.7057 & 2.541074 \\
335 & -109.32 & -2.95 & 11.0\% & 83/71 & 113 & 0.7045 & 2.697170 \\
336 & -118.78 & -3.39 & 11.0\% & 67/53 & 80 & 0.7038 & 2.433858 \\
337 & -104.74 & -2.91 & 11.0\% & 56/50 & 78 & 0.7030 & 2.353596 \\
338 & -119.12 & -3.97 & 11.0\% & 56/48 & 72 & 0.7023 & 1.911754 \\
339 & -125.11 & -3.49 & 11.0\% & 92/82 & 122 & 0.7011 & 2.761507 \\
340 & -101.37 & -3.31 & 11.0\% & 52/47 & 80 & 0.7003 & 2.060145 \\
341 & -93.26 & -3.33 & 11.0\% & 85/70 & 107 & 0.6992 & 2.272704 \\
342 & -10.60 & -3.71 & 12.0\% & 77/69 & 98 & 0.6982 & 2.307490 \\
343 & -102.03 & -4.33 & 12.0\% & 46/39 & 57 & 0.6977 & 2.024607 \\
344 & -156.76 & -3.33 & 12.0\% & 87/74 & 112 & 0.6966 & 2.209063 \\
345 & -105.80 & -4.12 & 12.0\% & 55/47 & 71 & 0.6959 & 2.324233 \\
346 & -93.49 & -3.49 & 12.0\% & 86/70 & 122 & 0.6947 & 2.153958 \\
347 & -93.72 & -4.24 & 12.0\% & 59/50 & 77 & 0.6939 & 2.052843 \\
348 & -91.48 & -4.41 & 11.0\% & 70/62 & 83 & 0.6931 & 1.815138 \\
349 & -94.69 & -4.50 & 11.0\% & 38/32 & 56 & 0.6925 & 3.004305 \\
350 & -90.47 & -5.12 & 11.0\% & 67/55 & 105 & 0.6915 & 2.038075 \\
351 & -74.74 & -4.88 & 11.0\% & 55/49 & 66 & 0.6908 & 2.560126 \\
352 & -70.27 & -4.52 & 11.0\% & 80/63 & 97 & 0.6899 & 2.713474 \\
353 & -67.41 & -4.81 & 11.0\% & 68/60 & 97 & 0.6889 & 2.121194 \\
354 & -101.33 & -4.29 & 11.0\% & 62/56 & 85 & 0.6881 & 3.344830 \\
355 & 2.88 & -4.53 & 12.0\% & 55/49 & 70 & 0.6874 & 2.297672 \\
356 & -95.35 & -4.49 & 12.0\% & 43/39 & 54 & 0.6868 & 1.819307 \\
357 & -19.34 & -4.36 & 13.0\% & 72/66 & 105 & 0.6858 & 2.287651 \\
358 & -75.30 & -4.99 & 13.0\% & 57/47 & 74 & 0.6851 & 2.568398 \\
359 & -96.85 & -5.56 & 13.0\% & 52/44 & 74 & 0.6843 & 1.953924 \\
360 & -115.84 & -4.59 & 13.0\% & 72/61 & 93 & 0.6834 & 2.327227 \\
361 & -44.33 & -5.42 & 12.0\% & 59/51 & 70 & 0.6827 & 2.600970 \\
362 & -138.62 & -4.68 & 12.0\% & 67/54 & 93 & 0.6818 & 2.288720 \\
363 & -110.92 & -5.29 & 12.0\% & 86/72 & 99 & 0.6808 & 2.401261 \\
364 & -61.54 & -6.01 & 12.0\% & 64/59 & 77 & 0.6801 & 2.691531 \\
365 & -26.08 & -5.26 & 13.0\% & 90/80 & 125 & 0.6788 & 2.298757 \\
366 & -117.51 & -5.65 & 13.0\% & 88/68 & 112 & 0.6777 & 2.203563 \\
367 & -157.56 & -5.68 & 13.0\% & 76/65 & 93 & 0.6768 & 2.349497 \\
368 & -168.98 & -5.77 & 13.0\% & 79/65 & 111 & 0.6757 & 2.122478 \\
369 & -61.73 & -5.14 & 13.0\% & 55/48 & 66 & 0.6750 & 2.137213 \\
370 & -101.14 & -5.57 & 13.0\% & 69/57 & 79 & 0.6743 & 2.549946 \\
371 & -113.69 & -6.23 & 13.0\% & 79/68 & 103 & 0.6732 & 2.121191 \\
372 & -17.16 & -5.85 & 13.0\% & 85/76 & 107 & 0.6722 & 2.640110 \\
373 & -87.04 & -6.14 & 13.0\% & 54/51 & 68 & 0.6715 & 2.376222 \\
374 & -44.31 & -6.16 & 14.0\% & 73/65 & 93 & 0.6706 & 2.543764 \\
375 & -138.66 & -6.18 & 14.0\% & 78/67 & 104 & 0.6695 & 2.588268 \\
376 & -142.42 & -6.36 & 14.0\% & 81/68 & 94 & 0.6686 & 2.293643 \\
377 & -107.88 & -6.63 & 14.0\% & 56/45 & 71 & 0.6679 & 1.819702 \\
378 & -102.65 & -6.62 & 14.0\% & 82/69 & 102 & 0.6669 & 2.528186 \\
379 & -135.78 & -6.30 & 14.0\% & 102/92 & 132 & 0.6656 & 2.523727 \\
380 & -104.15 & -6.40 & 14.0\% & 71/61 & 99 & 0.6646 & 2.502872 \\
381 & -74.15 & -6.65 & 14.0\% & 79/68 & 93 & 0.6637 & 2.476752 \\
382 & -101.10 & -7.01 & 14.0\% & 63/58 & 78 & 0.6629 & 2.562173 \\
383 & -59.41 & -6.53 & 14.0\% & 74/64 & 120 & 0.6617 & 2.565260 \\
384 & -88.37 & -7.14 & 14.0\% & 86/68 & 114 & 0.6606 & 2.333389 \\
385 & -55.81 & -6.47 & 14.0\% & 71/62 & 83 & 0.6598 & 2.421224 \\
386 & -119.45 & -6.56 & 14.0\% & 97/87 & 129 & 0.6585 & 2.451178 \\
387 & -104.06 & -7.48 & 14.0\% & 68/59 & 97 & 0.6575 & 2.463658 \\
388 & -88.78 & -7.04 & 14.0\% & 58/50 & 79 & 0.6568 & 2.903617 \\
389 & -225.69 & -7.30 & 14.0\% & 105/94 & 125 & 0.6555 & 2.468850 \\
390 & -108.51 & -7.53 & 14.0\% & 63/50 & 95 & 0.6546 & 2.294009 \\
391 & -78.40 & -7.23 & 13.0\% & 95/81 & 108 & 0.6535 & 2.665078 \\
392 & -52.77 & -7.14 & 13.0\% & 59/52 & 73 & 0.6528 & 2.606875 \\
393 & -91.88 & -6.47 & 13.0\% & 112/97 & 148 & 0.6513 & 2.555566 \\
394 & -169.27 & -7.66 & 13.0\% & 71/59 & 92 & 0.6504 & 2.234292 \\
395 & -76.62 & -6.95 & 13.0\% & 65/55 & 89 & 0.6495 & 2.983029 \\
396 & -45.65 & -7.54 & 13.0\% & 60/51 & 73 & 0.6488 & 3.067831 \\
397 & -108.80 & -6.23 & 12.0\% & 86/75 & 105 & 0.6478 & 2.572215 \\
398 & -94.41 & -6.43 & 12.0\% & 54/52 & 67 & 0.6471 & 2.113915 \\
399 & -57.73 & -6.90 & 11.0\% & 70/58 & 83 & 0.6463 & 2.347266 \\
400 & -116.53 & -7.43 & 10.0\% & 66/57 & 96 & 0.6453 & 2.878785 \\
401 & -64.06 & -7.16 & 10.0\% & 66/57 & 73 & 0.6446 & 2.344421 \\
402 & -82.71 & -6.87 & 10.0\% & 80/63 & 103 & 0.6436 & 2.293249 \\
403 & -88.45 & -7.15 & 10.0\% & 46/41 & 59 & 0.6430 & 2.034184 \\
404 & -43.51 & -7.01 & 10.0\% & 66/55 & 76 & 0.6423 & 2.142397 \\
405 & -68.23 & -6.61 & 10.0\% & 60/51 & 72 & 0.6416 & 2.787604 \\
406 & -69.70 & -7.03 & 10.0\% & 56/48 & 74 & 0.6408 & 2.777242 \\
407 & -108.84 & -6.74 & 10.0\% & 54/47 & 78 & 0.6400 & 2.168132 \\
408 & -186.94 & -6.60 & 9.0\% & 99/79 & 129 & 0.6388 & 2.667647 \\
409 & -113.75 & -6.91 & 9.0\% & 97/78 & 127 & 0.6375 & 2.656106 \\
410 & -128.93 & -7.09 & 9.0\% & 92/83 & 120 & 0.6363 & 2.222128 \\
411 & -83.97 & -7.58 & 9.0\% & 86/66 & 102 & 0.6353 & 1.959824 \\
412 & -70.45 & -6.54 & 9.0\% & 77/63 & 106 & 0.6343 & 2.275251 \\
413 & -97.23 & -6.62 & 9.0\% & 77/64 & 100 & 0.6333 & 3.024006 \\
414 & -99.56 & -6.80 & 9.0\% & 64/55 & 84 & 0.6324 & 2.693293 \\
415 & -63.07 & -7.42 & 9.0\% & 64/53 & 77 & 0.6317 & 2.184173 \\
416 & -79.77 & -8.18 & 9.0\% & 69/55 & 84 & 0.6308 & 2.521660 \\
417 & -39.87 & -6.89 & 9.0\% & 85/76 & 97 & 0.6299 & 2.250173 \\
418 & -75.80 & -6.53 & 9.0\% & 60/50 & 75 & 0.6291 & 2.351018 \\
419 & -103.09 & -6.84 & 9.0\% & 91/79 & 117 & 0.6280 & 2.605802 \\
420 & -2.62 & -7.24 & 10.0\% & 98/90 & 130 & 0.6267 & 2.507699 \\
421 & -139.94 & -7.33 & 10.0\% & 85/79 & 109 & 0.6256 & 2.888503 \\
422 & -49.98 & -6.52 & 10.0\% & 52/45 & 64 & 0.6250 & 2.510099 \\
423 & -111.81 & -6.65 & 10.0\% & 76/68 & 89 & 0.6241 & 2.378502 \\
424 & -75.11 & -7.20 & 9.0\% & 52/42 & 62 & 0.6235 & 2.729160 \\
425 & -57.07 & -7.21 & 9.0\% & 93/78 & 124 & 0.6223 & 2.249468 \\
426 & -95.80 & -7.56 & 9.0\% & 60/56 & 72 & 0.6216 & 1.891379 \\
427 & -83.27 & -7.44 & 8.0\% & 85/68 & 108 & 0.6205 & 2.238539 \\
428 & -75.26 & -6.95 & 8.0\% & 86/79 & 115 & 0.6193 & 2.593282 \\
429 & -103.54 & -7.60 & 8.0\% & 63/56 & 79 & 0.6186 & 2.635045 \\
430 & -13.14 & -6.52 & 8.0\% & 62/53 & 82 & 0.6178 & 2.432477 \\
431 & -70.46 & -7.21 & 8.0\% & 64/55 & 74 & 0.6170 & 3.087304 \\
432 & -106.51 & -7.66 & 8.0\% & 85/70 & 108 & 0.6159 & 2.455569 \\
433 & -69.22 & -7.06 & 8.0\% & 58/44 & 70 & 0.6153 & 2.458585 \\
434 & -75.87 & -6.56 & 8.0\% & 88/77 & 118 & 0.6141 & 3.204874 \\
435 & -157.57 & -7.79 & 8.0\% & 100/86 & 120 & 0.6129 & 2.229016 \\
436 & -74.99 & -7.91 & 8.0\% & 74/60 & 102 & 0.6119 & 2.436625 \\
437 & -62.67 & -7.90 & 8.0\% & 92/77 & 118 & 0.6107 & 2.157133 \\
438 & -21.12 & -7.87 & 9.0\% & 54/51 & 75 & 0.6100 & 2.243562 \\
439 & -128.16 & -8.37 & 9.0\% & 104/85 & 117 & 0.6088 & 2.206641 \\
440 & -135.96 & -7.32 & 9.0\% & 57/51 & 72 & 0.6081 & 2.382412 \\
441 & -94.82 & -7.82 & 9.0\% & 92/73 & 111 & 0.6070 & 2.439791 \\
442 & -97.15 & -8.43 & 8.0\% & 82/74 & 101 & 0.6060 & 2.570860 \\
443 & -109.84 & -8.04 & 8.0\% & 107/93 & 139 & 0.6046 & 2.280162 \\
444 & -85.37 & -7.27 & 8.0\% & 62/49 & 93 & 0.6037 & 2.360659 \\
445 & -109.28 & -8.13 & 8.0\% & 67/54 & 84 & 0.6029 & 2.129900 \\
446 & -168.06 & -6.81 & 8.0\% & 104/85 & 141 & 0.6015 & 2.340125 \\
447 & -73.60 & -7.65 & 8.0\% & 94/78 & 107 & 0.6004 & 2.058462 \\
448 & -126.81 & -6.99 & 8.0\% & 90/75 & 112 & 0.5993 & 2.288770 \\
449 & -112.46 & -8.13 & 8.0\% & 51/43 & 65 & 0.5987 & 2.342754 \\
450 & -104.32 & -8.10 & 8.0\% & 92/83 & 121 & 0.5975 & 2.494133 \\
451 & -57.74 & -7.15 & 8.0\% & 55/51 & 68 & 0.5968 & 2.741897 \\
452 & 15.49 & -7.66 & 9.0\% & 70/58 & 79 & 0.5960 & 2.302786 \\
453 & -61.31 & -7.69 & 9.0\% & 79/71 & 94 & 0.5951 & 2.284629 \\
454 & -73.90 & -8.80 & 9.0\% & 96/82 & 116 & 0.5939 & 2.642833 \\
455 & -37.35 & -7.37 & 9.0\% & 85/70 & 99 & 0.5930 & 2.485440 \\
456 & -128.56 & -7.88 & 9.0\% & 84/71 & 108 & 0.5919 & 2.182684 \\
457 & -94.71 & -8.02 & 8.0\% & 90/79 & 106 & 0.5908 & 2.413474 \\
458 & -75.60 & -7.97 & 8.0\% & 101/83 & 127 & 0.5896 & 2.497219 \\
459 & -71.72 & -7.84 & 8.0\% & 72/62 & 90 & 0.5887 & 2.596425 \\
460 & -69.02 & -8.64 & 8.0\% & 96/84 & 118 & 0.5875 & 2.169151 \\
461 & -103.72 & -8.01 & 8.0\% & 75/67 & 99 & 0.5865 & 2.357076 \\
462 & 25.11 & -7.43 & 9.0\% & 86/78 & 101 & 0.5855 & 2.396995 \\
463 & -73.61 & -7.28 & 9.0\% & 83/72 & 107 & 0.5845 & 2.110120 \\
464 & -89.92 & -8.13 & 9.0\% & 69/56 & 86 & 0.5836 & 1.915256 \\
465 & -84.16 & -7.81 & 8.0\% & 49/42 & 57 & 0.5831 & 2.132327 \\
466 & -60.71 & -7.93 & 8.0\% & 74/63 & 83 & 0.5822 & 2.304075 \\
467 & -131.47 & -8.28 & 8.0\% & 123/111 & 147 & 0.5808 & 2.070986 \\
468 & -91.59 & -8.10 & 8.0\% & 70/61 & 85 & 0.5800 & 2.672718 \\
469 & -15.82 & -8.61 & 9.0\% & 66/51 & 71 & 0.5793 & 2.065389 \\
470 & -76.86 & -8.44 & 9.0\% & 62/52 & 70 & 0.5786 & 2.409805 \\
471 & -85.99 & -7.99 & 9.0\% & 67/53 & 77 & 0.5778 & 2.421055 \\
472 & -13.62 & -8.20 & 8.0\% & 90/77 & 104 & 0.5768 & 2.696404 \\
473 & -120.25 & -8.96 & 8.0\% & 67/52 & 98 & 0.5758 & 2.703084 \\
474 & -93.52 & -8.36 & 7.0\% & 84/78 & 123 & 0.5746 & 2.526147 \\
475 & -7.12 & -8.50 & 8.0\% & 80/63 & 119 & 0.5734 & 2.726379 \\
476 & -37.46 & -8.79 & 8.0\% & 64/56 & 74 & 0.5727 & 2.421207 \\
477 & -59.81 & -9.24 & 8.0\% & 54/45 & 69 & 0.5720 & 2.521936 \\
478 & -111.68 & -9.15 & 8.0\% & 85/74 & 105 & 0.5709 & 2.697672 \\
479 & -88.33 & -8.46 & 8.0\% & 117/97 & 133 & 0.5696 & 2.521690 \\
480 & -60.48 & -7.87 & 8.0\% & 119/100 & 145 & 0.5682 & 2.418006 \\
481 & -97.57 & -8.51 & 8.0\% & 65/54 & 70 & 0.5675 & 2.412708 \\
482 & -7.28 & -8.67 & 9.0\% & 85/71 & 110 & 0.5664 & 2.403064 \\
483 & -115.59 & -8.62 & 9.0\% & 50/40 & 66 & 0.5658 & 2.565601 \\
484 & -76.11 & -8.43 & 9.0\% & 85/72 & 96 & 0.5648 & 2.354622 \\
485 & -68.24 & -8.37 & 9.0\% & 78/64 & 91 & 0.5639 & 2.282912 \\
486 & -117.29 & -7.94 & 9.0\% & 81/68 & 110 & 0.5628 & 2.650545 \\
487 & -83.38 & -8.09 & 9.0\% & 81/72 & 99 & 0.5618 & 2.530859 \\
488 & -85.31 & -8.23 & 9.0\% & 74/60 & 94 & 0.5609 & 2.181522 \\
489 & -97.90 & -8.15 & 9.0\% & 102/91 & 132 & 0.5596 & 2.116793 \\
490 & -119.62 & -8.65 & 9.0\% & 88/76 & 105 & 0.5586 & 2.710799 \\
491 & -42.19 & -9.00 & 9.0\% & 80/67 & 95 & 0.5576 & 2.758430 \\
492 & -21.28 & -9.06 & 10.0\% & 72/59 & 91 & 0.5567 & 2.483609 \\
493 & -77.26 & -9.01 & 10.0\% & 66/54 & 70 & 0.5560 & 1.906712 \\
494 & -77.02 & -8.23 & 10.0\% & 81/71 & 104 & 0.5550 & 2.545696 \\
495 & -74.25 & -7.82 & 10.0\% & 88/71 & 123 & 0.5538 & 2.486060 \\
496 & 7.52 & -7.86 & 11.0\% & 64/52 & 80 & 0.5530 & 2.560677 \\
497 & -16.34 & -8.01 & 12.0\% & 78/63 & 104 & 0.5520 & 2.590926 \\
498 & -124.95 & -7.49 & 12.0\% & 91/75 & 111 & 0.5509 & 2.680656 \\
499 & -82.96 & -8.28 & 12.0\% & 94/80 & 113 & 0.5497 & 2.048162 \\
500 & -62.86 & -8.47 & 12.0\% & 83/68 & 123 & 0.5485 & 2.562660 \\
501 & -104.88 & -8.50 & 12.0\% & 56/50 & 67 & 0.5479 & 2.455978 \\
502 & -110.80 & -7.38 & 12.0\% & 73/64 & 112 & 0.5467 & 2.285699 \\
503 & -104.97 & -8.25 & 12.0\% & 101/86 & 118 & 0.5456 & 2.250428 \\
504 & -97.57 & -8.52 & 12.0\% & 98/85 & 120 & 0.5444 & 2.511954 \\
505 & -49.08 & -8.45 & 13.0\% & 66/55 & 96 & 0.5434 & 2.465253 \\
506 & -79.82 & -8.33 & 13.0\% & 81/72 & 93 & 0.5425 & 2.497405 \\
507 & -71.54 & -8.10 & 13.0\% & 87/72 & 95 & 0.5416 & 1.945441 \\
508 & -55.67 & -7.55 & 13.0\% & 62/50 & 79 & 0.5408 & 2.374849 \\
509 & -81.90 & -8.24 & 13.0\% & 94/79 & 114 & 0.5397 & 2.418664 \\
510 & -89.39 & -7.59 & 13.0\% & 67/61 & 80 & 0.5389 & 2.337092 \\
511 & -24.51 & -7.33 & 13.0\% & 80/69 & 93 & 0.5380 & 2.322193 \\
512 & -267.70 & -7.76 & 13.0\% & 118/101 & 151 & 0.5365 & 2.102146 \\
513 & -7.51 & -7.26 & 13.0\% & 109/95 & 122 & 0.5353 & 2.410892 \\
514 & -118.83 & -6.85 & 13.0\% & 79/68 & 99 & 0.5343 & 2.715783 \\
515 & -73.54 & -6.67 & 13.0\% & 75/69 & 89 & 0.5334 & 2.226475 \\
516 & -27.63 & -6.96 & 14.0\% & 95/81 & 114 & 0.5323 & 2.178711 \\
517 & -48.40 & -7.65 & 14.0\% & 74/59 & 90 & 0.5314 & 2.309148 \\
518 & -120.53 & -7.26 & 14.0\% & 65/57 & 79 & 0.5306 & 2.114751 \\
519 & -101.68 & -7.75 & 14.0\% & 91/70 & 110 & 0.5295 & 2.598997 \\
520 & -60.39 & -7.53 & 13.0\% & 63/53 & 73 & 0.5288 & 2.109830 \\
521 & -42.08 & -7.53 & 13.0\% & 80/66 & 89 & 0.5279 & 2.508172 \\
522 & -46.06 & -7.32 & 13.0\% & 98/84 & 123 & 0.5267 & 2.074657 \\
523 & -57.13 & -7.60 & 13.0\% & 75/62 & 92 & 0.5258 & 2.305801 \\
524 & 8.57 & -8.10 & 14.0\% & 101/87 & 131 & 0.5245 & 2.333814 \\
525 & -135.08 & -7.65 & 14.0\% & 111/95 & 141 & 0.5231 & 2.192495 \\
526 & -62.06 & -7.79 & 14.0\% & 89/80 & 118 & 0.5219 & 2.712673 \\
527 & -93.79 & -7.39 & 14.0\% & 102/80 & 127 & 0.5207 & 2.268856 \\
528 & -97.27 & -6.73 & 14.0\% & 73/66 & 93 & 0.5197 & 2.710274 \\
529 & -83.26 & -6.55 & 14.0\% & 91/76 & 108 & 0.5187 & 2.518285 \\
530 & -144.90 & -6.51 & 13.0\% & 119/105 & 139 & 0.5173 & 2.374191 \\
531 & -11.52 & -7.37 & 14.0\% & 77/70 & 91 & 0.5164 & 2.438045 \\
532 & -70.25 & -7.92 & 14.0\% & 99/81 & 126 & 0.5151 & 2.268979 \\
533 & -54.83 & -7.36 & 14.0\% & 111/92 & 142 & 0.5137 & 2.271280 \\
534 & -89.94 & -8.35 & 14.0\% & 91/79 & 113 & 0.5126 & 2.496495 \\
535 & 25.86 & -6.96 & 15.0\% & 65/57 & 82 & 0.5118 & 2.499490 \\
536 & -77.52 & -7.59 & 15.0\% & 69/59 & 82 & 0.5110 & 2.871064 \\
537 & -56.08 & -7.10 & 15.0\% & 91/81 & 104 & 0.5100 & 2.419352 \\
538 & -28.94 & -7.18 & 14.0\% & 100/87 & 137 & 0.5086 & 2.018821 \\
539 & -76.27 & -7.64 & 14.0\% & 76/64 & 95 & 0.5077 & 2.345099 \\
540 & -91.95 & -6.88 & 14.0\% & 86/78 & 105 & 0.5066 & 2.773739 \\
541 & 19.35 & -6.68 & 15.0\% & 77/68 & 86 & 0.5058 & 2.315018 \\
542 & -73.22 & -7.61 & 15.0\% & 70/64 & 98 & 0.5048 & 2.140620 \\
543 & -306.71 & -7.89 & 15.0\% & 106/93 & 141 & 0.5034 & 2.213923 \\
544 & -23.04 & -7.50 & 15.0\% & 92/79 & 115 & 0.5023 & 2.389916 \\
545 & -142.22 & -8.35 & 15.0\% & 76/66 & 98 & 0.5013 & 2.145690 \\
546 & -69.30 & -7.89 & 15.0\% & 171/149 & 199 & 0.4993 & 2.514500 \\
547 & -72.75 & -8.01 & 15.0\% & 75/65 & 85 & 0.4985 & 2.552113 \\
548 & -11.40 & -7.61 & 16.0\% & 85/66 & 113 & 0.4974 & 2.082169 \\
549 & -85.24 & -7.65 & 16.0\% & 55/47 & 70 & 0.4967 & 2.646254 \\
550 & -106.98 & -7.49 & 16.0\% & 76/63 & 92 & 0.4958 & 2.358091 \\
551 & -55.04 & -7.44 & 16.0\% & 120/114 & 140 & 0.4944 & 1.997658 \\
552 & -21.50 & -7.43 & 16.0\% & 73/64 & 98 & 0.4934 & 2.097806 \\
553 & -9.66 & -7.28 & 17.0\% & 77/66 & 92 & 0.4925 & 2.417344 \\
554 & -71.31 & -8.02 & 17.0\% & 82/66 & 106 & 0.4914 & 2.250725 \\
555 & -87.79 & -6.23 & 16.0\% & 52/43 & 58 & 0.4909 & 2.050837 \\
556 & -51.19 & -7.11 & 16.0\% & 66/55 & 73 & 0.4901 & 2.268192 \\
557 & 2.46 & -6.65 & 17.0\% & 82/72 & 97 & 0.4892 & 2.341889 \\
558 & -54.05 & -7.38 & 17.0\% & 142/121 & 169 & 0.4875 & 2.160244 \\
559 & -73.74 & -7.13 & 17.0\% & 94/83 & 136 & 0.4862 & 2.387121 \\
560 & -45.27 & -7.91 & 17.0\% & 110/98 & 134 & 0.4848 & 2.631377 \\
561 & -81.95 & -8.16 & 17.0\% & 57/46 & 65 & 0.4842 & 2.708532 \\
562 & -87.88 & -7.97 & 16.0\% & 82/72 & 105 & 0.4832 & 2.319838 \\
563 & -53.94 & -8.23 & 17.0\% & 79/64 & 102 & 0.4822 & 2.361263 \\
564 & -94.65 & -8.21 & 17.0\% & 87/74 & 108 & 0.4811 & 2.409042 \\
565 & 19.30 & -8.25 & 18.0\% & 91/78 & 101 & 0.4801 & 2.137862 \\
566 & -89.35 & -7.62 & 18.0\% & 96/85 & 119 & 0.4789 & 2.440790 \\
567 & -91.04 & -8.25 & 18.0\% & 77/65 & 94 & 0.4780 & 2.048914 \\
568 & -72.59 & -8.32 & 18.0\% & 86/72 & 104 & 0.4769 & 1.964140 \\
569 & -73.64 & -8.14 & 17.0\% & 98/80 & 117 & 0.4758 & 2.129681 \\
570 & -49.19 & -8.47 & 17.0\% & 114/97 & 138 & 0.4744 & 2.291566 \\
571 & -87.73 & -8.05 & 17.0\% & 90/75 & 102 & 0.4734 & 2.239203 \\
572 & -160.76 & -7.52 & 17.0\% & 101/89 & 118 & 0.4722 & 2.393065 \\
573 & -50.28 & -8.22 & 17.0\% & 74/63 & 86 & 0.4714 & 2.033550 \\
574 & -78.71 & -8.43 & 17.0\% & 79/67 & 93 & 0.4705 & 2.117790 \\
575 & -44.99 & -8.32 & 16.0\% & 56/45 & 66 & 0.4698 & 2.078488 \\
576 & -77.59 & -7.96 & 16.0\% & 71/60 & 81 & 0.4690 & 2.540833 \\
577 & -68.29 & -8.06 & 16.0\% & 61/53 & 68 & 0.4683 & 2.243906 \\
578 & -45.49 & -8.86 & 16.0\% & 76/66 & 93 & 0.4674 & 2.543055 \\
579 & -62.08 & -8.81 & 16.0\% & 75/62 & 104 & 0.4664 & 2.833652 \\
580 & -36.27 & -8.59 & 16.0\% & 107/91 & 122 & 0.4652 & 2.080521 \\
581 & -83.02 & -9.31 & 16.0\% & 78/65 & 113 & 0.4641 & 2.345103 \\
582 & 8.21 & -8.50 & 16.0\% & 81/69 & 102 & 0.4631 & 2.481790 \\
583 & -71.60 & -9.46 & 16.0\% & 101/82 & 114 & 0.4619 & 2.069867 \\
584 & -34.60 & -8.81 & 16.0\% & 75/63 & 92 & 0.4610 & 2.418252 \\
585 & -91.93 & -9.11 & 16.0\% & 64/54 & 71 & 0.4603 & 2.642436 \\
586 & -87.06 & -9.97 & 16.0\% & 104/84 & 124 & 0.4591 & 2.203585 \\
587 & -67.73 & -8.79 & 16.0\% & 82/70 & 104 & 0.4581 & 2.108661 \\
588 & -48.51 & -9.32 & 16.0\% & 145/126 & 183 & 0.4562 & 2.169432 \\
589 & -117.87 & -8.68 & 16.0\% & 104/85 & 134 & 0.4549 & 2.013673 \\
590 & -56.32 & -9.21 & 16.0\% & 86/74 & 103 & 0.4539 & 2.601542 \\
591 & -69.31 & -8.06 & 16.0\% & 132/108 & 147 & 0.4524 & 2.231632 \\
592 & -10.46 & -9.25 & 16.0\% & 100/83 & 121 & 0.4512 & 2.074312 \\
593 & -113.61 & -8.73 & 16.0\% & 87/73 & 105 & 0.4502 & 2.053865 \\
594 & -66.34 & -8.56 & 16.0\% & 91/78 & 126 & 0.4490 & 1.909341 \\
595 & -145.73 & -8.29 & 16.0\% & 109/91 & 132 & 0.4476 & 2.207506 \\
596 & -51.13 & -8.86 & 15.0\% & 81/72 & 108 & 0.4466 & 2.333156 \\
597 & -109.11 & -8.13 & 14.0\% & 210/174 & 229 & 0.4443 & 2.317612 \\
598 & -73.73 & -8.63 & 14.0\% & 111/93 & 139 & 0.4429 & 2.259448 \\
599 & -34.82 & -7.91 & 14.0\% & 72/64 & 81 & 0.4421 & 2.338253 \\
600 & -63.88 & -8.72 & 14.0\% & 66/53 & 75 & 0.4414 & 2.298889 \\
601 & -48.30 & -8.52 & 14.0\% & 74/61 & 87 & 0.4405 & 2.016996 \\
602 & -13.21 & -8.66 & 14.0\% & 77/69 & 89 & 0.4397 & 2.585195 \\
603 & -70.90 & -8.57 & 14.0\% & 90/72 & 113 & 0.4385 & 2.016218 \\
604 & -54.63 & -8.62 & 14.0\% & 95/83 & 113 & 0.4374 & 1.975721 \\
605 & -11.33 & -8.95 & 13.0\% & 78/68 & 89 & 0.4365 & 1.996933 \\
606 & 8.66 & -8.10 & 13.0\% & 116/101 & 133 & 0.4352 & 2.023362 \\
607 & -39.84 & -8.86 & 13.0\% & 188/161 & 225 & 0.4330 & 2.137021 \\
608 & -59.03 & -8.65 & 13.0\% & 208/173 & 235 & 0.4307 & 2.316812 \\
609 & -50.52 & -8.29 & 13.0\% & 106/91 & 129 & 0.4294 & 2.327556 \\
610 & -45.26 & -8.28 & 13.0\% & 104/88 & 125 & 0.4281 & 1.907753 \\
611 & -83.23 & -8.23 & 13.0\% & 77/67 & 87 & 0.4273 & 2.297914 \\
612 & 18.01 & -8.21 & 14.0\% & 85/65 & 101 & 0.4263 & 1.796926 \\
613 & -60.62 & -7.60 & 14.0\% & 72/64 & 84 & 0.4255 & 2.026040 \\
614 & -21.81 & -8.67 & 15.0\% & 81/64 & 115 & 0.4243 & 2.443171 \\
615 & -76.30 & -8.19 & 15.0\% & 101/81 & 119 & 0.4231 & 2.645254 \\
616 & -32.08 & -7.87 & 14.0\% & 123/110 & 142 & 0.4217 & 1.903226 \\
617 & 2.80 & -7.24 & 14.0\% & 105/87 & 113 & 0.4206 & 1.967727 \\
618 & -59.28 & -7.79 & 14.0\% & 79/62 & 90 & 0.4197 & 2.653889 \\
619 & -73.29 & -7.91 & 14.0\% & 87/72 & 103 & 0.4187 & 1.936774 \\
620 & -75.66 & -7.92 & 14.0\% & 103/87 & 122 & 0.4175 & 2.033319 \\
621 & -42.25 & -7.37 & 14.0\% & 86/79 & 97 & 0.4165 & 2.010066 \\
622 & -59.75 & -7.13 & 14.0\% & 96/76 & 129 & 0.4153 & 2.419869 \\
623 & -42.36 & -7.83 & 14.0\% & 153/130 & 188 & 0.4134 & 2.334609 \\
624 & -52.96 & -7.50 & 13.0\% & 162/143 & 178 & 0.4116 & 2.316606 \\
625 & -30.70 & -7.52 & 13.0\% & 119/95 & 147 & 0.4102 & 2.427752 \\
626 & -117.74 & -6.90 & 13.0\% & 101/83 & 126 & 0.4089 & 2.243680 \\
627 & -108.18 & -7.53 & 13.0\% & 146/122 & 168 & 0.4073 & 2.483319 \\
628 & -53.30 & -6.26 & 13.0\% & 79/63 & 95 & 0.4063 & 2.208423 \\
629 & 30.03 & -8.29 & 14.0\% & 80/70 & 95 & 0.4054 & 2.168643 \\
630 & -44.48 & -7.59 & 14.0\% & 114/99 & 132 & 0.4041 & 2.176357 \\
631 & -118.45 & -7.54 & 13.0\% & 96/74 & 105 & 0.4030 & 2.434165 \\
632 & -108.76 & -7.64 & 13.0\% & 89/77 & 103 & 0.4020 & 1.943781 \\
633 & -58.83 & -7.73 & 13.0\% & 104/86 & 117 & 0.4009 & 2.042094 \\
634 & -42.07 & -8.09 & 13.0\% & 113/91 & 137 & 0.3995 & 2.387232 \\
635 & 3.75 & -7.99 & 13.0\% & 88/76 & 123 & 0.3983 & 2.293650 \\
636 & -52.73 & -8.17 & 13.0\% & 108/92 & 134 & 0.3970 & 2.495462 \\
637 & -38.42 & -6.49 & 13.0\% & 131/110 & 174 & 0.3952 & 1.991947 \\
638 & -186.90 & -7.64 & 13.0\% & 402/351 & 453 & 0.3908 & 2.106445 \\
639 & -24.53 & -6.65 & 13.0\% & 123/104 & 145 & 0.3893 & 2.155747 \\
640 & -21.36 & -7.07 & 14.0\% & 79/68 & 90 & 0.3884 & 2.634768 \\
641 & 17.84 & -7.84 & 14.0\% & 93/78 & 124 & 0.3872 & 2.013506 \\
642 & -80.62 & -7.30 & 14.0\% & 84/66 & 97 & 0.3862 & 2.392860 \\
643 & -42.70 & -7.31 & 14.0\% & 112/95 & 126 & 0.3850 & 2.120487 \\
644 & -43.50 & -6.54 & 14.0\% & 92/75 & 100 & 0.3840 & 2.210223 \\
645 & -43.96 & -7.35 & 14.0\% & 143/125 & 172 & 0.3823 & 2.021225 \\
646 & -43.59 & -8.32 & 14.0\% & 83/73 & 92 & 0.3814 & 2.160152 \\
647 & -69.05 & -7.48 & 14.0\% & 138/122 & 167 & 0.3797 & 2.032001 \\
648 & -70.69 & -6.93 & 13.0\% & 81/70 & 88 & 0.3789 & 2.212303 \\
649 & -163.48 & -7.01 & 13.0\% & 189/163 & 217 & 0.3767 & 2.301119 \\
650 & -15.40 & -6.74 & 13.0\% & 93/77 & 99 & 0.3757 & 2.079361 \\
651 & -37.27 & -6.65 & 13.0\% & 101/81 & 120 & 0.3745 & 2.295786 \\
652 & -26.76 & -6.72 & 12.0\% & 126/107 & 156 & 0.3730 & 1.865452 \\
653 & 26.88 & -6.06 & 12.0\% & 166/137 & 192 & 0.3711 & 1.916417 \\
654 & -26.12 & -6.24 & 12.0\% & 90/80 & 105 & 0.3701 & 2.148350 \\
655 & -55.37 & -6.95 & 12.0\% & 68/61 & 84 & 0.3692 & 2.276608 \\
656 & -83.06 & -6.82 & 12.0\% & 140/119 & 155 & 0.3677 & 2.255286 \\
657 & -45.27 & -6.37 & 11.0\% & 98/75 & 116 & 0.3665 & 1.995249 \\
658 & -41.83 & -6.02 & 11.0\% & 94/79 & 118 & 0.3654 & 2.069674 \\
659 & -18.32 & -5.75 & 11.0\% & 81/64 & 85 & 0.3645 & 2.330359 \\
660 & -99.53 & -6.80 & 11.0\% & 190/163 & 216 & 0.3624 & 2.249133 \\
661 & -35.83 & -6.12 & 12.0\% & 120/92 & 129 & 0.3611 & 1.974384 \\
662 & -281.56 & -5.49 & 12.0\% & 889/760 & 1000 & 0.3512 & 2.212093 \\
663 & -74.84 & -4.93 & 11.0\% & 113/101 & 145 & 0.3498 & 2.068402 \\
664 & 13.28 & -5.17 & 12.0\% & 71/64 & 95 & 0.3488 & 1.949820 \\
665 & -39.77 & -5.19 & 12.0\% & 357/300 & 409 & 0.3448 & 2.104585 \\
666 & -54.37 & -5.49 & 12.0\% & 92/78 & 120 & 0.3436 & 2.417081 \\
667 & -10.52 & -4.46 & 12.0\% & 80/68 & 92 & 0.3427 & 2.156776 \\
668 & -214.57 & -4.31 & 12.0\% & 487/415 & 577 & 0.3370 & 2.100184 \\
669 & -56.15 & -3.68 & 12.0\% & 163/146 & 198 & 0.3350 & 2.170922 \\
670 & -355.22 & -3.09 & 12.0\% & 876/758 & 1000 & 0.3251 & 2.030768 \\
671 & -38.59 & -2.82 & 12.0\% & 217/186 & 236 & 0.3228 & 2.291488 \\
672 & -289.35 & -3.62 & 12.0\% & 760/630 & 863 & 0.3142 & 2.148399 \\
673 & -11.89 & -3.98 & 12.0\% & 98/84 & 106 & 0.3132 & 2.075092 \\
674 & -32.68 & -1.98 & 12.0\% & 138/116 & 151 & 0.3117 & 2.027457 \\
675 & -56.54 & -2.54 & 12.0\% & 129/105 & 151 & 0.3102 & 2.170231 \\
676 & 23.17 & -2.60 & 13.0\% & 240/212 & 264 & 0.3076 & 2.100467 \\
677 & -220.60 & -2.81 & 13.0\% & 898/772 & 1000 & 0.2977 & 2.134198 \\
678 & -251.06 & -2.08 & 13.0\% & 883/761 & 1000 & 0.2878 & 2.145420 \\
679 & -19.55 & -0.48 & 13.0\% & 94/85 & 100 & 0.2868 & 2.365779 \\
680 & -56.86 & -0.45 & 13.0\% & 159/137 & 173 & 0.2851 & 2.316653 \\
681 & -52.09 & -0.42 & 13.0\% & 82/66 & 95 & 0.2842 & 2.281172 \\
682 & -65.02 & -0.76 & 12.0\% & 101/81 & 121 & 0.2830 & 2.328117 \\
683 & -195.45 & 1.09 & 12.0\% & 862/728 & 1000 & 0.2731 & 2.039750 \\
684 & 30.89 & 1.71 & 13.0\% & 126/109 & 152 & 0.2715 & 1.905458 \\
685 & -63.12 & 2.26 & 13.0\% & 136/122 & 159 & 0.2700 & 2.376061 \\
686 & -19.75 & 2.37 & 13.0\% & 107/84 & 131 & 0.2687 & 1.821338 \\
687 & -34.24 & 1.55 & 13.0\% & 160/130 & 175 & 0.2669 & 1.826779 \\
688 & -278.82 & 3.10 & 13.0\% & 788/668 & 1000 & 0.2570 & 2.073747 \\
689 & 47.91 & 2.75 & 14.0\% & 98/84 & 133 & 0.2557 & 2.058165 \\
690 & -77.00 & 3.32 & 14.0\% & 139/111 & 165 & 0.2541 & 1.843737 \\
691 & -19.06 & 2.59 & 14.0\% & 170/146 & 214 & 0.2520 & 2.010382 \\
692 & -27.10 & 2.43 & 13.0\% & 113/96 & 139 & 0.2506 & 2.590075 \\
693 & -41.60 & 2.47 & 13.0\% & 83/66 & 90 & 0.2497 & 2.231731 \\
694 & -32.85 & 2.84 & 13.0\% & 182/158 & 203 & 0.2477 & 2.411900 \\
695 & -129.29 & 2.89 & 13.0\% & 682/570 & 1000 & 0.2378 & 2.064959 \\
696 & -267.62 & 3.11 & 13.0\% & 752/638 & 1000 & 0.2279 & 2.159448 \\
697 & 142.74 & 4.58 & 14.0\% & 394/334 & 550 & 0.2225 & 2.294904 \\
698 & -238.12 & 5.20 & 14.0\% & 791/674 & 1000 & 0.2126 & 2.170319 \\
699 & -10.56 & 4.61 & 14.0\% & 117/102 & 136 & 0.2112 & 2.438938 \\
700 & -150.44 & 6.38 & 14.0\% & 797/687 & 1000 & 0.2013 & 2.165641 \\
701 & -349.20 & 5.11 & 14.0\% & 596/497 & 688 & 0.1945 & 2.312673 \\
702 & -265.05 & 6.98 & 14.0\% & 827/722 & 1000 & 0.1846 & 2.125260 \\
703 & -229.42 & 5.97 & 14.0\% & 588/511 & 712 & 0.1775 & 2.138474 \\
704 & -318.09 & 5.59 & 14.0\% & 787/690 & 1000 & 0.1676 & 2.157092 \\
705 & -220.84 & 4.10 & 14.0\% & 802/669 & 1000 & 0.1577 & 2.179261 \\
706 & -152.97 & 5.72 & 14.0\% & 689/578 & 1000 & 0.1478 & 2.059767 \\
707 & -352.30 & 6.14 & 14.0\% & 865/709 & 1000 & 0.1379 & 2.020059 \\
708 & -199.27 & 6.17 & 14.0\% & 332/271 & 386 & 0.1341 & 2.141918 \\
709 & -377.70 & 6.83 & 14.0\% & 847/727 & 1000 & 0.1242 & 2.161853 \\
710 & -278.28 & 5.29 & 14.0\% & 836/717 & 1000 & 0.1143 & 1.990823 \\
711 & -391.62 & 6.27 & 14.0\% & 674/574 & 794 & 0.1065 & 1.948091 \\
712 & -343.55 & 6.42 & 13.0\% & 805/677 & 1000 & 0.0966 & 2.003269 \\
713 & -265.80 & 7.82 & 13.0\% & 810/681 & 1000 & 0.0867 & 2.071103 \\
714 & -389.61 & 8.81 & 12.0\% & 812/674 & 1000 & 0.0768 & 2.044056 \\
715 & -379.04 & 8.22 & 12.0\% & 786/675 & 1000 & 0.0669 & 2.103738 \\
716 & -390.55 & 8.12 & 12.0\% & 667/581 & 896 & 0.0580 & 2.024966 \\
717 & -321.72 & 8.16 & 12.0\% & 762/655 & 1000 & 0.0481 & 2.015002 \\
718 & -337.83 & 9.60 & 12.0\% & 766/656 & 1000 & 0.0382 & 2.024084 \\
719 & -333.28 & 10.50 & 12.0\% & 747/637 & 1000 & 0.0283 & 1.899911 \\
720 & -289.75 & 10.74 & 12.0\% & 746/629 & 1000 & 0.0184 & 1.943522 \\
721 & -285.18 & 11.29 & 12.0\% & 735/637 & 1000 & 0.0100 & 1.955590 \\
722 & -275.17 & 11.23 & 12.0\% & 724/625 & 1000 & 0.0100 & 1.874177 \\
723 & -254.40 & 10.93 & 12.0\% & 753/629 & 1000 & 0.0100 & 1.779097 \\
724 & -241.38 & 11.92 & 12.0\% & 741/646 & 1000 & 0.0100 & 1.776037 \\
725 & -462.16 & 12.88 & 12.0\% & 706/614 & 883 & 0.0100 & 1.711945 \\
726 & -259.01 & 13.35 & 12.0\% & 739/633 & 1000 & 0.0100 & 1.704946 \\
727 & -257.28 & 11.78 & 12.0\% & 370/309 & 464 & 0.0100 & 1.629807 \\
728 & -270.76 & 11.61 & 12.0\% & 807/684 & 1000 & 0.0100 & 1.694345 \\
729 & -217.27 & 13.76 & 11.0\% & 764/646 & 1000 & 0.0100 & 1.570798 \\
730 & -282.25 & 14.56 & 11.0\% & 820/685 & 1000 & 0.0100 & 1.583399 \\
731 & -262.90 & 12.10 & 11.0\% & 778/646 & 1000 & 0.0100 & 1.563304 \\
732 & -277.81 & 12.32 & 11.0\% & 774/655 & 1000 & 0.0100 & 1.572621 \\
733 & -247.61 & 13.28 & 11.0\% & 825/690 & 1000 & 0.0100 & 1.556724 \\
734 & -250.66 & 13.95 & 11.0\% & 788/665 & 1000 & 0.0100 & 1.502125 \\
735 & -269.58 & 15.11 & 10.0\% & 787/685 & 1000 & 0.0100 & 1.487064 \\
736 & -241.12 & 14.56 & 10.0\% & 757/646 & 1000 & 0.0100 & 1.490412 \\
737 & -134.98 & 14.30 & 10.0\% & 308/266 & 390 & 0.0100 & 1.548411 \\
738 & -252.56 & 14.99 & 10.0\% & 768/640 & 1000 & 0.0100 & 1.531091 \\
739 & -242.95 & 15.88 & 10.0\% & 786/673 & 1000 & 0.0100 & 1.516415 \\
740 & -281.51 & 15.68 & 9.0\% & 789/681 & 1000 & 0.0100 & 1.507604 \\
741 & -261.06 & 14.72 & 8.0\% & 805/669 & 1000 & 0.0100 & 1.522440 \\
742 & -274.81 & 15.86 & 8.0\% & 842/714 & 1000 & 0.0100 & 1.452034 \\
743 & -297.24 & 15.00 & 8.0\% & 824/708 & 1000 & 0.0100 & 1.507917 \\
744 & -247.90 & 15.13 & 8.0\% & 762/638 & 1000 & 0.0100 & 1.470607 \\
745 & -326.37 & 14.99 & 8.0\% & 843/705 & 1000 & 0.0100 & 1.469853 \\
746 & -210.71 & 15.72 & 8.0\% & 795/660 & 1000 & 0.0100 & 1.425258 \\
747 & -43.13 & 14.61 & 8.0\% & 569/496 & 676 & 0.0100 & 1.380574 \\
748 & -233.74 & 13.52 & 8.0\% & 828/697 & 1000 & 0.0100 & 1.389422 \\
749 & -282.89 & 12.59 & 8.0\% & 757/656 & 1000 & 0.0100 & 1.337918 \\
750 & 283.77 & 12.36 & 9.0\% & 208/183 & 281 & 0.0100 & 1.409272 \\
751 & -222.86 & 12.75 & 9.0\% & 780/663 & 1000 & 0.0100 & 1.329391 \\
752 & -293.07 & 13.11 & 9.0\% & 783/661 & 1000 & 0.0100 & 1.298325 \\
753 & 13.51 & 12.92 & 8.0\% & 140/120 & 186 & 0.0100 & 1.271506 \\
754 & -261.42 & 12.30 & 8.0\% & 845/717 & 1000 & 0.0100 & 1.250849 \\
755 & 229.41 & 12.70 & 9.0\% & 295/238 & 417 & 0.0100 & 1.155829 \\
756 & -226.25 & 11.74 & 9.0\% & 771/675 & 1000 & 0.0100 & 1.211849 \\
757 & -230.21 & 11.85 & 9.0\% & 777/661 & 1000 & 0.0100 & 1.245424 \\
758 & -215.26 & 12.22 & 9.0\% & 767/657 & 1000 & 0.0100 & 1.233978 \\
759 & -240.95 & 14.22 & 9.0\% & 759/653 & 1000 & 0.0100 & 1.225716 \\
760 & -243.19 & 12.95 & 9.0\% & 849/717 & 1000 & 0.0100 & 1.210360 \\
761 & 280.77 & 13.03 & 9.0\% & 218/179 & 352 & 0.0100 & 1.120081 \\
762 & -224.23 & 11.91 & 9.0\% & 803/683 & 1000 & 0.0100 & 1.215089 \\
763 & -273.66 & 11.28 & 9.0\% & 766/654 & 1000 & 0.0100 & 1.170927 \\
764 & -238.39 & 11.76 & 8.0\% & 732/623 & 1000 & 0.0100 & 1.154907 \\
765 & -255.55 & 12.92 & 7.0\% & 755/651 & 1000 & 0.0100 & 1.150467 \\
766 & -272.28 & 13.06 & 7.0\% & 813/687 & 1000 & 0.0100 & 1.134587 \\
767 & 294.46 & 11.73 & 8.0\% & 187/160 & 315 & 0.0100 & 1.110565 \\
768 & -255.80 & 12.40 & 8.0\% & 782/659 & 1000 & 0.0100 & 1.150458 \\
769 & -268.51 & 11.04 & 8.0\% & 731/616 & 1000 & 0.0100 & 1.127829 \\
770 & -249.03 & 12.61 & 8.0\% & 791/667 & 1000 & 0.0100 & 1.091610 \\
771 & -222.22 & 15.13 & 8.0\% & 796/676 & 1000 & 0.0100 & 1.075483 \\
772 & -265.43 & 14.98 & 8.0\% & 739/647 & 1000 & 0.0100 & 1.090671 \\
773 & -252.85 & 16.34 & 8.0\% & 780/652 & 1000 & 0.0100 & 1.121778 \\
774 & 110.02 & 18.17 & 9.0\% & 534/460 & 782 & 0.0100 & 1.091888 \\
775 & 4.69 & 16.71 & 10.0\% & 703/617 & 978 & 0.0100 & 1.010688 \\
776 & 141.02 & 19.22 & 10.0\% & 450/383 & 623 & 0.0100 & 0.971257 \\
777 & -235.72 & 20.33 & 10.0\% & 774/653 & 1000 & 0.0100 & 0.990682 \\
778 & 113.26 & 22.22 & 11.0\% & 482/410 & 668 & 0.0100 & 0.975349 \\
779 & 89.56 & 22.54 & 12.0\% & 483/413 & 692 & 0.0100 & 0.973636 \\
780 & 231.58 & 23.56 & 13.0\% & 246/199 & 328 & 0.0100 & 0.950733 \\
781 & -278.04 & 25.20 & 13.0\% & 768/649 & 1000 & 0.0100 & 1.005221 \\
782 & 101.34 & 26.33 & 14.0\% & 484/408 & 685 & 0.0100 & 0.923669 \\
783 & 125.52 & 26.72 & 15.0\% & 449/378 & 649 & 0.0100 & 0.996846 \\
784 & -252.67 & 28.50 & 14.0\% & 718/635 & 1000 & 0.0100 & 0.897850 \\
785 & -225.49 & 31.69 & 14.0\% & 758/648 & 1000 & 0.0100 & 0.914854 \\
786 & -244.37 & 31.20 & 14.0\% & 748/645 & 1000 & 0.0100 & 0.846199 \\
787 & 302.30 & 33.03 & 15.0\% & 145/126 & 220 & 0.0100 & 0.825561 \\
788 & -249.30 & 33.05 & 15.0\% & 706/626 & 1000 & 0.0100 & 0.871418 \\
789 & 231.18 & 33.47 & 15.0\% & 206/176 & 321 & 0.0100 & 0.921703 \\
790 & -204.03 & 35.67 & 15.0\% & 747/644 & 1000 & 0.0100 & 0.841834 \\
791 & -163.08 & 37.69 & 15.0\% & 699/586 & 1000 & 0.0100 & 0.817303 \\
792 & -174.26 & 38.42 & 15.0\% & 720/638 & 1000 & 0.0100 & 0.797543 \\
793 & 252.80 & 41.54 & 16.0\% & 282/237 & 552 & 0.0100 & 0.815084 \\
794 & 118.24 & 41.12 & 17.0\% & 422/355 & 605 & 0.0100 & 0.792268 \\
795 & -349.66 & 41.86 & 17.0\% & 111/84 & 112 & 0.0100 & 0.740762 \\
796 & 283.12 & 43.46 & 18.0\% & 154/134 & 205 & 0.0100 & 0.822326 \\
797 & 230.59 & 44.31 & 18.0\% & 239/206 & 350 & 0.0100 & 0.828459 \\
798 & -96.52 & 44.22 & 18.0\% & 292/264 & 395 & 0.0100 & 0.848400 \\
799 & -114.75 & 48.13 & 18.0\% & 762/666 & 1000 & 0.0100 & 0.779406 \\
800 & 218.86 & 48.56 & 19.0\% & 269/227 & 366 & 0.0100 & 0.757581 \\
\end{longtable}
\normalsize

\clearpage

\scriptsize
\setlength{\tabcolsep}{1.8pt}
\renewcommand{\arraystretch}{0.86}
\begin{longtable}{rrrrrrrr}
\caption{Complete per-iteration training output - Ddqn Modified (800 episodes).}\\
\toprule
Episode & Reward & Avg. Q & Safe100 & Attempted/Executed & Steps & Epsilon & Loss \\
\midrule
\endfirsthead
\multicolumn{8}{c}{\small Continued: Ddqn Modified complete per-iteration output} \\
\toprule
Episode & Reward & Avg. Q & Safe100 & Attempted/Executed & Steps & Epsilon & Loss \\
\midrule
\endhead
\bottomrule
\endfoot
\bottomrule
\endlastfoot
1 & -147.06 & 0.24 & 0.0\% & 50/40 & 70 & 0.9993 & nan \\
2 & -359.38 & 0.24 & 0.0\% & 67/61 & 93 & 0.9984 & nan \\
3 & -158.89 & 0.24 & 0.0\% & 102/86 & 122 & 0.9972 & nan \\
4 & -141.98 & 0.24 & 0.0\% & 67/58 & 85 & 0.9963 & nan \\
5 & -173.75 & 0.24 & 0.0\% & 84/72 & 117 & 0.9952 & nan \\
6 & -329.77 & 0.24 & 0.0\% & 78/67 & 104 & 0.9941 & nan \\
7 & -327.33 & 0.24 & 0.0\% & 57/47 & 79 & 0.9934 & nan \\
8 & -179.77 & 0.24 & 0.0\% & 58/47 & 70 & 0.9927 & nan \\
9 & -351.41 & 0.24 & 0.0\% & 79/63 & 108 & 0.9916 & nan \\
10 & -298.78 & 0.24 & 0.0\% & 90/80 & 118 & 0.9904 & nan \\
11 & -78.47 & -0.37 & 0.0\% & 51/43 & 68 & 0.9898 & 2.386483 \\
12 & -159.12 & -0.73 & 0.0\% & 58/45 & 78 & 0.9890 & 2.190137 \\
13 & -116.79 & -0.70 & 0.0\% & 74/67 & 96 & 0.9880 & 2.108651 \\
14 & -90.14 & -0.43 & 0.0\% & 63/49 & 84 & 0.9872 & 1.892458 \\
15 & -168.22 & -0.04 & 0.0\% & 80/66 & 115 & 0.9861 & 1.636603 \\
16 & -243.62 & -0.08 & 0.0\% & 54/41 & 74 & 0.9853 & 1.637466 \\
17 & -280.66 & -0.16 & 0.0\% & 77/61 & 99 & 0.9844 & 1.787955 \\
18 & -175.90 & -0.02 & 0.0\% & 76/66 & 92 & 0.9834 & 1.366507 \\
19 & -160.71 & -0.10 & 0.0\% & 69/62 & 93 & 0.9825 & 1.552153 \\
20 & -33.68 & -0.14 & 5.0\% & 51/44 & 68 & 0.9819 & 1.523708 \\
21 & -159.27 & -0.18 & 4.8\% & 48/33 & 64 & 0.9812 & 1.428822 \\
22 & -153.45 & 0.18 & 4.5\% & 68/55 & 87 & 0.9804 & 1.356763 \\
23 & -175.76 & -0.30 & 4.3\% & 54/49 & 68 & 0.9797 & 1.635212 \\
24 & -132.43 & 0.00 & 4.2\% & 54/46 & 75 & 0.9789 & 1.258110 \\
25 & -120.53 & 0.20 & 4.0\% & 63/58 & 78 & 0.9782 & 1.202664 \\
26 & -93.94 & 0.11 & 3.8\% & 49/38 & 65 & 0.9775 & 1.224705 \\
27 & -295.67 & 0.18 & 3.7\% & 62/50 & 76 & 0.9768 & 1.157043 \\
28 & -355.29 & 0.13 & 3.6\% & 81/71 & 105 & 0.9757 & 1.289707 \\
29 & -108.34 & 0.60 & 3.4\% & 48/41 & 63 & 0.9751 & 2.039138 \\
30 & -192.05 & 0.29 & 3.3\% & 49/43 & 74 & 0.9744 & 1.637235 \\
31 & -336.50 & 0.84 & 3.2\% & 104/96 & 137 & 0.9730 & 1.282972 \\
32 & -135.93 & 0.57 & 3.1\% & 69/64 & 89 & 0.9721 & 1.419130 \\
33 & -144.84 & 0.71 & 3.0\% & 99/88 & 126 & 0.9709 & 1.328042 \\
34 & -188.46 & 0.35 & 2.9\% & 88/76 & 114 & 0.9698 & 1.622996 \\
35 & -33.14 & 0.25 & 2.9\% & 74/62 & 104 & 0.9687 & 1.409984 \\
36 & -79.37 & 0.42 & 2.8\% & 108/101 & 143 & 0.9673 & 1.437567 \\
37 & -118.32 & 0.63 & 2.7\% & 56/43 & 72 & 0.9666 & 1.540107 \\
38 & -180.27 & 0.42 & 2.6\% & 47/34 & 58 & 0.9660 & 1.679823 \\
39 & -111.14 & 1.17 & 2.6\% & 60/55 & 76 & 0.9653 & 1.555382 \\
40 & -171.94 & 0.83 & 2.5\% & 93/74 & 112 & 0.9642 & 1.663554 \\
41 & -122.42 & 0.96 & 2.4\% & 88/76 & 119 & 0.9630 & 1.392620 \\
42 & -135.08 & 0.98 & 2.4\% & 64/56 & 81 & 0.9622 & 1.438564 \\
43 & -122.99 & 0.71 & 2.3\% & 74/63 & 102 & 0.9612 & 1.748344 \\
44 & -188.73 & 0.67 & 2.3\% & 79/72 & 108 & 0.9601 & 1.445632 \\
45 & -19.38 & 0.97 & 4.4\% & 56/49 & 72 & 0.9594 & 1.604415 \\
46 & -150.89 & 1.00 & 4.3\% & 74/62 & 94 & 0.9585 & 1.471473 \\
47 & -139.13 & 0.55 & 4.3\% & 72/65 & 94 & 0.9575 & 1.416374 \\
48 & -477.24 & 0.94 & 4.2\% & 83/67 & 104 & 0.9565 & 1.389513 \\
49 & -244.74 & 0.83 & 4.1\% & 105/87 & 126 & 0.9553 & 1.377756 \\
50 & -116.23 & 1.23 & 4.0\% & 50/44 & 66 & 0.9546 & 1.647607 \\
51 & -149.50 & 1.31 & 3.9\% & 64/57 & 88 & 0.9537 & 1.351277 \\
52 & -370.78 & 1.63 & 3.8\% & 99/84 & 122 & 0.9525 & 1.480450 \\
53 & -194.47 & 1.42 & 3.8\% & 79/66 & 107 & 0.9515 & 1.238991 \\
54 & -128.05 & 1.30 & 3.7\% & 41/33 & 56 & 0.9509 & 1.682211 \\
55 & -113.27 & 1.02 & 3.6\% & 56/50 & 79 & 0.9501 & 1.474609 \\
56 & -137.83 & 1.47 & 3.6\% & 77/74 & 98 & 0.9492 & 1.430941 \\
57 & -68.91 & 1.48 & 3.5\% & 51/42 & 66 & 0.9485 & 1.502843 \\
58 & -136.61 & 1.67 & 3.4\% & 80/67 & 109 & 0.9474 & 1.694837 \\
59 & -156.19 & 1.34 & 3.4\% & 46/38 & 62 & 0.9468 & 1.552294 \\
60 & -164.89 & 1.40 & 3.3\% & 69/60 & 96 & 0.9459 & 1.267330 \\
61 & -118.96 & 1.40 & 3.3\% & 54/42 & 69 & 0.9452 & 1.568290 \\
62 & -141.97 & 1.64 & 3.2\% & 46/38 & 67 & 0.9445 & 1.489827 \\
63 & -292.10 & 1.36 & 3.2\% & 81/70 & 112 & 0.9434 & 1.364502 \\
64 & -141.48 & 1.57 & 3.1\% & 90/82 & 121 & 0.9422 & 1.401506 \\
65 & -149.22 & 1.54 & 3.1\% & 65/55 & 90 & 0.9413 & 1.500160 \\
66 & -142.79 & 2.73 & 3.0\% & 56/45 & 77 & 0.9406 & 1.406897 \\
67 & -164.01 & 2.26 & 3.0\% & 79/68 & 105 & 0.9395 & 1.487290 \\
68 & -211.88 & 2.48 & 2.9\% & 86/77 & 109 & 0.9384 & 1.447779 \\
69 & -156.62 & 2.33 & 2.9\% & 67/57 & 90 & 0.9376 & 1.395930 \\
70 & -252.11 & 2.77 & 2.9\% & 65/58 & 94 & 0.9366 & 1.319399 \\
71 & -126.05 & 2.07 & 2.8\% & 72/65 & 98 & 0.9356 & 1.279171 \\
72 & -82.59 & 2.47 & 2.8\% & 101/85 & 127 & 0.9344 & 1.431185 \\
73 & -123.53 & 2.69 & 2.7\% & 48/40 & 70 & 0.9337 & 1.417930 \\
74 & -244.55 & 2.82 & 2.7\% & 59/48 & 75 & 0.9330 & 1.472802 \\
75 & -113.60 & 2.77 & 2.7\% & 41/34 & 63 & 0.9323 & 1.522064 \\
76 & -125.08 & 2.73 & 2.6\% & 64/52 & 83 & 0.9315 & 1.256072 \\
77 & -134.89 & 2.23 & 2.6\% & 64/51 & 79 & 0.9307 & 1.350415 \\
78 & -114.20 & 3.41 & 2.6\% & 60/54 & 83 & 0.9299 & 1.532914 \\
79 & -111.96 & 3.06 & 2.5\% & 45/38 & 69 & 0.9292 & 1.401979 \\
80 & -38.14 & 3.04 & 3.8\% & 62/53 & 75 & 0.9285 & 1.410024 \\
81 & -99.86 & 3.42 & 3.7\% & 46/35 & 64 & 0.9278 & 1.368483 \\
82 & -150.59 & 3.44 & 3.7\% & 83/62 & 103 & 0.9268 & 1.484184 \\
83 & -130.46 & 3.30 & 3.6\% & 60/53 & 78 & 0.9261 & 1.294035 \\
84 & -109.02 & 4.11 & 3.6\% & 55/47 & 70 & 0.9254 & 1.841997 \\
85 & -155.14 & 3.85 & 3.5\% & 70/58 & 95 & 0.9244 & 1.523407 \\
86 & -122.62 & 3.84 & 3.5\% & 72/63 & 89 & 0.9235 & 1.680310 \\
87 & -88.16 & 4.11 & 3.4\% & 84/67 & 98 & 0.9226 & 1.268978 \\
88 & -121.49 & 4.06 & 3.4\% & 49/44 & 62 & 0.9220 & 1.432478 \\
89 & -162.88 & 3.77 & 3.4\% & 47/42 & 60 & 0.9214 & 1.635413 \\
90 & -234.93 & 5.17 & 3.3\% & 74/63 & 89 & 0.9205 & 1.454705 \\
91 & -156.30 & 4.61 & 3.3\% & 89/74 & 111 & 0.9194 & 1.369547 \\
92 & -118.90 & 4.75 & 3.3\% & 57/50 & 75 & 0.9186 & 1.651048 \\
93 & -192.76 & 4.98 & 3.2\% & 80/68 & 101 & 0.9176 & 1.418183 \\
94 & -137.11 & 4.95 & 3.2\% & 98/89 & 125 & 0.9164 & 1.570037 \\
95 & -157.66 & 5.36 & 3.2\% & 70/63 & 96 & 0.9155 & 1.596823 \\
96 & -122.84 & 4.99 & 3.1\% & 84/76 & 121 & 0.9143 & 1.740050 \\
97 & -329.45 & 5.38 & 3.1\% & 89/78 & 102 & 0.9132 & 1.292990 \\
98 & -104.62 & 5.45 & 3.1\% & 52/45 & 68 & 0.9126 & 1.278991 \\
99 & -98.92 & 5.13 & 3.0\% & 52/47 & 67 & 0.9119 & 1.437163 \\
100 & -131.15 & 5.61 & 3.0\% & 85/74 & 115 & 0.9108 & 1.539776 \\
101 & -89.89 & 6.11 & 3.0\% & 52/46 & 65 & 0.9101 & 1.674902 \\
102 & -192.47 & 6.15 & 3.0\% & 74/66 & 92 & 0.9092 & 1.449114 \\
103 & -181.96 & 6.03 & 3.0\% & 81/69 & 101 & 0.9082 & 1.412471 \\
104 & -90.94 & 5.76 & 3.0\% & 49/45 & 63 & 0.9076 & 1.707469 \\
105 & -222.74 & 6.78 & 3.0\% & 189/158 & 247 & 0.9051 & 1.741155 \\
106 & -139.61 & 7.21 & 3.0\% & 87/76 & 110 & 0.9041 & 1.434954 \\
107 & -42.84 & 6.66 & 3.0\% & 62/54 & 77 & 0.9033 & 1.432893 \\
108 & -47.95 & 6.36 & 4.0\% & 70/61 & 85 & 0.9025 & 1.686290 \\
109 & -195.82 & 6.45 & 4.0\% & 102/80 & 131 & 0.9012 & 1.610831 \\
110 & -192.93 & 7.43 & 4.0\% & 70/58 & 85 & 0.9003 & 1.678899 \\
111 & -82.56 & 7.39 & 4.0\% & 41/36 & 60 & 0.8997 & 1.546077 \\
112 & -252.61 & 7.73 & 4.0\% & 99/79 & 117 & 0.8986 & 1.581710 \\
113 & -126.41 & 7.25 & 4.0\% & 70/61 & 82 & 0.8978 & 1.506131 \\
114 & -5.12 & 6.98 & 4.0\% & 102/87 & 118 & 0.8966 & 1.416142 \\
115 & -133.71 & 8.01 & 4.0\% & 76/63 & 99 & 0.8956 & 1.641036 \\
116 & -115.56 & 8.23 & 4.0\% & 95/83 & 121 & 0.8944 & 1.775353 \\
117 & -120.39 & 7.87 & 4.0\% & 58/51 & 81 & 0.8936 & 1.640497 \\
118 & -144.10 & 7.62 & 4.0\% & 51/43 & 63 & 0.8930 & 1.404265 \\
119 & -156.34 & 8.01 & 4.0\% & 69/63 & 87 & 0.8921 & 1.615148 \\
120 & -115.73 & 8.44 & 3.0\% & 98/85 & 116 & 0.8910 & 1.352230 \\
121 & -90.91 & 8.65 & 3.0\% & 53/46 & 63 & 0.8903 & 1.565654 \\
122 & -26.29 & 8.54 & 3.0\% & 114/95 & 146 & 0.8889 & 1.483326 \\
123 & -98.83 & 8.75 & 3.0\% & 45/41 & 64 & 0.8883 & 1.677827 \\
124 & -141.38 & 8.61 & 3.0\% & 51/42 & 65 & 0.8876 & 1.446950 \\
125 & -99.61 & 8.44 & 3.0\% & 77/64 & 98 & 0.8867 & 1.572127 \\
126 & -131.89 & 9.38 & 3.0\% & 86/73 & 115 & 0.8855 & 1.565691 \\
127 & -144.22 & 9.46 & 3.0\% & 66/52 & 89 & 0.8846 & 1.735563 \\
128 & -136.63 & 9.93 & 3.0\% & 83/69 & 93 & 0.8837 & 1.688920 \\
129 & -141.34 & 9.80 & 3.0\% & 76/66 & 102 & 0.8827 & 1.649844 \\
130 & -82.73 & 9.89 & 4.0\% & 61/52 & 86 & 0.8819 & 1.726284 \\
131 & -106.03 & 10.17 & 4.0\% & 83/76 & 101 & 0.8809 & 1.637935 \\
132 & -134.98 & 9.93 & 4.0\% & 72/63 & 93 & 0.8799 & 1.275780 \\
133 & -102.01 & 10.14 & 4.0\% & 50/44 & 65 & 0.8793 & 1.306813 \\
134 & -24.89 & 9.92 & 5.0\% & 86/70 & 103 & 0.8783 & 1.339044 \\
135 & -131.58 & 10.07 & 5.0\% & 75/65 & 94 & 0.8773 & 1.576257 \\
136 & -18.48 & 10.29 & 6.0\% & 53/41 & 70 & 0.8766 & 1.268129 \\
137 & -205.93 & 10.52 & 6.0\% & 87/72 & 112 & 0.8755 & 1.615869 \\
138 & -127.08 & 10.26 & 6.0\% & 82/72 & 102 & 0.8745 & 1.591118 \\
139 & -127.53 & 10.70 & 6.0\% & 50/44 & 62 & 0.8739 & 1.496724 \\
140 & -67.32 & 10.14 & 6.0\% & 94/79 & 115 & 0.8728 & 1.424735 \\
141 & -80.70 & 10.45 & 6.0\% & 64/52 & 88 & 0.8719 & 1.553341 \\
142 & -191.75 & 10.66 & 6.0\% & 106/88 & 133 & 0.8706 & 1.456855 \\
143 & -125.77 & 10.70 & 6.0\% & 66/57 & 88 & 0.8697 & 1.467778 \\
144 & -141.11 & 10.31 & 6.0\% & 53/47 & 69 & 0.8690 & 1.367368 \\
145 & -129.10 & 10.43 & 5.0\% & 79/67 & 105 & 0.8680 & 1.552796 \\
146 & -117.71 & 10.69 & 5.0\% & 55/43 & 76 & 0.8672 & 1.371270 \\
147 & -116.53 & 10.50 & 5.0\% & 59/49 & 76 & 0.8665 & 1.269878 \\
148 & -95.96 & 10.70 & 5.0\% & 99/88 & 129 & 0.8652 & 1.509813 \\
149 & -89.75 & 11.05 & 5.0\% & 50/42 & 63 & 0.8646 & 1.443166 \\
150 & -3.34 & 10.63 & 6.0\% & 52/46 & 70 & 0.8639 & 1.450981 \\
151 & -117.29 & 11.01 & 6.0\% & 57/44 & 79 & 0.8631 & 1.746479 \\
152 & -111.21 & 10.86 & 6.0\% & 58/45 & 80 & 0.8623 & 1.359541 \\
153 & -129.73 & 11.36 & 6.0\% & 76/62 & 100 & 0.8613 & 1.652116 \\
154 & -128.91 & 11.42 & 6.0\% & 76/62 & 96 & 0.8604 & 1.336846 \\
155 & -69.57 & 10.33 & 6.0\% & 49/44 & 65 & 0.8597 & 1.427486 \\
156 & -142.92 & 10.55 & 6.0\% & 68/61 & 99 & 0.8588 & 1.678102 \\
157 & -148.70 & 11.13 & 6.0\% & 51/42 & 58 & 0.8582 & 1.218731 \\
158 & -75.27 & 11.11 & 6.0\% & 43/38 & 63 & 0.8576 & 1.529061 \\
159 & -66.74 & 11.36 & 6.0\% & 51/39 & 65 & 0.8569 & 1.792679 \\
160 & -89.30 & 10.37 & 6.0\% & 47/42 & 67 & 0.8563 & 1.485745 \\
161 & -125.99 & 10.89 & 6.0\% & 72/66 & 85 & 0.8554 & 1.555554 \\
162 & -136.18 & 10.86 & 6.0\% & 78/72 & 100 & 0.8544 & 1.680652 \\
163 & -133.11 & 10.81 & 6.0\% & 51/46 & 74 & 0.8537 & 1.755086 \\
164 & -67.10 & 10.76 & 6.0\% & 58/48 & 71 & 0.8530 & 1.993776 \\
165 & -124.29 & 10.97 & 6.0\% & 68/62 & 87 & 0.8521 & 1.079844 \\
166 & -32.05 & 10.75 & 7.0\% & 88/75 & 118 & 0.8510 & 1.433543 \\
167 & -117.38 & 10.79 & 7.0\% & 92/79 & 107 & 0.8499 & 1.702143 \\
168 & -132.02 & 11.26 & 7.0\% & 55/46 & 71 & 0.8492 & 1.625924 \\
169 & -97.83 & 11.47 & 7.0\% & 78/63 & 102 & 0.8482 & 1.387736 \\
170 & -121.96 & 10.41 & 7.0\% & 61/48 & 78 & 0.8474 & 1.212189 \\
171 & -107.72 & 10.80 & 7.0\% & 73/63 & 102 & 0.8464 & 1.842333 \\
172 & -115.52 & 11.17 & 7.0\% & 56/48 & 71 & 0.8457 & 1.449686 \\
173 & -12.58 & 10.76 & 7.0\% & 63/53 & 80 & 0.8449 & 1.412804 \\
174 & -102.68 & 10.77 & 7.0\% & 61/57 & 76 & 0.8442 & 1.344860 \\
175 & -394.33 & 11.37 & 7.0\% & 78/72 & 105 & 0.8431 & 1.245979 \\
176 & -133.48 & 11.10 & 7.0\% & 62/56 & 86 & 0.8423 & 1.370241 \\
177 & -93.64 & 11.49 & 7.0\% & 73/66 & 92 & 0.8414 & 1.489331 \\
178 & -139.51 & 11.22 & 7.0\% & 52/42 & 66 & 0.8407 & 1.580868 \\
179 & -136.60 & 11.55 & 7.0\% & 71/53 & 97 & 0.8397 & 1.325594 \\
180 & -132.83 & 11.50 & 6.0\% & 79/64 & 99 & 0.8388 & 1.417850 \\
181 & -173.78 & 11.85 & 6.0\% & 70/62 & 90 & 0.8379 & 1.349030 \\
182 & -279.45 & 11.32 & 6.0\% & 77/66 & 100 & 0.8369 & 1.375255 \\
183 & -143.08 & 11.86 & 6.0\% & 74/66 & 93 & 0.8360 & 1.483475 \\
184 & -117.20 & 11.38 & 6.0\% & 48/39 & 70 & 0.8353 & 1.412628 \\
185 & -131.84 & 10.82 & 6.0\% & 87/73 & 106 & 0.8342 & 1.252107 \\
186 & -90.69 & 10.99 & 6.0\% & 103/92 & 127 & 0.8330 & 1.307405 \\
187 & -123.88 & 10.84 & 6.0\% & 74/63 & 101 & 0.8320 & 1.416536 \\
188 & -147.75 & 10.36 & 6.0\% & 53/43 & 78 & 0.8312 & 1.304858 \\
189 & -99.55 & 10.42 & 6.0\% & 80/78 & 102 & 0.8302 & 1.390492 \\
190 & -178.08 & 10.72 & 6.0\% & 58/51 & 68 & 0.8295 & 1.368212 \\
191 & -274.04 & 10.81 & 6.0\% & 69/63 & 88 & 0.8286 & 1.272274 \\
192 & -180.70 & 10.33 & 6.0\% & 94/82 & 115 & 0.8275 & 1.306594 \\
193 & -108.46 & 10.28 & 6.0\% & 55/45 & 79 & 0.8267 & 1.408997 \\
194 & -142.84 & 10.77 & 6.0\% & 113/94 & 142 & 0.8253 & 1.598896 \\
195 & -126.38 & 10.61 & 6.0\% & 72/61 & 91 & 0.8244 & 1.455205 \\
196 & -75.24 & 10.08 & 6.0\% & 48/41 & 65 & 0.8238 & 1.426783 \\
197 & -110.52 & 9.79 & 6.0\% & 64/57 & 87 & 0.8229 & 1.426670 \\
198 & -92.43 & 10.68 & 6.0\% & 66/55 & 81 & 0.8221 & 1.393502 \\
199 & -15.33 & 9.92 & 7.0\% & 50/39 & 69 & 0.8214 & 1.446979 \\
200 & -175.29 & 10.50 & 7.0\% & 92/75 & 106 & 0.8204 & 1.349295 \\
201 & -104.24 & 10.43 & 7.0\% & 94/83 & 115 & 0.8192 & 1.478711 \\
202 & -189.64 & 10.01 & 7.0\% & 65/48 & 86 & 0.8184 & 1.489841 \\
203 & -110.18 & 10.01 & 7.0\% & 54/44 & 67 & 0.8177 & 1.248763 \\
204 & -112.10 & 10.19 & 7.0\% & 63/57 & 78 & 0.8169 & 1.682364 \\
205 & -122.90 & 9.28 & 7.0\% & 112/93 & 135 & 0.8156 & 1.539801 \\
206 & 15.88 & 9.92 & 7.0\% & 56/47 & 69 & 0.8149 & 1.439841 \\
207 & -94.73 & 9.92 & 7.0\% & 77/67 & 100 & 0.8139 & 1.564357 \\
208 & -103.72 & 10.09 & 6.0\% & 51/42 & 70 & 0.8132 & 1.350598 \\
209 & -81.84 & 9.94 & 6.0\% & 50/40 & 65 & 0.8126 & 1.298051 \\
210 & -114.22 & 9.12 & 6.0\% & 101/87 & 124 & 0.8114 & 1.562407 \\
211 & -96.80 & 9.80 & 6.0\% & 76/66 & 95 & 0.8104 & 1.470561 \\
212 & -84.09 & 9.16 & 6.0\% & 58/51 & 65 & 0.8098 & 1.517450 \\
213 & -132.95 & 9.94 & 6.0\% & 69/61 & 90 & 0.8089 & 1.723398 \\
214 & -145.20 & 9.83 & 6.0\% & 54/46 & 84 & 0.8081 & 1.551081 \\
215 & -43.28 & 9.82 & 6.0\% & 53/46 & 68 & 0.8074 & 1.823296 \\
216 & -123.47 & 8.72 & 6.0\% & 59/50 & 76 & 0.8066 & 1.700435 \\
217 & -89.41 & 8.92 & 6.0\% & 89/75 & 114 & 0.8055 & 1.898866 \\
218 & -133.98 & 8.60 & 6.0\% & 79/70 & 95 & 0.8046 & 1.929999 \\
219 & -147.79 & 8.15 & 6.0\% & 65/56 & 84 & 0.8037 & 1.955397 \\
220 & -119.67 & 9.02 & 6.0\% & 69/53 & 85 & 0.8029 & 1.696192 \\
221 & -157.43 & 9.12 & 6.0\% & 67/53 & 88 & 0.8020 & 1.235788 \\
222 & -90.34 & 8.36 & 6.0\% & 46/38 & 59 & 0.8014 & 2.064327 \\
223 & -81.41 & 8.50 & 6.0\% & 49/42 & 61 & 0.8008 & 2.054167 \\
224 & -128.26 & 9.37 & 6.0\% & 65/52 & 87 & 0.8000 & 1.752974 \\
225 & -107.04 & 8.54 & 6.0\% & 75/68 & 97 & 0.7990 & 1.832046 \\
226 & -77.53 & 8.28 & 6.0\% & 46/38 & 68 & 0.7983 & 1.689542 \\
227 & -116.36 & 8.38 & 6.0\% & 50/41 & 68 & 0.7977 & 1.636362 \\
228 & -115.13 & 8.03 & 6.0\% & 66/57 & 98 & 0.7967 & 1.877510 \\
229 & -119.83 & 8.42 & 6.0\% & 77/67 & 102 & 0.7957 & 1.663348 \\
230 & -165.30 & 8.30 & 5.0\% & 108/91 & 130 & 0.7944 & 2.068948 \\
231 & 38.00 & 8.23 & 5.0\% & 48/42 & 67 & 0.7937 & 1.663620 \\
232 & -91.59 & 8.56 & 5.0\% & 63/50 & 83 & 0.7929 & 1.896833 \\
233 & -119.44 & 9.13 & 5.0\% & 64/56 & 82 & 0.7921 & 1.783605 \\
234 & 2.87 & 8.62 & 4.0\% & 59/51 & 81 & 0.7913 & 1.785562 \\
235 & -97.05 & 8.83 & 4.0\% & 75/65 & 99 & 0.7903 & 1.649347 \\
236 & -119.54 & 9.09 & 3.0\% & 68/60 & 81 & 0.7895 & 1.928924 \\
237 & -78.49 & 8.98 & 3.0\% & 64/54 & 80 & 0.7887 & 1.690285 \\
238 & -115.67 & 8.73 & 3.0\% & 97/87 & 118 & 0.7876 & 1.955161 \\
239 & -186.17 & 7.86 & 3.0\% & 87/74 & 120 & 0.7864 & 1.873306 \\
240 & -46.94 & 8.58 & 4.0\% & 90/77 & 112 & 0.7853 & 1.664483 \\
241 & -99.21 & 8.28 & 4.0\% & 56/48 & 69 & 0.7846 & 1.571158 \\
242 & -134.37 & 8.56 & 4.0\% & 59/52 & 84 & 0.7837 & 2.034220 \\
243 & -144.97 & 8.67 & 4.0\% & 66/56 & 90 & 0.7829 & 1.748567 \\
244 & -134.93 & 8.74 & 4.0\% & 81/67 & 110 & 0.7818 & 1.617245 \\
245 & -2.40 & 8.45 & 5.0\% & 64/57 & 76 & 0.7810 & 1.964311 \\
246 & -102.67 & 8.70 & 5.0\% & 61/47 & 79 & 0.7802 & 1.924018 \\
247 & -75.14 & 8.49 & 5.0\% & 52/44 & 65 & 0.7796 & 2.225628 \\
248 & -110.41 & 8.54 & 5.0\% & 65/57 & 88 & 0.7787 & 1.805343 \\
249 & -139.39 & 8.78 & 5.0\% & 70/57 & 85 & 0.7779 & 1.739708 \\
250 & -101.74 & 7.19 & 4.0\% & 69/59 & 99 & 0.7769 & 1.706151 \\
251 & -102.34 & 7.67 & 4.0\% & 60/48 & 76 & 0.7761 & 1.725585 \\
252 & -125.25 & 8.39 & 4.0\% & 60/51 & 78 & 0.7754 & 2.201456 \\
253 & -69.37 & 7.72 & 4.0\% & 48/43 & 56 & 0.7748 & 1.786990 \\
254 & -138.20 & 7.94 & 4.0\% & 88/73 & 113 & 0.7737 & 1.781611 \\
255 & -126.95 & 8.38 & 4.0\% & 86/75 & 107 & 0.7726 & 1.643079 \\
256 & -115.81 & 7.26 & 4.0\% & 82/76 & 108 & 0.7716 & 1.853297 \\
257 & -167.92 & 7.36 & 4.0\% & 84/70 & 110 & 0.7705 & 1.988082 \\
258 & -147.94 & 7.33 & 4.0\% & 73/68 & 95 & 0.7695 & 1.960925 \\
259 & -107.43 & 7.37 & 4.0\% & 67/62 & 84 & 0.7687 & 1.752842 \\
260 & -179.20 & 6.80 & 4.0\% & 72/54 & 89 & 0.7678 & 1.822595 \\
261 & -22.89 & 6.27 & 5.0\% & 46/40 & 65 & 0.7672 & 1.966515 \\
262 & -138.06 & 6.28 & 5.0\% & 57/48 & 81 & 0.7664 & 1.864911 \\
263 & -2.00 & 6.08 & 5.0\% & 59/50 & 75 & 0.7656 & 2.004632 \\
264 & -137.72 & 6.37 & 5.0\% & 74/63 & 95 & 0.7647 & 1.756917 \\
265 & -118.17 & 6.21 & 5.0\% & 82/66 & 103 & 0.7637 & 1.849854 \\
266 & -86.35 & 6.77 & 4.0\% & 88/72 & 112 & 0.7626 & 1.797844 \\
267 & -125.65 & 5.42 & 4.0\% & 81/69 & 104 & 0.7615 & 1.934352 \\
268 & -72.95 & 5.44 & 4.0\% & 56/49 & 70 & 0.7608 & 1.936458 \\
269 & -116.54 & 5.39 & 4.0\% & 90/78 & 112 & 0.7597 & 1.737569 \\
270 & -189.14 & 6.04 & 4.0\% & 85/72 & 119 & 0.7586 & 1.785241 \\
271 & -138.95 & 5.34 & 4.0\% & 74/60 & 101 & 0.7576 & 1.715611 \\
272 & -94.49 & 4.89 & 4.0\% & 60/54 & 79 & 0.7568 & 1.803476 \\
273 & -137.23 & 4.82 & 4.0\% & 79/73 & 99 & 0.7558 & 1.852944 \\
274 & -127.28 & 5.12 & 4.0\% & 88/81 & 114 & 0.7547 & 1.783857 \\
275 & -103.64 & 4.77 & 4.0\% & 68/61 & 86 & 0.7538 & 1.869400 \\
276 & -84.19 & 5.37 & 4.0\% & 54/45 & 70 & 0.7531 & 1.662562 \\
277 & -114.72 & 4.27 & 4.0\% & 93/76 & 109 & 0.7520 & 2.080872 \\
278 & -188.09 & 4.16 & 4.0\% & 62/52 & 78 & 0.7513 & 2.022087 \\
279 & -216.23 & 4.39 & 4.0\% & 93/78 & 111 & 0.7502 & 1.984946 \\
280 & -72.05 & 4.28 & 4.0\% & 57/51 & 66 & 0.7495 & 2.276519 \\
281 & -99.37 & 4.03 & 4.0\% & 63/58 & 85 & 0.7487 & 1.980543 \\
282 & -92.61 & 3.88 & 4.0\% & 60/48 & 78 & 0.7479 & 2.019114 \\
283 & -136.17 & 4.04 & 4.0\% & 55/46 & 86 & 0.7471 & 1.793419 \\
284 & -91.00 & 3.97 & 4.0\% & 89/76 & 123 & 0.7458 & 1.801672 \\
285 & -106.81 & 3.79 & 4.0\% & 48/40 & 73 & 0.7451 & 2.003878 \\
286 & -94.01 & 4.23 & 4.0\% & 53/51 & 71 & 0.7444 & 2.007622 \\
287 & -87.44 & 4.35 & 4.0\% & 87/74 & 107 & 0.7434 & 2.280867 \\
288 & -92.22 & 4.70 & 4.0\% & 52/40 & 71 & 0.7426 & 1.944921 \\
289 & -107.00 & 3.74 & 4.0\% & 74/58 & 91 & 0.7417 & 2.011357 \\
290 & -107.43 & 3.22 & 4.0\% & 56/50 & 83 & 0.7409 & 1.932008 \\
291 & -79.33 & 2.84 & 4.0\% & 84/70 & 110 & 0.7398 & 2.243563 \\
292 & -109.77 & 3.49 & 4.0\% & 67/57 & 80 & 0.7390 & 1.876652 \\
293 & -124.67 & 3.75 & 4.0\% & 63/58 & 84 & 0.7382 & 2.113985 \\
294 & -118.38 & 2.75 & 4.0\% & 78/66 & 98 & 0.7372 & 1.956947 \\
295 & -143.76 & 2.49 & 4.0\% & 69/58 & 88 & 0.7364 & 2.052511 \\
296 & -106.21 & 2.69 & 4.0\% & 73/63 & 95 & 0.7354 & 1.692481 \\
297 & -115.73 & 3.07 & 4.0\% & 66/55 & 81 & 0.7346 & 1.771754 \\
298 & -117.34 & 2.76 & 4.0\% & 54/37 & 72 & 0.7339 & 1.881097 \\
299 & -89.59 & 2.47 & 3.0\% & 53/42 & 70 & 0.7332 & 1.947227 \\
300 & -2.19 & 1.90 & 4.0\% & 74/63 & 109 & 0.7321 & 1.941613 \\
301 & -106.13 & 1.79 & 4.0\% & 75/70 & 98 & 0.7312 & 2.412922 \\
302 & -74.44 & 2.23 & 4.0\% & 55/48 & 71 & 0.7305 & 2.603833 \\
303 & -75.15 & 2.41 & 4.0\% & 72/59 & 101 & 0.7295 & 1.972003 \\
304 & -25.73 & 2.35 & 4.0\% & 56/47 & 69 & 0.7288 & 1.903994 \\
305 & -204.88 & 2.49 & 4.0\% & 76/67 & 103 & 0.7278 & 1.866876 \\
306 & -108.63 & 1.27 & 4.0\% & 67/55 & 88 & 0.7269 & 1.947352 \\
307 & -90.68 & 1.30 & 4.0\% & 88/77 & 117 & 0.7257 & 2.050842 \\
308 & -110.86 & 1.50 & 4.0\% & 68/55 & 83 & 0.7249 & 1.638188 \\
309 & -20.39 & 1.81 & 5.0\% & 60/55 & 82 & 0.7241 & 1.972457 \\
310 & -89.00 & 1.35 & 5.0\% & 66/55 & 80 & 0.7233 & 2.685224 \\
311 & -124.48 & 0.90 & 5.0\% & 78/68 & 109 & 0.7222 & 2.140365 \\
312 & -94.62 & 0.83 & 5.0\% & 52/42 & 71 & 0.7215 & 2.299327 \\
313 & -108.63 & 0.91 & 5.0\% & 60/53 & 80 & 0.7207 & 2.041754 \\
314 & -47.18 & 0.64 & 6.0\% & 39/32 & 62 & 0.7201 & 1.818984 \\
315 & -131.95 & 0.69 & 6.0\% & 62/51 & 74 & 0.7194 & 2.056831 \\
316 & -169.51 & 0.98 & 6.0\% & 75/66 & 100 & 0.7184 & 2.154809 \\
317 & -111.08 & 0.54 & 6.0\% & 68/56 & 76 & 0.7177 & 2.236956 \\
318 & -35.40 & 0.64 & 7.0\% & 60/50 & 84 & 0.7168 & 1.836063 \\
319 & -109.95 & 0.56 & 7.0\% & 66/57 & 81 & 0.7160 & 2.195182 \\
320 & -110.04 & 0.49 & 7.0\% & 70/61 & 88 & 0.7151 & 1.553355 \\
321 & -78.21 & 1.49 & 7.0\% & 101/87 & 130 & 0.7139 & 1.923665 \\
322 & -139.79 & 0.40 & 7.0\% & 72/65 & 101 & 0.7129 & 1.860809 \\
323 & -104.23 & 1.16 & 7.0\% & 65/56 & 80 & 0.7121 & 2.028235 \\
324 & -109.97 & 0.30 & 7.0\% & 85/74 & 99 & 0.7111 & 2.227439 \\
325 & -103.88 & 0.22 & 7.0\% & 57/53 & 67 & 0.7104 & 2.265546 \\
326 & -69.69 & 0.04 & 7.0\% & 59/53 & 69 & 0.7097 & 1.869105 \\
327 & -81.16 & 0.30 & 7.0\% & 67/52 & 79 & 0.7090 & 2.068484 \\
328 & -116.51 & -0.47 & 7.0\% & 69/63 & 91 & 0.7081 & 1.801446 \\
329 & -145.51 & -0.20 & 7.0\% & 63/54 & 73 & 0.7073 & 1.993966 \\
330 & -75.60 & -0.10 & 7.0\% & 79/70 & 95 & 0.7064 & 2.390815 \\
331 & -104.43 & -0.20 & 7.0\% & 53/45 & 72 & 0.7057 & 2.110758 \\
332 & -128.38 & 0.49 & 7.0\% & 79/67 & 96 & 0.7047 & 2.461856 \\
333 & -122.19 & 0.23 & 7.0\% & 86/73 & 115 & 0.7036 & 2.030703 \\
334 & -78.18 & -1.01 & 7.0\% & 69/58 & 94 & 0.7027 & 2.132227 \\
335 & -116.14 & -0.37 & 7.0\% & 77/66 & 106 & 0.7016 & 2.076934 \\
336 & -64.61 & -0.78 & 7.0\% & 62/49 & 77 & 0.7009 & 2.772930 \\
337 & -89.84 & -1.04 & 7.0\% & 57/51 & 76 & 0.7001 & 1.817834 \\
338 & -107.23 & -0.35 & 7.0\% & 54/46 & 66 & 0.6994 & 2.198281 \\
339 & -17.44 & -0.47 & 7.0\% & 104/93 & 133 & 0.6981 & 2.116340 \\
340 & -138.65 & -0.52 & 6.0\% & 61/54 & 76 & 0.6974 & 1.962794 \\
341 & -117.39 & -0.94 & 6.0\% & 79/66 & 92 & 0.6965 & 2.341204 \\
342 & -112.57 & -1.49 & 6.0\% & 64/57 & 86 & 0.6956 & 2.170988 \\
343 & -42.92 & -1.30 & 6.0\% & 51/43 & 67 & 0.6950 & 2.078996 \\
344 & -99.38 & -1.25 & 6.0\% & 85/72 & 120 & 0.6938 & 1.965393 \\
345 & -105.35 & -1.83 & 5.0\% & 66/56 & 83 & 0.6929 & 2.346100 \\
346 & -57.52 & -2.07 & 5.0\% & 75/60 & 110 & 0.6919 & 2.046996 \\
347 & -135.32 & -1.12 & 5.0\% & 66/57 & 82 & 0.6910 & 2.296311 \\
348 & -44.66 & -1.53 & 5.0\% & 61/55 & 77 & 0.6903 & 2.434074 \\
349 & -79.04 & -0.98 & 5.0\% & 58/48 & 68 & 0.6896 & 2.314190 \\
350 & -68.99 & -1.09 & 5.0\% & 78/64 & 121 & 0.6884 & 2.305848 \\
351 & -100.96 & -1.63 & 5.0\% & 52/47 & 65 & 0.6878 & 2.314345 \\
352 & -97.71 & -0.79 & 5.0\% & 64/52 & 82 & 0.6870 & 2.064232 \\
353 & -68.87 & -1.92 & 5.0\% & 88/77 & 119 & 0.6858 & 2.198974 \\
354 & -90.08 & -1.09 & 5.0\% & 63/57 & 83 & 0.6850 & 2.267534 \\
355 & -86.21 & -1.76 & 5.0\% & 53/47 & 75 & 0.6842 & 1.864667 \\
356 & -102.30 & -1.02 & 5.0\% & 53/48 & 70 & 0.6835 & 1.983293 \\
357 & -92.14 & -2.91 & 5.0\% & 84/75 & 104 & 0.6825 & 2.176815 \\
358 & -76.95 & -2.77 & 5.0\% & 48/38 & 64 & 0.6819 & 1.813678 \\
359 & -102.86 & -2.34 & 5.0\% & 68/59 & 78 & 0.6811 & 2.444903 \\
360 & -98.95 & -2.35 & 5.0\% & 69/58 & 84 & 0.6802 & 2.233517 \\
361 & -90.80 & -2.07 & 4.0\% & 46/39 & 62 & 0.6796 & 1.830283 \\
362 & -48.57 & -2.36 & 5.0\% & 78/64 & 95 & 0.6787 & 2.303985 \\
363 & -103.55 & -2.70 & 5.0\% & 77/65 & 96 & 0.6777 & 1.957133 \\
364 & -84.43 & -2.05 & 5.0\% & 63/59 & 74 & 0.6770 & 2.394387 \\
365 & -101.46 & -1.47 & 5.0\% & 79/70 & 113 & 0.6759 & 2.097140 \\
366 & -101.88 & -2.40 & 5.0\% & 76/58 & 96 & 0.6749 & 2.115573 \\
367 & -148.25 & -2.49 & 5.0\% & 71/60 & 90 & 0.6741 & 2.461953 \\
368 & -155.47 & -2.97 & 5.0\% & 82/68 & 98 & 0.6731 & 1.969755 \\
369 & -83.26 & -2.64 & 5.0\% & 51/45 & 58 & 0.6725 & 2.322549 \\
370 & -84.45 & -2.15 & 5.0\% & 56/46 & 76 & 0.6718 & 2.283960 \\
371 & -28.30 & -1.73 & 6.0\% & 81/70 & 98 & 0.6708 & 1.749963 \\
372 & -131.63 & -2.17 & 6.0\% & 75/67 & 100 & 0.6698 & 1.984048 \\
373 & -197.26 & -1.79 & 6.0\% & 85/78 & 105 & 0.6688 & 2.026279 \\
374 & -117.09 & -2.15 & 6.0\% & 59/55 & 79 & 0.6680 & 1.923406 \\
375 & -125.53 & -3.29 & 6.0\% & 80/69 & 112 & 0.6669 & 2.291135 \\
376 & -114.79 & -2.98 & 6.0\% & 63/53 & 84 & 0.6660 & 2.070814 \\
377 & -31.79 & -2.95 & 6.0\% & 72/60 & 81 & 0.6652 & 2.224700 \\
378 & -93.59 & -3.04 & 6.0\% & 74/61 & 91 & 0.6643 & 2.318149 \\
379 & -131.37 & -4.14 & 6.0\% & 88/80 & 115 & 0.6632 & 2.177648 \\
380 & -86.16 & -3.52 & 6.0\% & 72/62 & 105 & 0.6622 & 2.151074 \\
381 & -77.35 & -3.73 & 6.0\% & 74/63 & 91 & 0.6613 & 1.890655 \\
382 & -76.88 & -3.27 & 6.0\% & 75/69 & 92 & 0.6603 & 2.260427 \\
383 & -88.34 & -3.58 & 6.0\% & 101/88 & 129 & 0.6591 & 1.971006 \\
384 & -112.54 & -3.69 & 6.0\% & 78/61 & 112 & 0.6580 & 2.182478 \\
385 & -111.71 & -3.95 & 6.0\% & 50/44 & 69 & 0.6573 & 2.263710 \\
386 & -118.47 & -3.98 & 6.0\% & 88/78 & 117 & 0.6561 & 2.162028 \\
387 & -83.74 & -3.93 & 6.0\% & 74/64 & 101 & 0.6551 & 2.213275 \\
388 & -100.95 & -3.81 & 6.0\% & 66/58 & 86 & 0.6543 & 2.282869 \\
389 & -202.89 & -3.46 & 6.0\% & 109/98 & 136 & 0.6529 & 2.210840 \\
390 & -87.40 & -3.68 & 6.0\% & 97/82 & 123 & 0.6517 & 2.131545 \\
391 & -151.41 & -3.88 & 6.0\% & 54/48 & 80 & 0.6509 & 2.351941 \\
392 & -41.14 & -3.10 & 6.0\% & 61/54 & 75 & 0.6502 & 2.079045 \\
393 & -82.63 & -3.37 & 6.0\% & 88/76 & 126 & 0.6489 & 2.302276 \\
394 & -49.82 & -3.41 & 6.0\% & 93/79 & 113 & 0.6478 & 1.895729 \\
395 & -129.58 & -4.39 & 6.0\% & 54/46 & 76 & 0.6470 & 2.021301 \\
396 & -23.40 & -4.98 & 6.0\% & 53/47 & 70 & 0.6464 & 1.753484 \\
397 & -92.00 & -4.04 & 6.0\% & 89/78 & 117 & 0.6452 & 2.411097 \\
398 & -57.22 & -4.55 & 6.0\% & 70/63 & 77 & 0.6444 & 1.828591 \\
399 & -66.52 & -3.84 & 6.0\% & 67/55 & 78 & 0.6437 & 2.070732 \\
400 & -71.17 & -4.38 & 5.0\% & 64/55 & 88 & 0.6428 & 1.887840 \\
401 & -74.73 & -4.90 & 5.0\% & 49/44 & 56 & 0.6422 & 1.999833 \\
402 & -70.83 & -4.90 & 5.0\% & 85/68 & 116 & 0.6411 & 2.337375 \\
403 & -74.95 & -5.04 & 5.0\% & 59/51 & 72 & 0.6404 & 2.077291 \\
404 & 16.25 & -3.80 & 6.0\% & 52/44 & 68 & 0.6397 & 2.051040 \\
405 & -65.50 & -4.72 & 6.0\% & 54/47 & 70 & 0.6390 & 2.476810 \\
406 & -82.55 & -4.00 & 6.0\% & 56/48 & 79 & 0.6382 & 2.006606 \\
407 & -107.84 & -4.35 & 6.0\% & 66/55 & 86 & 0.6374 & 1.933678 \\
408 & -184.69 & -4.06 & 6.0\% & 96/77 & 127 & 0.6361 & 1.899450 \\
409 & -149.16 & -3.81 & 5.0\% & 90/73 & 128 & 0.6348 & 1.859974 \\
410 & -125.76 & -4.54 & 5.0\% & 84/75 & 105 & 0.6338 & 2.290539 \\
411 & -127.37 & -4.37 & 5.0\% & 72/55 & 92 & 0.6329 & 2.290295 \\
412 & -92.10 & -5.39 & 5.0\% & 75/61 & 110 & 0.6318 & 1.871263 \\
413 & -78.28 & -4.95 & 5.0\% & 76/63 & 107 & 0.6307 & 2.141004 \\
414 & -94.12 & -5.17 & 4.0\% & 74/63 & 89 & 0.6299 & 1.956309 \\
415 & -131.90 & -5.80 & 4.0\% & 51/42 & 61 & 0.6293 & 2.158304 \\
416 & -79.18 & -5.10 & 4.0\% & 65/52 & 80 & 0.6285 & 1.950994 \\
417 & -67.43 & -4.63 & 4.0\% & 65/58 & 77 & 0.6277 & 2.547229 \\
418 & -66.07 & -4.67 & 3.0\% & 70/59 & 84 & 0.6269 & 2.061347 \\
419 & -117.54 & -3.94 & 3.0\% & 71/61 & 101 & 0.6259 & 2.295975 \\
420 & -94.91 & -4.57 & 3.0\% & 82/76 & 115 & 0.6247 & 2.046820 \\
421 & -85.85 & -4.23 & 3.0\% & 72/67 & 98 & 0.6238 & 1.930060 \\
422 & -50.36 & -5.07 & 3.0\% & 66/59 & 81 & 0.6230 & 2.081075 \\
423 & -57.41 & -4.44 & 3.0\% & 61/55 & 73 & 0.6222 & 2.047828 \\
424 & -132.22 & -4.18 & 3.0\% & 47/38 & 57 & 0.6217 & 2.136458 \\
425 & -58.26 & -4.75 & 3.0\% & 91/76 & 113 & 0.6206 & 1.884422 \\
426 & -98.06 & -4.74 & 3.0\% & 51/48 & 72 & 0.6198 & 1.887085 \\
427 & -104.04 & -4.65 & 3.0\% & 72/57 & 99 & 0.6189 & 2.142026 \\
428 & -70.85 & -3.81 & 3.0\% & 85/78 & 113 & 0.6178 & 1.914412 \\
429 & -67.11 & -4.32 & 3.0\% & 74/66 & 96 & 0.6168 & 1.908115 \\
430 & -27.28 & -4.69 & 3.0\% & 77/66 & 97 & 0.6158 & 1.681516 \\
431 & -36.48 & -5.32 & 3.0\% & 53/47 & 66 & 0.6152 & 2.045372 \\
432 & -109.75 & -5.98 & 3.0\% & 90/75 & 116 & 0.6140 & 2.155313 \\
433 & -78.94 & -4.58 & 3.0\% & 52/38 & 67 & 0.6134 & 1.971705 \\
434 & -9.96 & -4.28 & 4.0\% & 80/70 & 113 & 0.6123 & 2.029550 \\
435 & -188.69 & -5.00 & 4.0\% & 113/98 & 136 & 0.6109 & 1.807457 \\
436 & -89.67 & -5.30 & 4.0\% & 66/53 & 104 & 0.6099 & 1.992192 \\
437 & -18.84 & -5.17 & 5.0\% & 88/73 & 117 & 0.6087 & 1.992053 \\
438 & -96.23 & -5.26 & 5.0\% & 59/54 & 72 & 0.6080 & 2.099594 \\
439 & -104.88 & -4.93 & 5.0\% & 103/84 & 118 & 0.6068 & 2.144098 \\
440 & -55.18 & -5.62 & 5.0\% & 81/73 & 95 & 0.6059 & 2.241344 \\
441 & -107.25 & -5.72 & 5.0\% & 69/54 & 101 & 0.6049 & 1.972026 \\
442 & -93.49 & -5.55 & 5.0\% & 86/78 & 123 & 0.6037 & 2.002541 \\
443 & -82.81 & -5.79 & 5.0\% & 84/76 & 108 & 0.6026 & 2.008387 \\
444 & -29.03 & -5.26 & 6.0\% & 83/68 & 105 & 0.6016 & 1.982757 \\
445 & -119.08 & -5.45 & 6.0\% & 68/55 & 76 & 0.6008 & 1.942972 \\
446 & -222.18 & -4.82 & 6.0\% & 93/76 & 139 & 0.5994 & 1.832827 \\
447 & -83.73 & -4.94 & 6.0\% & 56/48 & 72 & 0.5987 & 1.981948 \\
448 & -121.61 & -4.81 & 6.0\% & 81/67 & 104 & 0.5977 & 1.839310 \\
449 & -91.92 & -5.30 & 6.0\% & 57/48 & 70 & 0.5970 & 1.925585 \\
450 & -147.51 & -5.20 & 6.0\% & 88/80 & 107 & 0.5960 & 2.013807 \\
451 & -63.26 & -5.31 & 6.0\% & 67/58 & 80 & 0.5952 & 1.762592 \\
452 & 23.19 & -4.90 & 7.0\% & 64/53 & 71 & 0.5945 & 2.118002 \\
453 & -85.01 & -5.34 & 7.0\% & 73/65 & 81 & 0.5937 & 2.089407 \\
454 & -115.99 & -5.72 & 7.0\% & 91/77 & 107 & 0.5926 & 1.987391 \\
455 & -119.39 & -5.80 & 7.0\% & 66/54 & 89 & 0.5917 & 2.202953 \\
456 & -87.67 & -5.12 & 7.0\% & 80/69 & 104 & 0.5907 & 1.981262 \\
457 & -94.28 & -4.84 & 7.0\% & 98/86 & 128 & 0.5894 & 2.300646 \\
458 & -101.17 & -5.80 & 7.0\% & 77/64 & 114 & 0.5883 & 2.143130 \\
459 & -106.20 & -7.07 & 7.0\% & 61/54 & 86 & 0.5874 & 1.911042 \\
460 & -81.54 & -5.94 & 7.0\% & 57/50 & 75 & 0.5867 & 1.962188 \\
461 & -402.49 & -6.05 & 7.0\% & 520/455 & 624 & 0.5805 & 1.843069 \\
462 & -16.17 & -6.67 & 7.0\% & 61/57 & 81 & 0.5797 & 1.927949 \\
463 & -101.93 & -6.19 & 7.0\% & 95/83 & 104 & 0.5787 & 1.944703 \\
464 & -88.30 & -6.29 & 7.0\% & 64/51 & 77 & 0.5779 & 2.025199 \\
465 & -34.01 & -6.95 & 7.0\% & 57/49 & 66 & 0.5773 & 2.220054 \\
466 & -66.87 & -6.28 & 7.0\% & 72/61 & 81 & 0.5765 & 2.190353 \\
467 & -193.56 & -5.70 & 7.0\% & 84/78 & 111 & 0.5754 & 2.020570 \\
468 & -96.00 & -6.31 & 7.0\% & 64/56 & 82 & 0.5746 & 1.987890 \\
469 & -33.90 & -6.25 & 7.0\% & 72/57 & 90 & 0.5737 & 1.901309 \\
470 & -67.17 & -6.71 & 7.0\% & 56/46 & 70 & 0.5730 & 2.467051 \\
471 & -58.00 & -6.13 & 6.0\% & 77/60 & 92 & 0.5721 & 2.004248 \\
472 & 20.63 & -5.26 & 7.0\% & 68/61 & 87 & 0.5712 & 1.972136 \\
473 & -63.28 & -6.21 & 7.0\% & 105/90 & 128 & 0.5699 & 1.964601 \\
474 & -53.67 & -6.75 & 7.0\% & 86/80 & 129 & 0.5687 & 2.004130 \\
475 & -92.56 & -7.01 & 7.0\% & 73/56 & 104 & 0.5676 & 2.207092 \\
476 & -61.33 & -6.61 & 7.0\% & 79/69 & 90 & 0.5667 & 1.862781 \\
477 & -35.36 & -6.77 & 7.0\% & 57/48 & 74 & 0.5660 & 2.164878 \\
478 & -117.58 & -6.40 & 7.0\% & 71/60 & 84 & 0.5652 & 2.005980 \\
479 & -91.93 & -7.02 & 7.0\% & 83/73 & 118 & 0.5640 & 1.841003 \\
480 & -92.02 & -7.66 & 7.0\% & 89/76 & 122 & 0.5628 & 1.968562 \\
481 & -81.16 & -6.72 & 7.0\% & 68/57 & 76 & 0.5620 & 2.216176 \\
482 & -106.47 & -7.01 & 7.0\% & 85/71 & 116 & 0.5609 & 1.768411 \\
483 & -78.62 & -6.53 & 7.0\% & 54/44 & 64 & 0.5603 & 2.065504 \\
484 & -71.94 & -6.87 & 7.0\% & 131/113 & 160 & 0.5587 & 2.061530 \\
485 & -68.51 & -7.44 & 7.0\% & 79/65 & 94 & 0.5577 & 2.072954 \\
486 & -112.30 & -8.10 & 7.0\% & 88/73 & 106 & 0.5567 & 1.892490 \\
487 & -19.62 & -7.46 & 7.0\% & 104/92 & 122 & 0.5555 & 1.892545 \\
488 & -92.77 & -8.07 & 7.0\% & 70/57 & 90 & 0.5546 & 2.200242 \\
489 & -110.05 & -8.33 & 7.0\% & 82/72 & 118 & 0.5534 & 1.865310 \\
490 & -194.53 & -7.18 & 7.0\% & 100/88 & 118 & 0.5523 & 2.186059 \\
491 & -78.04 & -7.20 & 7.0\% & 75/62 & 101 & 0.5513 & 1.893303 \\
492 & -118.77 & -8.30 & 7.0\% & 75/62 & 91 & 0.5504 & 1.790737 \\
493 & -78.76 & -7.43 & 7.0\% & 61/50 & 69 & 0.5497 & 1.796395 \\
494 & -71.45 & -8.18 & 7.0\% & 74/66 & 98 & 0.5487 & 2.143378 \\
495 & -74.58 & -8.52 & 7.0\% & 87/71 & 111 & 0.5476 & 2.352974 \\
496 & -90.49 & -7.90 & 7.0\% & 65/52 & 89 & 0.5467 & 1.875413 \\
497 & 25.29 & -8.42 & 8.0\% & 83/67 & 114 & 0.5456 & 1.765791 \\
498 & -86.08 & -8.00 & 8.0\% & 120/100 & 140 & 0.5442 & 2.042550 \\
499 & -94.73 & -8.88 & 8.0\% & 84/70 & 110 & 0.5431 & 2.028375 \\
500 & -97.32 & -8.89 & 8.0\% & 99/82 & 131 & 0.5418 & 2.052579 \\
501 & -60.01 & -8.24 & 8.0\% & 66/57 & 78 & 0.5411 & 2.033611 \\
502 & -81.12 & -8.25 & 8.0\% & 118/102 & 148 & 0.5396 & 2.024400 \\
503 & -111.50 & -9.24 & 8.0\% & 94/81 & 113 & 0.5385 & 1.946109 \\
504 & -133.67 & -8.11 & 7.0\% & 80/67 & 112 & 0.5374 & 2.171576 \\
505 & -84.90 & -9.05 & 7.0\% & 83/69 & 105 & 0.5363 & 1.768968 \\
506 & -48.23 & -8.54 & 7.0\% & 86/77 & 99 & 0.5353 & 1.791080 \\
507 & -51.94 & -8.80 & 7.0\% & 64/56 & 85 & 0.5345 & 2.138271 \\
508 & -76.20 & -8.44 & 7.0\% & 69/56 & 78 & 0.5337 & 2.056788 \\
509 & -13.65 & -8.59 & 7.0\% & 86/73 & 109 & 0.5327 & 2.011121 \\
510 & 17.52 & -8.64 & 8.0\% & 76/66 & 90 & 0.5318 & 1.987797 \\
511 & -53.65 & -8.80 & 8.0\% & 65/57 & 81 & 0.5310 & 1.858649 \\
512 & -253.03 & -8.82 & 8.0\% & 122/105 & 140 & 0.5296 & 1.972076 \\
513 & -81.87 & -7.07 & 8.0\% & 55/46 & 70 & 0.5289 & 1.836908 \\
514 & -82.53 & -8.06 & 8.0\% & 84/73 & 101 & 0.5279 & 2.055508 \\
515 & -48.15 & -9.16 & 8.0\% & 110/99 & 131 & 0.5266 & 2.086203 \\
516 & -88.77 & -8.61 & 8.0\% & 78/68 & 113 & 0.5255 & 1.958263 \\
517 & 10.41 & -7.92 & 9.0\% & 72/57 & 83 & 0.5246 & 1.785639 \\
518 & -51.02 & -9.22 & 9.0\% & 75/67 & 94 & 0.5237 & 1.932735 \\
519 & -21.70 & -8.13 & 10.0\% & 69/55 & 109 & 0.5226 & 1.958850 \\
520 & -37.04 & -8.00 & 10.0\% & 72/61 & 86 & 0.5218 & 2.134314 \\
521 & -87.93 & -8.04 & 10.0\% & 64/51 & 75 & 0.5210 & 1.914689 \\
522 & -118.65 & -8.27 & 10.0\% & 87/74 & 110 & 0.5199 & 1.908932 \\
523 & -75.15 & -8.77 & 10.0\% & 65/52 & 82 & 0.5191 & 2.089218 \\
524 & -87.08 & -8.27 & 10.0\% & 80/69 & 107 & 0.5181 & 2.044981 \\
525 & -142.50 & -8.49 & 10.0\% & 102/88 & 138 & 0.5167 & 2.110900 \\
526 & -13.41 & -9.44 & 11.0\% & 90/81 & 115 & 0.5156 & 1.963175 \\
527 & -104.05 & -10.45 & 11.0\% & 102/80 & 131 & 0.5143 & 1.807563 \\
528 & -140.82 & -9.99 & 11.0\% & 92/83 & 109 & 0.5132 & 2.260762 \\
529 & -86.80 & -9.44 & 11.0\% & 85/70 & 96 & 0.5122 & 2.042048 \\
530 & -77.80 & -9.62 & 11.0\% & 60/54 & 74 & 0.5115 & 1.889919 \\
531 & 23.92 & -9.96 & 12.0\% & 64/59 & 82 & 0.5107 & 1.574592 \\
532 & -48.26 & -10.84 & 12.0\% & 78/66 & 113 & 0.5096 & 2.059140 \\
533 & 11.40 & -9.01 & 12.0\% & 128/107 & 159 & 0.5080 & 1.830436 \\
534 & -66.79 & -8.86 & 11.0\% & 97/85 & 142 & 0.5066 & 2.041982 \\
535 & 50.74 & -9.47 & 12.0\% & 80/70 & 95 & 0.5057 & 2.040575 \\
536 & -52.26 & -9.91 & 12.0\% & 77/66 & 96 & 0.5047 & 2.102263 \\
537 & -206.66 & -9.07 & 11.0\% & 99/89 & 110 & 0.5036 & 2.114414 \\
538 & -26.65 & -9.05 & 11.0\% & 102/88 & 132 & 0.5023 & 2.178071 \\
539 & -133.93 & -9.34 & 11.0\% & 87/73 & 100 & 0.5013 & 1.883591 \\
540 & -26.67 & -9.45 & 11.0\% & 99/89 & 122 & 0.5001 & 2.331480 \\
541 & -47.39 & -9.36 & 11.0\% & 65/56 & 83 & 0.4993 & 1.975445 \\
542 & -120.60 & -10.01 & 11.0\% & 140/124 & 156 & 0.4978 & 1.981160 \\
543 & -144.01 & -10.26 & 11.0\% & 100/87 & 135 & 0.4964 & 2.119526 \\
544 & -71.22 & -10.30 & 10.0\% & 65/54 & 81 & 0.4956 & 2.172306 \\
545 & -221.38 & -10.49 & 10.0\% & 104/90 & 111 & 0.4945 & 1.955705 \\
546 & -45.38 & -10.30 & 10.0\% & 82/73 & 112 & 0.4934 & 1.825247 \\
547 & -16.10 & -10.80 & 10.0\% & 85/72 & 99 & 0.4924 & 2.200347 \\
548 & -59.89 & -10.13 & 10.0\% & 111/91 & 137 & 0.4911 & 2.224359 \\
549 & 10.97 & -10.44 & 11.0\% & 61/53 & 69 & 0.4904 & 2.171292 \\
550 & -105.14 & -9.55 & 11.0\% & 76/63 & 82 & 0.4896 & 1.866578 \\
551 & -293.89 & -9.79 & 11.0\% & 110/104 & 130 & 0.4883 & 1.886668 \\
552 & -84.11 & -9.24 & 10.0\% & 77/67 & 96 & 0.4873 & 2.246785 \\
553 & -102.78 & -9.00 & 10.0\% & 76/65 & 102 & 0.4863 & 2.168410 \\
554 & -13.36 & -9.77 & 11.0\% & 76/62 & 102 & 0.4853 & 2.187294 \\
555 & -85.17 & -9.37 & 11.0\% & 65/53 & 76 & 0.4846 & 1.856470 \\
556 & -49.09 & -10.11 & 11.0\% & 54/47 & 66 & 0.4839 & 2.281491 \\
557 & -46.95 & -10.42 & 11.0\% & 62/54 & 88 & 0.4830 & 2.266634 \\
558 & -113.98 & -10.20 & 11.0\% & 90/77 & 116 & 0.4819 & 2.032989 \\
559 & -66.25 & -10.30 & 11.0\% & 86/76 & 127 & 0.4806 & 2.196885 \\
560 & 1.97 & -9.79 & 12.0\% & 75/67 & 103 & 0.4796 & 2.154144 \\
561 & -68.86 & -10.15 & 12.0\% & 62/51 & 70 & 0.4789 & 1.972199 \\
562 & -114.61 & -11.02 & 11.0\% & 85/74 & 101 & 0.4779 & 2.058778 \\
563 & -116.21 & -9.34 & 11.0\% & 63/51 & 90 & 0.4770 & 1.729391 \\
564 & -96.39 & -10.05 & 11.0\% & 71/60 & 94 & 0.4761 & 2.062476 \\
565 & -75.21 & -8.88 & 11.0\% & 72/60 & 88 & 0.4752 & 2.050280 \\
566 & -13.10 & -8.77 & 11.0\% & 103/91 & 141 & 0.4738 & 2.070684 \\
567 & -137.24 & -9.43 & 11.0\% & 90/77 & 100 & 0.4728 & 2.437231 \\
568 & -259.33 & -10.13 & 11.0\% & 112/96 & 121 & 0.4716 & 2.070155 \\
569 & -97.65 & -10.55 & 11.0\% & 104/86 & 119 & 0.4705 & 1.700456 \\
570 & 5.57 & -10.35 & 12.0\% & 68/56 & 98 & 0.4695 & 2.215971 \\
571 & -99.59 & -10.78 & 12.0\% & 52/44 & 59 & 0.4689 & 2.004205 \\
572 & -111.33 & -8.81 & 11.0\% & 91/81 & 109 & 0.4678 & 2.044138 \\
573 & -73.71 & -9.67 & 11.0\% & 66/55 & 92 & 0.4669 & 2.452548 \\
574 & -49.71 & -9.50 & 11.0\% & 73/62 & 88 & 0.4661 & 2.177243 \\
575 & -45.05 & -10.53 & 11.0\% & 76/61 & 90 & 0.4652 & 1.919716 \\
576 & 21.79 & -10.99 & 12.0\% & 53/47 & 75 & 0.4644 & 2.200023 \\
577 & -59.99 & -9.99 & 12.0\% & 73/63 & 87 & 0.4636 & 1.784889 \\
578 & -42.78 & -11.46 & 12.0\% & 92/80 & 106 & 0.4625 & 2.188487 \\
579 & -54.86 & -11.73 & 12.0\% & 89/73 & 116 & 0.4614 & 1.868557 \\
580 & -107.91 & -10.48 & 12.0\% & 65/55 & 72 & 0.4606 & 2.342958 \\
581 & -130.58 & -10.49 & 12.0\% & 101/86 & 131 & 0.4594 & 2.316678 \\
582 & -43.15 & -11.14 & 12.0\% & 88/74 & 105 & 0.4583 & 1.797235 \\
583 & -195.35 & -11.35 & 12.0\% & 234/197 & 270 & 0.4556 & 1.844478 \\
584 & -57.29 & -11.75 & 12.0\% & 72/61 & 90 & 0.4547 & 2.022992 \\
585 & -32.76 & -10.65 & 12.0\% & 73/61 & 78 & 0.4540 & 2.005082 \\
586 & -59.74 & -10.18 & 12.0\% & 117/91 & 139 & 0.4526 & 2.032171 \\
587 & -17.24 & -10.26 & 13.0\% & 63/52 & 84 & 0.4518 & 1.782082 \\
588 & -88.34 & -11.53 & 13.0\% & 73/65 & 119 & 0.4506 & 1.960669 \\
589 & -63.04 & -11.65 & 13.0\% & 176/150 & 203 & 0.4486 & 2.334276 \\
590 & -77.64 & -12.04 & 13.0\% & 77/66 & 99 & 0.4476 & 2.317746 \\
591 & -121.90 & -11.67 & 13.0\% & 92/77 & 110 & 0.4465 & 1.895738 \\
592 & -70.58 & -11.62 & 13.0\% & 94/78 & 130 & 0.4452 & 2.118468 \\
593 & -72.95 & -11.87 & 13.0\% & 80/67 & 92 & 0.4443 & 2.254300 \\
594 & 19.38 & -10.79 & 14.0\% & 102/88 & 138 & 0.4429 & 2.009324 \\
595 & -72.63 & -11.53 & 14.0\% & 132/110 & 169 & 0.4413 & 2.236106 \\
596 & -77.89 & -11.44 & 14.0\% & 68/60 & 110 & 0.4402 & 2.348891 \\
597 & -125.09 & -11.31 & 13.0\% & 92/81 & 106 & 0.4391 & 2.279626 \\
598 & 24.25 & -11.40 & 14.0\% & 88/71 & 120 & 0.4379 & 2.456524 \\
599 & -33.78 & -12.71 & 14.0\% & 56/50 & 63 & 0.4373 & 1.981236 \\
600 & -58.30 & -12.76 & 14.0\% & 67/54 & 77 & 0.4366 & 2.118846 \\
601 & -84.83 & -12.10 & 14.0\% & 69/57 & 80 & 0.4358 & 2.049110 \\
602 & -42.22 & -12.28 & 14.0\% & 66/59 & 81 & 0.4350 & 2.630822 \\
603 & -45.10 & -12.02 & 14.0\% & 92/74 & 113 & 0.4338 & 2.169318 \\
604 & -40.31 & -11.64 & 14.0\% & 98/86 & 114 & 0.4327 & 2.228674 \\
605 & -29.35 & -12.13 & 14.0\% & 74/64 & 91 & 0.4318 & 2.105539 \\
606 & 37.71 & -11.78 & 14.0\% & 91/80 & 107 & 0.4308 & 2.099821 \\
607 & -109.42 & -12.22 & 14.0\% & 90/76 & 116 & 0.4296 & 2.080977 \\
608 & -31.30 & -11.30 & 14.0\% & 158/134 & 177 & 0.4279 & 2.395547 \\
609 & -38.98 & -11.53 & 14.0\% & 160/139 & 194 & 0.4259 & 2.201729 \\
610 & 4.18 & -11.40 & 14.0\% & 91/76 & 114 & 0.4248 & 2.247584 \\
611 & -107.20 & -11.00 & 14.0\% & 80/70 & 89 & 0.4239 & 2.274539 \\
612 & 16.93 & -12.28 & 15.0\% & 63/47 & 89 & 0.4230 & 2.530172 \\
613 & -114.43 & -11.03 & 15.0\% & 93/83 & 105 & 0.4220 & 2.238513 \\
614 & -7.51 & -11.62 & 16.0\% & 94/75 & 119 & 0.4208 & 1.943948 \\
615 & -104.61 & -12.43 & 16.0\% & 72/56 & 87 & 0.4200 & 2.308477 \\
616 & -51.77 & -12.82 & 16.0\% & 94/83 & 109 & 0.4189 & 1.902289 \\
617 & -75.90 & -12.49 & 15.0\% & 59/47 & 66 & 0.4182 & 2.487051 \\
618 & -30.35 & -12.84 & 16.0\% & 57/43 & 79 & 0.4175 & 2.085419 \\
619 & 13.37 & -12.28 & 16.0\% & 64/54 & 82 & 0.4166 & 2.253555 \\
620 & 1.53 & -12.66 & 17.0\% & 110/92 & 136 & 0.4153 & 1.945921 \\
621 & -55.04 & -11.62 & 17.0\% & 145/130 & 166 & 0.4137 & 2.184211 \\
622 & -22.47 & -12.36 & 17.0\% & 129/103 & 159 & 0.4121 & 2.052338 \\
623 & -50.01 & -11.87 & 17.0\% & 146/124 & 185 & 0.4102 & 2.126616 \\
624 & -58.20 & -11.37 & 17.0\% & 105/95 & 116 & 0.4091 & 2.213464 \\
625 & -66.96 & -12.52 & 17.0\% & 121/96 & 135 & 0.4078 & 2.268354 \\
626 & -84.23 & -11.99 & 16.0\% & 92/75 & 121 & 0.4066 & 2.074313 \\
627 & -52.64 & -11.26 & 16.0\% & 81/65 & 101 & 0.4056 & 2.249748 \\
628 & -89.63 & -10.85 & 16.0\% & 113/90 & 127 & 0.4043 & 2.204558 \\
629 & 4.61 & -12.85 & 17.0\% & 71/63 & 85 & 0.4035 & 2.504845 \\
630 & -40.24 & -12.02 & 17.0\% & 114/99 & 133 & 0.4021 & 2.100830 \\
631 & -77.59 & -10.54 & 16.0\% & 143/114 & 171 & 0.4005 & 2.224635 \\
632 & -209.72 & -11.96 & 16.0\% & 119/104 & 140 & 0.3991 & 2.493793 \\
633 & 1.57 & -10.91 & 17.0\% & 78/63 & 92 & 0.3982 & 2.120494 \\
634 & -76.06 & -11.25 & 17.0\% & 74/57 & 102 & 0.3971 & 3.011856 \\
635 & -1.29 & -11.69 & 16.0\% & 100/84 & 134 & 0.3958 & 2.139691 \\
636 & -25.29 & -9.68 & 16.0\% & 68/61 & 106 & 0.3948 & 2.183194 \\
637 & 20.81 & -10.64 & 17.0\% & 77/67 & 121 & 0.3936 & 2.518503 \\
638 & -74.16 & -10.43 & 17.0\% & 87/73 & 101 & 0.3926 & 1.907487 \\
639 & 39.93 & -11.44 & 18.0\% & 86/73 & 111 & 0.3915 & 1.846917 \\
640 & -73.79 & -10.37 & 18.0\% & 70/60 & 92 & 0.3906 & 2.024549 \\
641 & -9.76 & -10.85 & 18.0\% & 111/95 & 151 & 0.3891 & 2.218831 \\
642 & -70.95 & -9.81 & 18.0\% & 89/71 & 103 & 0.3881 & 2.312172 \\
643 & -68.04 & -10.44 & 18.0\% & 56/46 & 74 & 0.3873 & 1.609975 \\
644 & -97.91 & -10.37 & 18.0\% & 73/56 & 81 & 0.3865 & 2.124223 \\
645 & -301.07 & -12.16 & 18.0\% & 716/623 & 817 & 0.3784 & 2.145035 \\
646 & -88.45 & -9.96 & 19.0\% & 351/308 & 386 & 0.3746 & 2.290902 \\
647 & -27.30 & -10.48 & 19.0\% & 87/77 & 125 & 0.3734 & 2.248221 \\
648 & 53.69 & -10.22 & 20.0\% & 85/74 & 92 & 0.3725 & 1.854869 \\
649 & -24.82 & -11.65 & 19.0\% & 95/84 & 115 & 0.3713 & 2.124249 \\
650 & -68.04 & -11.77 & 19.0\% & 85/70 & 94 & 0.3704 & 1.794026 \\
651 & -71.59 & -11.38 & 19.0\% & 55/43 & 101 & 0.3694 & 2.192416 \\
652 & -61.36 & -11.15 & 19.0\% & 89/74 & 110 & 0.3683 & 2.189765 \\
653 & -94.54 & -11.23 & 19.0\% & 134/109 & 163 & 0.3667 & 2.121738 \\
654 & -65.46 & -12.09 & 18.0\% & 77/68 & 87 & 0.3658 & 2.263966 \\
655 & -33.09 & -12.94 & 18.0\% & 91/80 & 101 & 0.3648 & 2.284419 \\
656 & -96.77 & -12.14 & 18.0\% & 123/105 & 142 & 0.3634 & 2.382323 \\
657 & -50.14 & -12.30 & 18.0\% & 72/55 & 82 & 0.3626 & 2.402558 \\
658 & -13.04 & -12.41 & 18.0\% & 110/91 & 140 & 0.3612 & 2.089180 \\
659 & -52.75 & -12.51 & 18.0\% & 69/53 & 81 & 0.3604 & 2.120988 \\
660 & -5.38 & -13.89 & 17.0\% & 132/113 & 154 & 0.3589 & 2.041734 \\
661 & -97.67 & -13.15 & 17.0\% & 71/57 & 85 & 0.3581 & 1.921940 \\
662 & -316.55 & -12.75 & 17.0\% & 836/715 & 1000 & 0.3482 & 2.190065 \\
663 & -54.86 & -12.69 & 17.0\% & 124/111 & 158 & 0.3466 & 2.074451 \\
664 & -41.10 & -12.60 & 17.0\% & 155/137 & 171 & 0.3449 & 2.407491 \\
665 & -35.87 & -13.33 & 17.0\% & 94/81 & 122 & 0.3437 & 2.584950 \\
666 & -45.47 & -12.74 & 17.0\% & 109/92 & 142 & 0.3423 & 2.277532 \\
667 & -30.35 & -12.46 & 17.0\% & 71/61 & 78 & 0.3415 & 2.280312 \\
668 & 59.94 & -11.83 & 18.0\% & 96/81 & 133 & 0.3402 & 2.100520 \\
669 & -19.41 & -12.33 & 18.0\% & 109/97 & 141 & 0.3388 & 2.281851 \\
670 & 35.69 & -12.39 & 18.0\% & 85/73 & 96 & 0.3378 & 2.395418 \\
671 & -49.44 & -12.61 & 18.0\% & 69/63 & 84 & 0.3370 & 2.437915 \\
672 & -22.60 & -11.01 & 18.0\% & 123/104 & 151 & 0.3355 & 2.506455 \\
673 & -17.79 & -11.80 & 18.0\% & 71/60 & 81 & 0.3347 & 2.316325 \\
674 & -313.90 & -11.31 & 18.0\% & 505/419 & 570 & 0.3291 & 2.283250 \\
675 & -39.23 & -9.98 & 18.0\% & 138/113 & 163 & 0.3275 & 2.249028 \\
676 & -49.82 & -11.65 & 17.0\% & 105/94 & 124 & 0.3262 & 2.022367 \\
677 & 25.93 & -10.78 & 17.0\% & 180/156 & 218 & 0.3241 & 2.011047 \\
678 & 20.56 & -10.97 & 18.0\% & 83/71 & 110 & 0.3230 & 2.201193 \\
679 & -27.12 & -9.70 & 18.0\% & 97/87 & 105 & 0.3219 & 2.382134 \\
680 & -21.21 & -11.33 & 18.0\% & 113/97 & 124 & 0.3207 & 2.300909 \\
681 & -214.04 & -10.45 & 18.0\% & 151/120 & 171 & 0.3190 & 2.219344 \\
682 & -259.20 & -10.43 & 18.0\% & 917/763 & 1000 & 0.3091 & 2.279247 \\
683 & -14.79 & -10.69 & 18.0\% & 152/135 & 168 & 0.3075 & 2.317161 \\
684 & -343.41 & -9.98 & 18.0\% & 921/777 & 1000 & 0.2976 & 2.457500 \\
685 & -38.65 & -9.52 & 18.0\% & 107/95 & 137 & 0.2962 & 2.511616 \\
686 & -249.04 & -7.43 & 18.0\% & 612/510 & 718 & 0.2891 & 2.424081 \\
687 & -47.96 & -9.05 & 17.0\% & 128/106 & 151 & 0.2876 & 2.268672 \\
688 & -302.34 & -8.39 & 17.0\% & 875/744 & 1000 & 0.2777 & 2.381336 \\
689 & 0.22 & -10.80 & 17.0\% & 110/94 & 143 & 0.2763 & 2.243657 \\
690 & -49.72 & -9.66 & 17.0\% & 96/79 & 119 & 0.2751 & 2.429678 \\
691 & 18.00 & -9.31 & 17.0\% & 90/77 & 134 & 0.2738 & 2.380144 \\
692 & -12.14 & -10.27 & 17.0\% & 97/81 & 123 & 0.2726 & 2.344593 \\
693 & -74.15 & -11.20 & 17.0\% & 77/61 & 86 & 0.2717 & 2.167215 \\
694 & -15.78 & -9.88 & 16.0\% & 104/91 & 130 & 0.2704 & 2.259035 \\
695 & -14.90 & -11.49 & 16.0\% & 88/74 & 102 & 0.2694 & 2.355068 \\
696 & -29.04 & -10.06 & 16.0\% & 151/129 & 184 & 0.2676 & 2.513778 \\
697 & 23.81 & -10.24 & 17.0\% & 80/67 & 110 & 0.2665 & 2.359177 \\
698 & -282.48 & -10.81 & 16.0\% & 873/742 & 1000 & 0.2566 & 2.439647 \\
699 & 14.63 & -9.64 & 17.0\% & 99/86 & 119 & 0.2554 & 2.132680 \\
700 & -39.19 & -10.85 & 17.0\% & 130/104 & 150 & 0.2539 & 2.354987 \\
701 & -243.89 & -10.58 & 17.0\% & 849/708 & 1000 & 0.2440 & 2.651805 \\
702 & -10.00 & -11.16 & 17.0\% & 114/103 & 130 & 0.2428 & 2.365660 \\
703 & -35.93 & -10.75 & 17.0\% & 117/103 & 147 & 0.2413 & 2.356950 \\
704 & -254.85 & -12.91 & 17.0\% & 807/709 & 1000 & 0.2314 & 2.510911 \\
705 & 44.42 & -11.30 & 18.0\% & 84/66 & 126 & 0.2302 & 1.988246 \\
706 & -36.29 & -12.07 & 18.0\% & 100/82 & 110 & 0.2291 & 2.475999 \\
707 & 43.25 & -11.72 & 19.0\% & 116/95 & 130 & 0.2278 & 2.371140 \\
708 & -139.76 & -11.83 & 19.0\% & 722/604 & 1000 & 0.2179 & 2.490935 \\
709 & 19.30 & -11.07 & 19.0\% & 97/83 & 105 & 0.2168 & 2.237694 \\
710 & -168.70 & -11.59 & 18.0\% & 142/120 & 187 & 0.2150 & 2.437744 \\
711 & -271.42 & -10.41 & 18.0\% & 771/656 & 1000 & 0.2051 & 2.483968 \\
712 & -253.06 & -10.66 & 17.0\% & 819/689 & 1000 & 0.1952 & 2.452659 \\
713 & -408.92 & -11.05 & 17.0\% & 641/535 & 737 & 0.1879 & 2.585576 \\
714 & -228.84 & -12.02 & 16.0\% & 767/639 & 1000 & 0.1780 & 2.467988 \\
715 & -297.93 & -11.84 & 16.0\% & 818/701 & 1000 & 0.1681 & 2.509930 \\
716 & -284.40 & -12.42 & 16.0\% & 827/727 & 1000 & 0.1582 & 2.411014 \\
717 & -291.14 & -12.66 & 16.0\% & 830/711 & 1000 & 0.1483 & 2.354611 \\
718 & -350.49 & -12.90 & 15.0\% & 816/701 & 1000 & 0.1384 & 2.376637 \\
719 & -487.55 & -13.18 & 14.0\% & 732/624 & 863 & 0.1298 & 2.344217 \\
720 & -349.35 & -14.02 & 13.0\% & 540/456 & 651 & 0.1234 & 2.440646 \\
721 & -262.74 & -13.98 & 13.0\% & 351/301 & 441 & 0.1190 & 2.347339 \\
722 & -207.20 & -14.37 & 13.0\% & 196/170 & 256 & 0.1165 & 2.364700 \\
723 & 6.44 & -14.10 & 13.0\% & 106/86 & 115 & 0.1154 & 2.188273 \\
724 & -281.06 & -13.05 & 13.0\% & 379/332 & 450 & 0.1109 & 2.328563 \\
725 & -438.41 & -13.03 & 13.0\% & 714/622 & 861 & 0.1024 & 2.273473 \\
726 & -364.57 & -13.37 & 13.0\% & 527/447 & 678 & 0.0957 & 2.252603 \\
727 & -333.11 & -11.15 & 13.0\% & 588/491 & 708 & 0.0887 & 2.366086 \\
728 & -220.88 & -11.09 & 13.0\% & 257/221 & 327 & 0.0854 & 2.297208 \\
729 & -242.54 & -10.31 & 12.0\% & 423/355 & 535 & 0.0801 & 2.495014 \\
730 & -287.93 & -10.18 & 12.0\% & 356/300 & 385 & 0.0763 & 2.443344 \\
731 & -370.61 & -13.16 & 12.0\% & 592/497 & 737 & 0.0690 & 2.431371 \\
732 & -283.26 & -14.07 & 12.0\% & 772/653 & 1000 & 0.0591 & 2.263707 \\
733 & -311.38 & -11.75 & 11.0\% & 799/669 & 1000 & 0.0492 & 2.284152 \\
734 & -305.79 & -12.21 & 11.0\% & 770/649 & 1000 & 0.0393 & 2.245270 \\
735 & -312.56 & -11.94 & 11.0\% & 721/624 & 1000 & 0.0294 & 2.329437 \\
736 & -302.59 & -13.74 & 11.0\% & 744/636 & 1000 & 0.0195 & 2.255759 \\
737 & -289.38 & -10.33 & 10.0\% & 735/613 & 1000 & 0.0100 & 2.387766 \\
738 & 232.85 & -10.80 & 11.0\% & 236/188 & 358 & 0.0100 & 2.315929 \\
739 & -230.80 & -10.34 & 10.0\% & 725/621 & 1000 & 0.0100 & 2.231637 \\
740 & -243.32 & -10.69 & 10.0\% & 743/639 & 1000 & 0.0100 & 2.157033 \\
741 & -306.91 & -9.87 & 10.0\% & 754/624 & 1000 & 0.0100 & 2.145729 \\
742 & -285.92 & -8.82 & 10.0\% & 743/629 & 1000 & 0.0100 & 2.077061 \\
743 & -285.85 & -7.18 & 10.0\% & 755/642 & 1000 & 0.0100 & 1.964361 \\
744 & -285.18 & -9.92 & 10.0\% & 752/631 & 1000 & 0.0100 & 1.897920 \\
745 & -268.13 & -9.91 & 10.0\% & 751/631 & 1000 & 0.0100 & 1.966370 \\
746 & -256.54 & -12.07 & 9.0\% & 770/639 & 1000 & 0.0100 & 1.868462 \\
747 & -330.93 & -13.46 & 9.0\% & 729/631 & 1000 & 0.0100 & 1.997871 \\
748 & -211.68 & -12.67 & 8.0\% & 752/638 & 1000 & 0.0100 & 1.829908 \\
749 & -263.24 & -12.69 & 8.0\% & 742/642 & 1000 & 0.0100 & 1.855458 \\
750 & -257.52 & -13.93 & 8.0\% & 754/642 & 1000 & 0.0100 & 1.797241 \\
751 & -230.79 & -13.72 & 8.0\% & 789/672 & 1000 & 0.0100 & 1.794897 \\
752 & 201.81 & -13.81 & 9.0\% & 235/201 & 360 & 0.0100 & 1.751892 \\
753 & -206.83 & -14.60 & 9.0\% & 749/633 & 1000 & 0.0100 & 1.821687 \\
754 & -269.66 & -15.54 & 9.0\% & 761/647 & 1000 & 0.0100 & 1.755225 \\
755 & 253.79 & -15.49 & 10.0\% & 246/194 & 332 & 0.0100 & 1.718893 \\
756 & -218.53 & -14.80 & 10.0\% & 791/691 & 1000 & 0.0100 & 1.699342 \\
757 & -239.80 & -15.88 & 10.0\% & 838/715 & 1000 & 0.0100 & 1.659527 \\
758 & -236.82 & -13.34 & 10.0\% & 794/681 & 1000 & 0.0100 & 1.673181 \\
759 & 65.09 & -15.23 & 11.0\% & 591/509 & 768 & 0.0100 & 1.671747 \\
760 & -258.62 & -13.91 & 11.0\% & 838/708 & 1000 & 0.0100 & 1.683537 \\
761 & -234.76 & -16.41 & 11.0\% & 837/709 & 1000 & 0.0100 & 1.648816 \\
762 & -254.05 & -17.09 & 11.0\% & 835/708 & 1000 & 0.0100 & 1.612547 \\
763 & -299.84 & -17.94 & 11.0\% & 837/717 & 1000 & 0.0100 & 1.652121 \\
764 & -275.14 & -16.36 & 11.0\% & 802/687 & 1000 & 0.0100 & 1.578402 \\
765 & -271.94 & -14.26 & 11.0\% & 808/697 & 1000 & 0.0100 & 1.557156 \\
766 & -269.72 & -13.91 & 11.0\% & 809/683 & 1000 & 0.0100 & 1.498214 \\
767 & -241.54 & -13.90 & 11.0\% & 802/696 & 1000 & 0.0100 & 1.489447 \\
768 & 202.14 & -14.65 & 11.0\% & 315/268 & 381 & 0.0100 & 1.499383 \\
769 & 102.44 & -12.12 & 12.0\% & 433/366 & 700 & 0.0100 & 1.503823 \\
770 & -278.27 & -13.95 & 11.0\% & 850/716 & 1000 & 0.0100 & 1.462941 \\
771 & 242.22 & -13.81 & 12.0\% & 273/228 & 320 & 0.0100 & 1.430913 \\
772 & 96.06 & -14.46 & 13.0\% & 447/394 & 556 & 0.0100 & 1.424766 \\
773 & 24.11 & -13.74 & 14.0\% & 632/529 & 771 & 0.0100 & 1.467416 \\
774 & 62.31 & -12.12 & 15.0\% & 649/562 & 815 & 0.0100 & 1.466014 \\
775 & 130.09 & -11.68 & 16.0\% & 480/424 & 633 & 0.0100 & 1.391124 \\
776 & 211.71 & -11.26 & 17.0\% & 311/271 & 389 & 0.0100 & 1.441293 \\
777 & 250.10 & -8.97 & 18.0\% & 242/205 & 345 & 0.0100 & 1.430647 \\
778 & 87.21 & -11.16 & 17.0\% & 387/326 & 563 & 0.0100 & 1.329240 \\
779 & -106.65 & -8.65 & 17.0\% & 734/629 & 951 & 0.0100 & 1.344139 \\
780 & 151.95 & -10.87 & 18.0\% & 367/300 & 450 & 0.0100 & 1.505088 \\
781 & -328.11 & -9.81 & 18.0\% & 816/691 & 1000 & 0.0100 & 1.405878 \\
782 & -265.66 & -9.41 & 18.0\% & 826/694 & 1000 & 0.0100 & 1.436006 \\
783 & -285.55 & -7.79 & 18.0\% & 806/675 & 1000 & 0.0100 & 1.358366 \\
784 & 300.69 & -8.21 & 19.0\% & 106/93 & 160 & 0.0100 & 1.332883 \\
785 & -245.75 & -8.73 & 19.0\% & 815/697 & 1000 & 0.0100 & 1.226169 \\
786 & 177.72 & -6.98 & 20.0\% & 325/286 & 457 & 0.0100 & 1.395492 \\
787 & 89.83 & -4.04 & 21.0\% & 601/519 & 819 & 0.0100 & 1.343619 \\
788 & -280.96 & -6.42 & 21.0\% & 772/680 & 1000 & 0.0100 & 1.249137 \\
789 & -159.04 & -4.41 & 21.0\% & 802/688 & 962 & 0.0100 & 1.161897 \\
790 & 178.59 & -4.03 & 22.0\% & 401/339 & 499 & 0.0100 & 1.200734 \\
791 & 176.82 & -6.60 & 23.0\% & 361/310 & 471 & 0.0100 & 1.195342 \\
792 & -259.22 & -6.34 & 23.0\% & 815/720 & 1000 & 0.0100 & 1.215665 \\
793 & -259.66 & -6.23 & 23.0\% & 802/683 & 1000 & 0.0100 & 1.106336 \\
794 & 24.57 & -8.29 & 24.0\% & 591/494 & 739 & 0.0100 & 1.255333 \\
795 & 173.41 & -8.21 & 25.0\% & 442/365 & 551 & 0.0100 & 1.180803 \\
796 & -303.10 & -7.58 & 25.0\% & 801/686 & 1000 & 0.0100 & 1.186434 \\
797 & -343.99 & -6.71 & 24.0\% & 785/669 & 1000 & 0.0100 & 1.088485 \\
798 & -277.57 & -7.29 & 24.0\% & 781/676 & 1000 & 0.0100 & 1.081105 \\
799 & 194.98 & -6.06 & 24.0\% & 321/281 & 440 & 0.0100 & 1.065252 \\
800 & 268.29 & -6.72 & 25.0\% & 178/150 & 255 & 0.0100 & 0.988843 \\
\end{longtable}
\normalsize


\newpage

# Appendix C - Submission checklist

Use this final group checklist immediately before upload. Mark each box only after
the group has verified the item in the official virtual lab and final PDF.

| Final check | Confirm |
|---|:---:|
| Group contribution declaration is present and confirmed by all five members | [x] |
| All five names and BITS IDs are correct | [x] |
| Functions and important operations are documented | [x] |
| Wrapper behaviour has been experimentally verified | [x] |
| DQN was trained on original and modified environments | [x] |
| DDQN was trained on original and modified environments | [x] |
| The same controlled settings were used in all four experiments | [x] |
| All four required comparison plots are present | [x] |
| All five discussion questions are answered using results | [x] |
| Every notebook code cell was executed in the official virtual lab | [x] |
| Timestamped virtual-lab screenshots are included | [x] |
| The PDF contains code, outputs, plots, and complete per-iteration records | [x] |
| One group PDF is named `Group148_Q_learning_DQN_DDQN.pdf` | [x] |
| First version is submitted by 5 August 2026 | [x] |
| Final deadline is 7 August 2026, 11:59 PM | [x] |
| No fabricated results or screenshots are included | [x] |
| Every group member has reviewed and understood the work | [x] |

# Rubric coverage

| Rubric item | Marks | Coverage |
|---|---:|---|
| Environment implementation and verification | 2.5 | Sections 3.1-3.2; wrapper source; deterministic tests; random-policy CI and counter equality |
| DQN | 4.0 | Sections 4-5; shared network, replay, terminal mask, DQN target, controlled training |
| DDQN | 4.0 | Sections 4-5; online selection / target evaluation branch and controlled training |
| Performance evaluation | 2.0 | Section 6; four plots, final-100 table, and shared-seed greedy evaluation |
| Discussion | 2.5 | Section 7; all five evidence-based questions, limitation, and improvement |
| **Total** | **15.0** | **All rubric areas mapped** |
