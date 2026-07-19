# Robust LunarLander: DQN vs. DDQN under Stochastic Action Failure

Group 148 Assignment II - a reproducible study of hidden actuator failures,
attempted-fuel penalties, and value overestimation in Gymnasium LunarLander-v3.

## Result at a glance

Four agents were trained for 800 episodes with seed 148, the same 128 x 128
network, replay buffer, optimizer, exploration schedule, target-update cadence,
and fixed set of 512 validation states. The only algorithmic branch is the
next-state target calculation.

| Agent | Environment | Greedy mean reward | Safe landing rate | Mean attempted thrusters |
|---|---:|---:|---:|---:|
| DQN | Original | 67.15 | 24% | 138.07 |
| DDQN | Original | 169.05 | 61% | 250.72 |
| DQN | Modified | -217.58 | 3% | 710.45 |
| DDQN | Modified | 87.73 | 32% | 166.23 |

The mean absolute DQN-DDQN validation Q-gap over the final 100 training
episodes is 16.23 in the original environment and 58.30 in the modified
environment. Under stochastic action failure, DDQN is substantially more
robust in this seeded experiment.

![Four required assignment metrics](artifacts/plots/four_metric_overview.png)

## What is included

- Exact Gymnasium wrapper with 15% hidden thruster failures, a 0.3 penalty on
  every selected thruster action, and the strict +50 safe-landing bonus.
- External verification instrumentation that does not leak diagnostics through
  the agent-facing info dictionary.
- Shared modular implementations of Q-network, replay buffer, epsilon-greedy
  policy, target network, DQN targets, and DDQN targets.
- Per-episode logs for all 3,200 training episodes and 400 greedy evaluation
  episodes.
- Four required plots in PNG and editable SVG form.
- Executable notebook and one PDF submission with explanations, source
  listings, outputs, and first-five/last-five episode excerpts. Complete
  episode ledgers remain in the repository CSV files.
- Automated tests, style checks, provenance hashes, and a virtual-lab runbook.

## Reproduce

Python 3.11-3.13 is supported. The recorded run used Python 3.13.12,
Gymnasium 1.2.3, PyTorch 2.13.0, NumPy 2.4.4, and pandas 3.0.2.

    uv sync --extra dev
    source .venv/bin/activate
    pytest
    python -m robust_lunarlander.verification --episodes 250
    python -m robust_lunarlander.experiment --episodes 800 --evaluation-episodes 100
    python scripts/build_notebook.py
    jupyter nbconvert --execute --to notebook --inplace output/jupyter-notebook/Group_148_Q_learning_DQN_DDQN.ipynb
    python scripts/build_report.py
    bash scripts/render_report.sh

The complete study is deterministic for the recorded software/hardware stack,
although exact deep-learning floating-point trajectories can vary across
platforms.

## Repository map

| Path | Purpose |
|---|---|
| src/robust_lunarlander | Assignment implementation and experiment pipeline |
| tests | Fast environment, target-equation, replay, and schedule checks |
| artifacts/verification | Statistical and boundary-case wrapper evidence |
| artifacts/metrics | All per-episode training logs |
| artifacts/plots | Required figures |
| output/jupyter-notebook | Executable analysis notebook |
| output/pdf | Single PDF submission artifact |
| docs/virtual_lab_runbook.md | Timestamped virtual-lab execution procedure |

## Learn the ideas

- [First-principles story: The Flight Computer That Learned to Doubt Its Own Confidence](docs/first_principles_story.md)
  explains Q-learning, DQN, DDQN, maximization bias, hidden actuator failure,
  reward design, and fair evaluation as one narrative.
- [Reading and watch list](docs/reading_list.md) provides an ordered path through
  the best book chapters, primary papers, official tutorials, a YouTube lecture,
  a free course, and follow-up experiments.

## Evidence discipline

The report labels training observations as empirical evidence and treats
theoretical expectations separately. The controlled experiment uses one seed,
so it supports a reproducible within-seed comparison rather than a population
claim. A multi-seed confidence-interval study is the recommended extension.

## Submission warning

The source brief requires the exact group-member contribution declaration and
a timestamped screenshot from the institution's virtual lab. Those identity
and external-system artifacts must be genuine; see submission/group_details.json
and docs/virtual_lab_runbook.md before uploading the final PDF.

## References

- Mnih et al., Human-level control through deep reinforcement learning,
  Nature 518, 529-533 (2015), DOI 10.1038/nature14236.
- van Hasselt, Guez, and Silver, Deep Reinforcement Learning with Double
  Q-Learning, AAAI 30(1) (2016), DOI 10.1609/aaai.v30i1.10295.
- Gymnasium, LunarLander-v3 environment documentation.

## License

Code is released under the MIT License. The submitted analysis, generated
results, and assignment brief remain subject to their respective academic and
institutional policies.
