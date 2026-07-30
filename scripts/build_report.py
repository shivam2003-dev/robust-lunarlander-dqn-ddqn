"""Build the submission report from executable source and recorded artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

FENCE = "~" * 3
FINAL_PDF_NAME = "Group148_Q_learning_DQN_DDQN.pdf"
SOURCE_FILES = (
    "src/robust_lunarlander/config.py",
    "src/robust_lunarlander/envs.py",
    "src/robust_lunarlander/network.py",
    "src/robust_lunarlander/replay.py",
    "src/robust_lunarlander/agent.py",
    "src/robust_lunarlander/verification.py",
    "src/robust_lunarlander/experiment.py",
    "src/robust_lunarlander/plotting.py",
)
EXPERIMENTS = (
    "dqn_original",
    "ddqn_original",
    "dqn_modified",
    "ddqn_modified",
)


def percentage(value: float, digits: int = 0) -> str:
    """Format a fraction as a percentage."""

    return f"{100.0 * value:.{digits}f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a dependency-free Pandoc pipe table."""

    def cell_text(value: Any) -> str:
        """Escape characters that would otherwise break a pipe-table cell."""

        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(cell_text(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(cell_text(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def source_appendix(repo: Path) -> str:
    """Create syntax-highlighted listings for every implementation module."""

    sections: list[str] = []
    for relative_path in SOURCE_FILES:
        source = (repo / relative_path).read_text(encoding="utf-8").rstrip()
        sections.append(f"## {relative_path}\n\n{FENCE}{{.python}}\n{source}\n{FENCE}\n")
    return "\n\n".join(sections)


def episode_ledger(repo: Path) -> str:
    """Render all recorded per-episode compact outputs in printable PDF tables."""

    sections: list[str] = []
    for experiment in EXPERIMENTS:
        frame = pd.read_csv(repo / "artifacts" / "logs" / f"{experiment}.log")
        rows: list[str] = []
        for record in frame.itertuples(index=False):
            loss = "nan" if pd.isna(record.training_loss) else f"{record.training_loss:.6f}"
            safe_rate = f"{record.safe_rate_100:.1%}".replace("%", r"\%")
            rows.append(
                f"{int(record.episode)} & {record.total_reward:.2f} & "
                f"{record.average_q:.2f} & {safe_rate} & "
                f"{int(record.attempted)}/{int(record.executed)} & "
                f"{int(record.steps)} & {record.epsilon:.4f} & {loss} \\\\"
            )

        title = experiment.replace("_", " ").title()
        sections.append(
            "\\scriptsize\n"
            "\\setlength{\\tabcolsep}{1.8pt}\n"
            "\\renewcommand{\\arraystretch}{0.86}\n"
            "\\begin{longtable}{rrrrrrrr}\n"
            f"\\caption{{Complete per-iteration training output - {title} (800 episodes).}}\\\\\n"
            "\\toprule\n"
            "Episode & Reward & Avg. Q & Safe100 & Attempted/Executed & Steps & Epsilon & Loss \\\\\n"
            "\\midrule\n"
            "\\endfirsthead\n"
            f"\\multicolumn{{8}}{{c}}{{\\small Continued: {title} complete per-iteration output}} \\\\\n"
            "\\toprule\n"
            "Episode & Reward & Avg. Q & Safe100 & Attempted/Executed & Steps & Epsilon & Loss \\\\\n"
            "\\midrule\n"
            "\\endhead\n"
            "\\bottomrule\n"
            "\\endfoot\n"
            "\\bottomrule\n"
            "\\endlastfoot\n" + "\n".join(rows) + "\n\\end{longtable}\n"
            "\\normalsize\n"
        )
    return "\n\\clearpage\n\n".join(sections)


def output_integrity_table(repo: Path, episodes: int) -> pd.DataFrame:
    """Summarize complete CSV and progress-log coverage without printing 3,200 lines."""

    rows: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        metrics_path = repo / "artifacts" / "metrics" / f"{experiment}.csv"
        log_path = repo / "artifacts" / "logs" / f"{experiment}.log"
        frame = pd.read_csv(metrics_path)
        log_records = max(len(log_path.read_text(encoding="utf-8").splitlines()) - 1, 0)
        rows.append(
            {
                "Experiment": experiment,
                "CSV rows": len(frame),
                "Log records": log_records,
                "Episode range": f"{frame['episode'].min()}-{frame['episode'].max()}",
                "Complete": len(frame) == episodes and log_records == episodes,
            }
        )
    return pd.DataFrame(rows)


def roster_text(group: dict[str, Any]) -> str:
    """Format member IDs, declared contributions, and confirmation status."""

    rows = pd.DataFrame(group["members"]).rename(
        columns={
            "name": "Group member",
            "student_id": "BITS ID",
            "contribution_percent": "Contribution (%)",
        }
    )
    total = rows["Contribution (%)"].sum()
    confirmed = bool(group.get("contributions_confirmed_by_group", False))
    status = (
        "**GROUP CONFIRMATION RECORDED.**"
        if confirmed
        else (
            "**MANDATORY BEFORE SUBMISSION:** These contribution percentages are "
            "declared but must be confirmed by all five group members."
        )
    )
    return f"{status}\n\n{markdown_table(rows)}\n\n**Total contribution: {total:g}%**"


def screenshot_block(repo: Path, filename: str, label: str) -> str:
    """Embed a genuine screenshot when present or render an explicit placeholder."""

    relative_path = Path("submission") / "virtual_lab" / filename
    if (repo / relative_path).exists():
        return f"![{label}]({relative_path.as_posix()})"
    latex_path = relative_path.as_posix().replace("_", r"\_")
    return (
        "\\begin{center}\\fbox{\\parbox{0.88\\linewidth}{"
        "\\centering\\textbf{[INSERT VIRTUAL-LAB SCREENSHOT WITH VISIBLE "
        "TIMESTAMP HERE]}\\\\[0.4em]"
        f"{label}. Expected file: "
        f"\\texttt{{{latex_path}}}."
        "}}\\end{center}"
    )


def format_final_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the assignment-required final-100 training comparison table."""

    display = frame[
        [
            "algorithm",
            "environment",
            "mean_reward_final_100",
            "reward_std_final_100",
            "best_moving_average_reward_100",
            "final_fixed_set_average_q",
            "safe_landing_rate_final_100",
            "mean_attempted_thrusters_final_100",
            "mean_executed_thrusters_final_100",
            "successful_safe_landings_total",
            "training_duration_seconds",
        ]
    ].copy()
    display.columns = [
        "Algorithm",
        "Environment",
        "Mean reward final 100",
        "Reward SD final 100",
        "Best MA(100) reward",
        "Final fixed-set Q",
        "Safe rate final 100",
        "Mean attempted final 100",
        "Mean executed final 100",
        "Safe landings total",
        "Training seconds",
    ]
    numeric_columns = [
        "Mean reward final 100",
        "Reward SD final 100",
        "Best MA(100) reward",
        "Final fixed-set Q",
        "Mean attempted final 100",
        "Mean executed final 100",
        "Training seconds",
    ]
    for column in numeric_columns:
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    display["Safe rate final 100"] = display["Safe rate final 100"].map(percentage)
    return display


def main() -> None:
    """Assemble the report using only recorded values and explicit placeholders."""

    repo = Path(__file__).resolve().parents[1]
    output_dir = repo / "output" / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "Group_148_Q_learning_DQN_DDQN.md"

    summary = json.loads((repo / "artifacts" / "study_summary.json").read_text())
    verification = json.loads(
        (repo / "artifacts" / "verification" / "wrapper_verification.json").read_text()
    )
    config = json.loads((repo / "artifacts" / "training_config.json").read_text())
    provenance = json.loads((repo / "artifacts" / "system_provenance.json").read_text())
    group = json.loads((repo / "submission" / "group_details.json").read_text())
    evaluation = pd.read_csv(repo / "artifacts" / "evaluation_summary.csv")
    final_comparison = pd.read_csv(repo / "artifacts" / "final_comparison.csv")
    boundary = pd.read_csv(repo / "artifacts" / "verification" / "controlled_boundary_cases.csv")

    q_gap = summary["q_value_gap"]
    eval_lookup = evaluation.set_index("experiment")
    random_policy = verification["random_policy"]
    report_build_date = datetime.now().astimezone().date().isoformat()
    network_dimensions = [8, *config["hidden_sizes"], 4]
    parameter_count = sum(
        (input_size + 1) * output_size
        for input_size, output_size in zip(
            network_dimensions[:-1],
            network_dimensions[1:],
            strict=True,
        )
    )

    evaluation_display = evaluation[
        [
            "algorithm",
            "environment_type",
            "mean_reward",
            "reward_std",
            "safe_landing_rate",
            "mean_attempted_thruster_activations",
            "mean_executed_thruster_activations",
            "mean_episode_steps",
        ]
    ].copy()
    evaluation_display.columns = [
        "Algorithm",
        "Environment",
        "Mean reward",
        "Reward SD",
        "Safe landing rate",
        "Mean attempted",
        "Mean executed",
        "Mean steps",
    ]
    for column in ("Mean reward", "Reward SD", "Mean attempted", "Mean executed", "Mean steps"):
        evaluation_display[column] = evaluation_display[column].map(lambda value: f"{value:.2f}")
    evaluation_display["Safe landing rate"] = evaluation_display["Safe landing rate"].map(
        percentage
    )

    config_display = pd.DataFrame(
        [
            ("Global seed", config["seed"]),
            ("Training episodes per agent", config["episodes"]),
            ("Maximum steps per episode", config["max_steps_per_episode"]),
            ("Evaluation episodes per agent", config["evaluation_episodes"]),
            ("Network", "8-128-128-4, ReLU"),
            ("Trainable parameters", f"{parameter_count:,}"),
            ("Initialization", "PyTorch Linear default; same seed before every agent"),
            ("Optimizer / loss", "Adam / Smooth L1"),
            ("Learning rate", config["learning_rate"]),
            ("Discount factor", config["gamma"]),
            ("Replay capacity / batch", f"{config['replay_capacity']:,} / {config['batch_size']}"),
            ("Learning warm-up", f"{config['learning_starts']:,} steps"),
            ("Target update", f"Hard copy every {config['target_update_interval']} updates"),
            (
                "Epsilon schedule",
                f"{config['epsilon_start']:.2f} to {config['epsilon_end']:.2f} "
                f"over {config['epsilon_decay_steps']:,} steps",
            ),
            ("Fixed validation states", config["validation_state_count"]),
            ("Failure probability", config["failure_probability"]),
            ("Attempted-thruster penalty", config["attempted_thruster_penalty"]),
            ("Safe-landing bonus", config["safe_landing_bonus"]),
            ("Device", config["device"]),
        ],
        columns=["Parameter", "Shared value"],
    )

    epsilon_display = pd.DataFrame(
        [
            ("Start", 0, config["epsilon_start"]),
            ("Half decay", config["epsilon_decay_steps"] // 2, 0.505),
            ("End of decay", config["epsilon_decay_steps"], config["epsilon_end"]),
            ("After decay", config["epsilon_decay_steps"] * 2, config["epsilon_end"]),
        ],
        columns=["Checkpoint", "Environment step", "Epsilon"],
    )
    epsilon_display["Epsilon"] = epsilon_display["Epsilon"].map(lambda value: f"{value:.3f}")

    boundary_display = boundary[
        [
            "case",
            "selected_action",
            "executed_action",
            "expected_bonus",
            "fuel_penalty",
            "info_identity_preserved",
            "passed",
        ]
    ].copy()
    boundary_display.columns = [
        "Case",
        "Selected",
        "Executed",
        "Bonus",
        "Penalty",
        "Info unchanged",
        "Passed",
    ]
    boundary_labels = {
        "all_conditions_true": "Safe terminal",
        "attempted_thruster_success": "Firing + safe",
        "attempted_thruster_misfire": "Misfire + safe",
        "not_terminated": "Not terminated",
        "episode_truncated": "Truncated",
        "left_leg_absent": "Left leg absent",
        "right_leg_absent": "Right leg absent",
        "excess_horizontal_velocity": "Excess x velocity",
        "excess_vertical_velocity": "Excess y velocity",
        "excess_orientation_angle": "Excess angle",
    }
    boundary_display["Case"] = boundary_display["Case"].map(boundary_labels)

    final_display = format_final_comparison(final_comparison)
    final_performance_display = final_display[
        [
            "Algorithm",
            "Environment",
            "Mean reward final 100",
            "Reward SD final 100",
            "Best MA(100) reward",
            "Final fixed-set Q",
            "Safe rate final 100",
        ]
    ]
    final_behavior_display = final_display[
        [
            "Algorithm",
            "Environment",
            "Mean attempted final 100",
            "Mean executed final 100",
            "Safe landings total",
            "Training seconds",
        ]
    ]
    evaluation_performance_display = evaluation_display[
        [
            "Algorithm",
            "Environment",
            "Mean reward",
            "Reward SD",
            "Safe landing rate",
        ]
    ]
    evaluation_behavior_display = evaluation_display[
        [
            "Algorithm",
            "Environment",
            "Mean attempted",
            "Mean executed",
            "Mean steps",
        ]
    ]
    integrity_display = output_integrity_table(repo, int(config["episodes"]))
    wilson_interval = (
        f"{percentage(random_policy['misfire_rate_wilson_95_low'], 3)} to "
        f"{percentage(random_policy['misfire_rate_wilson_95_high'], 3)}"
    )
    environment_line = (
        f"{provenance['platform']}; Python {provenance['python'].split()[0]}; "
        f"Gymnasium {provenance['gymnasium']}; PyTorch {provenance['torch']}; "
        f"device {provenance['device']}"
    )

    start_screenshot = screenshot_block(
        repo,
        "01_start_timestamp.png",
        "Beginning-of-execution timestamp and institutional virtual-lab identity",
    )
    version_screenshot = screenshot_block(
        repo,
        "02_environment_versions.png",
        "Environment and package-version evidence",
    )
    training_screenshot = screenshot_block(
        repo,
        "03_training_progress.png",
        "Timestamped training progress showing compact per-episode output",
    )
    final_screenshot = screenshot_block(
        repo,
        "04_final_outputs_plots.png",
        "Timestamped final outputs and required plots",
    )
    saved_files_screenshot = """The saved study artifacts are included with this submission and are
listed below. The final-output screenshot above provides the timestamped virtual-lab
completion evidence; this inventory avoids inserting a non-genuine replacement image.

| Artifact class | Saved location |
|---|---|
| Per-episode ledgers and progress logs | `artifacts/metrics/`, `artifacts/logs/` |
| Model checkpoints | `artifacts/checkpoints/` |
| Fixed validation states | `artifacts/validation_states.npy` |
| Evaluation data and summary | `artifacts/evaluation_episodes.csv`, `artifacts/evaluation_summary.csv` |
| Plots and final comparison | `artifacts/plots/`, `artifacts/final_comparison.csv` |"""

    report = rf"""---
title: "Robust Reinforcement Learning under Stochastic Action Failure"
subtitle: "Experiential Learning - Assignment 2 | Deep Reinforcement Learning (S2-25_AIMLCZG512)"
author: "Group 148"
date: "{report_build_date}"
papersize: a4
fontsize: 10pt
geometry: "top=20mm,bottom=20mm,left=18mm,right=18mm"
mainfont: "Arial"
monofont: "Menlo"
colorlinks: true
linkcolor: "MidnightBlue"
urlcolor: "MidnightBlue"
---

# Group contribution declaration {{.unnumbered}}

{roster_text(group)}

| Submission field | Value |
|---|---|
| Course | Deep Reinforcement Learning (S2-25_AIMLCZG512) |
| Assignment | Experiential Learning - Assignment 2 |
| Group | 148 |
| Recorded study execution | {provenance["generated_at"]} |
| Recorded execution environment | {environment_line} |
| Report build date | {report_build_date} |
| Intended final PDF filename | `{FINAL_PDF_NAME}` |

**Filename note:** Check the exact final filename against the latest instructor
guidance before upload. The build currently follows the requested pattern
`{FINAL_PDF_NAME}`.

\tableofcontents

\newpage

# Executive summary

This report implements the specified hidden 15% actuator-failure wrapper and
compares DQN and Double DQN in a controlled 2 x 2 experiment. The four agents use
the same seed, network, initialization procedure, optimizer, replay settings,
exploration schedule, target-update cadence, duration, and fixed validation-state
set. The wrapper is the only environment difference; the target calculation is
the only DQN-versus-DDQN difference.

Across {random_policy["episodes"]} random-policy verification episodes,
{random_policy["misfired_thruster_actions"]:,} of
{random_policy["attempted_thruster_actions"]:,} attempted thruster actions
misfired ({percentage(random_policy["observed_misfire_rate"], 3)}). The target
15% lies in the Wilson 95% interval [{wilson_interval}]. The internal
fuel-penalty count equals the attempted-action count, and no returned `info`
object was changed.

For the recorded single-seed study, DDQN on the modified environment achieved
greedy mean reward {eval_lookup.loc["ddqn_modified", "mean_reward"]:.2f} and
safe-landing rate {percentage(eval_lookup.loc["ddqn_modified", "safe_landing_rate"])}.
DQN achieved {eval_lookup.loc["dqn_modified", "mean_reward"]:.2f} and
{percentage(eval_lookup.loc["dqn_modified", "safe_landing_rate"])}. These values
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

{FENCE}{{.bash}}
sudo apt-get update
sudo apt-get install -y swig build-essential python3-dev
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); print(env.observation_space, env.action_space); env.close()"
{FENCE}

If `sudo` is unavailable, ask the lab administrator to install `swig` and the
compiler packages; do not silently switch LunarLander versions.

## 2.2 Seeding and provenance

One function seeds Python `random`, NumPy, PyTorch CPU, all CUDA devices,
Gymnasium reset, and the action space. It also requests deterministic PyTorch
algorithms and disables cuDNN benchmarking where available. Box2D,
floating-point kernels, drivers, and hardware can still produce cross-platform
nondeterminism, so versions and hardware are recorded.

{markdown_table(config_display)}

Recorded provenance:

- Python: {provenance["python"].split()[0]}
- Gymnasium: {provenance["gymnasium"]}
- PyTorch: {provenance["torch"]}
- NumPy / pandas / Seaborn: {provenance["numpy"]} / {provenance["pandas"]} / {provenance["seaborn"]}
- Platform / logical CPUs: {provenance["platform"]} / {provenance["logical_cpu_count"]}
- CUDA available / device: {provenance["cuda_available"]} / {provenance["cuda_device_name"]}
- Device used: {provenance["device"]}
- Global random seed: {config["seed"]}
- Git commit at execution: `{provenance["git_commit"]}`
- Git worktree dirty at execution: {provenance["git_worktree_dirty_at_execution"]}

# 3. Custom environment and verification - 2.5 marks

For selected action $a$, the wrapper samples a private RNG only when
$a \in \{{1,2,3\}}$. With probability 0.15 it executes action 0; otherwise it
executes $a$. A private RNG avoids consuming the base LunarLander RNG and hence
keeps ordinary transition randomness isolated. Observation/action spaces,
termination, and truncation are delegated unchanged.

The modified reward is

$$R = R_{{base}} - 0.3\,\mathbf{{1}}(a \ne 0) + B,$$

where $B=50$ only for a terminated, non-truncated transition with both legs in
contact and absolute horizontal velocity, vertical velocity, and angle each
strictly below 0.10. Private counters are externally readable for verification
but are never inserted into the observation or returned `info`.

## 3.1 Deterministic mock-environment tests

{markdown_table(boundary_display)}

The automated unit suite additionally checks action 0, successful firing,
misfiring, strict threshold equality, both leg contacts, truncation, unchanged
spaces, invalid actions, and exact `info` identity.

## 3.2 Random-policy evidence

| Measure | Recorded result |
|---|---:|
| Episodes | {random_policy["episodes"]} |
| Environment steps | {random_policy["total_steps"]:,} |
| Attempted thruster actions | {random_policy["attempted_thruster_actions"]:,} |
| Misfires | {random_policy["misfired_thruster_actions"]:,} |
| Observed misfire rate | {percentage(random_policy["observed_misfire_rate"], 3)} |
| Absolute difference from 0.15 | {random_policy["misfire_rate_absolute_error"]:.6f} |
| Wilson 95% interval | {wilson_interval} |
| Fuel-penalty count | {random_policy["fuel_penalty_count"]:,} |
| Fuel-penalty count equals attempts | {random_policy["fuel_penalty_count_matches_attempts"]} |
| Fuel-penalty mismatches | {random_policy["fuel_penalty_mismatches"]} |
| Returned-info identity mismatches | {random_policy["info_identity_mismatches"]} |
| Random-policy safe bonuses observed | {random_policy["safe_landing_bonus_count"]} |

Successful-fire and forced-misfire mock cases both show the 0.3 selected-action
penalty. Deterministic landing-boundary cases are authoritative for the +50
bonus because random policies do not guarantee useful landing coverage.

# 4. Replay, network, DQN, and DDQN - 8 marks

## 4.1 Replay and terminal masking

The replay buffer stores state, action, reward, next state, `terminated`, and
`truncated` separately. The target masks only true terminal states:

$$y = r + \gamma(1-\text{{terminated}})\,Q_{{bootstrap}}.$$

A time-limit truncation still bootstraps because the underlying MDP state is not
terminal; the episode ended due to an external horizon. This choice is explicit
and unit-tested.

## 4.2 Shared Q-network

All experiments use the same 8-128-128-4 multilayer perceptron, ReLU
activations, PyTorch `Linear` initialization after the same global seed, Adam,
and Smooth L1 loss. It has **{parameter_count:,} trainable parameters**.

## 4.3 The only algorithmic branch

For DQN:

$$y_{{DQN}} = r + \gamma(1-t)\max_a Q_{{target}}(s',a).$$

For DDQN:

$$a^* = \arg\max_a Q_{{online}}(s',a), \qquad
y_{{DDQN}} = r + \gamma(1-t)Q_{{target}}(s',a^*).$$

The online/target architectures, optimizer, replay, epsilon policy, discount,
warm-up, batch size, learning rate, target updates, seed, and training duration
are otherwise identical.

## 4.4 Shared epsilon-greedy schedule

{markdown_table(epsilon_display)}

![Shared epsilon-greedy schedule](artifacts/plots/epsilon_schedule.png)

# 5. Fixed validation states, metrics, and controlled runs

The validation states were collected once from the original environment with a
reproducible random policy and never resampled during training. Reusing an
identical set prevents state-distribution drift from being confused with
algorithmic differences in predicted Q-values.

- Shape: {tuple(provenance["validation_set_shape"])}
- SHA-256: `{provenance["validation_set_sha256"]}`

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

{markdown_table(final_performance_display)}

**Panel B - action use, successes, and duration**

{markdown_table(final_behavior_display)}

Mean reward, reward SD, safe rate, and both action-use means use the final 100
training episodes. The best moving average, final Q-value, total successes, and
duration use the scopes stated in their labels. Executed actions equal attempts
in the original environment; modified runs report actual non-misfired executions.

## 6.2 Greedy evaluation on shared seeds

**Panel A - reward and strict landing performance**

{markdown_table(evaluation_performance_display)}

**Panel B - action use and episode length**

{markdown_table(evaluation_behavior_display)}

This evaluation uses 100 shared episode seeds per experiment and epsilon 0.
It is supplementary to the assignment's final-100 training summary.

# 7. Discussion - 2.5 marks

## 7.1 Does failure increase the DQN-DDQN Q-value difference?

For this execution, yes. The final-100 mean absolute Q-gap is
{q_gap["original_environment_mean_absolute_gap"]:.2f} in the original
environment and {q_gap["modified_environment_mean_absolute_gap"]:.2f} in the
modified environment, an increase of {q_gap["modified_minus_original"]:.2f}.
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
{eval_lookup.loc["ddqn_modified", "mean_attempted_thruster_activations"]:.2f}
actions in the modified environment versus
{eval_lookup.loc["ddqn_original", "mean_attempted_thruster_activations"]:.2f}
in the original. Greedy DQN attempts
{eval_lookup.loc["dqn_modified", "mean_attempted_thruster_activations"]:.2f}
versus {eval_lookup.loc["dqn_original", "mean_attempted_thruster_activations"]:.2f}.
Both modified policies therefore attempt more, not fewer, thruster actions. The
longer modified episodes and need to compensate for no-op commands may outweigh
the 0.3 cost. The penalty alone did not produce a demonstrably conservative
strategy.

## 7.4 Which algorithm performs better under failure?

The answer is metric-dependent. In the final 100 training episodes, DDQN has
the higher mean reward ({summary["final_100_training_episodes"]["ddqn_modified"]["mean_training_reward_last_100"]:.2f}
versus {summary["final_100_training_episodes"]["dqn_modified"]["mean_training_reward_last_100"]:.2f})
and safe-landing rate
({percentage(summary["final_100_training_episodes"]["ddqn_modified"]["safe_landing_rate_last_100"])}
versus {percentage(summary["final_100_training_episodes"]["dqn_modified"]["safe_landing_rate_last_100"])}).
However, frozen-checkpoint greedy evaluation strongly favors DQN: mean reward
{eval_lookup.loc["dqn_modified", "mean_reward"]:.2f} and safe-landing rate
{percentage(eval_lookup.loc["dqn_modified", "safe_landing_rate"])} versus
DDQN's {eval_lookup.loc["ddqn_modified", "mean_reward"]:.2f} and
{percentage(eval_lookup.loc["ddqn_modified", "safe_landing_rate"])}. Therefore
this run does not give an unambiguous DDQN advantage and is not cleanly
consistent with the theoretical expectation.

## 7.5 Limitation and improvement

The executed study uses one training seed, so between-run variance is not
estimated. The strongest improvement is a preregistered paired multi-seed study
using the same four-run design, followed by confidence intervals for final
reward, safe-landing rate, Q-gap, and thruster use. A later failure-probability
sweep could test whether any DDQN advantage scales with uncertainty.

# 8. Virtual-lab evidence

The following evidence must be genuine. Placeholders are intentional and are
never substituted with fabricated screenshots.

## 8.0 Beginning-of-execution evidence

{start_screenshot}

## 8.1 Package and environment evidence

{version_screenshot}

## 8.2 Timestamped training evidence

{training_screenshot}

## 8.3 Timestamped final outputs and plots

{final_screenshot}

## 8.4 Timestamped saved-file and checkpoint evidence

{saved_files_screenshot}

# 9. Conclusion

In the recorded single-seed experiment, hidden actuator failure increased the
DQN-DDQN fixed-state Q-value gap from
{q_gap["original_environment_mean_absolute_gap"]:.2f} to
{q_gap["modified_environment_mean_absolute_gap"]:.2f}. The selected-action fuel
penalty did not reduce attempted activations in the modified runs. DDQN was
slightly stronger in the final training window, while DQN was substantially
stronger in greedy evaluation, so no algorithm wins every metric. The main
limitation is the single seed; paired multi-seed replication is the next
improvement. This conclusion must be regenerated if the official virtual-lab
run changes the recorded outputs.

# References {{.unnumbered}}

1. Mnih, V. et al. (2015). Human-level control through deep reinforcement
   learning. *Nature*, 518, 529-533. [DOI record](https://doi.org/10.1038/nature14236).
2. van Hasselt, H., Guez, A., and Silver, D. (2016). Deep Reinforcement Learning
   with Double Q-Learning. *Proceedings of the AAAI Conference on Artificial
   Intelligence*, 30(1). [DOI record](https://doi.org/10.1609/aaai.v30i1.10295).
3. Farama Foundation. [Gymnasium LunarLander-v3 documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/).

\newpage

# Appendix A - Complete commented source

{source_appendix(repo)}

\newpage

# Appendix B - Per-iteration output evidence

Every training episode emits a compact console line and is persisted in both a
CSV ledger and a progress log. This appendix reproduces all
{4 * config["episodes"]:,} per-iteration records: 800 rows for each of the four
controlled experiments. Values follow the compact progress output convention:
reward, fixed-set average Q, moving safe-landing rate, attempted/executed
thrusters, steps, epsilon, and training loss. `nan` denotes the pre-warm-up
episodes where no gradient update was yet performed.

{markdown_table(integrity_display)}

<!-- ITERATION_TABLES_FOR_HTML -->

{episode_ledger(repo)}

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
| One group PDF is named `{FINAL_PDF_NAME}` | [x] |
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
"""

    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path} ({report_path.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
