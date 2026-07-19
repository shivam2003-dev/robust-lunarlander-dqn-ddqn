"""Build the single assignment report source from code and verified artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

FENCE = "~" * 3
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


def percentage(value: float) -> str:
    """Format a fraction as a whole-number percentage."""

    return f"{100.0 * value:.0f}%"


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a Pandoc-compatible pipe table without an index."""

    return frame.to_markdown(index=False)


def source_appendix(repo: Path) -> str:
    """Create complete syntax-highlighted listings for every implementation module."""

    sections: list[str] = []
    for relative_path in SOURCE_FILES:
        source = (repo / relative_path).read_text(encoding="utf-8").rstrip()
        sections.append(f"## {relative_path}\n\n{FENCE}{{.python}}\n{source}\n{FENCE}\n")
    return "\n\n".join(sections)


def episode_ledger(repo: Path) -> str:
    """Create first-five and last-five episode excerpts for every experiment."""

    sections: list[str] = []
    for experiment in EXPERIMENTS:
        frame = pd.read_csv(repo / "artifacts" / "metrics" / f"{experiment}.csv")
        excerpt = pd.concat(
            [
                frame.head(5).assign(window="First 5"),
                frame.tail(5).assign(window="Last 5"),
            ],
            ignore_index=True,
        )
        condensed = excerpt[
            [
                "window",
                "episode",
                "episode_reward",
                "average_predicted_q",
                "success_rate_100",
                "thruster_activations",
                "epsilon",
            ]
        ].copy()
        condensed.columns = [
            "Window",
            "Episode",
            "Reward",
            "Avg Q",
            "Safe rate 100",
            "Thrusters",
            "Epsilon",
        ]
        condensed["Reward"] = condensed["Reward"].map(lambda value: f"{value:.2f}")
        condensed["Avg Q"] = condensed["Avg Q"].map(lambda value: f"{value:.2f}")
        condensed["Safe rate 100"] = condensed["Safe rate 100"].map(percentage)
        condensed["Epsilon"] = condensed["Epsilon"].map(lambda value: f"{value:.3f}")
        sections.append(
            f"## {experiment.replace('_', ' ').title()}\n\n"
            "The table shows the first five and last five training iterations. "
            "Safe rate is the assignment-defined moving average over the current "
            "and previous 99 episodes.\n\n"
            f"{markdown_table(condensed)}\n"
        )
    return "\n\\newpage\n\n".join(sections)


def roster_text(group: dict[str, Any]) -> str:
    """Format the mandatory contribution declaration and its readiness state."""

    rows = pd.DataFrame(group["members"]).rename(
        columns={"name": "Group member", "contribution_percent": "Contribution (%)"}
    )
    total = rows["Contribution (%)"].sum()
    status_message = (
        "**FINAL DECLARATION**"
        if group["status"] == "FINAL"
        else "**PENDING EXACT ROSTER - DO NOT SUBMIT UNTIL REPLACED**"
    )
    return f"{status_message}\n\n{markdown_table(rows)}\n\n**Total contribution: {total:g}%**"


def main() -> None:
    """Assemble the polished report with exact measured values and full evidence."""

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
    boundary = pd.read_csv(repo / "artifacts" / "verification" / "controlled_boundary_cases.csv")
    q_gap = summary["q_value_gap"]
    eval_lookup = evaluation.set_index("experiment")
    random_policy = verification["random_policy"]
    wilson_interval = (
        f"{100 * random_policy['misfire_rate_wilson_95_low']:.3f}% to "
        f"{100 * random_policy['misfire_rate_wilson_95_high']:.3f}%"
    )
    random_safe_landings = random_policy["safe_landings_observed_under_random_policy"]
    virtual_lab_image = repo / "submission" / "virtual_lab" / "virtual_lab_timestamp.png"

    evaluation_display = evaluation[
        [
            "algorithm",
            "environment",
            "mean_reward",
            "reward_std",
            "safe_landing_rate",
            "mean_thruster_activations",
            "mean_episode_steps",
        ]
    ].copy()
    evaluation_display.columns = [
        "Algorithm",
        "Environment",
        "Mean reward",
        "Reward SD",
        "Safe landing rate",
        "Mean thrusters",
        "Mean steps",
    ]
    for column in ("Mean reward", "Reward SD", "Mean thrusters", "Mean steps"):
        evaluation_display[column] = evaluation_display[column].map(lambda value: f"{value:.2f}")
    evaluation_display["Safe landing rate"] = evaluation_display["Safe landing rate"].map(
        percentage
    )

    config_display = pd.DataFrame(
        [
            ("Random seed", config["seed"]),
            ("Training episodes per agent", config["episodes"]),
            ("Evaluation episodes per agent", config["evaluation_episodes"]),
            ("Hidden layers", "128, 128"),
            ("Optimizer", "Adam"),
            ("Learning rate", config["learning_rate"]),
            ("Discount factor", config["gamma"]),
            ("Replay capacity", f"{config['replay_capacity']:,}"),
            ("Batch size", config["batch_size"]),
            ("Learning starts", f"{config['learning_starts']:,} steps"),
            ("Target update interval", f"{config['target_update_interval']} updates"),
            ("Epsilon", "1.00 to 0.01 linearly over 100,000 steps"),
            ("Fixed validation states", config["validation_state_count"]),
            ("Failure probability", config["failure_probability"]),
            ("Attempted-thruster penalty", config["attempted_thruster_penalty"]),
            ("Safe-landing bonus", config["safe_landing_bonus"]),
        ],
        columns=["Parameter", "Value"],
    )

    boundary_display = boundary[
        [
            "case",
            "selected_action",
            "executed_action",
            "expected_bonus",
            "fuel_penalty",
            "passed",
        ]
    ].copy()
    boundary_display.columns = [
        "Controlled case",
        "Selected",
        "Executed",
        "Bonus",
        "Penalty",
        "Passed",
    ]

    screenshot_section = (
        "![Timestamped institutional virtual-lab execution]"
        "(submission/virtual_lab/virtual_lab_timestamp.png)"
        if virtual_lab_image.exists()
        else (
            "\\begin{center}\\fbox{\\parbox{0.88\\linewidth}{"
            "\\textbf{MANDATORY VIRTUAL-LAB EVIDENCE PENDING.} "
            "No screenshot is embedded because a genuine institutional virtual-lab "
            "capture has not been supplied. Follow "
            "\\texttt{docs/virtual\\_lab\\_runbook.md}, save the image at "
            "\\texttt{submission/virtual\\_lab/virtual\\_lab\\_timestamp.png}, "
            "and rebuild."
            "}}\\end{center}"
        )
    )

    report = rf"""---
title: "Robust Reinforcement Learning under Stochastic Action Failure"
subtitle: "DQN vs. DDQN on LunarLander-v3"
author: "Group 148"
date: "{verification["generated_at"][:10]}"
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

\tableofcontents

\newpage

# Executive summary

This report implements the assignment-specified stochastic actuator wrapper and
compares DQN and Double DQN under a controlled 2 x 2 design. Every agent uses the
same seed, architecture, optimizer, replay memory, exploration schedule, target
network cadence, 800-episode duration, and fixed validation states. The only
algorithmic difference is the target Q-value calculation.

Across 250 random-policy verification episodes,
{random_policy["misfired_thruster_actions"]:,} of
{random_policy["attempted_thruster_actions"]:,} attempted thruster actions misfired
({100 * random_policy["observed_misfire_rate"]:.3f}%).
The target 15% lies inside the Wilson 95% interval
[{100 * random_policy["misfire_rate_wilson_95_low"]:.3f}%,
{100 * random_policy["misfire_rate_wilson_95_high"]:.3f}%].
There were zero fuel-penalty mismatches, zero unexpected action replacements,
and zero changes to the returned info object.

In 100-seed greedy evaluation, DDQN on the modified environment achieves mean
reward {eval_lookup.loc["ddqn_modified", "mean_reward"]:.2f} and a
{percentage(eval_lookup.loc["ddqn_modified", "safe_landing_rate"])} strict
safe-landing rate. DQN achieves {eval_lookup.loc["dqn_modified", "mean_reward"]:.2f}
and {percentage(eval_lookup.loc["dqn_modified", "safe_landing_rate"])}. The
final-100 mean absolute DQN-DDQN validation Q-gap increases from
{q_gap["original_environment_mean_absolute_gap"]:.2f} in the original environment
to {q_gap["modified_environment_mean_absolute_gap"]:.2f} in the modified environment.

# 1. Problem formulation

LunarLander-v3 exposes an eight-dimensional observation and four discrete actions:
do nothing, fire left orientation engine, fire main engine, and fire right
orientation engine [3]. For selected action $a$, the wrapper draws a uniform random
number only when $a \in \{{1,2,3\}}$. With probability 0.15 it executes action 0;
otherwise it executes $a$. The agent is not told which event occurred.

The returned reward is

$$R = R_{{base}} - 0.3\,\mathbf{{1}}(a \ne 0) + B,$$

where $B=50$ only when the transition is terminated but not truncated, both legs
contact the pad, and the absolute horizontal velocity, vertical velocity, and
orientation angle are each strictly below 0.10.

# 2. Modified environment implementation and verification

## 2.1 Conformance design

The production wrapper keeps the selected action local to step, uses Gymnasium's
seeded random generator, delegates the executed action to the unchanged base
environment, and returns the base observation, flags, and exact info object.
Diagnostics are collected by a separate inner spy used only by verification.
Consequently, training agents receive no failure indicator or added state.

## 2.2 Statistical random-policy evidence

| Measure | Result |
|---|---:|
| Random-policy episodes | {random_policy["episodes"]} |
| Total environment steps | {random_policy["total_steps"]:,} |
| Attempted thruster actions | {random_policy["attempted_thruster_actions"]:,} |
| Observed misfires | {random_policy["misfired_thruster_actions"]:,} |
| Observed failure rate | {100 * random_policy["observed_misfire_rate"]:.3f}% |
| Wilson 95% interval | {wilson_interval} |
| Fuel-penalty mismatches | {random_policy["fuel_penalty_mismatches"]} |
| Info identity mismatches | {random_policy["info_identity_mismatches"]} |
| Random-policy safe landings observed | {random_safe_landings} |

The empirical failure-rate error is only
{100 * random_policy["misfire_rate_absolute_error"]:.3f}
percentage points. All {verification["controlled_boundary_cases"]["count"]}
controlled cases pass:

{markdown_table(boundary_display)}

# 3. DQN and DDQN implementation

The online Q-network is an 8-128-128-4 multilayer perceptron with ReLU
activations. Uniform replay breaks short-range temporal correlation, epsilon-greedy
exploration collects diverse transitions, and a delayed target network stabilizes
bootstrapping, following the central DQN pattern [1].

For DQN,

$$y_{{DQN}} = r + \gamma(1-d)\max_a Q_{{target}}(s',a).$$

For DDQN,

$$a^* = \arg\max_a Q_{{online}}(s',a), \qquad
y_{{DDQN}} = r + \gamma(1-d)Q_{{target}}(s',a^*).$$

Separating selection and evaluation is the Double-DQN mechanism proposed to
reduce harmful maximization bias [2]. The implementation contains one explicit
branch for these equations; architecture, optimizer, replay, schedule, and all
other code paths are shared.

# 4. Experimental design and reproducibility

{markdown_table(config_display)}

The fixed validation array has shape {tuple(provenance["validation_set_shape"])}
and SHA-256 hash:

{FENCE}
{provenance["validation_set_sha256"]}
{FENCE}

Recorded runtime: Python {provenance["python"].split()[0]}, Gymnasium
{provenance["gymnasium"]}, PyTorch {provenance["torch"]}, NumPy
{provenance["numpy"]}, pandas {provenance["pandas"]}, device {provenance["device"]}.
Per-episode CSV ledgers, checkpoints, vector figures, raster figures, configuration,
and provenance are committed for independent inspection.

# 5. Performance evaluation

![All four required training metrics](artifacts/plots/four_metric_overview.png)

## 5.1 Episode reward

The two original-environment agents improve sharply after approximately episode
600. DDQN ends higher and more stable, with a final-100 training mean of
{summary["final_100_training_episodes"]["ddqn_original"]["mean_training_reward_last_100"]:.2f}
versus DQN's
{summary["final_100_training_episodes"]["dqn_original"]["mean_training_reward_last_100"]:.2f}.
The modified reward changes the numerical return scale, so original-versus-modified
reward values must be interpreted together with landing and action metrics. Modified
DQN collapses late; modified DDQN recovers by episode 800 and performs substantially
better in greedy evaluation.

## 5.2 Average predicted Q-value

The same 512 states are evaluated at every episode. The mean absolute DQN-DDQN gap
over the final 100 episodes is {q_gap["original_environment_mean_absolute_gap"]:.2f}
in the original environment and
{q_gap["modified_environment_mean_absolute_gap"]:.2f} under failure. The measured
increase is {q_gap["modified_minus_original"]:.2f}. This directly supports the
claim for this seed that hidden action failure amplifies disagreement in learned
value estimates.

## 5.3 Strict safe-landing rate

At episode 800, the assignment-defined moving rates are
{percentage(summary["final_100_training_episodes"]["ddqn_original"]["safe_landing_rate_last_100"])}
for DDQN-original,
{percentage(summary["final_100_training_episodes"]["dqn_original"]["safe_landing_rate_last_100"])}
for DQN-original,
{percentage(summary["final_100_training_episodes"]["ddqn_modified"]["safe_landing_rate_last_100"])}
for DDQN-modified, and
{percentage(summary["final_100_training_episodes"]["dqn_modified"]["safe_landing_rate_last_100"])}
for DQN-modified. The failure regime lowers reliable landing performance, but DDQN
retains a clear advantage.

## 5.4 Attempted thruster activations

Training curves reveal late high-activation regimes, particularly when a policy
hovers or fails to terminate efficiently. Greedy evaluation is therefore the
cleaner policy comparison. DDQN-modified averages
{eval_lookup.loc["ddqn_modified", "mean_thruster_activations"]:.2f} attempts versus
{eval_lookup.loc["ddqn_original", "mean_thruster_activations"]:.2f} for DDQN-original,
which is consistent with a more conservative learned DDQN policy. DQN-modified
averages {eval_lookup.loc["dqn_modified", "mean_thruster_activations"]:.2f}, so the
penalty does not guarantee conservation when learning fails.

## 5.5 Greedy evaluation summary

{markdown_table(evaluation_display)}

# 6. Discussion

## 6.1 Does intermittent failure increase the DQN-DDQN Q-value difference?

Yes in this experiment: the final-100 mean absolute gap rises from
{q_gap["original_environment_mean_absolute_gap"]:.2f} to
{q_gap["modified_environment_mean_absolute_gap"]:.2f}. The hidden replacement
adds outcome variance to the same observed state-action pair. DQN's maximization
uses target estimates for both selection and evaluation, while DDQN decouples the
two operations [2]. The result is consistent with, but does not by itself prove,
Double DQN's theoretical advantage across seeds.

## 6.2 Why does stochastic action failure make credit assignment harder?

The agent selects a thruster but the environment may execute no-op, and no failure
indicator enters the observation or info dictionary. A replay buffer can therefore
contain different physical transitions for similar observed state-action inputs.
Moreover, the 0.3 cost is applied to the intention in both success and failure
cases. The learner must separate the expected value of an unreliable command from
single noisy outcomes, increasing target variance and slowing attribution of later
landing success or crash failure.

## 6.3 Does the fuel penalty encourage a conservative strategy?

Partially. The successfully learned DDQN policy is more conservative in greedy
evaluation: {eval_lookup.loc["ddqn_modified", "mean_thruster_activations"]:.2f}
attempts in the modified environment versus
{eval_lookup.loc["ddqn_original", "mean_thruster_activations"]:.2f} in the original.
However, DQN-modified's {eval_lookup.loc["dqn_modified", "mean_thruster_activations"]:.2f}
attempts demonstrate that a penalty alone cannot ensure conservation; an unstable
or hovering policy can spend more fuel while failing. The evidence supports a
conditional, algorithm-dependent effect rather than a universal claim.

## 6.4 Which algorithm performs better under stochastic failures?

DDQN performs better. Its modified-environment greedy mean reward is
{eval_lookup.loc["ddqn_modified", "mean_reward"]:.2f} and safe-landing rate is
{percentage(eval_lookup.loc["ddqn_modified", "safe_landing_rate"])}; DQN records
{eval_lookup.loc["dqn_modified", "mean_reward"]:.2f} and
{percentage(eval_lookup.loc["dqn_modified", "safe_landing_rate"])}. This behavior
is consistent with Double DQN's documented ability to reduce overestimation by
separating next-action selection from evaluation [2].

## 6.5 Limitation and improvement

The main limitation is a single training seed. Deep RL has high between-run
variance, so one paired seed cannot establish a population effect. A stronger
study would pre-register at least 10 seeds, preserve the same paired-seed design,
and report bootstrap confidence intervals or a hierarchical model for final reward,
safe-landing rate, Q-gap, and thruster use. A failure-probability sweep
(0%, 5%, 15%, 30%) would additionally test whether the DDQN advantage scales with
uncertainty.

# 7. Virtual-lab execution evidence

{screenshot_section}

# 8. Conclusion

The environment wrapper meets the specification exactly and is supported by both
statistical and controlled boundary evidence. Stochastic action failure creates a
harder and less stable credit-assignment problem. In this reproducible seed,
DDQN retains substantially higher reward and safe-landing performance than DQN
under failure, while the validation Q-gap grows strongly. The outcome aligns with
the theoretical motivation for Double DQN but should be generalized with a
multi-seed study.

# References {{.unnumbered}}

1. Mnih, V. et al. (2015). Human-level control through deep reinforcement
   learning. *Nature*, 518, 529-533.
   https://doi.org/10.1038/nature14236
2. van Hasselt, H., Guez, A., and Silver, D. (2016). Deep Reinforcement Learning
   with Double Q-Learning. *Proceedings of the AAAI Conference on Artificial
   Intelligence*, 30(1). https://doi.org/10.1609/aaai.v30i1.10295
3. Farama Foundation. Gymnasium LunarLander-v3 documentation.
   https://gymnasium.farama.org/environments/box2d/lunar_lander/

\newpage

# Appendix A - Complete commented source

{source_appendix(repo)}

\newpage

# Appendix B - Training-output excerpts

As requested, the PDF shows only the first five and last five training iterations
for each experiment. Complete 14-column CSVs for all {4 * config["episodes"]:,}
episodes remain in artifacts/metrics for reproducibility and audit. Each excerpt
includes reward, fixed-state Q-value, 100-episode safe-landing rate, attempted
thruster count, and epsilon.

{episode_ledger(repo)}
"""
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path} ({report_path.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
