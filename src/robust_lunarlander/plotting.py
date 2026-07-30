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
