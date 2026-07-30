"""Audit every committed experiment artifact against the assignment contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENTS = ("dqn_original", "ddqn_original", "dqn_modified", "ddqn_modified")
REQUIRED_METRIC_COLUMNS = {
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
    "episode_seconds",
}


def record(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    """Append one human-readable pass/fail audit result."""

    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def run_audit(repo: Path, *, require_submission_evidence: bool) -> list[dict[str, Any]]:
    """Inspect row counts, hashes, plots, checkpoints, and external submission inputs."""

    artifacts = repo / "artifacts"
    checks: list[dict[str, Any]] = []

    config = json.loads((artifacts / "training_config.json").read_text(encoding="utf-8"))
    episodes = int(config["episodes"])
    evaluation_episodes = int(config["evaluation_episodes"])
    record(checks, "training duration", episodes == 800, f"{episodes} episodes per agent")
    record(checks, "shared seed", int(config["seed"]) == 148, f"seed={config['seed']}")

    provenance = json.loads((artifacts / "system_provenance.json").read_text(encoding="utf-8"))
    validation_path = artifacts / "validation" / "fixed_validation_states.npz"
    states = np.load(validation_path)["states"]
    observed_hash = hashlib.sha256(states.tobytes()).hexdigest()
    record(
        checks,
        "fixed validation set shape",
        states.shape == (512, 8),
        f"shape={states.shape}",
    )
    record(
        checks,
        "fixed validation set hash",
        observed_hash == provenance["validation_set_sha256"],
        observed_hash,
    )

    for name in EXPERIMENTS:
        metrics_path = artifacts / "metrics" / f"{name}.csv"
        checkpoint_path = artifacts / "checkpoints" / f"{name}.pt"
        frame = pd.read_csv(metrics_path)
        expected_episodes = list(range(1, episodes + 1))
        record(
            checks,
            f"{name} episode ledger",
            len(frame) == episodes
            and frame["episode"].tolist() == expected_episodes
            and REQUIRED_METRIC_COLUMNS.issubset(frame.columns),
            (
                f"rows={len(frame)}, range={frame['episode'].min()}-"
                f"{frame['episode'].max()}, schema="
                f"{REQUIRED_METRIC_COLUMNS.issubset(frame.columns)}"
            ),
        )
        log_path = artifacts / "logs" / f"{name}.log"
        log_records = (
            max(len(log_path.read_text(encoding="utf-8").splitlines()) - 1, 0)
            if log_path.exists()
            else 0
        )
        record(
            checks,
            f"{name} per-episode progress log",
            log_records == episodes,
            f"records={log_records}",
        )
        record(
            checks,
            f"{name} checkpoint",
            checkpoint_path.exists() and checkpoint_path.stat().st_size > 0,
            f"{checkpoint_path.stat().st_size if checkpoint_path.exists() else 0} bytes",
        )

    evaluations = pd.read_csv(artifacts / "evaluation_episodes.csv")
    counts = evaluations.groupby("experiment").size().to_dict()
    record(
        checks,
        "greedy evaluation coverage",
        all(counts.get(name) == evaluation_episodes for name in EXPERIMENTS),
        str(counts),
    )

    required_plots = (
        "episode_reward",
        "average_predicted_q",
        "success_rate_100",
        "thruster_activations",
        "four_metric_overview",
        "epsilon_schedule",
    )
    for stem in required_plots:
        png = artifacts / "plots" / f"{stem}.png"
        svg = artifacts / "plots" / f"{stem}.svg"
        record(
            checks,
            f"plot {stem}",
            png.exists() and png.stat().st_size > 0 and svg.exists() and svg.stat().st_size > 0,
            f"png={png.exists()}, svg={svg.exists()}",
        )

    verification = json.loads(
        (artifacts / "verification" / "wrapper_verification.json").read_text(encoding="utf-8")
    )
    random_policy = verification["random_policy"]
    record(
        checks,
        "wrapper verification",
        verification["overall_passed"],
        f"rate={random_policy['observed_misfire_rate']:.6f}",
    )
    record(
        checks,
        "hidden info preserved",
        random_policy["info_identity_mismatches"] == 0,
        f"mismatches={random_policy['info_identity_mismatches']}",
    )
    record(
        checks,
        "attempted-action fuel penalty",
        random_policy["fuel_penalty_mismatches"] == 0
        and random_policy["fuel_penalty_count"] == random_policy["attempted_thruster_actions"],
        (
            f"mismatches={random_policy['fuel_penalty_mismatches']}, "
            f"count={random_policy['fuel_penalty_count']}, "
            f"attempts={random_policy['attempted_thruster_actions']}"
        ),
    )

    final_comparison_path = artifacts / "final_comparison.csv"
    final_comparison = pd.read_csv(final_comparison_path)
    required_final_columns = {
        "mean_reward_final_100",
        "reward_std_final_100",
        "best_moving_average_reward_100",
        "final_fixed_set_average_q",
        "safe_landing_rate_final_100",
        "mean_attempted_thrusters_final_100",
        "mean_executed_thrusters_final_100",
        "successful_safe_landings_total",
        "training_duration_seconds",
    }
    final_schema_ready = required_final_columns.issubset(final_comparison.columns)
    record(
        checks,
        "final comparison table",
        len(final_comparison) == 4 and final_schema_ready,
        f"rows={len(final_comparison)}, schema={final_schema_ready}",
    )

    group_details = json.loads(
        (repo / "submission" / "group_details.json").read_text(encoding="utf-8")
    )
    contribution_total = sum(
        float(member["contribution_percent"]) for member in group_details["members"]
    )
    names_complete = all("REPLACE_" not in member["name"] for member in group_details["members"])
    ids_complete = all(
        bool(member.get("student_id")) and "REPLACE_" not in member["student_id"]
        for member in group_details["members"]
    )
    group_ready = (
        group_details.get("contributions_confirmed_by_group") is True
        and names_complete
        and ids_complete
        and abs(contribution_total - 100.0) < 1e-9
    )
    record(
        checks,
        "group contribution declaration",
        group_ready if require_submission_evidence else True,
        (
            f"status={group_details['status']}, ids_complete={ids_complete}, "
            f"confirmed={group_details.get('contributions_confirmed_by_group')}, "
            f"total={contribution_total:g}%"
        ),
    )

    screenshot_names = (
        "01_start_timestamp.png",
        "02_environment_versions.png",
        "03_training_progress.png",
        "04_final_outputs_plots.png",
        "05_saved_artifacts.png",
    )
    missing_screenshots = [
        name
        for name in screenshot_names
        if not (repo / "submission" / "virtual_lab" / name).exists()
        or (repo / "submission" / "virtual_lab" / name).stat().st_size == 0
    ]
    screenshot_ready = not missing_screenshots
    record(
        checks,
        "virtual-lab screenshot set",
        screenshot_ready if require_submission_evidence else True,
        "all five present" if screenshot_ready else f"missing={missing_screenshots}",
    )

    final_pdf = repo / "output" / "pdf" / "Group148_Q_learning_DQN_DDQN.pdf"
    record(
        checks,
        "exact final PDF filename",
        (final_pdf.exists() and final_pdf.stat().st_size > 0)
        if require_submission_evidence
        else True,
        (f"{final_pdf.relative_to(repo)} ({'present' if final_pdf.exists() else 'not built yet'})"),
    )
    return checks


def main() -> None:
    """Run the audit, print a compact table, and fail on any required mismatch."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-submission-evidence",
        action="store_true",
        help="Also require final roster and genuine virtual-lab screenshot.",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    checks = run_audit(
        repo,
        require_submission_evidence=args.require_submission_evidence,
    )
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status:4s}  {check['check']}: {check['detail']}")
    failed = [check for check in checks if not check["passed"]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} required checks passed.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
