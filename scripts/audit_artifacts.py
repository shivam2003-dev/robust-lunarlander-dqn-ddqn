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
            len(frame) == episodes and frame["episode"].tolist() == expected_episodes,
            f"rows={len(frame)}, range={frame['episode'].min()}-{frame['episode'].max()}",
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
        random_policy["fuel_penalty_mismatches"] == 0,
        f"mismatches={random_policy['fuel_penalty_mismatches']}",
    )

    group_details = json.loads(
        (repo / "submission" / "group_details.json").read_text(encoding="utf-8")
    )
    contribution_total = sum(
        float(member["contribution_percent"]) for member in group_details["members"]
    )
    names_complete = all("REPLACE_" not in member["name"] for member in group_details["members"])
    group_ready = (
        group_details["status"] == "FINAL"
        and names_complete
        and abs(contribution_total - 100.0) < 1e-9
    )
    record(
        checks,
        "group contribution declaration",
        group_ready if require_submission_evidence else True,
        f"status={group_details['status']}, total={contribution_total:g}%",
    )

    screenshot = repo / "submission" / "virtual_lab" / "virtual_lab_timestamp.png"
    screenshot_ready = screenshot.exists() and screenshot.stat().st_size > 0
    record(
        checks,
        "virtual-lab timestamp screenshot",
        screenshot_ready if require_submission_evidence else True,
        "present" if screenshot_ready else "pending genuine virtual-lab capture",
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
