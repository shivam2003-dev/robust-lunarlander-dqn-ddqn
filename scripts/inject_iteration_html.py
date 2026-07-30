"""Insert complete per-episode training tables into the standalone HTML report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPERIMENTS = (
    "dqn_original",
    "ddqn_original",
    "dqn_modified",
    "ddqn_modified",
)
MARKER = "<!-- ITERATION_TABLES_FOR_HTML -->"


def iteration_sections(repo: Path) -> str:
    """Build four printable HTML tables from the recorded progress logs."""

    sections: list[str] = []
    for experiment in EXPERIMENTS:
        frame = pd.read_csv(repo / "artifacts" / "logs" / f"{experiment}.log")
        rows: list[str] = []
        for record in frame.itertuples(index=False):
            loss = "nan" if pd.isna(record.training_loss) else f"{record.training_loss:.6f}"
            rows.append(
                "<tr>"
                f"<td>{int(record.episode)}</td>"
                f"<td>{record.total_reward:.2f}</td>"
                f"<td>{record.average_q:.2f}</td>"
                f"<td>{record.safe_rate_100:.1%}</td>"
                f"<td>{int(record.attempted)}/{int(record.executed)}</td>"
                f"<td>{int(record.steps)}</td>"
                f"<td>{record.epsilon:.4f}</td>"
                f"<td>{loss}</td>"
                "</tr>"
            )
        title = experiment.replace("_", " ").title()
        sections.append(
            '<section class="iteration-run">\n'
            f"<h2>{title} - Complete per-iteration output (800 episodes)</h2>\n"
            '<table class="iteration-table">\n'
            "<thead><tr><th>Episode</th><th>Reward</th><th>Avg. Q</th>"
            "<th>Safe100</th><th>Attempted/Executed</th><th>Steps</th>"
            "<th>Epsilon</th><th>Loss</th></tr></thead>\n"
            "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>\n</section>"
        )
    return "\n\n".join(sections)


def main() -> None:
    """Replace the Pandoc marker in one standalone HTML report."""

    parser = argparse.ArgumentParser()
    parser.add_argument("html_path", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    source = args.html_path.read_text(encoding="utf-8")
    if MARKER not in source:
        message = f"HTML marker not found: {MARKER}"
        raise ValueError(message)
    args.html_path.write_text(source.replace(MARKER, iteration_sections(repo)), encoding="utf-8")


if __name__ == "__main__":
    main()
