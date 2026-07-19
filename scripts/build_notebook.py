"""Build the executable assignment notebook from verified study artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
from nbformat.v4 import new_code_cell, new_markdown_cell

FENCE = chr(96) * 3
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


def percentage(value: float) -> str:
    """Format a fraction as a whole-number percentage."""

    return f"{100.0 * value:.0f}%"


def source_listing(repo: Path, relative_path: str) -> str:
    """Return one complete source file as a fenced Markdown listing."""

    source = (repo / relative_path).read_text(encoding="utf-8")
    return f"### {relative_path}\n\n{FENCE}python\n{source.rstrip()}\n{FENCE}\n"


def main() -> None:
    """Populate the notebook with narrative, executable evidence, and full source."""

    repo = Path(__file__).resolve().parents[1]
    notebook_path = repo / "output" / "jupyter-notebook" / "Group_148_Q_learning_DQN_DDQN.ipynb"
    notebook = nbformat.from_dict(json.loads(notebook_path.read_text(encoding="utf-8")))
    summary = json.loads((repo / "artifacts" / "study_summary.json").read_text())
    verification = json.loads(
        (repo / "artifacts" / "verification" / "wrapper_verification.json").read_text()
    )
    evaluation = pd.read_csv(repo / "artifacts" / "evaluation_summary.csv")
    group = json.loads((repo / "submission" / "group_details.json").read_text())

    member_lines = "\n".join(
        f"- {member['name']}: {member['contribution_percent']:g}%" for member in group["members"]
    )
    eval_lookup = evaluation.set_index("experiment")
    q_gap = summary["q_value_gap"]

    cells = [
        new_markdown_cell(
            "# Group 148 - Robust Reinforcement Learning under Stochastic Action Failure\n\n"
            "**Assignment II | DQN and DDQN on LunarLander-v3**\n\n"
            "## Group contribution declaration\n\n"
            f"{member_lines}\n\n"
            f"**Declaration status:** {group['status']}. The percentages total "
            f"{sum(member['contribution_percent'] for member in group['members']):g}%.\n\n"
            "> The institutional virtual-lab screenshot is embedded only when a genuine "
            "capture exists at the documented evidence path."
        ),
        new_markdown_cell(
            "## Objective and experimental question\n\n"
            "We test whether hidden 15% thruster-command failures make value estimation "
            "and credit assignment more difficult, and whether Double DQN is more robust "
            "than DQN when every attempted thruster action also costs 0.3 reward units.\n\n"
            "The success criteria are exact wrapper conformance, fair algorithm comparison, "
            "four required training plots, strict safe-landing measurement, and reproducible "
            "episode-level evidence."
        ),
        new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n\n"
            "import pandas as pd\n"
            "from IPython.display import Image, Markdown, display\n\n"
            "repo_candidates = [Path.cwd(), *Path.cwd().parents]\n"
            "REPO = next(path for path in repo_candidates if (path / 'pyproject.toml').exists())\n"
            "sys.path.insert(0, str(REPO / 'src'))\n"
            "ARTIFACTS = REPO / 'artifacts'\n"
            "print('Repository:', REPO)\n"
            "print('Artifacts:', ARTIFACTS)\n"
        ),
        new_markdown_cell(
            "## Exact environment modification\n\n"
            "For selected action $a$, executed action $a_{exec}$ is replaced by 0 with "
            "probability 0.15 only when $a \\in \\{1,2,3\\}$. The agent receives no failure "
            "flag. The returned reward is\n\n"
            "$$R = R_{base} - 0.3\\,\\mathbf{1}(a \\ne 0) + B,$$\n\n"
            "where $B=50$ only for a non-truncated terminal transition with both legs in "
            "contact and absolute horizontal velocity, vertical velocity, and angle all "
            "strictly below 0.10."
        ),
        new_code_cell(
            "from robust_lunarlander.envs import StochasticActionFailureWrapper, is_safe_landing\n"
            "from robust_lunarlander.verification import run_controlled_boundary_verification\n\n"
            "boundary_results = run_controlled_boundary_verification()\n"
            "display(boundary_results)\n"
            "assert boundary_results['passed'].all()\n"
        ),
        new_markdown_cell(
            "## Random-policy verification\n\n"
            "The external spy observed "
            f"{verification['random_policy']['misfired_thruster_actions']:,} "
            f"misfires among {verification['random_policy']['attempted_thruster_actions']:,} "
            "attempted thruster actions across "
            f"{verification['random_policy']['episodes']} episodes: "
            f"**{percentage(verification['random_policy']['observed_misfire_rate'])}** "
            "(unrounded rate "
            f"{verification['random_policy']['observed_misfire_rate']:.6f}). "
            "The 15% target lies inside the Wilson 95% interval. Fuel-penalty mismatches, "
            "unexpected replacements, and info-object changes were all zero."
        ),
        new_code_cell(
            "verification = json.loads((ARTIFACTS / 'verification' / "
            "'wrapper_verification.json').read_text())\n"
            "random_policy_table = pd.json_normalize(verification['random_policy'])\n"
            "display(random_policy_table.T.rename(columns={0: 'value'}))\n"
            "assert verification['overall_passed']\n"
        ),
        new_markdown_cell(
            "## Fair DQN/DDQN design\n\n"
            "All four agents use seed 148, 800 episodes, an 8-128-128-4 ReLU network, "
            "Adam with learning rate 0.0005, replay capacity 100,000, batch size 64, "
            "discount 0.99, linear epsilon decay from 1.00 to 0.01 over 100,000 steps, "
            "and a hard target copy every 500 updates. A single SHA-256-hashed set of "
            "512 validation states is reused at every episode.\n\n"
            "DQN evaluates $\\max_a Q_{target}(s',a)$. DDQN selects the maximizing action "
            "with the online network and evaluates only that action with the target network. "
            "This is the sole algorithm branch."
        ),
        new_code_cell(
            "config = json.loads((ARTIFACTS / 'training_config.json').read_text())\n"
            "provenance = json.loads((ARTIFACTS / 'system_provenance.json').read_text())\n"
            "display(pd.DataFrame({'value': config}).head(30))\n"
            "print('Fixed validation SHA-256:', provenance['validation_set_sha256'])\n"
        ),
        new_markdown_cell("## Required performance plots"),
        new_code_cell(
            "overview_path = ARTIFACTS / 'plots' / 'four_metric_overview.png'\n"
            "display(Image(filename=str(overview_path), width=1100))\n"
        ),
        new_markdown_cell(
            "### Plot interpretations\n\n"
            "1. **Episode reward.** Both original-environment agents improve sharply after "
            "roughly episode 600; DDQN finishes higher. Modified-environment learning is "
            "less stable. DDQN recovers late, while DQN remains in a poor high-activation "
            "policy.\n\n"
            "2. **Average predicted Q-value.** The final-100 mean absolute DQN-DDQN gap is "
            f"{q_gap['original_environment_mean_absolute_gap']:.2f} on the original "
            f"environment and {q_gap['modified_environment_mean_absolute_gap']:.2f} on the "
            "modified environment. The measured increase is "
            f"{q_gap['modified_minus_original']:.2f}.\n\n"
            "3. **Safe-landing rate.** The strict 100-episode moving rate ends at 71% for "
            "DDQN-original, 36% for DQN-original, 18% for DDQN-modified, and 3% for "
            "DQN-modified.\n\n"
            "4. **Thruster activations.** Counts reveal late high-activation regimes. In "
            "greedy evaluation, DDQN-modified uses fewer attempts than DDQN-original; "
            "DQN-modified instead shows a failed, inefficient policy."
        ),
        new_markdown_cell("## Greedy evaluation on 100 shared seeds"),
        new_code_cell(
            "evaluation = pd.read_csv(ARTIFACTS / 'evaluation_summary.csv')\n"
            "display(evaluation.style.format({'mean_reward': '{:.2f}', "
            "'reward_std': '{:.2f}', 'safe_landing_rate': '{:.0%}', "
            "'mean_thruster_activations': '{:.2f}', 'mean_episode_steps': '{:.2f}'}))\n"
        ),
        new_markdown_cell(
            "## Discussion\n\n"
            "### 1. Does failure increase the DQN-DDQN Q-value difference?\n\n"
            f"Yes for this seeded run. The final-100 mean absolute gap increases from "
            f"{q_gap['original_environment_mean_absolute_gap']:.2f} to "
            f"{q_gap['modified_environment_mean_absolute_gap']:.2f}. This is empirical "
            "evidence for the configured run, not a multi-seed population estimate.\n\n"
            "### 2. Why is credit assignment harder?\n\n"
            "The same selected thruster action can produce either a physical impulse or no "
            "impulse, yet the agent sees neither the replacement nor a failure flag. Replay "
            "therefore contains higher-variance outcomes for apparently identical "
            "state-action pairs. The attempted-action penalty is charged in both cases, "
            "which further separates immediate cost from uncertain physical effect.\n\n"
            "### 3. Does the penalty produce a conservative strategy?\n\n"
            "Evidence is conditional. DDQN's greedy modified policy averages "
            f"{eval_lookup.loc['ddqn_modified', 'mean_thruster_activations']:.2f} attempts, "
            "below DDQN-original's "
            f"{eval_lookup.loc['ddqn_original', 'mean_thruster_activations']:.2f}. "
            "DQN-modified averages "
            f"{eval_lookup.loc['dqn_modified', 'mean_thruster_activations']:.2f}, so the "
            "penalty alone does not guarantee conservation when learning collapses.\n\n"
            "### 4. Which algorithm is better under failures?\n\n"
            "DDQN. Its modified-environment greedy mean reward is "
            f"{eval_lookup.loc['ddqn_modified', 'mean_reward']:.2f} with "
            f"{percentage(eval_lookup.loc['ddqn_modified', 'safe_landing_rate'])} safe "
            "landings, versus DQN's "
            f"{eval_lookup.loc['dqn_modified', 'mean_reward']:.2f} and "
            f"{percentage(eval_lookup.loc['dqn_modified', 'safe_landing_rate'])}. This is "
            "consistent with Double DQN's theoretical motivation of decoupling action "
            "selection and evaluation to reduce harmful maximization bias.\n\n"
            "### 5. Limitation and improvement\n\n"
            "The principal limitation is one training seed. Repeat the full 2 x 2 design "
            "over at least 10 pre-registered seeds and report bootstrap confidence intervals "
            "or a hierarchical model for reward, safe-landing rate, Q-gap, and thruster use."
        ),
        new_markdown_cell(
            "## Virtual-lab timestamp evidence\n\n"
            "The following cell embeds the required screenshot only when the genuine "
            "institutional virtual-lab capture is present."
        ),
        new_code_cell(
            "virtual_lab_image = (\n"
            "    REPO / 'submission' / 'virtual_lab' / 'virtual_lab_timestamp.png'\n"
            ")\n"
            "if virtual_lab_image.exists():\n"
            "    display(Image(filename=str(virtual_lab_image), width=1000))\n"
            "else:\n"
            "    print('MANDATORY EVIDENCE PENDING: follow docs/virtual_lab_runbook.md')\n"
        ),
        new_markdown_cell(
            "## References\n\n"
            "1. Mnih et al. (2015), *Human-level control through deep reinforcement "
            "learning*, Nature 518, 529-533. DOI: 10.1038/nature14236.\n"
            "2. van Hasselt, Guez, and Silver (2016), *Deep Reinforcement Learning with "
            "Double Q-Learning*, AAAI 30(1). DOI: 10.1609/aaai.v30i1.10295.\n"
            "3. Farama Foundation, *Gymnasium LunarLander-v3 documentation*."
        ),
        new_markdown_cell(
            "# Complete commented source appendix\n\n"
            "Every implementation function is documented. The listings below are generated "
            "directly from the same modules used to produce the committed outputs."
        ),
    ]
    cells.extend(new_markdown_cell(source_listing(repo, path)) for path in SOURCE_FILES)
    notebook.cells = cells
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nbformat.write(notebook, notebook_path)
    print(f"Wrote {notebook_path} with {len(cells)} cells.")


if __name__ == "__main__":
    main()
