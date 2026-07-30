"""Build the executable assignment notebook from source and recorded artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd
from nbformat.v4 import new_code_cell, new_markdown_cell

FINAL_PDF_NAME = "Group148_Q_learning_DQN_DDQN.pdf"
EXPERIMENTS = (
    "dqn_original",
    "ddqn_original",
    "dqn_modified",
    "ddqn_modified",
)
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


def percentage(value: float, digits: int = 0) -> str:
    """Format a fraction as a percentage."""

    return f"{100.0 * value:.{digits}f}%"


def section_markdown(
    title: str,
    *,
    purpose: str,
    expected: str,
    capture: str,
    marks: str,
    body: str = "",
) -> str:
    """Build a consistent notebook section introduction."""

    details = (
        f"## {title}\n\n"
        f"**Purpose.** {purpose}\n\n"
        f"**Expected output after execution.** {expected}\n\n"
        f"**Virtual-lab evidence.** {capture}\n\n"
        f"**Assignment mapping.** {marks}"
    )
    return f"{details}\n\n{body}" if body else details


def member_table(group: dict[str, Any]) -> str:
    """Return the five-member declaration as a Markdown table."""

    lines = [
        "| Group member | BITS ID | Contribution |",
        "|---|---|---:|",
    ]
    for member in group["members"]:
        lines.append(
            f"| {member['name']} | {member['student_id']} | {member['contribution_percent']:g}% |"
        )
    return "\n".join(lines)


def main() -> None:
    """Populate a clean notebook with alternating narrative and executable cells."""

    repo = Path(__file__).resolve().parents[1]
    notebook_path = repo / "output" / "jupyter-notebook" / ("Group_148_Q_learning_DQN_DDQN.ipynb")
    notebook = nbformat.from_dict(json.loads(notebook_path.read_text(encoding="utf-8")))
    summary = json.loads((repo / "artifacts" / "study_summary.json").read_text())
    provenance = json.loads((repo / "artifacts" / "system_provenance.json").read_text())
    evaluation = pd.read_csv(repo / "artifacts" / "evaluation_summary.csv")
    group = json.loads((repo / "submission" / "group_details.json").read_text())
    eval_lookup = evaluation.set_index("experiment")
    q_gap = summary["q_value_gap"]

    confirmation_note = (
        "Confirmed by all members."
        if group.get("contributions_confirmed_by_group", False)
        else (
            "**PENDING HUMAN CONFIRMATION:** all five members must confirm these "
            "percentages before submission."
        )
    )

    cells = [
        new_markdown_cell(
            "# Robust Reinforcement Learning under Stochastic Action Failure\n\n"
            "**Course:** Deep Reinforcement Learning (S2-25_AIMLCZG512)  \n"
            "**Assignment:** Experiential Learning - Assignment 2  \n"
            "**Group:** 148  \n"
            f"**Recorded execution:** {provenance['generated_at']}  \n"
            f"**Execution environment:** {provenance['platform']}; "
            f"Python {provenance['python'].split()[0]}; Gymnasium "
            f"{provenance['gymnasium']}; PyTorch {provenance['torch']}; "
            f"device {provenance['device']}  \n"
            f"**Intended final PDF:** `{FINAL_PDF_NAME}`  \n\n"
            "Check the exact filename against the latest instructor guidance.\n\n"
            "## Group contribution declaration\n\n"
            f"{member_table(group)}\n\n"
            f"**Contribution confirmation:** {confirmation_note}\n\n"
            "> [INSERT VIRTUAL-LAB SCREENSHOT WITH VISIBLE TIMESTAMP HERE]\n\n"
            "Expected file: `submission/virtual_lab/01_start_timestamp.png`."
        ),
        new_code_cell(
            "from pathlib import Path\n"
            "import hashlib\n"
            "import inspect\n"
            "import json\n"
            "import os\n"
            "import platform\n"
            "import subprocess\n"
            "import sys\n\n"
            "import gymnasium as gym\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import seaborn as sns\n"
            "import torch\n"
            "from IPython.display import Image, Markdown, display\n\n"
            "repo_candidates = [Path.cwd(), *Path.cwd().parents]\n"
            "REPO = next(path for path in repo_candidates if (path / 'pyproject.toml').exists())\n"
            "sys.path.insert(0, str(REPO / 'src'))\n"
            "ARTIFACTS = REPO / 'artifacts'\n"
            "print('Repository:', REPO)\n"
            "print('Artifacts:', ARTIFACTS)\n"
            "print('Notebook interpreter:', sys.version.split()[0])\n"
        ),
        new_markdown_cell(
            "## Concise notebook blueprint\n\n"
            "1. Reproducibility and dependency evidence\n"
            "2. LunarLander-v3 inspection\n"
            "3. Exact wrapper implementation and deterministic tests\n"
            "4. Random-policy verification with confidence interval\n"
            "5. Replay buffer, terminal masking, Q-network, and shared agent\n"
            "6. Epsilon schedule and fixed validation states\n"
            "7. Four controlled experiments and per-episode metrics\n"
            "8. Required plots and final comparison table\n"
            "9. Evidence-based discussion, conclusion, screenshots, and checklist\n\n"
            "The notebook loads only recorded artifacts unless "
            "`RUN_FULL_TRAINING=1` is set before execution. The authoritative "
            "virtual-lab workflow runs training first, rebuilds this notebook from "
            "those artifacts, and then executes it top to bottom."
        ),
        new_code_cell(
            "required_paths = [\n"
            "    ARTIFACTS / 'training_config.json',\n"
            "    ARTIFACTS / 'system_provenance.json',\n"
            "    ARTIFACTS / 'study_summary.json',\n"
            "    ARTIFACTS / 'final_comparison.csv',\n"
            "]\n"
            "missing = [path for path in required_paths if not path.exists()]\n"
            "assert not missing, f'Missing required recorded artifacts: {missing}'\n"
            "print('Artifact preflight passed:', len(required_paths), 'required files found.')\n"
        ),
        new_markdown_cell(
            section_markdown(
                "1. Installation and package evidence",
                purpose=(
                    "Document the exact LunarLander-v3/Box2D environment and verify "
                    "the required Python stack."
                ),
                expected=(
                    "A version table and a successful LunarLander-v3 construction. "
                    "Exact version values are produced by the virtual lab."
                ),
                capture=(
                    "Capture the timestamp, lab identity, Python/Gymnasium/PyTorch "
                    "versions, CPU/GPU details, and device used."
                ),
                marks="Reproducibility setup and virtual-lab evidence.",
                body=(
                    "For Debian/Ubuntu virtual labs:\n\n"
                    "```bash\n"
                    "sudo apt-get update\n"
                    "sudo apt-get install -y swig build-essential python3-dev\n"
                    "python -m pip install --upgrade pip setuptools wheel\n"
                    'python -m pip install -e ".[dev]"\n'
                    "```\n\n"
                    "If `sudo` is unavailable, request `swig` and compiler support "
                    "from the lab administrator. Do not switch LunarLander versions."
                ),
            )
        ),
        new_code_cell(
            "versions = pd.DataFrame(\n"
            "    {\n"
            "        'Component': ['Python', 'Gymnasium', 'PyTorch', 'NumPy', "
            "'pandas', 'Matplotlib', 'Seaborn'],\n"
            "        'Version': [sys.version.split()[0], gym.__version__, "
            "torch.__version__, np.__version__, pd.__version__, "
            "plt.matplotlib.__version__, sns.__version__],\n"
            "    }\n"
            ")\n"
            "display(versions)\n"
            "print('Platform:', platform.platform())\n"
            "print('Logical CPUs:', os.cpu_count())\n"
            "print('CUDA available:', torch.cuda.is_available())\n"
            "print('Device used:', json.loads((ARTIFACTS / "
            "'training_config.json').read_text())['device'])\n"
            "inspection_env = gym.make('LunarLander-v3')\n"
            "print('Constructed:', inspection_env.spec.id)\n"
            "inspection_env.close()\n"
        ),
        new_markdown_cell(
            section_markdown(
                "2. Reproducibility setup",
                purpose=(
                    "Seed Python, NumPy, PyTorch CPU/CUDA, Gymnasium, and the action "
                    "space from one function and expose the full shared configuration."
                ),
                expected=(
                    "The central hyperparameter table, deterministic-setting status, "
                    "and seed confirmation."
                ),
                capture="Include this output in the package/version screenshot.",
                marks="Controlled settings shared by all DQN/DDQN experiments.",
                body=(
                    "Deterministic algorithms are requested where practical. GPU "
                    "kernels, Box2D, drivers, and cross-platform floating-point behavior "
                    "can still introduce nondeterminism."
                ),
            )
        ),
        new_code_cell(
            "from robust_lunarlander.config import TrainingConfig\n"
            "from robust_lunarlander.envs import make_environment\n"
            "from robust_lunarlander.experiment import set_reproducible_seeds\n\n"
            "CONFIG = TrainingConfig(**{\n"
            "    **json.loads((ARTIFACTS / 'training_config.json').read_text()),\n"
            "    'output_dir': ARTIFACTS,\n"
            "    'hidden_sizes': tuple(json.loads((ARTIFACTS / "
            "'training_config.json').read_text())['hidden_sizes']),\n"
            "})\n"
            "seed_check_env = make_environment(modified=False)\n"
            "set_reproducible_seeds(CONFIG.seed, seed_check_env)\n"
            "seed_check_env.close()\n"
            "display(pd.DataFrame(CONFIG.as_serializable_dict().items(), "
            "columns=['Hyperparameter', 'Value']))\n"
            "print('Deterministic algorithms requested:', "
            "torch.are_deterministic_algorithms_enabled())\n"
        ),
        new_markdown_cell(
            section_markdown(
                "3. LunarLander-v3 environment inspection",
                purpose="Confirm the exact observation shape, action count, and meanings.",
                expected="Observation shape `(8,)`, action count `4`, and the four meanings.",
                capture="Keep the executed output visible in the notebook/PDF.",
                marks="Environment inspection supporting the 2.5-mark environment section.",
            )
        ),
        new_code_cell(
            "env = gym.make('LunarLander-v3')\n"
            "action_meanings = {\n"
            "    0: 'Do nothing',\n"
            "    1: 'Fire left orientation engine',\n"
            "    2: 'Fire main engine',\n"
            "    3: 'Fire right orientation engine',\n"
            "}\n"
            "print('Environment ID:', env.spec.id)\n"
            "print('Observation shape:', env.observation_space.shape)\n"
            "print('Action count:', env.action_space.n)\n"
            "display(pd.DataFrame(action_meanings.items(), columns=['Action', 'Meaning']))\n"
            "assert env.observation_space.shape == (8,)\n"
            "assert env.action_space.n == 4\n"
            "env.close()\n"
        ),
        new_markdown_cell(
            section_markdown(
                "4. Custom stochastic-action-failure wrapper",
                purpose=(
                    "Show the complete wrapper that changes only executed action and "
                    "reward while keeping diagnostics private."
                ),
                expected="The class source and safe-landing helper are printed from the module.",
                capture="Capture source plus later pass tables; never fabricate a misfire.",
                marks="Environment implementation and verification - 2.5 marks.",
                body=(
                    "$R=R_{base}-0.3\\mathbf{1}(a\\ne0)+B$, where $B=50$ only "
                    "when every strict landing predicate is true. A private RNG prevents "
                    "failure draws from consuming the base environment RNG."
                ),
            )
        ),
        new_code_cell(
            "from robust_lunarlander.envs import (\n"
            "    StochasticActionFailureWrapper,\n"
            "    is_safe_landing,\n"
            ")\n"
            "print(inspect.getsource(is_safe_landing))\n"
            "print(inspect.getsource(StochasticActionFailureWrapper))\n"
        ),
        new_markdown_cell(
            section_markdown(
                "5. Deterministic wrapper unit tests",
                purpose=(
                    "Verify no-op, firing, misfire, exact +50 bonus, truncation, "
                    "leg contacts, velocity/angle limits, info identity, and spaces."
                ),
                expected="All pytest tests and all controlled boundary rows pass.",
                capture="Capture the pass summary and controlled-case table.",
                marks="Environment implementation and verification - 2.5 marks.",
            )
        ),
        new_code_cell(
            "test_run = subprocess.run(\n"
            "    [sys.executable, '-m', 'pytest', '-q'],\n"
            "    cwd=REPO,\n"
            "    env={**os.environ, 'PYTHONPATH': str(REPO / 'src')},\n"
            "    capture_output=True,\n"
            "    text=True,\n"
            "    check=True,\n"
            ")\n"
            "print(test_run.stdout.strip())\n"
            "from robust_lunarlander.verification import "
            "run_controlled_boundary_verification\n"
            "boundary_results = run_controlled_boundary_verification()\n"
            "display(boundary_results)\n"
            "assert boundary_results['passed'].all()\n"
        ),
        new_markdown_cell(
            section_markdown(
                "6. Random-policy verification",
                purpose=(
                    "Estimate the 15% misfire probability and prove that every "
                    "attempted thruster action receives the fuel penalty."
                ),
                expected=(
                    "Episodes, attempts, misfires, rate, absolute error, Wilson 95% "
                    "interval, count equality, and zero info mismatches."
                ),
                capture="Capture the full summary table with a visible timestamp.",
                marks="Random-policy verification - 2.5 marks.",
                body=(
                    "Deterministic mock tests remain the primary landing-bonus evidence "
                    "because random agents may not reliably land safely."
                ),
            )
        ),
        new_code_cell(
            "verification = json.loads((ARTIFACTS / 'verification' / "
            "'wrapper_verification.json').read_text())\n"
            "random_summary = verification['random_policy']\n"
            "display(pd.DataFrame(random_summary.items(), columns=['Measure', 'Value']))\n"
            "assert verification['overall_passed']\n"
            "assert random_summary['fuel_penalty_count'] == "
            "random_summary['attempted_thruster_actions']\n"
            "assert random_summary['target_inside_wilson_interval']\n"
        ),
        new_markdown_cell(
            section_markdown(
                "7. Replay buffer and terminal masking",
                purpose=(
                    "Store both Gymnasium ending flags and make bootstrapping policy explicit."
                ),
                expected=(
                    "A sampled batch with state/action/reward/next-state/terminated/"
                    "truncated tensors and the source of the target-mask calculation."
                ),
                capture="Keep the shapes and target-mask source visible.",
                marks="DQN 4 marks and DDQN 4 marks.",
                body=(
                    "True terminals do not bootstrap. Time-limit truncations do bootstrap "
                    "because the underlying MDP state is not terminal."
                ),
            )
        ),
        new_code_cell(
            "from robust_lunarlander.agent import ValueAgent\n"
            "from robust_lunarlander.replay import ReplayBuffer\n"
            "buffer = ReplayBuffer(capacity=8, observation_size=8, seed=CONFIG.seed)\n"
            "zero = np.zeros(8, dtype=np.float32)\n"
            "buffer.add(zero, 0, 1.0, zero, terminated=False, truncated=True)\n"
            "batch = buffer.sample(1, torch.device('cpu'))\n"
            "print('Sample shapes:', [tuple(t.shape) for t in batch])\n"
            "print(inspect.getsource(ValueAgent._target_values))\n"
        ),
        new_markdown_cell(
            section_markdown(
                "8. Shared Q-network",
                purpose=("Instantiate the common 8-to-4 network and count trainable parameters."),
                expected="Printed architecture and exact parameter count.",
                capture="Capture architecture and parameter count once.",
                marks="Shared architecture evidence for DQN and DDQN.",
            )
        ),
        new_code_cell(
            "from robust_lunarlander.network import QNetwork\n"
            "network = QNetwork(8, 4, CONFIG.hidden_sizes)\n"
            "parameter_count = sum(p.numel() for p in network.parameters() if p.requires_grad)\n"
            "print(network)\n"
            "print('Trainable parameters:', f'{parameter_count:,}')\n"
        ),
        new_markdown_cell(
            section_markdown(
                "9. One reusable DQN/DDQN agent",
                purpose=("Expose the sole algorithmic difference: the next-state target Q-value."),
                expected="The shared bootstrap function shows DQN and DDQN side by side.",
                capture="Keep this code visible as direct implementation evidence.",
                marks="DQN - 4 marks; DDQN - 4 marks.",
                body=(
                    "**DQN:** $\\max_a Q_{target}(s',a)$.  \n"
                    "**DDQN:** $a^*=\\arg\\max_a Q_{online}(s',a)$, then "
                    "$Q_{target}(s',a^*)$."
                ),
            )
        ),
        new_code_cell(
            "print(inspect.getsource(ValueAgent._bootstrap_values))\n"
            "dqn_agent = ValueAgent(8, 4, 'dqn', CONFIG)\n"
            "ddqn_agent = ValueAgent(8, 4, 'ddqn', CONFIG)\n"
            "print('Shared network types:', type(dqn_agent.online_network).__name__, "
            "type(ddqn_agent.online_network).__name__)\n"
            "print('Algorithms:', dqn_agent.algorithm.upper(), ddqn_agent.algorithm.upper())\n"
        ),
        new_markdown_cell(
            section_markdown(
                "10. Epsilon-greedy exploration",
                purpose="Apply one reproducible linear schedule to all four experiments.",
                expected="Schedule checkpoints and a labelled schedule plot.",
                capture="Capture the plot or table with start, final, and decay duration.",
                marks="Controlled-experiment fairness evidence.",
            )
        ),
        new_code_cell(
            "from robust_lunarlander.config import linear_epsilon\n"
            "epsilon_steps = [0, CONFIG.epsilon_decay_steps // 2, "
            "CONFIG.epsilon_decay_steps, CONFIG.epsilon_decay_steps * 2]\n"
            "epsilon_table = pd.DataFrame({\n"
            "    'Environment step': epsilon_steps,\n"
            "    'Epsilon': [linear_epsilon(step, CONFIG) for step in epsilon_steps],\n"
            "})\n"
            "display(epsilon_table)\n"
            "display(Image(filename=str(ARTIFACTS / 'plots' / "
            "'epsilon_schedule.png'), width=900))\n"
        ),
        new_markdown_cell(
            section_markdown(
                "11. Fixed validation-state set",
                purpose=(
                    "Prove that one immutable state set is reused for every Q-value comparison."
                ),
                expected="Shape `(512, 8)` and a SHA-256 digest matching provenance.",
                capture="Capture shape and digest once.",
                marks="Fair fixed-set Q-value comparison.",
            )
        ),
        new_code_cell(
            "validation_path = ARTIFACTS / 'validation' / 'fixed_validation_states.npz'\n"
            "validation_states = np.load(validation_path)['states']\n"
            "validation_hash = hashlib.sha256(validation_states.tobytes()).hexdigest()\n"
            "provenance = json.loads((ARTIFACTS / 'system_provenance.json').read_text())\n"
            "print('Validation shape:', validation_states.shape)\n"
            "print('Validation SHA-256:', validation_hash)\n"
            "assert validation_hash == provenance['validation_set_sha256']\n"
            "validation_states.setflags(write=False)\n"
            "print('Array writeable:', validation_states.flags.writeable)\n"
        ),
        new_markdown_cell(
            section_markdown(
                "12. Controlled four-run experiment",
                purpose=(
                    "Train fresh DQN/DDQN agents on original/modified environments "
                    "with identical settings and persist every episode."
                ),
                expected=(
                    "One compact line per episode, four 800-row CSVs/logs, checkpoints, "
                    "plots, configuration, validation states, and summaries."
                ),
                capture=(
                    "Capture timestamped progress during each run and final saved files. "
                    "Do not present interrupted runs as complete."
                ),
                marks="DQN 4 marks, DDQN 4 marks, and performance evaluation 2 marks.",
                body=(
                    "Set `RUN_FULL_TRAINING=1` before launching Jupyter only when a "
                    "fresh full run is intended. The recommended lab workflow runs "
                    "`make study` first, then rebuilds and executes this notebook."
                ),
            )
        ),
        new_code_cell(
            "RUN_FULL_TRAINING = os.environ.get('RUN_FULL_TRAINING', '0') == '1'\n"
            "if RUN_FULL_TRAINING:\n"
            "    from robust_lunarlander.experiment import run_complete_study\n"
            "    run_complete_study(CONFIG, force=True)\n"
            "else:\n"
            "    print('Reusing recorded complete artifacts. Set RUN_FULL_TRAINING=1 '\n"
            "          'before launch for a fresh run.')\n"
            "for name in " + repr(EXPERIMENTS) + ":\n"
            "    frame = pd.read_csv(ARTIFACTS / 'metrics' / f'{name}.csv')\n"
            "    log_lines = (ARTIFACTS / 'logs' / f'{name}.log').read_text().splitlines()\n"
            "    print(f'{name}: CSV rows={len(frame)}, progress records={len(log_lines)-1}, '\n"
            "          f'episodes={frame.episode.min()}-{frame.episode.max()}')\n"
            "    assert len(frame) == CONFIG.episodes\n"
            "    assert len(log_lines) - 1 == CONFIG.episodes\n"
        ),
        new_markdown_cell(
            section_markdown(
                "13. Per-episode metrics and compact iteration evidence",
                purpose=(
                    "Show the required schema and representative notebook output; "
                    "the PDF appendix holds the complete 3,200-row record."
                ),
                expected="Schema audit plus first/last rows for each experiment.",
                capture="Capture representative beginning/end rows and row-count evidence.",
                marks="Code quality and reproducible execution evidence.",
                body=(
                    "All iterations remain in `artifacts/metrics/*.csv` and "
                    "`artifacts/logs/*.log`. Appendix B of the submission PDF includes "
                    "all 3,200 compact per-iteration records; this notebook displays "
                    "only beginning/end rows for readability."
                ),
            )
        ),
        new_code_cell(
            "required_metric_columns = {\n"
            "    'episode', 'episode_reward', 'average_predicted_q',\n"
            "    'successful_safe_landing', 'moving_safe_landing_rate_100',\n"
            "    'attempted_thruster_activations', 'executed_thruster_activations',\n"
            "    'average_attempted_thruster_activations_per_episode',\n"
            "    'episode_steps', 'epsilon', 'training_loss', 'environment_type',\n"
            "    'algorithm', 'random_seed', 'episode_seconds',\n"
            "}\n"
            "for name in " + repr(EXPERIMENTS) + ":\n"
            "    frame = pd.read_csv(ARTIFACTS / 'metrics' / f'{name}.csv')\n"
            "    assert required_metric_columns.issubset(frame.columns)\n"
            "    excerpt = pd.concat([frame.head(5), frame.tail(5)])\n"
            '    display(Markdown(f\'### {name.replace("_", " ").title()}\'))\n'
            "    display(excerpt[[\n"
            "        'episode', 'episode_reward', 'average_predicted_q',\n"
            "        'moving_safe_landing_rate_100',\n"
            "        'attempted_thruster_activations',\n"
            "        'executed_thruster_activations', 'epsilon'\n"
            "    ]])\n"
        ),
        new_markdown_cell(
            section_markdown(
                "14. Required comparison plots",
                purpose="Compare all four agents on the four assignment metrics.",
                expected=(
                    "Reward, fixed-state Q, moving safe-landing rate, and average "
                    "attempted-thruster plots with titles, axes, legend, grid, and caption."
                ),
                capture="Capture the timestamped final-output and plot screen.",
                marks="Performance evaluation - 2 marks.",
                body=(
                    "Raw traces are light where useful; moderate smoothing is overlaid "
                    "without hiding instability."
                ),
            )
        ),
        new_code_cell(
            "plot_names = [\n"
            "    'episode_reward.png', 'average_predicted_q.png',\n"
            "    'success_rate_100.png', 'thruster_activations.png',\n"
            "]\n"
            "for plot_name in plot_names:\n"
            "    display(Image(filename=str(ARTIFACTS / 'plots' / plot_name), width=950))\n"
        ),
        new_markdown_cell(
            section_markdown(
                "15. Evaluation table",
                purpose=(
                    "Report final-100 training statistics and supplementary greedy evaluation."
                ),
                expected=(
                    "Mean/SD reward, best MA(100), final fixed-set Q, final safe rate, "
                    "attempted/executed actions, successes, and duration for every run."
                ),
                capture="Capture the full comparison table.",
                marks="Performance evaluation - 2 marks.",
            )
        ),
        new_code_cell(
            "final_comparison = pd.read_csv(ARTIFACTS / 'final_comparison.csv')\n"
            "evaluation = pd.read_csv(ARTIFACTS / 'evaluation_summary.csv')\n"
            "display(Markdown('### Final-100 training comparison'))\n"
            "display(final_comparison)\n"
            "display(Markdown('### Greedy evaluation on 100 shared seeds'))\n"
            "display(evaluation)\n"
        ),
        new_markdown_cell(
            section_markdown(
                "16. Evidence-based discussion",
                purpose="Answer all five assignment questions using the recorded study.",
                expected="Claims tied to plot panels and numerical table values.",
                capture="Capture after the final tables/plots exist.",
                marks="Discussion - 2.5 marks.",
                body=(
                    "### 1. Does failure increase the DQN-DDQN Q-value difference?\n\n"
                    f"For this run, the final-100 mean absolute gap changes from "
                    f"{q_gap['original_environment_mean_absolute_gap']:.2f} in the "
                    f"original environment to "
                    f"{q_gap['modified_environment_mean_absolute_gap']:.2f} in the "
                    f"modified environment (difference "
                    f"{q_gap['modified_minus_original']:.2f}). This is single-seed "
                    "evidence, not a population estimate.\n\n"
                    "### 2. Why is temporal credit assignment harder?\n\n"
                    "The same selected action can produce a thruster impulse or a no-op "
                    "without a failure indicator. Replay therefore contains noisier "
                    "outcomes for similar state-action inputs, while the selected-action "
                    "cost is certain and the physical effect is uncertain.\n\n"
                    "### 3. Does the penalty encourage conservation?\n\n"
                    f"No in the attempted-action metric for this run. Greedy DDQN attempts "
                    f"{eval_lookup.loc['ddqn_modified', 'mean_attempted_thruster_activations']:.2f} "
                    "actions when modified versus "
                    f"{eval_lookup.loc['ddqn_original', 'mean_attempted_thruster_activations']:.2f} "
                    "when original. Greedy DQN attempts "
                    f"{eval_lookup.loc['dqn_modified', 'mean_attempted_thruster_activations']:.2f} "
                    "versus "
                    f"{eval_lookup.loc['dqn_original', 'mean_attempted_thruster_activations']:.2f}. "
                    "Both modified policies attempt more actions. Longer episodes and "
                    "compensation for no-op commands may outweigh the 0.3 cost.\n\n"
                    "### 4. Which algorithm performs better under failure?\n\n"
                    "The result is metric-dependent. DDQN is slightly better in the final "
                    "100 training episodes, but frozen-checkpoint greedy evaluation favors "
                    f"DQN: reward {eval_lookup.loc['dqn_modified', 'mean_reward']:.2f} and "
                    f"safe rate {percentage(eval_lookup.loc['dqn_modified', 'safe_landing_rate'])} "
                    f"versus DDQN reward {eval_lookup.loc['ddqn_modified', 'mean_reward']:.2f} "
                    f"and safe rate "
                    f"{percentage(eval_lookup.loc['ddqn_modified', 'safe_landing_rate'])}. "
                    "This execution does not show an unambiguous DDQN advantage and is not "
                    "cleanly consistent with the theoretical expectation.\n\n"
                    "### 5. Limitation and improvement\n\n"
                    "The study uses one training seed. A paired multi-seed replication "
                    "with confidence intervals is the most direct improvement."
                ),
            )
        ),
        new_code_cell(
            "discussion_evidence = {\n"
            "    'q_gap_original': "
            f"{q_gap['original_environment_mean_absolute_gap']!r},\n"
            "    'q_gap_modified': "
            f"{q_gap['modified_environment_mean_absolute_gap']!r},\n"
            "    'ddqn_modified_mean_reward': "
            f"{float(eval_lookup.loc['ddqn_modified', 'mean_reward'])!r},\n"
            "    'dqn_modified_mean_reward': "
            f"{float(eval_lookup.loc['dqn_modified', 'mean_reward'])!r},\n"
            "}\n"
            "display(pd.Series(discussion_evidence, name='Recorded value'))\n"
        ),
        new_markdown_cell(
            section_markdown(
                "17. Virtual-lab screenshot placeholders",
                purpose="Reserve exact locations for genuine institutional evidence.",
                expected="Existing images display; missing images remain explicit placeholders.",
                capture="Use visible timestamps and retain institutional lab identity.",
                marks="Virtual-lab evidence and submission readiness.",
                body=(
                    "- `01_start_timestamp.png`: beginning of execution\n"
                    "- `02_environment_versions.png`: package/environment evidence\n"
                    "- `03_training_progress.png`: timestamped training\n"
                    "- `04_final_outputs_plots.png`: final tables and plots\n"
                    "- `05_saved_artifacts.png`: CSV/checkpoints/validation/plots\n\n"
                    "Do not fabricate or substitute local screenshots."
                ),
            )
        ),
        new_code_cell(
            "screenshot_dir = REPO / 'submission' / 'virtual_lab'\n"
            "screenshot_specs = {\n"
            "    '01_start_timestamp.png': 'Beginning timestamp',\n"
            "    '02_environment_versions.png': 'Environment versions',\n"
            "    '03_training_progress.png': 'Training progress',\n"
            "    '04_final_outputs_plots.png': 'Final outputs and plots',\n"
            "    '05_saved_artifacts.png': 'Saved artifacts',\n"
            "}\n"
            "for filename, label in screenshot_specs.items():\n"
            "    path = screenshot_dir / filename\n"
            "    if path.exists():\n"
            "        display(Markdown(f'### {label}'))\n"
            "        display(Image(filename=str(path), width=1000))\n"
            "    else:\n"
            "        print(f'[INSERT VIRTUAL-LAB SCREENSHOT WITH VISIBLE TIMESTAMP HERE] '\n"
            "              f'{label}: {path.relative_to(REPO)}')\n"
        ),
        new_markdown_cell(
            section_markdown(
                "18. Conclusion",
                purpose="Summarize only the strongest recorded evidence and limitation.",
                expected="A concise result-dependent conclusion.",
                capture="Capture only after the official run and final plots complete.",
                marks="Conclusion and Discussion - 2.5 marks.",
                body=(
                    f"In the recorded run, hidden actuator failure changes the final-100 "
                    f"DQN-DDQN fixed-state Q-gap from "
                    f"{q_gap['original_environment_mean_absolute_gap']:.2f} to "
                    f"{q_gap['modified_environment_mean_absolute_gap']:.2f}. The attempted "
                    "fuel penalty did not reduce attempted activations in the modified "
                    "runs. DDQN is slightly stronger in the final training window, while "
                    "DQN is substantially stronger in greedy evaluation, so no algorithm "
                    "wins every metric. The main limitation is the single seed. Rebuild "
                    "this notebook if the official virtual-lab outputs change."
                ),
            )
        ),
        new_code_cell(
            "audit_run = subprocess.run(\n"
            "    [sys.executable, 'scripts/audit_artifacts.py'],\n"
            "    cwd=REPO,\n"
            "    capture_output=True,\n"
            "    text=True,\n"
            "    check=True,\n"
            ")\n"
            "print(audit_run.stdout)\n"
        ),
        new_markdown_cell(
            section_markdown(
                "19. Complete commented source appendix",
                purpose="Expose the exact modular implementation used by the study.",
                expected="All eight source modules printed from disk without replay tensors.",
                capture="The PDF report contains the same syntax-highlighted listings.",
                marks="Code quality across all rubric sections.",
            )
        ),
        new_code_cell(
            "source_files = " + repr(SOURCE_FILES) + "\n"
            "for relative_path in source_files:\n"
            "    display(Markdown(f'### {relative_path}'))\n"
            "    source = (REPO / relative_path).read_text(encoding='utf-8')\n"
            "    display(Markdown(f'```python\\n{source}\\n```'))\n"
        ),
        new_markdown_cell(
            "# Submission checklist\n\n"
            "- [ ] Contribution percentages confirmed by all five members\n"
            "- [ ] All five names and BITS IDs checked\n"
            "- [x] Functions documented and important operations explained\n"
            "- [x] Wrapper verified statistically and deterministically\n"
            "- [x] DQN and DDQN code supports both environments with controlled settings\n"
            "- [x] Four required plots and evaluation table generated\n"
            "- [x] Five discussion questions tied to recorded evidence\n"
            "- [ ] Every cell executed in the official virtual lab\n"
            "- [ ] Five genuine timestamped screenshots inserted\n"
            f"- [ ] Upload only `{FINAL_PDF_NAME}` after checking instructor naming guidance\n"
            "- [ ] First version submitted by 5 August 2026\n"
            "- [ ] Final submission by 7 August 2026, 11:59 PM\n"
            "- [ ] Every member reviewed and understood the original work\n"
            "- [x] No fabricated outputs inserted by this notebook builder\n\n"
            "# Rubric coverage\n\n"
            "| Rubric item | Marks | Notebook coverage |\n"
            "|---|---:|---|\n"
            "| Environment implementation and verification | 2.5 | Sections 3-6 |\n"
            "| DQN | 4.0 | Sections 7-13 |\n"
            "| DDQN | 4.0 | Sections 7-13 |\n"
            "| Performance evaluation | 2.0 | Sections 14-15 |\n"
            "| Discussion | 2.5 | Sections 16 and 18 |\n"
            "| **Total** | **15.0** | **All rubric areas mapped; human evidence gates remain** |"
        ),
    ]

    notebook.cells = cells
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }
    nbformat.write(notebook, notebook_path)
    print(f"Wrote {notebook_path} with {len(cells)} cells.")


if __name__ == "__main__":
    main()
