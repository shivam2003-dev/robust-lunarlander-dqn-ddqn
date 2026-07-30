# Virtual-lab execution and timestamp evidence

The assignment requires execution in the institution's virtual lab and a
timestamped screenshot in the final PDF. This repository does not label a
local-machine or GitHub Actions capture as virtual-lab evidence.

## 1. Prepare the virtual lab

Clone the final repository and enter it:

    git clone https://github.com/shivam2003-dev/robust-lunarlander-dqn-ddqn.git
    cd robust-lunarlander-dqn-ddqn
    uv sync --extra dev
    source .venv/bin/activate

If uv is unavailable, create a Python 3.11-3.13 environment and install the
project with pip:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

For a Debian/Ubuntu lab, Box2D may also require:

    sudo apt-get update
    sudo apt-get install -y swig build-essential python3-dev

If `sudo` is unavailable, ask the lab administrator for these packages. Keep
the environment ID fixed at `LunarLander-v3`.

## 2. Run the authoritative checks

Before execution, show the lab identity and timestamp and save:

    submission/virtual_lab/01_start_timestamp.png

    make test
    make verify

For a full retraining run:

    make study | tee submission/virtual_lab/full_training_console.log

`make study` uses `--force`; it does not silently reuse committed development
artifacts. For a quick separate smoke run:

    make smoke

After the full run, rebuild and execute the notebook and report:

    make notebook
    make report

For an integrity check of the resulting artifacts:

    python scripts/audit_artifacts.py

## 3. Display timestamped provenance

Keep the complete command and output visible in one terminal window:

    date '+%Y-%m-%d %H:%M:%S %Z (%z)'
    hostname
    python --version
    python -c "import gymnasium, torch; print('Gymnasium', gymnasium.__version__); print('PyTorch', torch.__version__)"
    git rev-parse HEAD
    cat artifacts/verification/wrapper_verification.json

The screenshot must visibly include:

- the virtual-lab desktop or institutional lab marker;
- timestamp and timezone;
- hostname;
- Python, Gymnasium, and PyTorch versions;
- Git commit SHA;
- the overall_passed verification result.

## 4. Save genuine evidence

Save five genuine screenshots:

    submission/virtual_lab/01_start_timestamp.png
    submission/virtual_lab/02_environment_versions.png
    submission/virtual_lab/03_training_progress.png
    submission/virtual_lab/04_final_outputs_plots.png
    submission/virtual_lab/05_saved_artifacts.png

Do not crop away the lab identity, timestamp, command, or pass result. Rebuild
the notebook and PDF after the genuine screenshots exist:

    make notebook
    make report

## 5. Final upload checks

- Verify all names/IDs and ask all members to confirm the declared percentages.
- Only then set `contributions_confirmed_by_group` to `true` in
  `submission/group_details.json` and rebuild.
- Confirm all five screenshots are readable at 100% zoom in the PDF.
- Open the PDF from output/pdf and inspect every page.
- Run `python scripts/audit_artifacts.py --require-submission-evidence`.
- Upload only `Group148_Q_learning_DQN_DDQN.pdf`, after checking the latest
  instructor filename guidance.
