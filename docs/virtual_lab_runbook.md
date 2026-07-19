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

## 2. Run the authoritative checks

    make test
    make verify

For a full retraining run:

    make study

For a faster integrity check of the committed experiment artifacts, run:

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

Save the screenshot as:

    submission/virtual_lab/virtual_lab_timestamp.png

Do not crop away the lab identity, timestamp, command, or pass result. Rebuild
the notebook and PDF only after the genuine screenshot exists:

    make notebook
    make report

## 5. Final upload checks

- Verify the contribution declaration contains every member and totals 100%.
- Confirm the screenshot is readable at 100% zoom in the PDF.
- Open the PDF from output/pdf and inspect every page.
- Upload only Group_148_Q_learning_DQN_DDQN.pdf.
