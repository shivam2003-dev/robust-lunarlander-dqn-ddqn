.PHONY: install test verify smoke study notebook notebook-html report html all

install:
	uv sync --extra dev

test:
	python -m ruff check src tests scripts
	python -m ruff format --check src tests scripts
	PYTHONPATH=src pytest

verify:
	PYTHONPATH=src python -m robust_lunarlander.verification --episodes 250

smoke:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m robust_lunarlander.experiment --episodes 5 --evaluation-episodes 2 --output-dir tmp/smoke-study --force

study:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m robust_lunarlander.experiment --episodes 800 --evaluation-episodes 100 --force

notebook:
	PYTHONPATH=src python scripts/build_notebook.py
	PYTHONPATH=src uv run jupyter nbconvert --execute --to notebook --inplace output/jupyter-notebook/Group_148_Q_learning_DQN_DDQN.ipynb

notebook-html:
	bash scripts/render_notebook_html.sh

report:
	PYTHONPATH=src python scripts/build_report.py
	bash scripts/render_report.sh

html:
	PYTHONPATH=src python scripts/build_report.py
	bash scripts/render_report_html.sh

all: test verify study notebook notebook-html report html
