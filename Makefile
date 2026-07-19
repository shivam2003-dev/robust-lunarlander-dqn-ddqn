.PHONY: install test verify study notebook report all

install:
	uv sync --extra dev

test:
	python -m ruff check src tests scripts
	python -m ruff format --check src tests scripts
	PYTHONPATH=src pytest

verify:
	PYTHONPATH=src python -m robust_lunarlander.verification --episodes 250

study:
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m robust_lunarlander.experiment --episodes 800 --evaluation-episodes 100

notebook:
	PYTHONPATH=src python scripts/build_notebook.py
	PYTHONPATH=src jupyter nbconvert --execute --to notebook --inplace output/jupyter-notebook/Group_148_Q_learning_DQN_DDQN.ipynb

report:
	PYTHONPATH=src python scripts/build_report.py
	bash scripts/render_report.sh

all: test verify study notebook report
