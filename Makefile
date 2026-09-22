# Use the SAME interpreter that has the package installed. A bare `pytest` or
# `csat` can resolve to a different environment's shim (pipx/uv tool installs are
# a common cause), which then fails with "No module named torch".
PYTHON ?= python3

.PHONY: help install install-cpu test lint fmt check notebook tracking run run-full clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras
	$(PYTHON) -m pip install -e ".[dev,notebook]"

install-cpu:  ## Install CPU-only torch, then the package
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e ".[dev,notebook]"

test:  ## Run the unit tests
	$(PYTHON) -m pytest -v

lint:  ## Lint
	$(PYTHON) -m ruff check src tests

fmt:  ## Auto-fix lint issues
	$(PYTHON) -m ruff check --fix src tests

check: test lint  ## Everything CI checks
	$(PYTHON) -m csat.cli check
	$(PYTHON) scripts/gen_tracking_doc.py --check
	$(PYTHON) scripts/build_notebook.py --check

notebook:  ## Regenerate the Kaggle notebook from package sources
	$(PYTHON) scripts/build_notebook.py

tracking:  ## Regenerate docs/REPRODUCTION_TRACKING.md from the table in code
	$(PYTHON) scripts/gen_tracking_doc.py

run:  ## Run all experiments (quick sweeps)
	$(PYTHON) -m csat.cli run all

run-full:  ## Run all experiments (full sweeps)
	$(PYTHON) -m csat.cli run all --full

clean:  ## Remove caches and generated output
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf results/*.csv results/*.json figures/*.png notebooks/executed.ipynb
