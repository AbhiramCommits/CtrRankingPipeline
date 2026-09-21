SHELL := /bin/bash
PYTHON := .venv/bin/python
ROWS ?= 1000000

.PHONY: install data features train eval serve test lint clean

$(PYTHON):
	@$(MAKE) install

install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "using uv"; \
		uv sync --extra dev; \
	else \
		echo "uv not found, falling back to venv + pip"; \
		( command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3 ) -m venv .venv && \
		$(PYTHON) -m pip install --upgrade pip && \
		$(PYTHON) -m pip install -e ".[dev]"; \
	fi

data: $(PYTHON)
	$(PYTHON) scripts/download_data.py --config configs/data.yaml --rows $(ROWS)
	$(PYTHON) scripts/run_ingest.py --config configs/spark.yaml

features: $(PYTHON)
	$(PYTHON) scripts/run_features.py --config configs/features.yaml

train: $(PYTHON)
	$(PYTHON) scripts/train_retrieval.py --config configs/retrieval.yaml
	$(PYTHON) scripts/train_ranker.py --config configs/ranker.yaml

eval: $(PYTHON)
	$(PYTHON) scripts/run_eval.py --config configs/eval.yaml

serve: $(PYTHON)
	$(PYTHON) scripts/run_serve.py

test: $(PYTHON)
	$(PYTHON) -m pytest --cov=ctr --cov-report=term-missing

lint: $(PYTHON)
	@if $(PYTHON) -m ruff --version >/dev/null 2>&1; then \
		$(PYTHON) -m ruff check ctr scripts tests; \
	else \
		echo "ruff not installed; falling back to syntax check"; \
		$(PYTHON) -m compileall -q ctr scripts tests; \
	fi

clean:
	rm -rf data/raw data/parquet .pytest_cache .coverage htmlcov
