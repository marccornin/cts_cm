PY ?= python
PKG := cts_cm

.PHONY: help install dev lint format type test smoke run docker clean

help:
	@echo "install  - pip install the package"
	@echo "dev      - install with dev extras + pre-commit"
	@echo "lint     - ruff + isort check"
	@echo "format   - black + isort write"
	@echo "type     - mypy --strict"
	@echo "test     - pytest"
	@echo "smoke    - end-to-end run on the smoke experiment"
	@echo "run      - main experiment on a synthetic OAI cohort"
	@echo "docker   - build the container image"
	@echo "clean    - remove caches and build artefacts"

install:
	$(PY) -m pip install .

dev:
	$(PY) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	isort --check-only .

format:
	black .
	isort .

type:
	$(PY) -m mypy --strict $(PKG)

test:
	$(PY) -m pytest

smoke:
	$(PY) -m $(PKG).observer run-all --config configs/experiment/_smoke.yaml

run:
	$(PY) -m $(PKG).observer run-all --config configs/experiment/main.yaml

docker:
	docker build -t cts_cm:0.1.0 .

clean:
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
