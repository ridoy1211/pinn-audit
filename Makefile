.PHONY: help setup lint fmt typecheck test smoke figures convergence clean

help:
	@echo "setup        install package + dev tools (editable)"
	@echo "lint         ruff check + format check"
	@echo "typecheck    mypy"
	@echo "test         pytest with coverage"
	@echo "smoke        run the tiny CI experiment"
	@echo "convergence  measure observed order for both baselines"
	@echo "figures      regenerate figures/ from scratch"

setup:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

fmt:
	ruff format src tests scripts
	ruff check --fix src tests scripts

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

typecheck:
	mypy

test:
	pytest --cov=pinnaudit --cov-report=term-missing

smoke:
	pinnaudit list
	pinnaudit run --config configs/smoke.yaml

convergence:
	pinnaudit convergence --config configs/heat_fd.yaml --levels 4
	pinnaudit convergence --config configs/wave_fd.yaml --levels 4

figures:
	python scripts/plot_baselines.py --out figures

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
