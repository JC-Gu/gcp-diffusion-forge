.PHONY: install install-cuda lint format typecheck test test-unit test-cov clean dev-up dev-down

install:
	uv sync --all-packages

install-cuda:
	uv sync --all-packages --extra cuda

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy packages/

test:
	uv run pytest tests/

test-unit:
	uv run pytest tests/unit/ -v

test-cov:
	uv run pytest tests/unit/ --cov=packages --cov-report=term-missing --cov-report=html

# Start local GCS emulator + Prefect server
dev-up:
	docker compose up -d
	@echo "GCS emulator:  http://localhost:4443"
	@echo "Prefect UI:    http://localhost:4200"

dev-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
	rm -rf htmlcov/ .coverage
