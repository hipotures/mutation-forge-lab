.PHONY: check test smoke

check:
	uv run ruff check .
	uv run mypy
	uv run pytest

test:
	uv run pytest

smoke:
	uv run mforge baseline run --config configs/stage1-smoke.toml
