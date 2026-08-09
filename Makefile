.PHONY: check test smoke appserver-artifact-parity

check:
	uv run ruff check .
	uv run mypy
	uv run pytest

test:
	uv run pytest

smoke:
	uv run mforge baseline run --config configs/stage1-smoke.toml

appserver-artifact-parity:
	uv run python scripts/appserver_artifact_parity.py
	uv run pytest tests/unit/test_appserver_artifact_parity.py
