from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mutation_forge.experiment.native_v3 import NativeV3ExperimentAdapter
from mutation_forge.experiment.service import ExperimentService


class _BatchedAstProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._lock = threading.Lock()

    @staticmethod
    def _program(slot_id: str) -> str:
        ordinal = int(slot_id.rsplit("-", 1)[1])
        return json.dumps(
            {
                "schema_version": "mforge.native.program.v3",
                "entry": {
                    "op": "if",
                    "condition": {
                        "op": "equal",
                        "left": {"op": "ctx", "field": "step_index"},
                        "right": ordinal,
                    },
                    "then": {"op": "no_plan", "reason": "EXPLICIT"},
                    "else": {"op": "no_plan", "reason": "NO_MATCH"},
                },
            },
            separators=(",", ":"),
        )

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt = json.loads(str(request["prompt"]))
        slot_ids = tuple(str(slot["slot_id"]) for slot in prompt["slots"])
        with self._lock:
            self.calls.append(slot_ids)
        envelope = {
            "schema_version": "mforge.native.program_batch.v3",
            "programs": [
                {
                    "slot_id": slot_id,
                    "program_json_raw": self._program(slot_id),
                    "design_summary": f"Deterministic no-plan integration policy {slot_id}.",
                }
                for slot_id in slot_ids
            ],
        }
        return {
            "response_text": json.dumps(envelope, separators=(",", ":")),
            "usage": {
                "inputTokens": 1,
                "cachedInputTokens": 0,
                "outputTokens": 1,
                "reasoningOutputTokens": 0,
                "totalTokens": 2,
            },
        }

    def repair(
        self,
        _request: Mapping[str, Any],
        _diagnostics: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        raise AssertionError("the valid integration batch must not be repaired")

    def close(self) -> None:
        return None


def _write_config(path: Path, workspace: Path) -> None:
    path.write_text(
        f'''schema_version = "mforge.experiment.v3"
exp_id = "native-v3-integration"
workspace = "{workspace.as_posix()}"
kind = "heg"
preset = "native"

[run]
wall_seconds = 120
output = "json"
turn_timeout_base_seconds = 30
max_total_tokens_per_hour = "unbounded"

[model]
provider = "codex"
name = "gpt-5.6-luna"
effort = "high"
concurrency = 2
max_repairs = 1

[search]
population_size = 8
max_generations = 1
max_model_turns = 2
selection = "persistent-elite-weighted-diversity"

[evaluation]
graph_mode = "unrestricted_min_degree_3"
order_schedule = "static"
orders = [4]
graph_seeds = [11]
policy_seeds = [21]
validation_graph_seeds = [12]
validation_policy_seeds = [22]
horizon = 1
baselines = [
  "add-low-local-cycle-risk",
  "remove-low-bridge-risk",
  "random-valid",
  "degree-fanout",
]
replay = false

[resources]
workers = 2
thread_count = 2

[native_v3]
provider_batch_size = 4
candidate_queue_capacity = 16
evaluation_queue_capacity = 8
target_evaluation_backlog = 4
candidate_shard_size = 1
auxiliary_shard_size = 1
witness_cap = 8
''',
        encoding="utf-8",
    )


def test_native_v3_runs_batched_ast_epoch_end_to_end(tmp_path: Path) -> None:
    provider = _BatchedAstProvider()
    config_path = tmp_path / "experiment.toml"
    _write_config(config_path, tmp_path / "workspace")

    result = ExperimentService(
        adapter=NativeV3ExperimentAdapter(provider=provider),
    ).run(config_path)

    assert result["schema_version"] == "mforge.experiment.run.v3"
    assert result["status"] == "exhausted"
    assert result["stop_reason"] == "generation_limit"
    assert result["generation"] == 1
    assert isinstance(result["semantic_checkpoint_hash"], str)
    assert provider.calls == [
        ("slot-00", "slot-01", "slot-02", "slot-03"),
        ("slot-04", "slot-05", "slot-06", "slot-07"),
    ]
    semantic_db = (
        tmp_path / "workspace" / "native-v3-integration" / "artifacts" / "native-v3-state.sqlite3"
    )
    assert semantic_db.is_file()
