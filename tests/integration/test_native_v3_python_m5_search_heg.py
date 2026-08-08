from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mutation_forge.backends.heg import HegBackend
from mutation_forge.native_v3_python import (
    DevelopmentCaseV1,
    M5ProviderContextV1,
    M5ProviderResultV1,
    PythonPanelScientificEvaluator,
    run_m5_search,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _source(label: str) -> str:
    return (
        "def propose(ctx, graph, api, seed):\n"
        "    candidates = api.non_edges_legal()\n"
        f"    selected = api.pick(candidates, seed, \"{label}\")\n"
        "    if not selected:\n"
        "        return api.no_plan(\"NO_MATCH\")\n"
        "    api.add_edge(selected)\n"
        "    return api.emit()\n"
    )


class _RecordedProvider:
    model = "recorded-m5"
    effort = "high"

    def __init__(self) -> None:
        self.anchor = M5ProviderContextV1(
            "recorded-anchor",
            "recorded-anchor-turn",
            "/opaque/recorded-anchor",
            ("recorded-anchor-turn",),
        )

    def ensure_specification_anchor(
        self,
        *,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del prompt, system_prompt, output_schema, artifact_dir
        return M5ProviderResultV1(
            response_text=json.dumps(
                {
                    "schema_version": "mforge.native.python_m5_specification_ack.v1",
                    "ack": "specification-retained",
                }
            ),
            context=self.anchor,
            usage={
                "inputTokens": 0,
                "cachedInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "outputTokens": 0,
                "reasoningOutputTokens": 0,
                "totalTokens": 0,
                "final": True,
                "partial": False,
            },
            duration_ms=0,
            warnings=0,
        )

    def _result(
        self,
        generation: int,
        slot: str,
        history: tuple[str, ...],
    ) -> M5ProviderResultV1:
        turn = f"recorded-g{generation}-{slot}"
        source = _source(f"recorded-g{generation}-{slot}")
        response = json.dumps(
            {
                "schema_version": "mforge.native.python_policy_response.v1",
                "source": source,
            },
            separators=(",", ":"),
        )
        return M5ProviderResultV1(
            response_text=response,
            context=M5ProviderContextV1(
                thread_id=f"recorded-thread-g{generation}-{slot}",
                turn_id=turn,
                thread_path=f"/opaque/g{generation}/{slot}",
                included_turn_ids=history + (turn,),
            ),
            usage={
                "inputTokens": 1,
                "cachedInputTokens": 0,
                "cacheWriteInputTokens": 0,
                "outputTokens": 1,
                "reasoningOutputTokens": 0,
                "totalTokens": 2,
                "final": True,
                "partial": False,
            },
            duration_ms=0,
            warnings=0,
        )

    def generate_root(
        self,
        *,
        anchor: M5ProviderContextV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del prompt, system_prompt, output_schema, idempotency_key, artifact_dir
        return self._result(generation, slot, anchor.included_turn_ids)

    def generate_child(
        self,
        *,
        parent: M5ProviderContextV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del prompt, system_prompt, output_schema, idempotency_key, artifact_dir
        return self._result(generation, slot, parent.included_turn_ids)

    def repair(
        self,
        *,
        previous: M5ProviderResultV1,
        generation: int,
        slot: str,
        prompt: str,
        system_prompt: str,
        output_schema: Mapping[str, Any],
        idempotency_key: str,
        artifact_dir: Path,
    ) -> M5ProviderResultV1:
        del prompt, system_prompt, output_schema, idempotency_key, artifact_dir
        return self._result(generation, slot, previous.context.included_turn_ids)

    def close(self) -> None:
        return None


def test_recorded_two_generation_python_search_uses_current_heg(
    tmp_path: Path,
) -> None:
    backend = HegBackend(PROJECT_ROOT.parent / "heg")
    try:
        panel = (
            DevelopmentCaseV1(
                case_id="order-30-seed-101",
                order=30,
                graph_seed=101,
                policy_seed=17,
                horizon=1,
                witness_cap=64,
                forbidden_lengths=backend.target_forbidden_lengths(30),
            ),
        )
        evaluator = PythonPanelScientificEvaluator(
            backend=backend,
            artifact_root=tmp_path / "scientific",
        )
        report = run_m5_search(
            provider=_RecordedProvider(),
            evaluator=evaluator,
            workspace=tmp_path / "search",
            panel=panel,
            system_prompt="Return only the required structured response.",
            specification_prompt="Retain the accepted ordinary-Python specification.",
            specification_ack_schema={"type": "object"},
            policy_schema={"type": "object"},
        )
    finally:
        backend.close()
    assert report["candidate_count"] == 16
    assert report["generation_allocations"] == {
        "0": {"children": 0, "roots": 8},
        "1": {"children": 4, "roots": 4},
    }
    profiles = report["behavior_profiles"]
    assert isinstance(profiles, Mapping)
    assert all(
        isinstance(profile, Mapping)
        and profile["propose_calls"] == 1
        and profile["selector_frequencies"]["non_edges_legal"] == 1
        and profile["action_frequencies"]["add_edge"] == 1
        for profile in profiles.values()
    )
    assert backend.score_implementation == "heg-cpp-score-worker"
    assert backend.dirty is False
    assert report["dsl_runtime_used"] is False
    assert report["safe_api_expanded"] is False
    assert report["preview_active"] is False
    assert all(report["acceptance_checks"].values())
    assert report["provider_turns"] == 17
    assert report["candidate_program_turns"] == 16
    assert report["specification_anchor_turns"] == 1
    assert report["usage"] == {
        "inputTokens": 16,
        "cachedInputTokens": 0,
        "cacheWriteInputTokens": 0,
        "outputTokens": 16,
        "reasoningOutputTokens": 0,
        "totalTokens": 32,
    }
