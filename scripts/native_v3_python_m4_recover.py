"""Finalize the exact retained M4 root after a post-evaluation artifact failure."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mutation_forge.backends.heg import HegBackend
from mutation_forge.experiment.json_io import read_json
from mutation_forge.experiment.provider import LocalCodexAppServerProvider
from mutation_forge.native_v3_python import PythonSerialEpisodeConfigV1
from mutation_forge.native_v3_python.provider_evaluation import run_m4_single_root


class _RetainedRootTransport:
    def __init__(self, result: Mapping[str, Any]) -> None:
        self.result = dict(result)
        self.used = False

    def generate(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        del request
        if self.used:
            raise RuntimeError("retained M4 root may be replayed only once")
        self.used = True
        return dict(self.result)

    def repair(
        self,
        request: Mapping[str, Any],
        diagnostics: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        raise RuntimeError((request, diagnostics, "retained root is already valid"))

    def close(self) -> None:
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--heg-repo", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.workspace.resolve()
    turn = (
        root
        / "artifacts"
        / "generations"
        / "generation-0000"
        / "slot-00"
        / "initial"
    )
    request = read_json(turn / "slot-00.request.json.gz")
    provider_raw = read_json(turn / "slot-00.provider-raw.json.gz")
    if not isinstance(request, Mapping) or not isinstance(provider_raw, Mapping):
        raise RuntimeError("retained M4 provider artifacts are invalid")
    response_text = provider_raw.get("response_text")
    usage = provider_raw.get("usage")
    if not isinstance(response_text, str) or not isinstance(usage, Mapping):
        raise RuntimeError("retained M4 response or final usage is missing")
    provider_duration_ms = 0
    for line in (turn / "slot-00.events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if not isinstance(event, Mapping) or event.get("method") != "turn/completed":
            continue
        params = event.get("params")
        completed_turn = params.get("turn") if isinstance(params, Mapping) else None
        duration = (
            completed_turn.get("durationMs")
            if isinstance(completed_turn, Mapping)
            else None
        )
        if isinstance(duration, int) and not isinstance(duration, bool):
            provider_duration_ms = duration
    failure_report = (
        root
        / "native-v3-python-output"
        / "root-0000"
        / "m4-report.json.gz"
    )
    retained_failure_report = failure_report.with_name(
        "m4-report.pre-recovery.json.gz"
    )
    if retained_failure_report.exists():
        raise RuntimeError("M4 retained root was already recovered")
    if failure_report.exists():
        failure_report.replace(retained_failure_report)
    raw_result = {
        "status": "completed",
        "accepted": True,
        "charged": int(usage.get("totalTokens", 0)) > 0,
        "content": True,
        "response": json.loads(response_text),
        "response_text": response_text,
        "response_projection_valid": True,
        "response_diagnostics": [],
        "transport_diagnostics": provider_raw.get("transport_diagnostics", []),
        "usage": dict(usage),
        "provider_request_id": provider_raw.get("request_id"),
        "provider_thread_id": provider_raw.get("thread_id"),
        "provider_turn_id": provider_raw.get("turn_id"),
        "provider_duration_ms": provider_duration_ms,
        "model": request.get("model"),
        "effort": request.get("effort"),
    }
    transport = _RetainedRootTransport(raw_result)
    provider = LocalCodexAppServerProvider(
        model=str(request["model"]),
        effort=str(request["effort"]),
        concurrency=1,
        max_repairs=1,
        transport=transport,
        persist_artifacts=False,
    )
    backend = HegBackend(args.heg_repo)
    try:
        report = run_m4_single_root(
            provider,
            root,
            backend_factory=lambda: backend,
            config=PythonSerialEpisodeConfigV1(
                order=30,
                graph_seed=101,
                policy_seed=17,
                horizon=1,
                witness_cap=64,
                episode_id="native-v3-python-m4-single-root",
                forbidden_lengths=backend.target_forbidden_lengths(30),
            ),
        )
    finally:
        provider.close()
    print(
        json.dumps(
            {
                **report,
                "workspace": str(root),
                "recovered_from": str(retained_failure_report),
                "model_calls_during_recovery": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] in {"completed", "program_failure"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
