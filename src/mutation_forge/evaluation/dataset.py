from __future__ import annotations

from mutation_forge.artifacts import canonical_json_hash
from mutation_forge.backends.base import GraphBackend
from mutation_forge.config import LabConfig
from mutation_forge.models import DatasetEntry, JsonValue

DATASET_SCHEMA_VERSION = "1.0"


def build_dataset(
    config: LabConfig,
    backend: GraphBackend,
    *,
    heg_commit: str,
) -> dict[str, JsonValue]:
    entries: list[DatasetEntry] = []
    for order in config.dataset.orders:
        for graph_seed in config.dataset.graph_seeds:
            graph = backend.generate_seed(order=order, seed=graph_seed)
            validation = backend.validate(graph)
            if not validation.valid:
                raise RuntimeError(
                    f"generated graph failed validation: {'; '.join(validation.errors)}"
                )
            graph_hash = backend.canonical_hash(graph)
            entries.append(
                DatasetEntry(
                    entry_id=f"n{order}-g{graph_seed}-{graph_hash[:12]}",
                    order=order,
                    graph_seed=graph_seed,
                    graph6=backend.serialize_graph6(graph),
                    graph_hash=graph_hash,
                    generator_version="heg-cubic-first-v1",
                    backend_id=backend.backend_id,
                    heg_commit=heg_commit,
                    split=config.dataset.split,
                )
            )
    payload: dict[str, JsonValue] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "split": config.dataset.split,
        "backend_id": backend.backend_id,
        "heg_commit": heg_commit,
        "orders": list(config.dataset.orders),
        "graph_seeds": list(config.dataset.graph_seeds),
        "policy_seeds": list(config.dataset.policy_seeds),
        "entries": [entry.as_dict() for entry in entries],
    }
    payload["manifest_hash"] = canonical_json_hash(payload)
    return payload
