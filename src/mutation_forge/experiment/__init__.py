"""Experiment workspaces and stage-free orchestration.

The experiment package is deliberately a small persistence boundary around the
existing scientific engines.  It owns identity, continuation, sessions, and
evidence layout; adapters supplied by later stages perform the actual search.
"""

from .config import (
    EXPERIMENT_SCHEMA_VERSION,
    MAX_EXPERIMENT_ID_BYTES,
    ExperimentConfig,
    ExperimentEvaluationConfig,
    ExperimentModelConfig,
    ExperimentResourcesConfig,
    ExperimentRunConfig,
    ExperimentSearchConfig,
    load_experiment_config,
    validate_experiment_id,
)
from .layout import ExperimentLayout, WorkspaceError
from .lock import LOCK_SCHEMA_VERSION, LockError
from .service import (
    ExperimentAdapter,
    ExperimentService,
    LegacyStage4Adapter,
    NullExperimentAdapter,
    run_experiment,
)
from .state import StateStore
from .status import STATUS_SCHEMA_VERSION, experiment_status, render_status

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "MAX_EXPERIMENT_ID_BYTES",
    "ExperimentConfig",
    "ExperimentEvaluationConfig",
    "ExperimentModelConfig",
    "ExperimentResourcesConfig",
    "ExperimentRunConfig",
    "ExperimentSearchConfig",
    "ExperimentAdapter",
    "ExperimentLayout",
    "ExperimentService",
    "LegacyStage4Adapter",
    "LOCK_SCHEMA_VERSION",
    "LockError",
    "NullExperimentAdapter",
    "STATUS_SCHEMA_VERSION",
    "StateStore",
    "WorkspaceError",
    "experiment_status",
    "load_experiment_config",
    "render_status",
    "run_experiment",
    "validate_experiment_id",
]
