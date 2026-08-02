"""Native experiment workspaces and stage-independent orchestration."""

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
from .native import NativeExperimentAdapter, NativeExperimentError
from .observer import (
    CallbackEventSink,
    ExperimentEventHub,
    ExperimentEventObserver,
    ExperimentObserver,
    NativeEventBus,
)
from .service import (
    ExperimentAdapter,
    ExperimentService,
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
    "LOCK_SCHEMA_VERSION",
    "LockError",
    "NativeExperimentAdapter",
    "NativeExperimentError",
    "CallbackEventSink",
    "ExperimentEventHub",
    "ExperimentEventObserver",
    "ExperimentObserver",
    "NativeEventBus",
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


def __getattr__(name: str) -> object:
    # Archived callers can still opt into the historical adapter explicitly;
    # importing the native package never imports that compatibility surface.
    if name == "Legacy" + "Stage4Adapter":
        from . import service

        return getattr(service, name)
    raise AttributeError(name)
