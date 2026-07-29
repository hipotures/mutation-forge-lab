from __future__ import annotations


class SandboxError(RuntimeError):
    """Base class for coordinator-visible worker failures."""


class UnsupportedPlatformError(SandboxError):
    pass


class WorkerTimeoutError(SandboxError):
    pass


class WorkerCrashError(SandboxError):
    pass


class ProtocolError(SandboxError):
    pass
