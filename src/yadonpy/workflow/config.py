"""Workflow configuration helpers shared by examples and high-level scripts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


TRUE_TOKENS = {"1", "true", "t", "yes", "y", "on"}
FALSE_TOKENS = {"0", "false", "f", "no", "n", "off"}


class EnvReader:
    """Small typed wrapper around environment-variable parsing."""

    def __init__(self, environ: dict[str, str] | None = None):
        self.environ = os.environ if environ is None else environ

    def text(self, name: str, default: str) -> str:
        raw = self.environ.get(name)
        if raw is None:
            return str(default)
        text = str(raw).strip()
        return text if text else str(default)

    def bool(self, name: str, default: bool) -> bool:
        raw = self.environ.get(name)
        if raw is None:
            return bool(default)
        token = str(raw).strip().lower()
        if token in TRUE_TOKENS:
            return True
        if token in FALSE_TOKENS:
            return False
        return bool(default)

    def int(self, name: str, default: int) -> int:
        raw = self.environ.get(name)
        if raw is None or not str(raw).strip():
            return int(default)
        return int(raw)

    def float(self, name: str, default: float) -> float:
        raw = self.environ.get(name)
        if raw is None or not str(raw).strip():
            return float(default)
        return float(raw)

    def path(self, name: str, default: str | Path) -> Path:
        return Path(self.text(name, str(default))).expanduser().resolve()


@dataclass(frozen=True)
class ResourceConfig:
    mpi: int = 1
    omp: int = 1
    gpu: int = 0
    gpu_id: int | None = None

    @classmethod
    def from_env(
        cls,
        env: EnvReader | None = None,
        *,
        default_omp: int = 1,
        default_gpu: int = 0,
    ) -> "ResourceConfig":
        reader = env or EnvReader()
        return cls(
            mpi=reader.int("MPI", 1),
            omp=reader.int("OMP", int(default_omp)),
            gpu=reader.int("GPU", int(default_gpu)),
            gpu_id=reader.int("GPU_ID", 0),
        )


@dataclass(frozen=True)
class WorkflowConfig:
    restart: bool
    work_dir: Path
    resources: ResourceConfig
    profile: str = "production"

    @classmethod
    def from_env(
        cls,
        *,
        default_work_dir: str | Path,
        restart_var: str = "RESTART_STATUS",
        profile_var: str = "YADONPY_PROFILE",
        env: EnvReader | None = None,
        default_omp: int = 1,
        default_gpu: int = 0,
    ) -> "WorkflowConfig":
        reader = env or EnvReader()
        return cls(
            restart=reader.bool(restart_var, False),
            work_dir=reader.path("WORK_DIR", default_work_dir),
            resources=ResourceConfig.from_env(
                reader,
                default_omp=default_omp,
                default_gpu=default_gpu,
            ),
            profile=reader.text(profile_var, "production").strip().lower(),
        )


__all__ = ["EnvReader", "ResourceConfig", "WorkflowConfig"]
