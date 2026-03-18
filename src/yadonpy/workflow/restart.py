"""Tiny helper that makes ResumeManager easier to use in scripts.

This is intentionally *very* small so example scripts can look like:

    rst = Restart(work_dir, restart=True)
    rst.step("resp_monomer_A", outputs=[...], run=lambda: ...)

If ``restart=False``, steps are always executed (no skipping). If
``restart=True``, the helper will:

  - check whether outputs already exist
  - if yes, skip and optionally call ``load`` to return an in-memory object
  - otherwise run ``run`` and then record a success marker

This is a sequential-script helper (not a scheduler).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union

from .resume import ResumeManager, StepSpec


def _as_paths(outputs: Sequence[Union[str, Path]]) -> list[Path]:
    return [p if isinstance(p, Path) else Path(p) for p in outputs]


@dataclass
class Restart:
    """A small wrapper around :class:`~yadonpy.workflow.ResumeManager`.

    Parameters
    ----------
    work_dir:
        Workflow directory.
    restart:
        If True, enable skipping of completed steps.
    strict_inputs:
        If True, re-run a step when recorded input signature mismatches.
    """

    work_dir: Union[Path, str]
    restart: bool = True
    strict_inputs: bool = False

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir).expanduser().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Note: ResumeManager has its own skip logic, so we must pass enabled=self.restart
        # to ensure restart=False always runs steps.
        self._mgr = ResumeManager(self.work_dir, strict_inputs=bool(self.strict_inputs), enabled=bool(self.restart))

    @property
    def state_path(self) -> Path:
        return self._mgr.state_path

    def is_done(self, name: str, outputs: Sequence[Union[str, Path]], inputs: Optional[Dict[str, Any]] = None) -> bool:
        if not self.restart:
            return False
        return self._mgr.is_done(StepSpec(name=name, outputs=_as_paths(outputs), inputs=inputs))

    def step(
        self,
        name: str,
        *,
        outputs: Sequence[Union[str, Path]],
        run: Callable[[], Any],
        load: Optional[Callable[[], Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        description: str = "",
        verbose: bool = True,
    ) -> Any:
        """Run a step if needed, otherwise skip.

        If skipped and ``load`` is provided, returns ``load()``.
        """
        spec = StepSpec(name=name, outputs=_as_paths(outputs), inputs=inputs, description=description)
        if self.restart and self._mgr.is_done(spec):
            if verbose:
                print(f"[SKIP] {name} (already done)")
            return load() if load is not None else None
        # Run the step (even if restart=False)
        out = self._mgr.run(spec, run, verbose=verbose)
        return out
