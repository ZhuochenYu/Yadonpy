"""Restart-aware filesystem helpers for script-first workflows.

`WorkDir` wraps a root directory and provides predictable child-directory
creation, restart cleanup, and JSON bookkeeping. The goal is to let examples
express scientific steps in order while keeping file operations idempotent and
safe across reruns.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from ..runtime import resolve_restart


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _safe_child_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(name or "").strip())
    s = s.strip("._")
    return s or "job"


@dataclass
class WorkDir(os.PathLike[str]):
    """Path-like workflow directory handle.

    The object behaves like a normal path for existing YadonPy APIs while also
    tracking lightweight workflow metadata under ``work_dir/.yadonpy/workdir.json``.
    It never deletes an existing directory unless ``clean=True`` is passed
    explicitly.
    """

    path: Path | str
    restart: Optional[bool] = None
    clean: bool = False
    create: bool = True
    metadata_name: str = '.yadonpy/workdir.json'
    _path: Path = field(init=False, repr=False)
    _meta: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._path = Path(self.path).expanduser().resolve()
        self.restart = bool(resolve_restart(self.restart))
        existed_before = self._path.exists()
        if self.clean and (not self.restart) and existed_before:
            shutil.rmtree(self._path)
            existed_before = False
        if self.create:
            self._path.mkdir(parents=True, exist_ok=True)
        self._meta = self._load_meta(existed_before=existed_before)
        self.touch()

    # ---- path-like behavior -------------------------------------------------
    def __fspath__(self) -> str:
        return str(self._path)

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self) -> str:
        return f"WorkDir(path={self._path!s}, restart={self.restart}, initialized={self.initialized})"

    def __truediv__(self, other):
        return self._path / other

    def __rtruediv__(self, other):
        return Path(other) / self._path

    def __getattr__(self, name: str):
        return getattr(self._path, name)

    # ---- metadata -----------------------------------------------------------
    @property
    def metadata_path(self) -> Path:
        return self._path / self.metadata_name

    @property
    def initialized(self) -> bool:
        try:
            return bool(self._meta.get('initialized', False))
        except Exception:
            return False

    @property
    def existed_before(self) -> bool:
        try:
            return bool(self._meta.get('existed_before', False))
        except Exception:
            return False

    @property
    def flags(self) -> Dict[str, Any]:
        v = self._meta.get('flags', {})
        return v if isinstance(v, dict) else {}

    def _load_meta(self, *, existed_before: bool) -> Dict[str, Any]:
        mp = self.metadata_path
        if mp.exists():
            try:
                data = json.loads(mp.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {
            'schema_version': 1,
            'path': str(self._path),
            'initialized': False,
            'created_at': _now_iso(),
            'updated_at': _now_iso(),
            'restart': bool(self.restart),
            'existed_before': bool(existed_before),
            'flags': {},
        }

    def save(self) -> 'WorkDir':
        mp = self.metadata_path
        mp.parent.mkdir(parents=True, exist_ok=True)
        self._meta['path'] = str(self._path)
        self._meta['restart'] = bool(self.restart)
        self._meta['updated_at'] = _now_iso()
        tmp = mp.with_suffix(mp.suffix + '.tmp')
        tmp.write_text(json.dumps(self._meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        os.replace(tmp, mp)
        return self

    def touch(self) -> 'WorkDir':
        self._meta['initialized'] = True
        self._meta['last_opened_at'] = _now_iso()
        return self.save()

    def mark(self, key: str, value: Any = True) -> 'WorkDir':
        flags = self.flags
        flags[str(key)] = value
        self._meta['flags'] = flags
        return self.save()

    def get(self, key: str, default: Any = None) -> Any:
        return self.flags.get(str(key), default)

    @property
    def path_obj(self) -> Path:
        return self._path

    def child(self, name: str, *, restart: Optional[bool] = None, clean: bool = False, create: bool = True) -> "WorkDir":
        child_path = self._path / _safe_child_name(name)
        return workdir(child_path, restart=self.restart if restart is None else restart, clean=clean, create=create)


def workdir(path: str | os.PathLike | WorkDir, *, restart: Optional[bool] = None, clean: bool = False, create: bool = True) -> WorkDir:
    """Return a path-like workflow directory handle.

    Parameters
    ----------
    path:
        Existing path or a previously created :class:`WorkDir`.
    restart:
        Optional override for the global YadonPy restart flag.
    clean:
        Delete the directory first. This is opt-in and is never implied by
        ``restart=False``.
    create:
        Create the directory if needed.
    """
    if isinstance(path, WorkDir):
        if restart is not None:
            path.restart = bool(resolve_restart(restart))
        if clean and (not path.restart) and path.path_obj.exists():
            shutil.rmtree(path.path_obj)
            path.path_obj.mkdir(parents=True, exist_ok=True)
            path._meta['existed_before'] = False
        if create:
            path.path_obj.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path
    return WorkDir(path=path, restart=restart, clean=clean, create=create)


def workunit(parent: str | os.PathLike | WorkDir, name: str, *, restart: Optional[bool] = None, clean: bool = False, create: bool = True) -> WorkDir:
    base = workdir(parent, restart=restart, clean=False, create=create)
    return base.child(name, restart=restart, clean=clean, create=create)


__all__ = ['WorkDir', 'workdir', 'workunit']
