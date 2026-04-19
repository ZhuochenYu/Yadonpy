"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..core.logging_utils import yadon_print


class GromacsError(RuntimeError):
    """Raised when a GROMACS command fails."""


def _which(cmd: str) -> Optional[str]:
    from shutil import which

    return which(cmd)


@dataclass(frozen=True)
class GromacsExec:
    """How to invoke GROMACS.

    We deliberately keep this lightweight and allow cluster-specific wrappers.
    """

    # NOTE:
    # We intentionally default to **gmx** (thread-MPI builds are very common).
    # Many HPC clusters also provide `gmx_mpi`, but that binary typically
    # requires `mpirun/srun` to actually use multiple MPI ranks.
    #
    # YadonPy's default workflow controls parallelism via `-ntmpi/-ntomp`, which
    # matches the thread-MPI `gmx` binary.
    gmx_cmd: str = "gmx"  # or gmx_mpi (advanced)

    @staticmethod
    def autodetect() -> "GromacsExec":
        """Auto-detect a usable `gmx` executable.

        Priority (most predictable first):
          1) $YADONPY_GMX_CMD (explicit override)
          2) gmx
          3) gmx_mpi

        Rationale:
          - In many installs both `gmx` and `gmx_mpi` exist.
          - `gmx_mpi` usually needs `mpirun/srun` to make use of multi-rank MPI.
          - YadonPy defaults to controlling parallelism via `-ntmpi/-ntomp`,
            which aligns with `gmx` thread-MPI builds.
        """
        override = os.environ.get("YADONPY_GMX_CMD")
        if override:
            return GromacsExec(override)
        if _which("gmx"):
            return GromacsExec("gmx")
        if _which("gmx_mpi"):
            return GromacsExec("gmx_mpi")
        return GromacsExec("gmx")


class GromacsRunner:
    @staticmethod
    def _normalize_energy_term_name(name: str) -> str:
        """Normalize GROMACS energy term names for robust matching."""
        s = str(name or '').strip().lower()
        s = re.sub(r'\([^)]*\)', '', s)
        s = s.replace('_', ' ')
        s = re.sub(r'[^a-z0-9]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    @classmethod
    def _semantic_energy_aliases(cls, requested: str) -> list[str]:
        base = cls._normalize_energy_term_name(requested)
        aliases = {base}
        if base == 'temperature':
            aliases.update({'temperature', 'temp', 't system'})
        elif base == 'pressure':
            aliases.update({'pressure', 'pres dc', 'pressure dc', 'pres', 'pressure bar'})
        elif base == 'density':
            aliases.update({'density', 'density kg m 3', 'density kg m3', 'mass density'})
        elif base == 'volume':
            aliases.update({'volume', 'box volume'})
        elif base == 'kinetic en':
            aliases.update({'kinetic en', 'kinetic energy'})
        elif base == 'total energy':
            aliases.update({'total energy', 'tot energy'})
        return [a for a in aliases if a]

    """Small helper to run GROMACS commands with good error messages."""

    def __init__(
        self,
        exec_: Optional[GromacsExec] = None,
        *,
        env: Optional[dict[str, str]] = None,
        verbose: bool = True,
    ):
        self.exec = exec_ or GromacsExec.autodetect()
        self.env = env
        self.verbose = bool(verbose)
        self._help_cache: dict[str, str] = {}
        # Cache expensive `gmx energy` term probes to avoid repeated calls.
        # Key: (edr_path, cwd_path_or_empty)
        self._energy_term_cache: dict[tuple[str, str], dict[str, int]] = {}

    def _log(self, msg: str) -> None:
        if self.verbose:
            yadon_print(str(msg), level=1)

    def _gmx_help(self, subcmd: str, *, cwd: Optional[Path] = None) -> str:
        """Return cached help output for `gmx <subcmd> -h` (best-effort)."""
        if subcmd in self._help_cache:
            return self._help_cache[subcmd]
        try:
            # Help probes are option-detection metadata, not part of the real
            # workflow. Keep them quiet and bounded so a wrapper-specific
            # `gmx mdrun -h` stall cannot block a simulation stage.
            proc = subprocess.run(
                [self.exec.gmx_cmd, subcmd, "-h"],
                cwd=str(cwd) if cwd else None,
                env={**os.environ, **(self.env or {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=8,
            )
            out = (proc.stdout or b"") + (proc.stderr or b"")
            text = out.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            text = ""
        except Exception:
            text = ""
        self._help_cache[subcmd] = text
        return text

    def _tool_has_option(self, subcmd: str, opt: str, *, cwd: Optional[Path] = None) -> bool:
        """Return True if help text suggests the tool supports `opt`.

        If help cannot be obtained, return True to avoid false negatives on
        restricted clusters.
        """
        help_txt = self._gmx_help(subcmd, cwd=cwd)
        if not help_txt:
            return True
        return opt in help_txt

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        stdin_text: Optional[str] = None,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        cmd = [self.exec.gmx_cmd, *args]
        if self.verbose:
            self._log("[CMD] " + " ".join(map(str, cmd)))
        proc = subprocess.run(
            cmd,
            input=stdin_text.encode("utf-8") if stdin_text is not None else None,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            cwd=str(cwd) if cwd else None,
            env={**os.environ, **(self.env or {})},
        )
        if check and proc.returncode != 0:
            raise GromacsError(
                "GROMACS command failed\n"
                f"  cmd: {' '.join(cmd)}\n"
                f"  cwd: {str(cwd) if cwd else os.getcwd()}\n"
                f"  stdout:\n{(proc.stdout or b'').decode('utf-8', errors='replace')}\n"
                f"  stderr:\n{(proc.stderr or b'').decode('utf-8', errors='replace')}\n"
            )
        return proc

    def _run_capture_tee(
        self,
        args: Sequence[str],
        *,
        cwd: Optional[Path] = None,
        env: Optional[dict[str, str]] = None,
        tail_chars: int = 8000,
    ) -> tuple[int, str]:
        """Run a command while streaming stdout to the console and capturing a tail.

        This avoids storing the full output in memory while still giving us
        enough information to produce actionable error messages.
        """
        cmd = [self.exec.gmx_cmd, *map(str, args)]
        self._log("[CMD] " + " ".join(cmd))

        env_final = os.environ.copy()
        env_final.update(self.env or {})
        if env:
            env_final.update({str(k): str(v) for k, v in env.items()})

        # IMPORTANT:
        #   Do NOT enable universal_newlines/text mode here.
        #   GROMACS (notably `mdrun -v`) uses carriage returns ("\r") to update
        #   progress *in-place* on a single line. Python's universal newlines
        #   converts standalone "\r" into "\n", which would incorrectly create
        #   a new console line for every progress update.
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env_final,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        assert proc.stdout is not None

        tail = deque()  # stores decoded text chunks
        tail_len = 0

        # Stream raw bytes to preserve \r behavior, while capturing a tail for errors.
        for chunk in iter(lambda: proc.stdout.read(4096), b""):
            if self.verbose:
                try:
                    import sys

                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except Exception:
                    # Best-effort fallback
                    print(chunk.decode("utf-8", errors="replace"), end="")

            text = chunk.decode("utf-8", errors="replace")
            tail.append(text)
            tail_len += len(text)
            while tail_len > tail_chars and tail:
                old = tail.popleft()
                tail_len -= len(old)

        proc.wait()
        return int(proc.returncode), "".join(tail)

    # ----------------------
    # Core GROMACS actions
    # ----------------------
    def grompp(
        self,
        *,
        mdp: Path,
        gro: Path,
        top: Path,
        out_tpr: Path,
        ndx: Optional[Path] = None,
        cpt: Optional[Path] = None,
        maxwarn: int = 5,
        cwd: Optional[Path] = None,
    ) -> None:
        args = [
            "grompp",
            "-f",
            str(mdp),
            "-c",
            str(gro),
            "-p",
            str(top),
            "-o",
            str(out_tpr),
            "-maxwarn",
            str(maxwarn),
        ]
        if ndx:
            args += ["-n", str(ndx)]
        if cpt:
            args += ["-t", str(cpt)]
        self.run(args, cwd=cwd)

    def mdrun(
        self,
        *,
        tpr: Path,
        deffnm: str,
        cwd: Path,
        ntomp: Optional[int] = None,
        ntmpi: Optional[int] = None,
        use_gpu: bool = True,
        nb: Optional[str] = None,
        prefer_gpu_update: bool = True,
        gpu_id: Optional[str] = None,
        append: bool = True,
        cpi: Optional[Path] = None,
        checkpoint_minutes: Optional[float] = None,
        mdrun_extra_args: Optional[Sequence[str]] = None,
        # "auto" is generally the safest default across clusters; "on" can
        # over-constrain pinning policies under some schedulers/cgroups.
        pin: str = "auto",
        loud: bool = True,
    ) -> None:
        """Run `gmx mdrun` with robust GPU defaults.

        When `use_gpu=True`, YadonPy offloads the hot kernels to GPU by default.
        If `prefer_gpu_update=True` (default), it will also request GPU update:

          -nb gpu -bonded gpu -pme gpu -pmefft gpu -update gpu -pin auto

        The optional `nb` parameter can override *only* the nonbonded kernel
        placement ("cpu" or "gpu"). This is useful for EM-like runs where you
        want CPU for bonded/PME/update, but still prefer GPU nonbonded.

        Some systems (e.g., with interdispersed constraint groups + domain
        decomposition) cannot use GPU update. In that case, YadonPy will
        automatically fall back to `-update cpu` and retry once.
        """

        # Base cmd
        args: list[str] = ["mdrun", "-s", str(tpr), "-deffnm", str(deffnm)]

        # Console progress output.
        # Always add `-v` so progress is printed to stdout.
        # Relying on help-text detection for `-v` has proven unreliable on
        # some clusters/GROMACS wrappers.
        args += ["-v"]

        # Write coordinates to the log at a coarse interval.
        # Requested default: -stepout 10000 (applies to all runs).
        # We add it unconditionally; if a very old GROMACS build rejects it,
        # the mdrun retry logic will drop it and re-run.
        args += ["-stepout", "10000"]


        # Pinning improves CPU-side efficiency on most nodes.
        if self._tool_has_option("mdrun", "-pin", cwd=cwd):
            args += ["-pin", str(pin)]

        # Always write an explicit log file for debugging.
        if self._tool_has_option("mdrun", "-g", cwd=cwd):
            args += ["-g", f"{deffnm}.log"]

        if (
            checkpoint_minutes is not None
            and float(checkpoint_minutes) > 0.0
            and self._tool_has_option("mdrun", "-cpt", cwd=cwd)
        ):
            args += ["-cpt", str(float(checkpoint_minutes))]

        # Threads / ranks (thread-MPI)
        if ntomp is not None and self._tool_has_option("mdrun", "-ntomp", cwd=cwd):
            args += ["-ntomp", str(int(ntomp))]
        if ntmpi is not None and self._tool_has_option("mdrun", "-ntmpi", cwd=cwd):
            args += ["-ntmpi", str(int(ntmpi))]

        # Determine restart checkpoint.
        # If caller didn't pass cpi explicitly, auto-resume when append=True and
        # a checkpoint exists in cwd.
        if cpi is None and append:
            guess = Path(cwd) / f"{deffnm}.cpt"
            if guess.exists():
                cpi = guess

        # Offload targets
        if not use_gpu:
            # Default: force CPU execution even if the GROMACS build defaults to GPU.
            off = {
                "-nb": "cpu",
                "-bonded": "cpu",
                "-update": "cpu",
                "-pme": "cpu",
                "-pmefft": "cpu",
            }
        else:
            off = {
                "-nb": "gpu",
                "-bonded": "gpu",
                "-update": "gpu" if prefer_gpu_update else "cpu",
                "-pme": "gpu",
                "-pmefft": "gpu",
            }

        # Allow a targeted override of *only* the nonbonded kernel placement.
        # This is useful for "EM-like" runs where we want CPU for everything else,
        # but still prefer GPU nonbonded.
        if nb is not None:
            nb_norm = str(nb).strip().lower()
            if nb_norm in {"cpu", "gpu"}:
                off["-nb"] = nb_norm
            else:
                raise ValueError(f"Invalid nb={nb!r}. Expected 'cpu' or 'gpu'.")

        for k, v in off.items():
            if self._tool_has_option("mdrun", k, cwd=cwd):
                args += [k, v]

        # GPU selection
        env_run: dict[str, str] = {}
        # Whether this specific run actually uses GPU kernels.
        uses_gpu = any(v == "gpu" for v in off.values())

        if uses_gpu and gpu_id is not None and str(gpu_id).strip() != "":
            gid_raw = str(gpu_id).strip()

            # If desired, isolate the requested device(s) by restricting CUDA_VISIBLE_DEVICES.
            # This can help on some clusters, but can also interact poorly with local GPU/MIG setups.
            # Therefore it is **opt-in** via YADONPY_ISOLATE_GPU=1.
            isolate = os.environ.get("YADONPY_ISOLATE_GPU", "0").strip().lower() in ("1", "true", "yes")
            if (
                isolate
                and "CUDA_VISIBLE_DEVICES" not in os.environ
                and re.fullmatch(r"\d+(,\d+)*", gid_raw)
            ):
                env_run["CUDA_VISIBLE_DEVICES"] = gid_raw
                # Remap ids to 0..N-1 for the isolated view.
                ndev = len(gid_raw.split(","))
                gid_raw = ",".join(str(i) for i in range(ndev))
                self._log(
                    f"[INFO] Set CUDA_VISIBLE_DEVICES={env_run['CUDA_VISIBLE_DEVICES']} and remapped -gpu_id to {gid_raw} for isolation."
                )


            if self._tool_has_option("mdrun", "-gpu_id", cwd=cwd):
                args += ["-gpu_id", gid_raw]

        if not uses_gpu:
            # Avoid creating CUDA contexts on shared nodes when this run is CPU-only.
            env_run["GMX_DISABLE_GPU_DETECTION"] = "1"

        # Restart handling
        if cpi is not None and Path(cpi).exists() and self._tool_has_option("mdrun", "-cpi", cwd=cwd):
            args += ["-cpi", str(cpi)]
            if append and self._tool_has_option("mdrun", "-append", cwd=cwd):
                args += ["-append"]

        # Extra args (filter out duplicates of managed options)
        # Allow users to specify -npme via mdrun_extra_args, but keep it managed here to avoid duplicates.
        user_npme: Optional[str] = None
        if mdrun_extra_args:
            extra = list(mdrun_extra_args)
            for j, a in enumerate(extra):
                if str(a) == "-npme" and j + 1 < len(extra):
                    user_npme = str(extra[j + 1])
                    break

        protected = {
            "-nb",
            "-bonded",
            "-pme",
            "-pmefft",
            "-update",
            "-gpu_id",
            "-ntomp",
            "-ntmpi",
            "-npme",
            "-nt",
            "-pin",
            "-stepout",
            "-g",
            "-cpt",
            "-deffnm",
            "-s",
            "-cpi",
            "-append",
        }

        def _filter_extra(extra: Sequence[str]) -> list[str]:
            out: list[str] = []
            i = 0
            while i < len(extra):
                a = str(extra[i])
                if a in protected:
                    # Drop the option and its value (if any)
                    if i + 1 < len(extra) and not str(extra[i + 1]).startswith("-"):
                        i += 2
                    else:
                        i += 1
                    self._log(f"[WARN] Ignoring mdrun extra arg '{a} ...' (managed by workflow).")
                    continue
                out.append(a)
                i += 1
            return out

        if mdrun_extra_args:
            args += _filter_extra(list(mdrun_extra_args))

        # PME GPU with multiple (thread-MPI) ranks requires -npme to be set in recent GROMACS.
        # See: "PME tasks were required to run on GPUs with multiple ranks but the -npme option was not specified."
        if uses_gpu and self._tool_has_option("mdrun", "-npme", cwd=cwd):
            if user_npme is not None and user_npme.strip() != "":
                # Respect user-provided value
                args += ["-npme", user_npme.strip()]
                self._log(f"[INFO] Using user-provided -npme {user_npme.strip()}.")
            else:
                # Auto-select a safe default when thread-MPI ranks > 1 and PME is on GPU.
                if ntmpi is not None and int(ntmpi) > 1:
                    if "-pme" in args:
                        k = args.index("-pme")
                        if k + 1 < len(args) and args[k + 1] == "gpu":
                            # Avoid adding duplicates
                            if "-npme" not in args:
                                args += ["-npme", "1"]
                                self._log("[INFO] Added -npme 1 (required for PME GPU with multiple thread-MPI ranks).")

        def _render(cmd_: Sequence[str]) -> str:
            return " ".join(str(x) for x in cmd_)

        def _is_user_input_error(out: str) -> bool:
            s = (out or "").lower()
            needles = (
                "error in user input",
                "inconsistency in user input",
                "requested mdrun to use gpu devices",
                "incompatible devices",
                "cannot compute pme interactions on a gpu",
                "non-dynamical integrator",
                "pme tasks were required to run on gpus with multiple ranks",
                "-npme option was not specified",
            )
            return any(n in s for n in needles)

        def _is_cufft_failure(out: str) -> bool:
            s = (out or "").lower()
            needles = (
                "cufft",
                "gpu_3dfft",
                "3d fft",
            )
            return any(n in s for n in needles)

        def _is_update_gpu_incompatible(out: str) -> bool:
            s = (out or "").lower()
            needles = (
                "update groups can not be used",
                "update groups cannot be used",
                "cannot use gpu update",
                "can not use gpu update",
                "gpu update is not supported",
            )
            return any(n in s for n in needles)

        def _is_cuda_error(out: str, rc: int) -> bool:
            if rc == -6:
                # SIGABRT is common on CUDA illegal-address errors.
                pass
            s = (out or "").lower()
            needles = (
                "cuda error",
                "illegal memory access",
                "out of memory",
                "cudaerrormemoryallocation",
                "cudaerrorillegaladdress",
                "cudaerrorinvalidvalue",
                "freeing of the device buffer failed",
                "gmx::internalerror",
            )
            return any(n in s for n in needles)

        def _replace_kv(cmd: list[str], key: str, val: str) -> None:
            if key in cmd:
                j = cmd.index(key)
                if j + 1 < len(cmd):
                    cmd[j + 1] = val

        def _drop_opt(cmd: list[str], key: str) -> None:
            if key in cmd:
                j = cmd.index(key)
                if j + 1 < len(cmd) and not str(cmd[j + 1]).startswith("-"):
                    del cmd[j : j + 2]
                else:
                    del cmd[j : j + 1]

        def _fallback_to_cpu(reason: str) -> None:
            nonlocal uses_gpu
            self._log(reason)
            for key in ("-nb", "-bonded", "-update", "-pme", "-pmefft"):
                _replace_kv(args, key, "cpu")
            _drop_opt(args, "-gpu_id")
            _drop_opt(args, "-npme")
            _drop_opt(args, "-cpi")
            _drop_opt(args, "-append")
            env_run["GMX_DISABLE_GPU_DETECTION"] = "1"
            try:
                deffnm_idx = args.index("-deffnm")
                deffnm_prefix = str(args[deffnm_idx + 1]) if deffnm_idx + 1 < len(args) else "md"
            except Exception:
                deffnm_prefix = "md"
            for suffix in (".log", ".xtc", ".trr", ".edr", ".cpt", "_prev.cpt"):
                try:
                    stale = Path(cwd) / f"{deffnm_prefix}{suffix}"
                    if stale.exists():
                        stale.unlink()
                except Exception:
                    pass
            uses_gpu = False

        # Run with streaming output and limited captured tail.
        max_attempts = 3
        did_fft_fallback = False

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self._log(f"[WARN] mdrun failed (attempt {attempt-1}/{max_attempts}). Retrying (attempt {attempt}/{max_attempts})...")
            rc, tail = self._run_capture_tee(args, cwd=cwd, env=env_run)
            if rc == 0:
                return

            # Very old GROMACS builds may not support -stepout. Retry without it.
            if "-stepout" in args and "unknown option" in (tail or "").lower() and "-stepout" in (tail or "").lower():
                try:
                    j = args.index("-stepout")
                    # remove flag + value if present
                    del args[j:j+2]
                    self._log("[WARN] Detected unsupported -stepout. Retrying without -stepout.")
                    continue
                except Exception:
                    pass

            # GPU update incompatibility: retry once with -update cpu.
            if uses_gpu and prefer_gpu_update and _is_update_gpu_incompatible(tail):
                if "-update" in args:
                    j = args.index("-update")
                    if j + 1 < len(args) and args[j + 1] == "gpu":
                        self._log(
                            "[WARN] Detected GPU-update incompatibility for this system. "
                            "Retrying with -update cpu (keeping other kernels on GPU)."
                        )
                        args[j + 1] = "cpu"
                        continue

            # cuFFT failures: keep short-range on GPU, move PME/FFT to CPU.
            if uses_gpu and (not did_fft_fallback) and _is_cufft_failure(tail):
                did_fft_fallback = True
                self._log(
                    "[WARN] Detected cuFFT-related failure with GPU PME/FFT. "
                    "Retrying with -pme cpu -pmefft cpu (keeping -nb/-bonded/-update on GPU)."
                )
                _replace_kv(args, "-pme", "cpu")
                _replace_kv(args, "-pmefft", "cpu")
                continue

            # Missing -npme with PME GPU + multiple ranks (GROMACS requires explicit -npme in this mode).
            if uses_gpu and ("-npme option was not specified" in (tail or "").lower()):
                if self._tool_has_option("mdrun", "-npme", cwd=cwd) and "-npme" not in args:
                    self._log("[WARN] GROMACS requires -npme when running PME on GPU with multiple ranks. Retrying with -npme 1.")
                    args += ["-npme", "1"]
                    continue
            # If GPU was requested but is not usable, fall back to CPU once.
            if uses_gpu and _is_user_input_error(tail):
                _fallback_to_cpu(
                    "[WARN] GPU offload appears unsupported/misconfigured on this node. "
                    "Falling back to CPU kernels for this stage."
                )
                continue

            # CUDA/internal GPU runtime failures tend to persist with the same
            # offload layout on GROMACS 2025.x. Prefer a deterministic CPU
            # fallback over blind same-command retries.
            if uses_gpu and _is_cuda_error(tail, rc):
                _fallback_to_cpu(
                    "[WARN] Detected CUDA/internal GPU runtime failure during mdrun. "
                    "Falling back to CPU kernels for this stage."
                )
                continue

            # non-retryable error
            desc = f"exit code {rc}"
            if rc < 0:
                sig = -rc
                try:
                    sname = signal.Signals(sig).name
                except Exception:
                    sname = "UNKNOWN"
                desc = f"terminated by signal {sig} ({sname})"
            raise GromacsError(
                "GROMACS mdrun failed\n"
                f"  cmd: {self.exec.gmx_cmd} {_render(args)}\n"
                f"  cwd: {str(cwd)}\n"
                f"  reason: {desc}\n"
                f"  output tail:\n{tail}"
            )

        # retries exhausted
        raise GromacsError(
            "GROMACS mdrun failed after retries\n"
            f"  cmd: {self.exec.gmx_cmd} {_render(args)}\n"
            f"  cwd: {str(cwd)}\n"
            f"  output tail:\n{tail}"
        )

    def energy_xvg(
        self,
        *,
        edr: Path,
        out_xvg: Path,
        terms: Sequence[str],
        cwd: Optional[Path] = None,
        allow_missing: bool = False,
    ) -> dict:
        """Extract time series from an .edr file into XVG.

        GROMACS uses an interactive menu for selecting energy terms.
        Numeric IDs can change across versions, so we:

        1) run a cheap probe call to capture the term list (name -> id)
        2) resolve requested names to IDs
        3) run the actual extraction using those IDs

        This is much more robust than hard-coding IDs or hoping name matching
        works the same across clusters.
        """
        ids, missing = self._resolve_energy_term_ids(edr=edr, terms=terms, cwd=cwd, allow_missing=allow_missing)
        menu = "\n".join([*map(str, ids), "0", ""])  # trailing newline
        args = ["energy", "-f", str(edr), "-o", str(out_xvg), "-xvg", "xmgrace"]
        self.run(args, cwd=cwd, stdin_text=menu)
        return {"resolved_terms": list(terms) if not missing else [t for t in terms if t not in missing], "missing_terms": missing}

    # ----------------------
    # Structural analyses
    # ----------------------
    def rdf(
        self,
        *,
        tpr: Path,
        xtc: Path,
        ndx: Path,
        ref_group: str,
        sel_group: str,
        out_rdf_xvg: Path,
        out_cn_xvg: Optional[Path] = None,
        bin_nm: float = 0.002,
        begin_ps: Optional[float] = None,
        end_ps: Optional[float] = None,
        cwd: Optional[Path] = None,
    ) -> None:
        # NOTE (2026-02): `gmx rdf` does NOT support a `-nojump` CLI option in
        # modern GROMACS (e.g., 2025.x). Any unwrapping must be done via
        # `gmx trjconv -pbc nojump` on the trajectory beforehand.
        args = [
            "rdf",
            "-s",
            str(tpr),
            "-f",
            str(xtc),
            "-n",
            str(ndx),
            "-o",
            str(out_rdf_xvg),
            "-bin",
            str(bin_nm),
            "-xvg",
            "none",
        ]
        if out_cn_xvg is not None:
            args += ["-cn", str(out_cn_xvg)]
        if begin_ps is not None:
            args += ["-b", str(begin_ps)]
        if end_ps is not None:
            args += ["-e", str(end_ps)]
        # group selection can accept group names
        stdin = f"{ref_group}\n{sel_group}\n"
        self.run(args, cwd=cwd, stdin_text=stdin)

    def msd(
        self,
        *,
        tpr: Path,
        xtc: Path,
        ndx: Path,
        group: str,
        out_xvg: Path,
        # NOTE: gmx msd enforces -dt <= -trestart, where -dt is effectively the
        # trajectory frame interval. Some GROMACS versions default -trestart to
        # 10 ps, which can break MSD on trajectories written every 20 ps or more.
        # YadonPy therefore:
        #   - uses a safer default trestart (20 ps), and
        #   - lets higher-level code auto-tune it from the written frame interval.
        trestart_ps: float = 20.0,
        dt_ps: Optional[float] = None,
        rmcomm: bool = True,
        begin_ps: Optional[float] = None,
        end_ps: Optional[float] = None,
        cwd: Optional[Path] = None,
    ) -> None:
        args = [
            "msd",
            "-s",
            str(tpr),
            "-f",
            str(xtc),
            "-n",
            str(ndx),
            "-o",
            str(out_xvg),
            "-xvg",
            "none",
            # Remove center-of-mass drift (recommended for diffusion/MSD).
            # This matches the typical robust analysis setup.
            "-trestart",
            str(trestart_ps),
        ]

        # GROMACS 2025+ enforces that -trestart is divisible by -dt.
        # Many versions default -dt to the trajectory frame interval. Passing it
        # explicitly avoids ambiguity and makes our auto-tuned trestart robust.
        if dt_ps is not None and self._tool_has_option("msd", "-dt", cwd=cwd):
            args += ["-dt", f"{float(dt_ps):.6f}"]
        # If supported, ask `gmx msd` to unwrap internally. This is a safety net
        # (we also generate a nojump trajectory upstream).
        if self._tool_has_option("msd", "-pbc", cwd=cwd):
            args += ["-pbc", "nojump"]
        # NOTE (v0.5.23): Do NOT add `-mol` by default.
        # Users may prefer atom-based MSD (especially for polymers / custom selections)
        # and `-mol` can change the meaning of the group selection.
        if rmcomm and self._tool_has_option("msd", "-rmcomm", cwd=cwd):
            args += ["-rmcomm"]
        if begin_ps is not None:
            args += ["-b", str(begin_ps)]
        if end_ps is not None:
            args += ["-e", str(end_ps)]
        self.run(args, cwd=cwd, stdin_text=f"{group}\n")

    def density_number_profile(
        self,
        *,
        tpr: Path,
        xtc: Path,
        ndx: Path,
        group: str,
        out_xvg: Path,
        axis: str = "Z",
        begin_ps: Optional[float] = None,
        end_ps: Optional[float] = None,
        cwd: Optional[Path] = None,
    ) -> None:
        axis = axis.upper()
        if axis not in {"X", "Y", "Z"}:
            raise ValueError("axis must be one of X/Y/Z")
        args = [
            "density",
            "-s",
            str(tpr),
            "-f",
            str(xtc),
            "-n",
            str(ndx),
            "-o",
            str(out_xvg),
            "-dens",
            "number",
            "-d",
            axis,
            "-xvg",
            "none",
        ]
        if begin_ps is not None:
            args += ["-b", str(begin_ps)]
        if end_ps is not None:
            args += ["-e", str(end_ps)]
        self.run(args, cwd=cwd, stdin_text=f"{group}\n")

    def gyrate(
        self,
        *,
        tpr: Path,
        xtc: Path,
        out_xvg: Path,
        ndx: Optional[Path] = None,
        group: str | int = 0,
        begin_ps: Optional[float] = None,
        end_ps: Optional[float] = None,
        cwd: Optional[Path] = None,
    ) -> None:
        """Compute radius of gyration time series via `gmx gyrate`.

        Notes
        -----
        - `group` can be an integer group index or a string (group name/index) accepted by GROMACS.
        - When ndx is not provided, group is interpreted in the default index file (System is usually 0).
        """
        args = ["gyrate", "-s", str(tpr), "-f", str(xtc), "-o", str(out_xvg), "-xvg", "none"]
        if ndx is not None:
            args += ["-n", str(ndx)]
        if begin_ps is not None:
            args += ["-b", str(begin_ps)]
        if end_ps is not None:
            args += ["-e", str(end_ps)]
        self.run(args, cwd=cwd, stdin_text=f"{group}\n")

    def list_energy_terms(self, *, edr: Path, cwd: Optional[Path] = None) -> dict[str, int]:
        """Return mapping {term_name: term_id} by probing `gmx energy` output."""
        cache_key = (str(edr.resolve()), str((cwd or Path("")).resolve()) if cwd else "")
        cached = self._energy_term_cache.get(cache_key)
        if cached:
            return cached

        # We must provide an output file path even when we only want the list.
        # Use a unique temporary filename (avoid collisions in MPI / multi-run situations)
        # and prefer a fast local scratch directory when available.
        env_tmp = os.environ.get("TMPDIR") or os.environ.get("TMP") or os.environ.get("TEMP")
        tmp_dir: Path | None = None
        if env_tmp:
            try:
                tmp_dir = Path(env_tmp)
                tmp_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                tmp_dir = None
        if tmp_dir is None:
            tmp_dir = cwd or edr.parent

        tmp_xvg = tmp_dir / f"_yadonpy_energy_list_tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}.xvg"

        # NOTE (2026-02): In some GROMACS builds, the interactive energy-term list
        # is printed to stderr (or split across stdout/stderr) when stdin is piped.
        # We therefore capture and parse *both* streams.
        try:
            proc = subprocess.run(
                [self.exec.gmx_cmd, "energy", "-f", str(edr), "-o", str(tmp_xvg), "-xvg", "none"],
                input=b"0\n",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(cwd) if cwd else None,
                env={**os.environ, **(self.env or {})},
            )
        finally:
            if tmp_xvg.exists():
                # NFS / parallel IO can sporadically fail on first attempt
                for _ in range(5):
                    try:
                        tmp_xvg.unlink()
                        break
                    except Exception:
                        try:
                            import time

                            time.sleep(0.05)
                        except Exception:
                            pass

        text = (proc.stdout or b"").decode("utf-8", errors="replace")
        mapping: dict[str, int] = {}
        # Some GROMACS builds print multiple "id  name" entries on the same line, e.g.
        #   14  Temperature   15  Pres-XX   16  Pressure   20  Volume   21  Density
        # Parse all entries on each line instead of assuming one entry per line.
        entry_pat = re.compile(r'(?:^|\s)(\d+)\s+([^\d].*?)(?=(?:\s{2,}\d+\s+[^\d])|$)')
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            for m in entry_pat.finditer(line):
                try:
                    idx = int(m.group(1))
                except Exception:
                    continue
                name = re.sub(r'\s+', ' ', (m.group(2) or '').strip())
                if not name:
                    continue
                if name.lower() in {"end", "0"}:
                    continue
                mapping[name] = idx

        # Save to cache for this runner instance.
        self._energy_term_cache[cache_key] = mapping
        return mapping

    def _resolve_energy_term_ids(self, *, edr: Path, terms: Sequence[str], cwd: Optional[Path], allow_missing: bool = False) -> tuple[list[int], list[str]]:
        mapping = self.list_energy_terms(edr=edr, cwd=cwd)
        if not mapping:
            raise GromacsError(
                "Failed to parse energy terms from `gmx energy` probe output. "
                "Please check your GROMACS installation and the .edr file."
            )

        # Case-insensitive + normalized semantic lookup.
        by_lower = {k.lower(): v for k, v in mapping.items()}
        by_norm = {self._normalize_energy_term_name(k): v for k, v in mapping.items()}

        resolved: list[int] = []
        missing: list[str] = []
        for t in terms:
            key = t.strip()
            if not key:
                continue
            v = by_lower.get(key.lower())
            if v is None:
                for alias in self._semantic_energy_aliases(key):
                    v = by_norm.get(alias)
                    if v is not None:
                        break
            if v is None:
                key_norm = self._normalize_energy_term_name(key)
                hits = [v2 for k2, v2 in by_norm.items() if key_norm and (key_norm in k2 or k2 in key_norm)]
                hits = list(dict.fromkeys(hits))
                if len(hits) == 1:
                    v = hits[0]
            if v is None:
                missing.append(t)
            else:
                resolved.append(v)

        if missing and (not allow_missing):
            sample = ", ".join(list(mapping.keys())[:12])
            raise GromacsError(
                "Requested energy terms were not found in the .edr file: "
                f"{missing}. Available examples: {sample} ..."
            )
        return resolved, missing

    def trjconv(
        self,
        *,
        tpr: Path,
        xtc: Path,
        out: Path,
        pbc: str = "mol",
        center: bool = True,
        ur: Optional[str] = None,
        group: str = "System",
        cwd: Optional[Path] = None,
    ) -> None:
        """Convenience wrapper for gmx trjconv with typical PBC handling."""
        args = ["trjconv", "-s", str(tpr), "-f", str(xtc), "-o", str(out), "-pbc", pbc]
        if center:
            args += ["-center"]
        if ur:
            args += ["-ur", str(ur)]
        # trjconv asks for 2 groups if center; choose same group twice
        stdin = f"{group}\n{group}\n" if center else f"{group}\n"
        self.run(args, cwd=cwd, stdin_text=stdin)


    def _ndx_group_index(self, *, ndx: Path, group: str) -> int:
        """Return 0-based group index in an .ndx file by group name (exact match)."""
        name = str(group).strip()
        idx = -1
        cur = None
        for raw in ndx.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                cur = line.strip("[]").strip()
                idx += 1
                if cur == name:
                    return idx
        raise KeyError(f"Group '{name}' not found in {ndx}")

    def _trjconv_nojump(
        self,
        *,
        tpr: Path,
        traj: Path,
        out_traj: Path,
        cwd: Optional[Path] = None,
        group: str = "System",
    ) -> None:
        """Make a no-jump trajectory for `gmx current` (best-effort)."""
        args = ["trjconv", "-s", str(tpr), "-f", str(traj), "-o", str(out_traj), "-pbc", "nojump"]
        # Preserve velocities when the output format supports it (required for conductivity).
        if out_traj.suffix.lower() in (".trr", ".tng"):
            args += ["-vel"]
        # trjconv asks for a group selection
        self.run(args, cwd=cwd, stdin_text=f"{group}\n")

    def current(
        self,
        *,
        tpr: Path,
        traj: Path,
        ndx: Path,
        group: str,
        out_xvg: Path,
        out_dsp: Optional[Path] = None,
        temp_k: Optional[float] = None,
        sh: Optional[int] = None,
        bfit_ps: Optional[float] = None,
        efit_ps: Optional[float] = None,
        cwd: Optional[Path] = None,
    ) -> subprocess.CompletedProcess:
        """Run `gmx current` (for ionic conductivity).

        Notes:
            - `gmx current` typically requires velocities; a `.trr` trajectory is recommended.
            - We pre-generate a `-pbc nojump` trajectory to reduce artifacts.
            - The nojump trajectory is cached under the analysis directory so
              repeated post-processing runs do not pay the conversion cost again.
        """
        out_xvg.parent.mkdir(parents=True, exist_ok=True)

        traj_in = Path(traj)
        nojump = out_xvg.parent / f"_nojump{traj_in.suffix}"
        try:
            sources = [Path(tpr), Path(traj)]
            cache_fresh = (
                nojump.exists()
                and nojump.stat().st_size > 0
                and nojump.stat().st_mtime >= max(p.stat().st_mtime for p in sources if p.exists())
            )
            if not cache_fresh:
                self._trjconv_nojump(tpr=tpr, traj=traj_in, out_traj=nojump, cwd=cwd)
            traj_in = nojump
        except Exception:
            # fall back to raw trajectory
            traj_in = Path(traj)

        args = [
            "current",
            "-s",
            str(tpr),
            "-f",
            str(traj_in),
            "-n",
            str(ndx),
            "-o",
            str(out_xvg),
            "-xvg",
            "none",
        ]
        if out_dsp is not None:
            args += ["-dsp", str(out_dsp)]
        if temp_k is not None:
            args += ["-temp", str(float(temp_k))]
        if sh is not None:
            args += ["-sh", str(int(sh))]
        if bfit_ps is not None:
            args += ["-bfit", str(float(bfit_ps))]
        if efit_ps is not None:
            args += ["-efit", str(float(efit_ps))]

        # gmx current prompts multiple times. Most GROMACS tools accept either an index or a group name
        # at the prompt. Using the group name avoids fragile index mapping between .tpr built-in groups and
        # external .ndx groups (a common cause of selecting the wrong group silently).
        stdin = "\n".join([str(group)] * 20) + "\n"
        proc = self.run(args, cwd=cwd, stdin_text=stdin, check=False)

        # Sanity check: some failures (e.g., missing velocities) may still exit 0 but produce an empty -dsp file.
        if out_dsp is not None:
            try:
                dsp_path = Path(out_dsp)
                if (not dsp_path.exists()) or dsp_path.stat().st_size < 16:
                    raise GromacsError(
                        "gmx current produced an empty -dsp output (required for EH conductivity). "
                        "This typically indicates that velocities are not present in the trajectory or the selected group has no charges."
                    )
            except OSError:
                raise GromacsError("gmx current did not produce a readable -dsp output.")

        if proc.returncode != 0:
            raise GromacsError(
                "GROMACS command failed\n"
                f"  cmd: {self.exec.gmx_cmd} {' '.join(args)}\n"
                f"  stdout:\n{proc.stdout.decode('utf-8', errors='replace')}\n"
                f"  stderr:\n{proc.stderr.decode('utf-8', errors='replace')}\n"
            )
        return proc
