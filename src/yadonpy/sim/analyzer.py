"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..gmx.engine import GromacsRunner
from ..gmx.analysis.xvg import read_xvg
from ..gmx.analysis.thermo import (
    stats_from_xvg,
    kappa_t_from_volume,
    bulk_modulus_gpa_from_kappa_t,
    cp_molar_from_enthalpy,
    cv_molar_from_total_energy,
)
from ..core import utils
from ..gmx.analysis.conductivity import conductivity_from_current_dsp, EHFit, plot_eh_fit_svg
from ..gmx.analysis.auto_plot import (
    plot_msd,
    plot_msd_overlay,
    plot_msd_summary,
    plot_rdf_cn,
    plot_rdf_cn_summary,
    _plot_overlay_from_xvgs,
)
from ..gmx.analysis.rg_convergence import find_rg_convergence, plot_rg_convergence_svg
from ..gmx.topology import parse_system_top, SystemTopology
from .polymer_metrics import compute_cell_summary, compute_polymer_metrics


def _integrate_cn(r: np.ndarray, g: np.ndarray, rho_nm3: float) -> np.ndarray:
    """Compute running coordination number from g(r)."""
    # CN(r) = 4*pi*rho * integral_0^r g(r')*r'^2 dr'
    dr = np.gradient(r)
    cn = 4.0 * np.pi * rho_nm3 * np.cumsum(g * r * r * dr)
    return cn




def _system_total_mass_amu(top: SystemTopology) -> float:
    total = 0.0
    try:
        for name, count in list(getattr(top, 'molecules', []) or []):
            mt = (getattr(top, 'moleculetypes', {}) or {}).get(name)
            if mt is None:
                continue
            total += float(getattr(mt, 'total_mass', 0.0)) * float(count)
    except Exception:
        return 0.0
    return float(total)


def _density_from_mass_and_volume(total_mass_amu: float, volume_nm3: float) -> Optional[float]:
    try:
        m_kg = float(total_mass_amu) * 1.66053906660e-27
        v_m3 = float(volume_nm3) * 1.0e-27
        if m_kg <= 0.0 or v_m3 <= 0.0:
            return None
        return float(m_kg / v_m3)
    except Exception:
        return None

def _first_shell_from_gr(r: np.ndarray, g: np.ndarray, cn: np.ndarray) -> Dict[str, Any]:
    """Estimate first shell location: peak and first minimum after peak."""
    if len(r) < 5:
        return {"r_peak_nm": None, "g_peak": None, "r_shell_nm": None, "cn_shell": None}
    # peak as global max (excluding r=0)
    i0 = int(np.argmax(g))
    r_peak = float(r[i0])
    g_peak = float(g[i0])
    # first minimum after peak: scan forward until slope changes sign (local min)
    r_shell = None
    cn_shell = None
    for i in range(i0 + 1, len(g) - 1):
        if g[i] <= g[i - 1] and g[i] <= g[i + 1]:
            r_shell = float(r[i])
            cn_shell = float(cn[i])
            break
    if r_shell is None:
        r_shell = float(r[-1])
        cn_shell = float(cn[-1])
    return {"r_peak_nm": r_peak, "g_peak": g_peak, "r_shell_nm": r_shell, "cn_shell": cn_shell}


@dataclass
class AnalyzeResult:
    work_dir: Path
    tpr: Path
    xtc: Path
    edr: Path
    top: Path
    ndx: Path
    trr: Optional[Path] = None
    frac_last: float = 0.5

    def __post_init__(self) -> None:
        """Best-effort resolution of analysis artifacts.

        In some workflows (notably multi-round additional equilibration), the
        analysis step may be handed paths that are valid *conceptually* but not
        physically present (e.g., when a stage is resumed, renamed, or moved).

        To avoid false "cannot check equilibrium" outcomes (which can cause
        repeated additional rounds), we aggressively try to resolve missing
        artifacts by searching under the work directory.
        """

        def _resolve_missing(p: Path, *, suffix: str) -> Path:
            p = Path(p)
            if p.exists():
                return p

            # 1) search by basename under work_dir
            hits = list(Path(self.work_dir).rglob(p.name)) if p.name else []
            # 2) fallback: search for common filenames (md.*) for the suffix
            if not hits:
                common = [f"md{suffix}", f"md_steep{suffix}", f"md_cg{suffix}"]
                for nm in common:
                    hits.extend(Path(self.work_dir).rglob(nm))

            # Prefer the most recently modified file.
            hits = [h for h in hits if h.is_file()]
            if not hits:
                return p
            hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return hits[0]

        # Resolve core artifacts first
        self.tpr = _resolve_missing(self.tpr, suffix=".tpr")
        self.xtc = _resolve_missing(self.xtc, suffix=".xtc")
        self.edr = _resolve_missing(self.edr, suffix=".edr")

        # Optional artifacts
        if self.trr is not None:
            self.trr = _resolve_missing(self.trr, suffix=".trr")

    @staticmethod
    def _normalize_term_name(name: object) -> str:
        import re

        s = str(name or '').strip().lower()
        s = re.sub(r'\([^)]*\)', '', s)
        s = s.replace('_', ' ')
        s = re.sub(r'[^a-z0-9]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _find_thermo_key(self, thermo: Dict[str, Any], *candidates: str) -> Optional[str]:
        if not isinstance(thermo, dict):
            return None
        norm_map = {self._normalize_term_name(k): k for k in thermo.keys()}
        for cand in candidates:
            norm = self._normalize_term_name(cand)
            if norm in norm_map:
                return norm_map[norm]
        for cand in candidates:
            norm = self._normalize_term_name(cand)
            for k_norm, k_raw in norm_map.items():
                if norm and (norm in k_norm or k_norm in norm):
                    return k_raw
        return None

    def _thermo_mean(self, thermo: Dict[str, Any], *candidates: str) -> tuple[Optional[float], Optional[str], Optional[Dict[str, Any]]]:
        key = self._find_thermo_key(thermo, *candidates)
        if key is None:
            return None, None, None
        entry = thermo.get(key) or {}
        try:
            mean = float(entry.get('mean'))
        except Exception:
            mean = None
        return mean, key, (entry if isinstance(entry, dict) else None)

    def _analysis_dir(self) -> Path:
        # Keep analysis outputs grouped and sortable.
        d = self.work_dir / "06_analysis"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _system_dir(self) -> Path:
        """Locate the exported GROMACS system directory.

        YadonPy v0.2.7+ organizes module outputs under numbered folders.
        We keep a small backward-compatible probe for older layouts.
        """
        for name in ("02_system", "00_system"):
            d = self.work_dir / name
            if d.exists():
                return d
        # default (even if not yet created)
        return self.work_dir / "02_system"

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        seconds = float(max(0.0, seconds))
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        minutes, sec = divmod(seconds, 60.0)
        if minutes < 60.0:
            return f"{int(minutes)}m {sec:.1f}s"
        hours, minutes = divmod(minutes, 60.0)
        return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"

    def _analysis_log(self, message: str, *, level: int = 1) -> None:
        utils.radon_print(f"[ANALYZE] {message}", level=level)

    @staticmethod
    def _compact_path(path_like: object) -> str:
        try:
            p = Path(path_like)
            parts = list(p.parts)
            if len(parts) <= 4:
                return str(p)
            return str(Path(*parts[-4:]))
        except Exception:
            return str(path_like)

    @staticmethod
    def _shorten(text: object, *, max_len: int = 96) -> str:
        s = str(text)
        if len(s) <= max_len:
            return s
        return s[: max(0, max_len - 3)] + '...'

    def _rule(self, char: str = '-', width: int = 78) -> None:
        self._analysis_log(char * int(max(24, width)), level=1)

    def _section_begin(self, title: str, *, detail: Optional[str] = None) -> float:
        self._rule('=')
        self._analysis_log(f"[SECTION] {title}", level=1)
        if detail:
            self._analysis_log(f"[NOTE] {detail}", level=1)
        return time.perf_counter()

    def _section_done(self, title: str, t0: float, *, detail: Optional[str] = None) -> None:
        msg = f"[DONE] {title} | elapsed={self._format_elapsed(time.perf_counter() - float(t0))}"
        if detail:
            msg += f" | {detail}"
        self._analysis_log(msg, level=1)
        self._rule('=')

    def _item(self, label: str, value: object, *, level: int = 1) -> None:
        self._analysis_log(f"[ITEM] {label:<20}: {self._shorten(value)}", level=level)

    def _stat(self, label: str, value: object, *, level: int = 1) -> None:
        self._analysis_log(f"[STAT] {label:<20}: {self._shorten(value)}", level=level)

    def _progress_title(self, prefix: str, idx: int, total: int) -> str:
        if total <= 0:
            return prefix
        frac = 100.0 * float(idx) / float(total)
        return f"{prefix} [{idx}/{total} | {frac:5.1f}%]"

    @staticmethod
    def _compose_step_message(tag: str, title: str, *, detail: Optional[str] = None, elapsed: Optional[str] = None) -> str:
        msg = f"[{tag}] {title}"
        if elapsed is not None:
            msg += f" | elapsed={elapsed}"
        if detail:
            msg += f" | {detail}"
        return msg

    def _step_begin(self, title: str, *, detail: Optional[str] = None) -> float:
        self._analysis_log(self._compose_step_message("STEP", title, detail=detail), level=1)
        return time.perf_counter()

    def _step_done(self, title: str, t0: float, *, detail: Optional[str] = None) -> None:
        self._analysis_log(
            self._compose_step_message(
                "DONE",
                title,
                detail=detail,
                elapsed=self._format_elapsed(time.perf_counter() - float(t0)),
            ),
            level=1,
        )

    def _step_skip(self, title: str, *, detail: Optional[str] = None) -> None:
        self._analysis_log(self._compose_step_message("SKIP", title, detail=detail), level=1)

    def _step_warn(self, title: str, *, detail: Optional[str] = None) -> None:
        self._analysis_log(self._compose_step_message("WARN", title, detail=detail), level=2)

    def _load_summary(self) -> Dict[str, Any]:
        summary_path = self._analysis_dir() / "summary.json"
        if summary_path.exists():
            try:
                return json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _write_summary(self, data: Dict[str, Any]) -> None:
        summary_path = self._analysis_dir() / "summary.json"
        summary_path.write_text(
            json.dumps(self._prune_raw_paths(data), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _update_summary_sections(self, **sections: Any) -> Dict[str, Any]:
        summary = self._load_summary()
        for k, v in sections.items():
            summary[k] = v
        self._write_summary(summary)
        return summary

    @staticmethod
    def _prune_raw_paths(obj):
        """Prune raw artifact paths from user-facing summary payloads.

        Requested behavior:
          - In summary.json, raw data paths (absolute .xvg/.gro/.edr/.tpr/.xtc/.trr, etc.) are not necessary.
          - Keep plot outputs (e.g., *.svg) but store only the basename.

        This function recursively walks dict/list structures.
        """

        DROP_KEYS = {
            "xvg",
            "rdf_xvg",
            "cn_xvg",
            "gro",
            "edr",
            "tpr",
            "xtc",
            "trr",
        }
        DROP_SUFFIX = ("_xvg", "_gro", "_edr", "_tpr", "_xtc", "_trr")

        def _is_pathlike(s: str) -> bool:
            return isinstance(s, str) and ("/" in s or "\\" in s) and (s.endswith(".xvg") or s.endswith(".gro") or s.endswith(".edr") or s.endswith(".tpr") or s.endswith(".xtc") or s.endswith(".trr"))

        if isinstance(obj, list):
            return [AnalyzeResult._prune_raw_paths(v) for v in obj]
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                ks = str(k)
                if ks in DROP_KEYS or ks.endswith(DROP_SUFFIX):
                    continue
                # Keep svg but shorten to basename for portability.
                if isinstance(v, str) and (v.endswith(".svg") or v.endswith(".png")):
                    out[ks] = Path(v).name
                    continue
                if isinstance(v, str) and _is_pathlike(v):
                    # Drop remaining raw artifact paths.
                    continue
                out[ks] = AnalyzeResult._prune_raw_paths(v)
            return out
        return obj

    def get_all_prop(self, *, temp: float, press: float, save: bool = True) -> Dict[str, Any]:
        """Compute core thermo + polymer metrics and save them into analysis outputs."""
        t_all = self._section_begin(
            "Post-analysis workflow",
            detail=f"target_T={float(temp):.2f} K | target_P={float(press):.2f} bar",
        )
        runner = GromacsRunner()
        out = self._analysis_dir() / "thermo.xvg"
        self._item("analysis_dir", self._compact_path(self._analysis_dir()))
        self._item("trajectory", self._compact_path(self.xtc))
        self._item("energy_file", self._compact_path(self.edr))
        terms = [
            "Temperature",
            "Pressure",
            "Density",
            "Volume",
            "Potential",
            "Kinetic En.",
            "Total Energy",
            "Enthalpy",
        ]

        t0 = self._step_begin("Step 1/5 thermo extraction", detail="gmx energy -> 06_analysis/thermo.xvg")
        self._item("requested_terms", ", ".join(terms))
        res_energy = runner.energy_xvg(edr=self.edr, out_xvg=out, terms=terms, allow_missing=True)
        self._item("resolved_terms", ", ".join(res_energy.get("resolved_terms") or []) or "(none)")
        missing_terms = list(res_energy.get("missing_terms") or [])
        if missing_terms:
            self._item("missing_terms", ", ".join(missing_terms))
        df = read_xvg(out).df
        thermo: Dict[str, Any] = {}
        for col in df.columns:
            if col == "x":
                continue
            s = stats_from_xvg(out, col=col, frac_last=self.frac_last)
            thermo[col] = {"mean": float(s.mean), "std": float(s.std), "n": int(s.n)}
        if not thermo:
            self._step_warn("Step 1/5 thermo extraction", detail="no thermo series extracted from EDR")
            return {}
        self._step_done("Step 1/5 thermo extraction", t0, detail=f"series={len(thermo)} | output={out.name}")

        t0 = self._step_begin("Step 2/5 bulk thermodynamic properties", detail="production averages / compressibility / bulk modulus / Cp / Cv")
        density_mean, density_key, density_stats = self._thermo_mean(thermo, "Density", "Mass Density")
        temperature_mean, temperature_key, temperature_stats = self._thermo_mean(thermo, "Temperature", "Temp", "T-System")
        pressure_mean, pressure_key, pressure_stats = self._thermo_mean(thermo, "Pressure", "Pres. DC", "Pressure DC")
        volume_mean, volume_key, volume_stats = self._thermo_mean(thermo, "Volume", "Box Volume")

        basic: Dict[str, Any] = {
            "input_conditions": {"target_temperature_K": float(temp), "target_pressure_bar": float(press)},
            "density_kg_m3": density_mean,
            "temperature_K": temperature_mean,
            "pressure_bar": pressure_mean,
            "volume_nm3": volume_mean,
            "density_stats": density_stats,
            "temperature_stats": temperature_stats,
            "pressure_stats": pressure_stats,
            "volume_stats": volume_stats,
            "source_terms": {
                "density": density_key,
                "temperature": temperature_key,
                "pressure": pressure_key,
                "volume": volume_key,
            },
        }
        try:
            t_mean = float(temperature_mean if temperature_mean is not None else temp)
        except Exception:
            t_mean = float(temp)
        if volume_key and volume_key in df.columns:
            try:
                kappa = kappa_t_from_volume(df[volume_key].to_numpy(dtype=float), t_mean, frac_last=self.frac_last)
                basic["kappa_t_1_Pa"] = float(kappa)
                basic["bulk_modulus_GPa"] = float(bulk_modulus_gpa_from_kappa_t(kappa))
            except Exception:
                pass
        enthalpy_key = self._find_thermo_key(thermo, "Enthalpy")
        if enthalpy_key and enthalpy_key in df.columns:
            try:
                basic["Cp_J_mol_K"] = float(cp_molar_from_enthalpy(df[enthalpy_key].to_numpy(dtype=float), t_mean, frac_last=self.frac_last))
            except Exception:
                pass
        total_energy_key = self._find_thermo_key(thermo, "Total Energy", "Tot Energy")
        if total_energy_key and total_energy_key in df.columns:
            try:
                basic["Cv_J_mol_K"] = float(cv_molar_from_total_energy(df[total_energy_key].to_numpy(dtype=float), t_mean, frac_last=self.frac_last))
            except Exception:
                pass
        detail = f"density={density_mean:.3f} kg/m^3" if density_mean is not None else "density unavailable"
        self._step_done("Step 2/5 bulk thermodynamic properties", t0, detail=detail)
        try:
            self._stat("temperature_mean", f"{float(temperature_mean):.3f} K")
        except Exception:
            pass
        try:
            self._stat("pressure_mean", f"{float(pressure_mean):.3f} bar")
        except Exception:
            pass
        try:
            self._stat("density_mean", f"{float(density_mean):.3f} kg/m^3")
        except Exception:
            pass
        try:
            self._stat("volume_mean", f"{float(volume_mean):.4f} nm^3")
        except Exception:
            pass

        t0 = self._step_begin("Step 3/5 cell summary", detail="box lengths / volume statistics")
        cell_summary: Dict[str, Any] = {}
        try:
            cell_summary = compute_cell_summary(gro_path=self._system_dir() / "system.gro", xtc_path=self.xtc)
        except Exception as e:
            cell_summary = {"warning": str(e)}
        if (not cell_summary) or ("lengths_nm" not in cell_summary):
            try:
                a, b, c = _read_box_dims_nm(self._system_dir() / "system.gro")
                cell_summary = {
                    **(cell_summary or {}),
                    "source": "system.gro",
                    "lengths_nm": {
                        "a": {"mean": float(a), "std": 0.0},
                        "b": {"mean": float(b), "std": 0.0},
                        "c": {"mean": float(c), "std": 0.0},
                    },
                    "volume_nm3": {"mean": float(_read_box_volume_nm3(self._system_dir() / "system.gro")), "std": 0.0},
                }
            except Exception:
                pass
        self._step_done("Step 3/5 cell summary", t0)
        if isinstance(cell_summary, dict) and cell_summary.get("lengths_nm"):
            try:
                ln = cell_summary["lengths_nm"]
                self._stat("cell_lengths_nm", f"a={float(ln['a']['mean']):.3f}, b={float(ln['b']['mean']):.3f}, c={float(ln['c']['mean']):.3f}")
            except Exception:
                pass

        fallback_notes: Dict[str, Any] = {}
        try:
            vol_entry = (cell_summary or {}).get("volume_nm3") if isinstance(cell_summary, dict) else None
            cell_volume_mean = None
            if isinstance(vol_entry, dict):
                cell_volume_mean = vol_entry.get("mean")
            elif vol_entry is not None:
                cell_volume_mean = vol_entry
            if (volume_mean is None) and (cell_volume_mean is not None):
                volume_mean = float(cell_volume_mean)
                basic["volume_nm3"] = volume_mean
                basic["volume_stats"] = {"mean": float(volume_mean), "std": 0.0, "n": 1}
                basic.setdefault("source_terms", {})["volume"] = "cell_summary.volume_nm3"
                fallback_notes["volume_nm3"] = "cell_summary.volume_nm3"
            if density_mean is None and volume_mean is not None:
                try:
                    top_obj = parse_system_top(self.top)
                    total_mass_amu = _system_total_mass_amu(top_obj)
                    rho = _density_from_mass_and_volume(total_mass_amu, float(volume_mean))
                    if rho is not None:
                        density_mean = float(rho)
                        basic["density_kg_m3"] = density_mean
                        basic["density_stats"] = {"mean": float(density_mean), "std": 0.0, "n": 1}
                        basic.setdefault("source_terms", {})["density"] = "derived_from_topology_and_volume"
                        fallback_notes["density_kg_m3"] = "derived_from_topology_and_volume"
                except Exception:
                    pass
            if temperature_mean is None:
                temperature_mean = float(temp)
                basic["temperature_K"] = temperature_mean
                basic["temperature_stats"] = {"mean": float(temperature_mean), "std": None, "n": 0}
                basic.setdefault("source_terms", {})["temperature"] = "target_condition_fallback"
                fallback_notes["temperature_K"] = "target_condition_fallback"
            if pressure_mean is None:
                pressure_mean = float(press)
                basic["pressure_bar"] = pressure_mean
                basic["pressure_stats"] = {"mean": float(pressure_mean), "std": None, "n": 0}
                basic.setdefault("source_terms", {})["pressure"] = "target_condition_fallback"
                fallback_notes["pressure_bar"] = "target_condition_fallback"
            if fallback_notes:
                basic["fallbacks"] = fallback_notes
                self._item("basic_property_fallbacks", json.dumps(fallback_notes, ensure_ascii=False))
        except Exception:
            pass

        polymer_all: Dict[str, Any] = {}
        poly_mts = self._polymer_moltypes_from_meta()
        if poly_mts:
            t0 = self._step_begin("Step 4/5 polymer metrics", detail="Rg / end-to-end distance / persistence length")
            try:
                polymer_raw = compute_polymer_metrics(
                    gro_path=self._system_dir() / "system.gro",
                    xtc_path=self.xtc,
                    top_path=self.top,
                    system_meta_path=self._system_dir() / "system_meta.json",
                )
                if isinstance(polymer_raw, dict):
                    polymer_all = {k: v for k, v in polymer_raw.items() if not str(k).startswith("_")}
                    if (not cell_summary) and isinstance(polymer_raw.get("_cell"), dict):
                        cell_summary = dict(polymer_raw["_cell"])
            except Exception as e:
                polymer_all = {"warning": str(e)}
            self._step_done("Step 4/5 polymer metrics", t0, detail=f"polymer_moltypes={len(polymer_all)}")
        else:
            self._step_skip("Step 4/5 polymer metrics", detail="no polymer moltypes in system_meta.json")
        polymer_rg = {k: v.get("radius_of_gyration_nm") for k, v in polymer_all.items() if isinstance(v, dict) and "radius_of_gyration_nm" in v}
        polymer_e2e = {k: v.get("end_to_end_distance_nm") for k, v in polymer_all.items() if isinstance(v, dict) and "end_to_end_distance_nm" in v}
        polymer_pl = {
            k: {
                "persistence_length_nm": v.get("persistence_length_nm"),
                "average_backbone_bond_length_nm": v.get("average_backbone_bond_length_nm"),
                "n_chains": v.get("n_chains"),
                "n_frames": v.get("n_frames"),
            }
            for k, v in polymer_all.items()
            if isinstance(v, dict) and "persistence_length_nm" in v
        }
        if polymer_rg:
            self._item("Rg_entries", len(polymer_rg))
        if polymer_pl:
            self._item("persistence_entries", len(polymer_pl))

        result = {
            "thermo": thermo,
            "basic_properties": basic,
            "cell": cell_summary,
            "polymer_radius_of_gyration": polymer_rg,
            "polymer_end_to_end_distance": polymer_e2e,
            "polymer_persistence_length": polymer_pl,
            "polymer_metrics": polymer_all,
        }

        if save:
            t0 = self._step_begin("Step 5/5 save analysis summary", detail="write JSON summaries into 06_analysis/")
            ad = self._analysis_dir()
            (ad / "thermo_summary.json").write_text(json.dumps(thermo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (ad / "basic_properties.json").write_text(json.dumps(basic, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (ad / "cell_summary.json").write_text(json.dumps(cell_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (ad / "polymer_radius_of_gyration.json").write_text(json.dumps(polymer_rg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (ad / "polymer_end_to_end_distance.json").write_text(json.dumps(polymer_e2e, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (ad / "polymer_persistence_length.json").write_text(json.dumps(polymer_pl, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (ad / "polymer_metrics.json").write_text(json.dumps(polymer_all, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self._update_summary_sections(
                thermo=thermo,
                basic_properties=basic,
                cell=cell_summary,
                polymer_radius_of_gyration=polymer_rg,
                polymer_end_to_end_distance=polymer_e2e,
                polymer_persistence_length=polymer_pl,
                polymer_metrics=polymer_all,
            )
            self._step_done("Step 5/5 save analysis summary", t0)
            self._item("summary_json", self._compact_path(self._analysis_dir() / "summary.json"))

        detail = f"outputs={self._compact_path(self._analysis_dir())}"
        if density_mean is not None:
            detail += f" | mean_density={density_mean:.3f} kg/m^3"
        self._section_done("Post-analysis workflow", t_all, detail=detail)
        return result

    def _ndx_group_order(self) -> list[str]:
        """Return group names in index file order."""
        names: list[str] = []
        try:
            txt = Path(self.ndx).read_text(encoding="utf-8", errors="replace")
            for line in txt.splitlines():
                s = line.strip()
                if s.startswith("[") and s.endswith("]"):
                    names.append(s.strip("[]").strip())
        except Exception:
            return []
        return names

    def _pick_rg_group(self) -> str | int:
        """Pick a group for Rg check.

        Rule (v0.5.3):
        - If system_meta.json indicates polymer species (psmiles with '*' or kind=='polymer'),
          prefer a whole-polymer index group for the first polymer moltype.
          We try (in order): MOL_<moltype>, <moltype>, then common fallbacks
          like "Polymer".
        - Otherwise, fall back to group 0 (usually "System").
        """
        names = self._ndx_group_order()
        poly_mts = self._polymer_moltypes_from_meta()
        if poly_mts:
            mt = str(poly_mts[0])
            candidates = [f"MOL_{mt}", mt, "Polymer", "Polymers", f"REP_{mt}"]
            for cand in candidates:
                for i, n in enumerate(names):
                    if n.strip() == cand:
                        return i
            # Fuzzy fallback: any group that contains the moltype token
            for i, n in enumerate(names):
                ns = str(n).strip()
                if mt in ns and not ns.startswith("REP_"):
                    return i
            for i, n in enumerate(names):
                if mt in str(n):
                    return i
        return 0

    def _polymer_moltypes_from_meta(self) -> list[str]:
        """Return polymer moltypes inferred from system_meta.json.

        We treat a system as polymer-containing if any species has:
          - kind == 'polymer', OR
          - a (p)SMILES containing '*'
        """
        try:
            sys_dir = self._system_dir()
            meta_path = sys_dir / "system_meta.json"
            if not meta_path.exists():
                return []
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out: list[str] = []
            for sp in meta.get("species", []) or []:
                smi = str(sp.get("smiles") or "")
                kind = str(sp.get("kind") or "")
                mt = sp.get("moltype") or sp.get("mol_name") or sp.get("mol_id")
                if not mt:
                    continue
                if (kind.lower() == "polymer") or ("*" in smi):
                    out.append(str(mt))
            # stable ordering
            return sorted(set(out))
        except Exception:
            return []

    def _has_polymer_group(self) -> bool:
        """Whether this system contains polymer species (meta-driven)."""
        return bool(self._polymer_moltypes_from_meta())

    def _density_plateau_kwargs(self, *, has_polymer: bool) -> dict[str, float]:
        """Return density plateau criteria for polymer vs liquid-like systems."""
        if has_polymer:
            return {
                "min_window_frac": 0.2,
                "step_frac": 0.02,
                # Polymer and polyelectrolyte systems relax more slowly than simple liquids.
                # Their density tail during additional NPT rounds is often still drifting mildly
                # even when the box is already usable for downstream interface preparation.
                "slope_threshold_per_ps": 1.5e-1,
                "rel_std_threshold": 0.03,
            }
        return {
            "min_window_frac": 0.2,
            "step_frac": 0.02,
            "slope_threshold_per_ps": 8e-2,
            "rel_std_threshold": 0.02,
        }



    def _read_mdp_kv(self, mdp_path: Path) -> dict[str, str]:
        """Parse a GROMACS .mdp into a {key: value} mapping (best-effort)."""
        out: dict[str, str] = {}
        try:
            for raw in mdp_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.split(";", 1)[0].strip()
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip().lower()] = v.strip()
        except Exception:
            return {}
        return out

    def _auto_msd_trestart_ps(self, *, default_ps: float = 10.0) -> float:
        """Choose a safe -trestart for `gmx msd`.

        `gmx msd` requires dt <= trestart, where dt is effectively the trajectory frame interval.
        We parse dt and nstxout-compressed from the run's md.mdp when available.

        For smooth MSD curves, using an *overly large* `-trestart` can reduce the
        number of time origins and make the MSD look noisy / jagged. We therefore
        choose a value that is:
          - always >= frame_interval_ps (GROMACS requirement),
          - typically a small multiple of the frame interval to increase averaging,
          - capped to avoid pathological values when output is very sparse.

        Heuristic (v0.5.23):
            trestart ≈ 5 × (frame interval), but:
              - at least 10 ps (to keep enough time origins when frames are dense),
              - capped at 200 ps (to avoid pathological values when output is very sparse),
              - always an integer multiple of the frame interval (GROMACS 2025+ requirement).

        If parsing fails, we fall back to default_ps.
        """
        mdp_path = Path(self.tpr).with_suffix(".mdp")
        if not mdp_path.exists():
            return float(default_ps)

        kv = self._read_mdp_kv(mdp_path)
        try:
            dt = float(kv.get("dt", "0"))
        except Exception:
            dt = 0.0

        nst_raw = kv.get("nstxout-compressed") or kv.get("nstxout") or "0"
        try:
            nst = int(float(nst_raw))
        except Exception:
            nst = 0

        if dt > 0 and nst > 0:
            # Frame interval written to the compressed trajectory (xtc).
            frame_interval_ps = float(dt) * float(nst)
            # Choose a value that improves averaging but stays modest.
            # NOTE: Using too small trestart on sparse trajectories breaks `gmx msd`.
            # Using too large trestart reduces the number of time origins and can
            # make MSD look jagged. 5× frame interval is a good compromise.
            base = float(min(200.0, max(10.0, 5.0 * frame_interval_ps)))
            # GROMACS 2025+ requires -trestart to be divisible by -dt (which often defaults
            # to the trajectory frame interval). Ensure we return an *integer multiple*
            # of the frame interval to avoid runtime errors.
            try:
                import math
                n = int(math.ceil(base / max(frame_interval_ps, 1e-12)))
                n = max(n, 1)
                trestart = float(n) * float(frame_interval_ps)
                return float(trestart)
            except Exception:
                return float(base)
        return float(default_ps)
    def _rg_series(self) -> Optional[dict[str, Any]]:
        """Compute radius of gyration time series via `gmx gyrate` (best-effort).

        Returns a dict with:
          - t_ps: (N,) time in ps
          - rg_nm: (N,) Rg in nm
          - rg_components_nm: (N,3) optional components (RgX,RgY,RgZ)
        """
        if not self._has_polymer_group():
            return None
        runner = GromacsRunner()
        out = self._analysis_dir() / "rg.xvg"
        grp = self._pick_rg_group()
        try:
            runner.gyrate(tpr=self.tpr, xtc=self.xtc, ndx=self.ndx, group=grp, out_xvg=out)
            df = read_xvg(out).df
        except Exception:
            return None

        if "x" not in df.columns:
            return None
        t_ps = np.asarray(df["x"].values, dtype=float)
        cols = [c for c in df.columns if c != "x"]
        if not cols:
            return None

        rg_nm = np.asarray(df[cols[0]].values, dtype=float)
        rg_components_nm: Optional[np.ndarray] = None
        if len(cols) >= 4:
            # Common GROMACS output: Rg, RgX, RgY, RgZ
            try:
                rg_components_nm = np.vstack([
                    np.asarray(df[cols[1]].values, dtype=float),
                    np.asarray(df[cols[2]].values, dtype=float),
                    np.asarray(df[cols[3]].values, dtype=float),
                ]).T
            except Exception:
                rg_components_nm = None

        if rg_nm.size < 10:
            return None
        return {"t_ps": t_ps, "rg_nm": rg_nm, "rg_components_nm": rg_components_nm, "group": grp}

    def _rg_series_nm(self) -> Optional[np.ndarray]:
        """Backward compatible: return Rg (nm) only."""
        s = self._rg_series()
        return None if s is None else np.asarray(s["rg_nm"], dtype=float)

    def check_eq(self) -> bool:
        """Equilibration convergence check (yzc-gmx-gen style).

        v0.5.1 rule:
        - Always require *density* convergence (plateau) when Density is available.
        - If the system contains polymer species (detected from system_meta.json: kind=='polymer' or pSMILES contains '*'),
          additionally require polymer *Rg* convergence.

        This function never raises; it writes analysis/equilibrium.json and returns a boolean.
        """

        ok = True
        density_gate = None
        rg_gate = None
        poly_mts = self._polymer_moltypes_from_meta()
        density_kwargs = self._density_plateau_kwargs(has_polymer=bool(poly_mts))

        # --- density gate (always when available) ---
        try:
            runner = GromacsRunner()
            out_xvg = self._analysis_dir() / "thermo.xvg"
            mapping = runner.list_energy_terms(edr=self.edr)
            dens_term = None
            for have in mapping.keys():
                if "density" in str(have).lower():
                    dens_term = str(have)
                    break
            if dens_term is not None:
                runner.energy_xvg(edr=self.edr, out_xvg=out_xvg, terms=[dens_term])
                df = read_xvg(out_xvg).df
                if "x" in df.columns and len(df.columns) >= 2:
                    t_ps = np.asarray(df["x"].values, dtype=float)
                    y = np.asarray(df[df.columns[1]].values, dtype=float)
                    from ..gmx.analysis.plateau import find_plateau_start

                    # Density in GROMACS is typically kg/m^3. Use a relaxed slope gate for
                    # polymer and electrolyte systems where the tail-window mean can still drift
                    # slightly even after the short-time fluctuations have settled.
                    res = find_plateau_start(t_ps, y, **density_kwargs)
                    density_gate = {
                        "ok": bool(res.ok),
                        "window_start_time_ps": float(res.window_start_time_ps),
                        "mean": float(res.mean),
                        "std": float(res.std),
                        "rel_std": float(res.rel_std),
                        "slope_per_ps": float(res.slope),
                        "term": str(dens_term),
                        "criteria": {k: float(v) for k, v in density_kwargs.items()},
                        "system_class": ("polymer" if poly_mts else "liquid"),
                    }
                    if not res.ok:
                        ok = False
                        utils.radon_print(
                            f"Density gate does not converge. mean={res.mean:.6g}, std={res.std:.6g}, rel_std={res.rel_std:.3g}, slope={res.slope:.3g} /ps",
                            level=2,
                        )
        except Exception:
            # If density cannot be computed, we do not fail hard, but report.
            density_gate = {"ok": False, "reason": "Density term not available or parse failed"}
            ok = False

        # --- polymer Rg gate (only for polymer systems) ---
        if poly_mts:
            s = self._rg_series()
            if s is None:
                utils.radon_print(
                    "Polymer Rg gate enabled but Rg could not be computed; cannot check equilibrium reliably.",
                    level=2,
                )
                ok = False
            else:
                res_rg = find_rg_convergence(
                    np.asarray(s["t_ps"], dtype=float),
                    np.asarray(s["rg_nm"], dtype=float),
                    s.get("rg_components_nm"),
                )
                rg_gate = {
                    "ok": bool(res_rg.ok),
                    "converged_by": str(res_rg.converged_by),
                    "plateau_start_time_ps": float(res_rg.plateau_start_time),
                    "mean_nm": float(res_rg.mean),
                    "std_nm": float(res_rg.std),
                    "rel_std": float(res_rg.rel_std),
                    "slope_per_ps": float(res_rg.slope),
                    "sma_sd_nm": float(res_rg.sma_sd),
                    "sd_max_nm": float(res_rg.sd_max),
                    "group": str(s.get("group")),
                    "polymer_moltypes": poly_mts,
                }
                if not res_rg.ok:
                    ok = False
                    utils.radon_print(
                        f"Rg gate does not converge (by={res_rg.converged_by}). mean={res_rg.mean:.6g} nm, std={res_rg.std:.6g} nm, slope={res_rg.slope:.3g} /ps",
                        level=2,
                    )
        # --- print a clear summary (so users can see why additional rounds run) ---
        try:
            utils.radon_print("================ Equilibration convergence check ================", level=1)

            # Density
            if density_gate is None:
                utils.radon_print("[EQ-CHECK] Density: SKIPPED (no density term found in EDR)", level=2)
            else:
                st = "PASS" if density_gate.get("ok") else "FAIL"
                lvl = 1 if density_gate.get("ok") else 2
                utils.radon_print(
                    "[EQ-CHECK] Density plateau: %s | term=%s | mean=%.6g | rel_std=%.3g | slope=%.3g /ps | window_start=%.3g ps" % (
                        st,
                        density_gate.get("term"),
                        float(density_gate.get("mean", float('nan'))),
                        float(density_gate.get("rel_std", float('nan'))),
                        float(density_gate.get("slope_per_ps", float('nan'))),
                        float(density_gate.get("window_start_time_ps", float('nan'))),
                    ),
                    level=lvl,
                )

            # Polymer / Rg
            if not poly_mts:
                utils.radon_print("[EQ-CHECK] Polymer: NO (Rg gate disabled)", level=1)
            else:
                utils.radon_print("[EQ-CHECK] Polymer: YES | moltypes=%s" % (", ".join(map(str, poly_mts))), level=1)
                if rg_gate is None:
                    utils.radon_print("[EQ-CHECK] Rg plateau: FAIL (Rg could not be computed)", level=2)
                else:
                    st = "PASS" if rg_gate.get("ok") else "FAIL"
                    lvl = 1 if rg_gate.get("ok") else 2
                    utils.radon_print(
                        "[EQ-CHECK] Rg plateau: %s | by=%s | group=%s | mean=%.6g nm | rel_std=%.3g | slope=%.3g /ps | window_start=%.3g ps" % (
                            st,
                            rg_gate.get("converged_by"),
                            rg_gate.get("group"),
                            float(rg_gate.get("mean_nm", float('nan'))),
                            float(rg_gate.get("rel_std", float('nan'))),
                            float(rg_gate.get("slope_per_ps", float('nan'))),
                            float(rg_gate.get("plateau_start_time_ps", float('nan'))),
                        ),
                        level=lvl,
                    )

            # Overall
            st = "PASS" if ok else "FAIL"
            utils.radon_print("[EQ-CHECK] Overall: %s" % st, level=1 if ok else 2)
            utils.radon_print("[EQ-CHECK] Details saved to analysis/equilibrium.json and analysis/plots/", level=1)
            utils.radon_print("=================================================================", level=1)
        except Exception:
            pass

        # Dedicated marker for resumable workflows (best-effort)
        try:
            payload = {"ok": bool(ok)}
            if density_gate is not None:
                payload["density_gate"] = density_gate
            if rg_gate is not None:
                payload["rg_gate"] = rg_gate
            (self._analysis_dir() / "equilibrium.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

        # 不论是否收敛，都绘制收敛性曲线（best-effort）。
        try:
            from ..gmx.analysis.auto_plot import plot_thermo_stage

            plots_dir = self._analysis_dir() / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            thermo_xvg = self._analysis_dir() / "thermo.xvg"
            if thermo_xvg.exists():
                plot_thermo_stage(thermo_xvg, out_dir=plots_dir, title_prefix="eq")
        except Exception:
            pass
        try:
            # Rg 收敛曲线（yzc-gmx-gen style）
            s = self._rg_series()
            if s is not None and rg_gate is not None:
                plots_dir = self._analysis_dir() / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                # Recompute full result for plotting (cheap)
                res_rg = find_rg_convergence(
                    np.asarray(s["t_ps"], dtype=float),
                    np.asarray(s["rg_nm"], dtype=float),
                    s.get("rg_components_nm"),
                )
                plot_rg_convergence_svg(
                    t=np.asarray(s["t_ps"], dtype=float),
                    rg=np.asarray(s["rg_nm"], dtype=float),
                    rg_components=s.get("rg_components_nm"),
                    res=res_rg,
                    out_svg=plots_dir / "rg_convergence.svg",
                )
            elif poly_mts:
                # fallback: keep legacy plot if available
                self.rg()
        except Exception:
            pass
        return ok

    def rdf(
        self,
        mol_or_mols: object,
        *,
        center_mol: Optional[object] = None,
        include_h: bool = False,
        bin_nm: float = 0.002,
    ) -> Dict[str, Any]:
        """Compute RDF between center group and each species' atomtypes."""
        t_all = self._section_begin("RDF/CN analysis", detail=f"bin={float(bin_nm):.4f} nm")
        if center_mol is None:
            center_mol = mol_or_mols
        try:
            from rdkit import Chem
            smi = center_mol.GetProp('_yadonpy_smiles') if center_mol.HasProp('_yadonpy_smiles') else Chem.MolToSmiles(center_mol, isomericSmiles=True)
        except Exception:
            smi = ""

        topo = parse_system_top(self.top)
        center_group = None
        sys_dir = self._system_dir()
        meta_path = sys_dir / "system_meta.json"
        if meta_path.exists() and smi:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                for sp in meta.get("species", []):
                    if sp.get("smiles") == smi:
                        center_group = sp.get("moltype") or sp.get("mol_name") or sp.get("mol_id")
                        break
            except Exception:
                center_group = None
        if not center_group:
            center_group = "IONS"
            try:
                q_formal = sum(int(a.GetFormalCharge()) for a in center_mol.GetAtoms())
                if q_formal > 0:
                    center_group = "CATIONS"
                elif q_formal < 0:
                    center_group = "ANIONS"
            except Exception:
                pass

        analysis_dir = self._analysis_dir() / "rdf"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        runner = GromacsRunner()
        self._item("center_group", center_group)
        self._item("topology", self._compact_path(self.top))
        self._item("index", self._compact_path(self.ndx))

        try:
            atmap = {}
            for _mn, _count in topo.molecules:
                _mt = topo.moleculetypes.get(_mn)
                if _mt is None:
                    continue
                atmap[_mn] = {
                    "n_molecules": int(_count),
                    "natoms": int(_mt.natoms),
                    "unique_atomtypes": sorted(set(_mt.atomtypes)),
                }
            (analysis_dir / "atomtypes_by_moltype.json").write_text(
                json.dumps(atmap, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

        gro_path = sys_dir / "system.gro"
        vol_nm3 = _read_box_volume_nm3(gro_path) if gro_path.exists() else None
        ndx_groups: set[str] = set()
        try:
            ndx_path = Path(self.ndx)
            if ndx_path.exists():
                for raw in ndx_path.read_text(encoding='utf-8', errors='replace').splitlines():
                    s = raw.strip()
                    if s.startswith('[') and s.endswith(']'):
                        ndx_groups.add(s.strip('[]').strip())
        except Exception:
            ndx_groups = set()
        if not ndx_groups:
            try:
                from ..gmx.index import generate_system_ndx
                generate_system_ndx(top_path=self.top, ndx_path=self.ndx)
                ndx_path = Path(self.ndx)
                if ndx_path.exists():
                    for raw in ndx_path.read_text(encoding='utf-8', errors='replace').splitlines():
                        s = raw.strip()
                        if s.startswith('[') and s.endswith(']'):
                            ndx_groups.add(s.strip('[]').strip())
            except Exception:
                pass

        tasks: list[tuple[str, str, Optional[float], str]] = []
        for moltype, count in topo.molecules:
            mt = topo.moleculetypes.get(moltype)
            if mt is None:
                continue
            for atype in sorted(set(mt.atomtypes)):
                rho = None
                if vol_nm3 and vol_nm3 > 0:
                    n_atoms_type_per_mol = sum(1 for _at in mt.atomtypes if _at == atype)
                    rho = float(count) * float(n_atoms_type_per_mol) / float(vol_nm3)
                if not include_h:
                    all_h = True
                    for an, at in zip(mt.atomnames, mt.atomtypes):
                        if at == atype and not (an.upper().startswith('H') or at.lower().startswith('h')):
                            all_h = False
                            break
                    if all_h:
                        continue
                tasks.append((str(moltype), str(atype), rho, f"TYPE_{moltype}_{atype}"))

        total_tasks = len(tasks)
        self._item("rdf_targets", total_tasks)
        out_summary: Dict[str, Any] = {}
        rdf_xvgs: Dict[str, Path] = {}
        cn_xvgs: Dict[str, Path] = {}
        for idx, (moltype, atype, rho, target_group) in enumerate(tasks, start=1):
            tag = f"{moltype}:{atype}"
            title = self._progress_title("RDF/CN", idx, total_tasks)
            t0 = self._step_begin(title, detail=f"{center_group} -> {target_group}")
            rdf_xvg = analysis_dir / f"rdf_{moltype}_{atype}.xvg"
            cn_xvg = analysis_dir / f"cn_{moltype}_{atype}.xvg"
            if ndx_groups and (target_group not in ndx_groups):
                out_summary[tag] = {
                    "center_group": center_group,
                    "target_group": target_group,
                    "moltype": moltype,
                    "atomtype": atype,
                    "rho_target_nm3": rho,
                    "rdf_xvg": None,
                    "cn_xvg": None,
                    "rdf_cn_svg": None,
                    "note": f"ndx group missing: {target_group}",
                }
                self._step_skip(title, detail=f"missing ndx group {target_group}")
                continue
            try:
                runner.rdf(
                    tpr=self.tpr,
                    xtc=self.xtc,
                    ndx=self.ndx,
                    ref_group=center_group,
                    sel_group=target_group,
                    out_rdf_xvg=rdf_xvg,
                    out_cn_xvg=cn_xvg,
                    bin_nm=bin_nm,
                    cwd=analysis_dir,
                )
            except Exception as e:
                out_summary[tag] = {
                    "center_group": center_group,
                    "target_group": target_group,
                    "moltype": moltype,
                    "atomtype": atype,
                    "rho_target_nm3": rho,
                    "rdf_xvg": str(rdf_xvg) if rdf_xvg.exists() else None,
                    "cn_xvg": str(cn_xvg) if cn_xvg.exists() else None,
                    "rdf_cn_svg": None,
                    "note": f"gmx rdf failed: {e.__class__.__name__}",
                }
                self._step_warn(title, detail=f"gmx rdf failed: {e.__class__.__name__}")
                continue

            try:
                plots_dir = analysis_dir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                svg = plot_rdf_cn(
                    rdf_xvg=rdf_xvg,
                    cn_xvg=cn_xvg if cn_xvg.exists() else None,
                    out_svg=plots_dir / f"rdf_{moltype}_{atype}.svg",
                    title=f"RDF/CN: {center_group} vs {target_group}",
                )
            except Exception:
                svg = None
            df = read_xvg(rdf_xvg).df
            r = df["x"].to_numpy(dtype=float)
            ycols = [c for c in df.columns if c != "x"]
            g = df[ycols[0]].to_numpy(dtype=float) if ycols else np.zeros_like(r)
            cn_est = _integrate_cn(r, g, rho_nm3=float(rho or 0.0)) if rho else np.zeros_like(r)
            shell = _first_shell_from_gr(r, g, cn_est)
            if cn_xvg.exists() and shell.get("r_shell_nm") is not None:
                try:
                    df_cn = read_xvg(cn_xvg).df
                    r_cn = df_cn["x"].to_numpy(dtype=float)
                    y_cn_cols = [c for c in df_cn.columns if c != "x"]
                    cn_curve = df_cn[y_cn_cols[0]].to_numpy(dtype=float) if y_cn_cols else None
                    if cn_curve is not None and len(r_cn) == len(cn_curve) and len(r_cn) > 1:
                        shell["cn_shell"] = float(np.interp(float(shell["r_shell_nm"]), r_cn, cn_curve))
                except Exception:
                    pass
            out_summary[tag] = {
                "center_group": center_group,
                "target_group": target_group,
                "moltype": moltype,
                "atomtype": atype,
                "rho_target_nm3": rho,
                "rdf_xvg": str(rdf_xvg),
                "cn_xvg": str(cn_xvg),
                "rdf_cn_svg": str(svg) if svg is not None else None,
                **shell,
            }
            rdf_xvgs[tag] = rdf_xvg
            if cn_xvg.exists():
                cn_xvgs[tag] = cn_xvg
            detail = None
            if shell.get("r_shell_nm") is not None:
                detail = f"r_shell={float(shell['r_shell_nm']):.3f} nm"
                if shell.get("cn_shell") is not None:
                    detail += f" | CN={float(shell['cn_shell']):.3f}"
            self._step_done(title, t0, detail=detail)

        overlay: Dict[str, Any] = {}
        try:
            plots_dir = analysis_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            if rdf_xvgs:
                ov_rdf = _plot_overlay_from_xvgs(
                    xvg_map=rdf_xvgs,
                    out_svg=plots_dir / "rdf_all_types.svg",
                    title=f"RDF summary ({center_group})",
                    xlabel="r (nm)",
                    ylabel="g(r)",
                    smooth=True,
                )
                ov_cn = _plot_overlay_from_xvgs(
                    xvg_map=cn_xvgs,
                    out_svg=plots_dir / "cn_all_types.svg",
                    title=f"CN summary ({center_group})",
                    xlabel="r (nm)",
                    ylabel="CN",
                    smooth=True,
                ) if cn_xvgs else None
                ov_combo = plot_rdf_cn_summary(
                    rdf_xvgs=rdf_xvgs,
                    cn_xvgs=cn_xvgs,
                    out_svg=plots_dir / "rdf_cn_all_types.svg",
                    title=f"RDF/CN summary ({center_group})",
                )
                if ov_rdf is not None:
                    overlay["rdf_all_types_svg"] = str(ov_rdf)
                if ov_cn is not None:
                    overlay["cn_all_types_svg"] = str(ov_cn)
                if ov_combo is not None:
                    overlay["rdf_cn_all_types_svg"] = str(ov_combo)
        except Exception as e:
            self._step_warn("RDF/CN overlay plots", detail=str(e))
        if overlay:
            out_summary["_overlay"] = overlay

        summary_path = self._analysis_dir() / "summary.json"
        summary_all = {}
        if summary_path.exists():
            try:
                summary_all = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_all = {}
        summary_all.setdefault("rdf_first_shell", {})
        summary_all["rdf_first_shell"].update(out_summary)
        summary_path.write_text(
            json.dumps(self._prune_raw_paths(summary_all), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self._analysis_dir() / "rdf_first_shell.json").write_text(
            json.dumps(out_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._item("rdf_outputs", self._compact_path(analysis_dir))
        self._section_done("RDF/CN analysis", t_all, detail=f"targets={total_tasks} | outputs={self._compact_path(analysis_dir)}")
        return out_summary

    def msd(
        self,
        mols: Optional[Sequence[object]] = None,
        *,
        begin_ps: Optional[float] = None,
        end_ps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute MSD for each moltype group and estimate diffusion coefficient."""
        t_all = self._section_begin("MSD analysis", detail="per-moltype diffusion coefficients")
        topo = parse_system_top(self.top)
        runner = GromacsRunner()
        outdir = self._analysis_dir() / "msd"
        outdir.mkdir(parents=True, exist_ok=True)

        xtc_for_msd = Path(self.xtc)
        try:
            cand = xtc_for_msd.with_name(xtc_for_msd.name + ".pbc_tmp" + xtc_for_msd.suffix)
            if cand.exists() and cand.stat().st_size > 0:
                xtc_for_msd = cand
        except Exception:
            xtc_for_msd = Path(self.xtc)

        trestart_ps = self._auto_msd_trestart_ps(default_ps=20.0)
        self._item("xtc_used", self._compact_path(xtc_for_msd))
        self._item("trestart_ps", f"{trestart_ps:.3f}")
        dt_frame_ps: Optional[float] = None
        try:
            mdp_path = Path(self.tpr).with_suffix(".mdp")
            if mdp_path.exists():
                kv = self._read_mdp_kv(mdp_path)
                dt = float(kv.get("dt", "0") or 0)
                nst_raw = kv.get("nstxout-compressed") or kv.get("nstxout") or "0"
                nst = int(float(nst_raw))
                if dt > 0 and nst > 0:
                    dt_frame_ps = float(dt) * float(nst)
        except Exception:
            dt_frame_ps = None

        res: Dict[str, Any] = {}
        msd_xvgs: Dict[str, Path] = {}

        def _best_linear_fit(t_ps: np.ndarray, y_nm2: np.ndarray) -> dict[str, float]:
            if t_ps.size < 8:
                a, b = np.polyfit(t_ps, y_nm2, 1)
                return {"slope": float(a), "intercept": float(b), "r2": float("nan"), "t_start_ps": float(t_ps[0]), "t_end_ps": float(t_ps[-1])}
            start_fracs = [0.20, 0.30, 0.40, 0.50, 0.60]
            end_fracs = [0.85, 0.90, 0.95, 1.00]
            best = None
            n = int(t_ps.size)
            for sf in start_fracs:
                i0 = max(0, min(n - 3, int(round(sf * n))))
                for ef in end_fracs:
                    i1 = max(i0 + 3, min(n, int(round(ef * n))))
                    tt = t_ps[i0:i1]
                    yy = y_nm2[i0:i1]
                    if tt.size < 8:
                        continue
                    a, b = np.polyfit(tt, yy, 1)
                    yhat = a * tt + b
                    ss_res = float(np.sum((yy - yhat) ** 2))
                    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                    cand = {"slope": float(a), "intercept": float(b), "r2": float(r2), "t_start_ps": float(tt[0]), "t_end_ps": float(tt[-1]), "n_points": int(tt.size)}
                    if best is None or (cand["r2"] > best["r2"] + 1e-6) or (abs(cand["r2"] - best["r2"]) <= 1e-6 and cand["n_points"] > best["n_points"]):
                        best = cand
            if best is None:
                a, b = np.polyfit(t_ps, y_nm2, 1)
                return {"slope": float(a), "intercept": float(b), "r2": float("nan"), "t_start_ps": float(t_ps[0]), "t_end_ps": float(t_ps[-1])}
            best.pop("n_points", None)
            return best

        total = len(topo.molecules)
        self._item("msd_groups", total)
        for idx, (moltype, _count) in enumerate(topo.molecules, start=1):
            title = self._progress_title("MSD", idx, total)
            t0 = self._step_begin(title, detail=f"group={moltype}")
            xvg = outdir / f"msd_{moltype}.xvg"
            runner.msd(
                tpr=self.tpr,
                xtc=xtc_for_msd,
                ndx=self.ndx,
                group=moltype,
                out_xvg=xvg,
                begin_ps=begin_ps,
                end_ps=end_ps,
                trestart_ps=trestart_ps,
                dt_ps=dt_frame_ps,
                rmcomm=True,
                cwd=outdir,
            )
            df = read_xvg(xvg).df
            t = df["x"].to_numpy(dtype=float)
            ycols = [c for c in df.columns if c != "x"]
            msd_arr = df[ycols[0]].to_numpy(dtype=float) if ycols else np.zeros_like(t)
            if len(t) < 5:
                self._step_warn(title, detail=f"group={moltype} has too few MSD points")
                continue
            fit = _best_linear_fit(t, msd_arr)
            a = float(fit["slope"])
            D_nm2_ps = float(a) / 6.0
            D_m2_s = D_nm2_ps * 1e-6
            rec = {
                "D_m2_s": D_m2_s,
                "D_nm2_ps": D_nm2_ps,
                "fit_slope": float(a),
                "fit_intercept": float(fit["intercept"]),
                "fit_r2": float(fit.get("r2")) if fit.get("r2") is not None else None,
                "xvg": str(xvg),
                "fit_t_start_ps": float(fit["t_start_ps"]),
                "fit_t_end_ps": float(fit["t_end_ps"]),
                "xtc_used": str(xtc_for_msd),
            }
            try:
                plots_dir = outdir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                created = plot_msd(xvg, out_dir=plots_dir, group=str(moltype), fit_t_start_ps=float(fit["t_start_ps"]), fit_t_end_ps=float(fit["t_end_ps"]))
                if created:
                    rec.setdefault("plots", {}).update(created)
            except Exception:
                pass
            res[str(moltype)] = rec
            msd_xvgs[str(moltype)] = xvg
            self._step_done(title, t0, detail=f"group={moltype} | D={D_m2_s:.3e} m^2/s")

        try:
            if msd_xvgs:
                plots_dir = outdir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                ov1 = plot_msd_overlay(msd_xvgs=msd_xvgs, out_svg=plots_dir / "msd_overlay.svg", title="MSD overlay", loglog=False)
                ov2 = plot_msd_overlay(msd_xvgs=msd_xvgs, out_svg=plots_dir / "msd_overlay_loglog.svg", title="MSD overlay (log-log)", loglog=True)
                ov3 = plot_msd_summary(msd_xvgs=msd_xvgs, out_svg=plots_dir / "msd_all_types.svg", title="MSD summary (all moltypes)")
                if ov1 is not None or ov2 is not None or ov3 is not None:
                    res.setdefault("_overlay", {})
                    if ov1 is not None:
                        res["_overlay"]["msd_overlay_svg"] = str(ov1)
                    if ov2 is not None:
                        res["_overlay"]["msd_overlay_loglog_svg"] = str(ov2)
                    if ov3 is not None:
                        res["_overlay"]["msd_all_types_svg"] = str(ov3)
        except Exception as e:
            self._step_warn("MSD overlay plots", detail=str(e))

        summary_path = self._analysis_dir() / "summary.json"
        summary_all: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary_all = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_all = {}
        summary_all.setdefault("msd", {})
        summary_all["msd"].update(res)
        summary_path.write_text(json.dumps(self._prune_raw_paths(summary_all), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (self._analysis_dir() / "msd.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._item("msd_outputs", self._compact_path(outdir))
        self._section_done("MSD analysis", t_all, detail=f"groups={len(msd_xvgs)} | outputs={self._compact_path(outdir)}")
        return res

    def rg(self, *, begin_ps: Optional[float] = None, end_ps: Optional[float] = None) -> Dict[str, Any]:
        """Compute and plot radius of gyration time series (best-effort)."""
        t_all = self._section_begin("Rg analysis", detail="polymer radius of gyration time series")
        if not self._has_polymer_group():
            outdir = self._analysis_dir() / 'rg'
            outdir.mkdir(parents=True, exist_ok=True)
            rec: Dict[str, Any] = {
                'skipped': True,
                'reason': 'no polymer moltypes in system_meta.json',
            }
            (outdir / 'rg.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            try:
                self._update_summary_sections(rg=rec)
            except Exception:
                pass
            self._section_done("Rg analysis", t_all, detail="skipped=no polymer moltypes")
            return rec
        runner = GromacsRunner()
        outdir = self._analysis_dir() / 'rg'
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / 'rg.xvg'
        rec: Dict[str, Any] = {'xvg': str(out)}
        try:
            grp = self._pick_rg_group()
            rec['group'] = str(grp)
            runner.gyrate(tpr=self.tpr, xtc=self.xtc, ndx=self.ndx, group=grp, out_xvg=out, begin_ps=begin_ps, end_ps=end_ps, cwd=outdir)
            df = read_xvg(out).df
            # pick first non-x column
            ycols = [c for c in df.columns if c != 'x']
            if ycols:
                y = df[ycols[0]].to_numpy(dtype=float)
                rec['mean_nm'] = float(np.mean(y))
                rec['std_nm'] = float(np.std(y))
        except Exception as e:
            rec['error'] = str(e)

        # plots
        try:
            from ..gmx.analysis.auto_plot import plot_rg
            plots_dir = outdir / 'plots'
            plots_dir.mkdir(parents=True, exist_ok=True)
            created = plot_rg(out, out_dir=plots_dir, group=str(rec.get('group', 'polymer')))
            if created:
                rec.setdefault('plots', {}).update(created)
        except Exception:
            pass

        # write rg.json
        (outdir / 'rg.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        # update summary
        try:
            self._update_summary_sections(rg=rec)
        except Exception:
            pass
        if "mean_nm" in rec:
            self._stat("Rg_mean", f"{float(rec['mean_nm']):.4f} nm")
        self._section_done("Rg analysis", t_all, detail=f"outputs={self._compact_path(outdir)}")
        return rec

    def sigma(self, *, temp_k: Optional[float] = None, msd: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute ionic conductivity."""
        t_all = self._section_begin("Conductivity analysis", detail="NE + EH conductivity")
        topo = parse_system_top(self.top)
        if msd is None:
            self._item("msd_source", "not provided -> recompute")
            msd = self.msd()
        else:
            self._item("msd_source", "caller-provided MSD results")

        if temp_k is None:
            temp_k = 300.0
            try:
                p = self._analysis_dir() / "thermo_summary.json"
                if p.exists():
                    thermo = json.loads(p.read_text(encoding="utf-8"))
                    if "Temperature" in thermo and isinstance(thermo["Temperature"], dict):
                        temp_k = float(thermo["Temperature"].get("mean", temp_k))
            except Exception:
                temp_k = 300.0

        self._item("temperature_K", f"{float(temp_k):.2f}")
        t0 = self._step_begin("Step 1/2 Nernst-Einstein conductivity", detail=f"temperature={float(temp_k):.2f} K")
        vol_nm3 = None
        try:
            p = self._analysis_dir() / "thermo_summary.json"
            if p.exists():
                thermo = json.loads(p.read_text(encoding="utf-8"))
                if "Volume" in thermo and isinstance(thermo["Volume"], dict):
                    vol_nm3 = float(thermo["Volume"].get("mean"))
        except Exception:
            vol_nm3 = None
        if vol_nm3 is None:
            vol_nm3 = _read_box_volume_nm3(self._system_dir() / "system.gro")
        e_c = 1.602176634e-19
        k_b = 1.380649e-23
        vol_m3 = vol_nm3 * 1e-27

        ne_components = []
        ignored_components = []
        for moltype, count in topo.molecules:
            mt = topo.moleculetypes.get(moltype)
            if mt is None:
                continue
            q_e = float(np.sum(mt.charges))
            if abs(q_e) < 1e-12:
                continue
            rec = msd.get(str(moltype), {}) if isinstance(msd, dict) else {}
            D_m2_s = rec.get("D_m2_s") if isinstance(rec, dict) else None
            if D_m2_s is None:
                continue
            term = (float(count) * (q_e * e_c) ** 2 * float(D_m2_s)) / (max(float(vol_m3), 1e-300) * k_b * float(temp_k))
            comp = {
                "moltype": str(moltype),
                "count": int(count),
                "charge_e": float(q_e),
                "D_m2_s": float(D_m2_s),
                "sigma_component_S_m": float(term),
            }
            if abs(q_e) >= 5.0 and int(getattr(mt, 'natoms', 0) or 0) >= 30:
                comp["ignored_as_polyionic_macromolecule"] = True
                comp["reason"] = "large net charge on a large molecule; excluded from NE by default"
                ignored_components.append(comp)
            else:
                ne_components.append(comp)
        sigma_ne = float(sum(c["sigma_component_S_m"] for c in ne_components))
        ne_out = {
            "sigma_S_m": sigma_ne,
            "temperature_K": float(temp_k),
            "volume_nm3": float(vol_nm3),
            "components": ne_components,
            "ignored_components": ignored_components,
        }
        self._step_done("Step 1/2 Nernst-Einstein conductivity", t0, detail=f"sigma_NE={sigma_ne:.3e} S/m | active_components={len(ne_components)}")
        self._stat("sigma_NE", f"{sigma_ne:.3e} S/m")

        t0 = self._step_begin("Step 2/2 Einstein-Helfand conductivity", detail="gmx current -dsp (velocity trajectory required)")
        eh_out: Dict[str, Any] = {"sigma_S_m": None, "reason": None}
        try:
            def _ndx_groups(ndx_path: Path) -> set[str]:
                out = set()
                try:
                    for raw in Path(ndx_path).read_text(encoding='utf-8', errors='replace').splitlines():
                        s = raw.strip()
                        if s.startswith('[') and s.endswith(']'):
                            out.add(s.strip('[]').strip())
                except Exception:
                    return set()
                return out

            groups = _ndx_groups(self.ndx)
            if not groups:
                try:
                    from ..gmx.index import generate_system_ndx
                    generate_system_ndx(top_path=self.top, ndx_path=self.ndx)
                    groups = _ndx_groups(self.ndx)
                except Exception:
                    pass
            if "IONS" not in groups:
                raise KeyError("No IONS group in system.ndx; cannot run EH conductivity.")
            group = "IONS"
            trr = self.trr
            if trr is None:
                cand = self.tpr.parent / "md.trr"
                trr = cand if cand.exists() else None
            outdir = self._analysis_dir() / "sigma"
            outdir.mkdir(parents=True, exist_ok=True)
            main_xvg = outdir / f"current_{group}.xvg"
            dsp_xvg = outdir / f"current_dsp_{group}.xvg"
            if trr is None or (not Path(trr).exists()):
                raise FileNotFoundError(
                    "No .trr trajectory found for EH conductivity. "
                    "EH requires velocities; enable writing TRR (nstxout/nstvout) in mdp or analyze a stage that produces md.trr."
                )
            proc = GromacsRunner().current(
                tpr=self.tpr,
                traj=Path(trr),
                ndx=self.ndx,
                group=group,
                out_xvg=main_xvg,
                out_dsp=dsp_xvg,
                temp_k=float(temp_k),
                cwd=outdir,
            )
            fit: EHFit = conductivity_from_current_dsp(dsp_xvg)
            eh_out = {
                "sigma_S_m": float(fit.sigma_S_m),
                "window_start_ps": float(fit.window_start_ps),
                "window_end_ps": float(fit.window_end_ps),
                "r2": float(fit.r2),
                "note": str(fit.note),
                "group": str(group),
                "reason": None,
                "method": "gmx current -dsp",
                "dsp_xvg": str(dsp_xvg),
            }
            try:
                eh_svg = plot_eh_fit_svg(dsp_xvg, fit, out_svg=outdir / f"eh_fit_{group}.svg", title=f"EH ({group})")
                eh_out["eh_fit_svg"] = str(eh_svg)
            except Exception as _pe:
                eh_out["eh_fit_plot_warning"] = str(_pe)
            try:
                if proc is not None:
                    eh_out["gmx_current_stdout_tail"] = proc.stdout.decode("utf-8", errors="replace").splitlines()[-20:]
            except Exception:
                pass
            self._step_done("Step 2/2 Einstein-Helfand conductivity", t0, detail=f"sigma_EH={float(fit.sigma_S_m):.3e} S/m")
        except Exception as e:
            eh_out["reason"] = str(e)
            self._step_warn("Step 2/2 Einstein-Helfand conductivity", detail=str(e))

        out = {"ne": ne_out, "eh": eh_out}
        summary_path = self._analysis_dir() / "summary.json"
        summary_all: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary_all = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_all = {}
        summary_all["sigma"] = out
        summary_path.write_text(json.dumps(self._prune_raw_paths(summary_all), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (self._analysis_dir() / "sigma.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        eh_detail = eh_out.get("sigma_S_m")
        eh_str = f"{float(eh_detail):.3e} S/m" if eh_detail is not None else "unavailable"
        self._section_done("Conductivity analysis", t_all, detail=f"sigma_NE={sigma_ne:.3e} S/m | sigma_EH={eh_str}")
        return out

    def density_distribution(
        self,
        mols: Optional[Sequence[object]] = None,
        *,
        axes: Sequence[str] = ("X", "Y", "Z"),
        begin_ps: Optional[float] = None,
        end_ps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Number density profile (gmx density -d number) for moltypes."""
        t_all = self._section_begin("Number-density distribution analysis", detail=f"axes={','.join(map(str, axes))}")
        topo = parse_system_top(self.top)
        runner = GromacsRunner()
        outdir = self._analysis_dir() / "number_density_distribution"
        outdir.mkdir(parents=True, exist_ok=True)
        plots_dir = outdir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        want_moltypes: Optional[set[str]] = None
        if mols is not None:
            want_moltypes = set()
            try:
                from rdkit import Chem
                sys_dir = self._system_dir()
                meta_path = sys_dir / "system_meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                sp_list = meta.get("species", []) or []
                def _mol_to_smi(m):
                    try:
                        return m.GetProp('_yadonpy_smiles') if m.HasProp('_yadonpy_smiles') else Chem.MolToSmiles(m, isomericSmiles=True)
                    except Exception:
                        return ""
                for m in mols:
                    smi = _mol_to_smi(m)
                    for sp in sp_list:
                        if str(sp.get("smiles") or "") == smi:
                            mt = sp.get("moltype") or sp.get("mol_name") or sp.get("mol_id")
                            if mt:
                                want_moltypes.add(str(mt))
                            break
            except Exception:
                want_moltypes = None

        selected = [str(moltype) for moltype, _count in topo.molecules if want_moltypes is None or str(moltype) in want_moltypes]
        total = len(selected) * len(tuple(axes))
        self._item("selected_moltypes", ', '.join(selected) if selected else 'none')
        self._item("density_tasks", total)
        res: Dict[str, Any] = {}
        idx = 0
        for moltype, _count in topo.molecules:
            if want_moltypes is not None and str(moltype) not in want_moltypes:
                continue
            for ax in axes:
                idx += 1
                title = self._progress_title("Number-density", idx, total)
                t0 = self._step_begin(title, detail=f"group=REP_{moltype} | axis={ax.upper()}")
                xvg = outdir / f"ndens_{moltype}_{ax.upper()}.xvg"
                runner.density_number_profile(
                    tpr=self.tpr,
                    xtc=self.xtc,
                    ndx=self.ndx,
                    group=f"REP_{moltype}",
                    out_xvg=xvg,
                    axis=ax.upper(),
                    begin_ps=begin_ps,
                    end_ps=end_ps,
                    cwd=outdir,
                )
                res.setdefault(str(moltype), {})[ax.upper()] = str(xvg)
                try:
                    from ..gmx.analysis.plot import plot_xvg_svg
                    svg = plot_xvg_svg(xvg, out_svg=plots_dir / f"ndens_{moltype}_{ax.upper()}.svg", title=f"ndens {moltype} {ax.upper()}")
                    res.setdefault(str(moltype), {}).setdefault("plots", {})[ax.upper()] = str(svg)
                except Exception:
                    pass
                self._step_done(title, t0, detail=f"output={xvg.name}")

        summary_path = self._analysis_dir() / "summary.json"
        summary_all: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary_all = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_all = {}
        summary_all["number_density_profile"] = res
        summary_path.write_text(json.dumps(self._prune_raw_paths(summary_all), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (self._analysis_dir() / "number_density_profile.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._item("density_outputs", self._compact_path(outdir))
        self._section_done("Number-density distribution analysis", t_all, detail=f"outputs={self._compact_path(outdir)}")
        return res

    def den_dis(
        self,
        mols: Optional[Sequence[object]] = None,
        *,
        axes: Sequence[str] = ("X", "Y", "Z"),
        begin_ps: Optional[float] = None,
        end_ps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Alias for :meth:`density_distribution` (yzc-gmx-gen style).

        Examples
        --------
        density_distributionr = analy.den_dis()  # all moltypes
        density_distributionr = analy.den_dis([poly, Li])
        """
        return self.density_distribution(mols, axes=axes, begin_ps=begin_ps, end_ps=end_ps)

def _read_box_volume_nm3(gro_path: Path) -> float:
    """Read box volume (nm^3) from the last line of a .gro file.

    Supports both orthorhombic (3 floats) and triclinic (9 floats) formats.
    For triclinic boxes, we compute det([v1, v2, v3]).
    """
    lines = gro_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return 0.0
    parts = lines[-1].split()
    if len(parts) < 3:
        return 0.0

    vals = [float(x) for x in parts]
    if len(vals) >= 9:
        # GROMACS .gro triclinic box: v1x v2y v3z v1y v1z v2x v2z v3x v3y
        v1 = np.array([vals[0], vals[3], vals[4]], dtype=float)
        v2 = np.array([vals[5], vals[1], vals[6]], dtype=float)
        v3 = np.array([vals[7], vals[8], vals[2]], dtype=float)
        vol = float(abs(np.linalg.det(np.vstack([v1, v2, v3]))))
        return vol

    # Orthorhombic or unknown (use first 3 as lengths)
    lx, ly, lz = float(vals[0]), float(vals[1]), float(vals[2])
    return float(lx * ly * lz)


def _read_box_dims_nm(gro_path: Path) -> tuple[float, float, float]:
    """Read box lengths (nm) from a .gro file.

    For triclinic boxes, returns the norms of the three box vectors.
    """
    lines = gro_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return 0.0, 0.0, 0.0
    parts = lines[-1].split()
    if len(parts) < 3:
        return 0.0, 0.0, 0.0
    vals = [float(x) for x in parts]
    if len(vals) >= 9:
        v1 = np.array([vals[0], vals[3], vals[4]], dtype=float)
        v2 = np.array([vals[5], vals[1], vals[6]], dtype=float)
        v3 = np.array([vals[7], vals[8], vals[2]], dtype=float)
        return float(np.linalg.norm(v1)), float(np.linalg.norm(v2)), float(np.linalg.norm(v3))
    return float(vals[0]), float(vals[1]), float(vals[2])
