"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..gmx.engine import GromacsRunner
from ..gmx.analysis.xvg import read_xvg
from ..gmx.analysis.thermo import stats_from_xvg
from ..core import utils
from ..gmx.analysis.conductivity import conductivity_from_current_dsp, EHFit, plot_eh_fit_svg, parse_gmx_current_sigmas
from ..gmx.analysis.auto_plot import plot_msd, plot_msd_overlay, plot_rdf_cn
from ..gmx.analysis.rg_convergence import find_rg_convergence, plot_rg_convergence_svg
from ..gmx.topology import parse_system_top


def _integrate_cn(r: np.ndarray, g: np.ndarray, rho_nm3: float) -> np.ndarray:
    """Compute running coordination number from g(r)."""
    # CN(r) = 4*pi*rho * integral_0^r g(r')*r'^2 dr'
    dr = np.gradient(r)
    cn = 4.0 * np.pi * rho_nm3 * np.cumsum(g * r * r * dr)
    return cn


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
        """Basic thermo properties from edr (mean/std over last frac)."""
        runner = GromacsRunner()
        out = self._analysis_dir() / "thermo.xvg"
        terms = ["Temperature", "Pressure", "Density", "Volume", "Total Energy", "Enthalpy"]
        # filter by availability
        mapping = runner.list_energy_terms(edr=self.edr)
        # GROMACS term names can include units or minor variations across versions.
        # Prefer substring match and then pass the *actual* term names back to `gmx energy`.
        terms_av: list[str] = []
        for want in terms:
            for have in mapping.keys():
                if want.lower() in have.lower():
                    terms_av.append(have)
                    break
        if not terms_av:
            return {}
        runner.energy_xvg(edr=self.edr, out_xvg=out, terms=terms_av)
        df = read_xvg(out).df
        res: Dict[str, Any] = {}
        for col in df.columns:
            if col == "x":
                continue
            s = stats_from_xvg(out, col=col, frac_last=self.frac_last)
            # Store as plain dict for JSON safety and stable public API
            res[col] = {"mean": float(s.mean), "std": float(s.std), "n": int(s.n)}
        if save:
            p = self._analysis_dir() / "thermo_summary.json"
            p.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return res

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
        """Pick a sensible group for Rg check.

        Heuristic:
        1) Any group containing 'POLY' (case-insensitive)
        2) Any group containing 'POLYMER'
        3) Fallback to 'System' (0)
        """
        names = self._ndx_group_order()
        for i, n in enumerate(names):
            if "poly" in n.lower():
                return i
        return 0

    def _has_polymer_group(self) -> bool:
        """Heuristic: treat as polymer system if index file contains a 'poly*' group."""
        for n in self._ndx_group_order():
            if "poly" in n.lower():
                return True
        return False



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

    def _auto_msd_trestart_ps(self, *, default_ps: float = 20.0) -> float:
        """Choose a safe -trestart for `gmx msd`.

        `gmx msd` requires dt <= trestart, where dt is effectively the trajectory frame interval.
        We parse dt and nstxout-compressed from the run's md.mdp when available and pick:
            trestart = max(default_ps, 10 * frame_interval_ps)

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
            frame_interval_ps = dt * float(nst)
            return float(max(default_ps, 10.0 * frame_interval_ps))
        return float(default_ps)
    def _rg_series(self) -> Optional[dict[str, Any]]:
        """Compute radius of gyration time series via `gmx gyrate` (best-effort).

        Returns a dict with:
          - t_ps: (N,) time in ps
          - rg_nm: (N,) Rg in nm
          - rg_components_nm: (N,3) optional components (RgX,RgY,RgZ)
        """
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
        """RadonPy-style equilibrium check using thermo fluctuations.

        This follows the spirit of RadonPy's `Equilibration_analyze.check_eq`:
        a property is considered converged when its fluctuation (here: std of the
        last `frac_last` segment) is sufficiently small.

        Notes:
        - GROMACS doesn't expose the same detailed per-term energy decomposition as
          RadonPy/LAMMPS presets in every workflow; we therefore focus on the most
          robust, always-available terms.
        """
        try:
            thermo = self.get_all_prop(temp=0, press=0, save=False)
        except Exception:
            return False

        # RadonPy default criteria (approx.):
        #   totene_sma_sd_crit = 0.0005, dens_sma_sd_crit = 0.001
        # We approximate sma_sd by std over the last window.
        # NOTE: term names vary across GROMACS versions (units, spacing, etc.).
        # We therefore match by substring against keys present in `thermo`.
        rel_crit = {
            "Total Energy": 0.0005,
            "Kinetic En.": 0.0005,
            "Potential": 0.0005,
            "Density": 0.001,
            # Temperature is not explicitly checked in RadonPy (kinetic energy is),
            # but it's a stable and useful proxy in GROMACS.
            "Temperature": 0.001,
        }

        def _find_key(want: str) -> Optional[str]:
            w = want.lower()
            for k in thermo.keys():
                if w in str(k).lower():
                    return str(k)
            return None

        ok = True
        resolved: dict[str, str] = {}
        for term in rel_crit.keys():
            k = _find_key(term)
            if k is not None:
                resolved[term] = k

        for term, crit in rel_crit.items():
            k = resolved.get(term)
            if not k:
                continue
            mean = float(thermo[k].get("mean", 0.0))
            std = float(thermo[k].get("std", 0.0))
            if mean == 0.0:
                continue
            if std > abs(mean) * float(crit):
                utils.radon_print(
                    f"{term} does not converge. mean={mean:.6g}, std={std:.6g}, crit={crit}",
                    level=2,
                )
                ok = False

        # Polymer systems: always apply an Rg gate (yzc-gmx-gen criterion),
        # regardless of whether Density is available.
        rg_gate = None
        if self._has_polymer_group():
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
                }
                if not res_rg.ok:
                    utils.radon_print(
                        f"Rg gate does not converge (by={res_rg.converged_by}). mean={res_rg.mean:.6g} nm, std={res_rg.std:.6g} nm, slope={res_rg.slope:.3g} /ps",
                        level=2,
                    )
                    ok = False
        # Dedicated marker for resumable workflows (best-effort)
        try:
            payload = {"ok": bool(ok)}
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
            else:
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
        """Compute RDF between center group and each species' atomtypes.

        Usage
        -----
        Preferred (RadonPy-style, linear scripts):
            rdf = analy.rdf(center_molecule)

        Backward compatible form:
            rdf = analy.rdf(species_mols, center_mol=center_molecule)

        Notes
        -----
        - Center group is found by SMILES matching (via _yadonpy_smiles or RDKit MolToSmiles)
        - For each moltype in system.ndx, we compute RDF vs each TYPE_<moltype>_<atomtype>
        """
        # Backward compatibility: previous signature was rdf(species_mols, center_mol=...)
        if center_mol is None:
            center_mol = mol_or_mols
        # resolve center moltype by smiles
        try:
            from rdkit import Chem

            smi = center_mol.GetProp('_yadonpy_smiles') if center_mol.HasProp('_yadonpy_smiles') else Chem.MolToSmiles(center_mol, isomericSmiles=True)
        except Exception:
            smi = ""

        topo = parse_system_top(self.top)

        # Resolve center group by SMILES -> moltype mapping written at system export time.
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
            # Fallback: ionic group by net formal charge
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

        # density of targets: approximate from counts / box volume
        # We'll compute rho for each moltype from system.top and box length from gro.
        gro_path = sys_dir / "system.gro"
        vol_nm3 = _read_box_volume_nm3(gro_path) if gro_path.exists() else None

        out_summary: Dict[str, Any] = {}
        for moltype, count in topo.molecules:
            mt = topo.moleculetypes.get(moltype)
            if mt is None:
                continue
            # build rho for this moltype in nm^-3 based on molecule count
            rho = None
            if vol_nm3 and vol_nm3 > 0:
                rho = float(count) / vol_nm3

            # for each atomtype group
            for atype in sorted(set(mt.atomtypes)):
                # hydrogens skipped unless include_h
                if not include_h:
                    # if all atoms with this type are hydrogens, skip
                    all_h = True
                    for an, at in zip(mt.atomnames, mt.atomtypes):
                        if at == atype and not (an.upper().startswith('H') or at.lower().startswith('h')):
                            all_h = False
                            break
                    if all_h:
                        continue
                target_group = f"TYPE_{moltype}_{atype}"
                tag = f"{moltype}:{atype}"
                rdf_xvg = analysis_dir / f"rdf_{moltype}_{atype}.xvg"
                cn_xvg = analysis_dir / f"cn_{moltype}_{atype}.xvg"
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

                # --- auto plot (RDF + CN)
                try:
                    plots_dir = analysis_dir / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    svg = plot_rdf_cn(
                        rdf_xvg=rdf_xvg,
                        cn_xvg=cn_xvg if cn_xvg.exists() else None,
                        out_svg=plots_dir / f"rdf_cn_{moltype}_{atype}.svg",
                        title=f"RDF/CN: {center_group} vs {target_group}",
                    )
                except Exception:
                    svg = None
                df = read_xvg(rdf_xvg).df
                r = df["x"].to_numpy(dtype=float)
                # rdf output might have single y column named 'y0' or similar
                ycols = [c for c in df.columns if c != "x"]
                g = df[ycols[0]].to_numpy(dtype=float) if ycols else np.zeros_like(r)
                # compute CN ourselves for robustness
                cn = _integrate_cn(r, g, rho_nm3=float(rho or 0.0)) if rho else np.zeros_like(r)
                shell = _first_shell_from_gr(r, g, cn)
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

        # write summary.json
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

        # Dedicated marker/output for resumable workflows
        (self._analysis_dir() / "rdf_first_shell.json").write_text(
            json.dumps(out_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return out_summary

    def msd(
        self,
        mols: Optional[Sequence[object]] = None,
        *,
        begin_ps: Optional[float] = None,
        end_ps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute MSD for each moltype group and estimate diffusion coefficient.

        Notes
        - Uses `gmx msd` on each moltype group.
        - Diffusion is estimated from a linear fit on the last `frac_last` portion:
          MSD(t) ≈ a t + b, D = a / 6  (3D diffusion).
        """
        # `mols` is kept only for backward compatibility with earlier APIs.
        topo = parse_system_top(self.top)
        runner = GromacsRunner()
        outdir = self._analysis_dir() / "msd"
        outdir.mkdir(parents=True, exist_ok=True)

        trestart_ps = self._auto_msd_trestart_ps(default_ps=20.0)
        utils.yadon_print(f"YadonPy info: MSD trestart(auto) = {trestart_ps:.3f} ps", level=1)

        res: Dict[str, Any] = {}
        msd_xvgs: Dict[str, Path] = {}
        for moltype, _count in topo.molecules:
            xvg = outdir / f"msd_{moltype}.xvg"
            runner.msd(
                tpr=self.tpr,
                xtc=self.xtc,
                ndx=self.ndx,
                group=moltype,
                out_xvg=xvg,
                begin_ps=begin_ps,
                end_ps=end_ps,
                trestart_ps=trestart_ps,
                rmcomm=True,
                cwd=outdir,
            )
            df = read_xvg(xvg).df
            t = df["x"].to_numpy(dtype=float)
            ycols = [c for c in df.columns if c != "x"]
            msd_arr = df[ycols[0]].to_numpy(dtype=float) if ycols else np.zeros_like(t)

            if len(t) < 5:
                continue
            n0 = int(len(t) * (1 - self.frac_last))
            t_fit = t[n0:]
            msd_fit = msd_arr[n0:]

            a, b = np.polyfit(t_fit, msd_fit, 1)
            D = float(a) / 6.0
            rec = {"D_nm2_ps": D, "fit_slope": float(a), "fit_intercept": float(b), "xvg": str(xvg),
                   "fit_t_start_ps": float(t_fit[0]), "fit_t_end_ps": float(t_fit[-1])}

            # --- auto plots (yzc-gmx-gen style)
            try:
                plots_dir = outdir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                created = plot_msd(xvg, out_dir=plots_dir, group=str(moltype), fit_t_start_ps=float(t_fit[0]), fit_t_end_ps=float(t_fit[-1]))
                if created:
                    rec.setdefault("plots", {}).update(created)
            except Exception:
                pass

            res[moltype] = rec
            msd_xvgs[str(moltype)] = xvg

        # MSD overlay plots
        try:
            if msd_xvgs:
                plots_dir = outdir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                ov1 = plot_msd_overlay(msd_xvgs=msd_xvgs, out_svg=plots_dir / "msd_overlay.svg", title="MSD overlay", loglog=False)
                ov2 = plot_msd_overlay(msd_xvgs=msd_xvgs, out_svg=plots_dir / "msd_overlay_loglog.svg", title="MSD overlay (log-log)", loglog=True)
                if ov1 is not None or ov2 is not None:
                    res.setdefault("_overlay", {})
                    if ov1 is not None:
                        res["_overlay"]["msd_overlay_svg"] = str(ov1)
                    if ov2 is not None:
                        res["_overlay"]["msd_overlay_loglog_svg"] = str(ov2)
        except Exception:
            pass

        summary_path = self._analysis_dir() / "summary.json"
        summary_all: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary_all = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_all = {}
        summary_all.setdefault("msd", {})
        summary_all["msd"].update(res)
        summary_path.write_text(
            json.dumps(self._prune_raw_paths(summary_all), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Dedicated marker/output for resumable workflows
        (self._analysis_dir() / "msd.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return res

    def rg(self, *, begin_ps: Optional[float] = None, end_ps: Optional[float] = None) -> Dict[str, Any]:
        """Compute and plot radius of gyration time series (best-effort)."""
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
            summary_path = self._analysis_dir() / 'summary.json'
            summary_all: Dict[str, Any] = {}
            if summary_path.exists():
                summary_all = json.loads(summary_path.read_text(encoding='utf-8'))
            summary_all['rg'] = rec
            summary_path.write_text(
                json.dumps(self._prune_raw_paths(summary_all), indent=2, ensure_ascii=False) + '\n',
                encoding='utf-8',
            )
        except Exception:
            pass
        return rec

    def sigma(self, *, temp_k: Optional[float] = None, msd: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute ionic conductivity.

        Returns both:
          - NE: Nernst–Einstein conductivity from MSD-derived diffusion
          - EH: Einstein–Helfand conductivity from `gmx current -dsp` (linear fit)

        EH requires a trajectory with velocities (typically `.trr`).
        """
        topo = parse_system_top(self.top)
        if msd is None:
            msd = self.msd()

        if temp_k is None:
            # Best-effort: read mean temperature from previous thermo summary if available.
            temp_k = 300.0
            try:
                p = self._analysis_dir() / "thermo_summary.json"
                if p.exists():
                    thermo = json.loads(p.read_text(encoding="utf-8"))
                    if "Temperature" in thermo and isinstance(thermo["Temperature"], dict):
                        temp_k = float(thermo["Temperature"].get("mean", temp_k))
            except Exception:
                temp_k = 300.0

        # ---------- NE (diffusion-based) ----------
        # Prefer mean volume from thermo if present (more accurate than initial box).
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

        ne_sigma_s_m = 0.0
        ne_parts: Dict[str, Any] = {}
        for moltype, count in topo.molecules:
            mt = topo.moleculetypes.get(moltype)
            if mt is None:
                continue
            q_e = mt.net_charge
            if abs(q_e) < 1e-8:
                continue
            D_nm2_ps = float(msd.get(moltype, {}).get("D_nm2_ps", 0.0))
            D_m2_s = D_nm2_ps * 1e-6  # 1 nm^2/ps = 1e-6 m^2/s
            n_number = float(count) / vol_m3  # 1/m^3
            sigma_i = n_number * (q_e * e_c) ** 2 * D_m2_s / (k_b * float(temp_k))
            ne_sigma_s_m += sigma_i
            ne_parts[moltype] = {
                "count": int(count),
                "q_e": float(q_e),
                "D_nm2_ps": D_nm2_ps,
                "sigma_S_m": float(sigma_i),
            }

        ne_out = {"sigma_S_m": float(ne_sigma_s_m), "components": ne_parts}

        # If the system has no ionic species, do not report conductivity in summary.
        has_ions = bool(ne_parts)
        if not has_ions:
            out = {"note": "no ionic species (net charge == 0 for all molecule types)"}
            # Write a dedicated marker for reproducibility, but keep summary.json clean.
            (self._analysis_dir() / "sigma.json").write_text(
                json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            return out

        # ---------- EH (gmx current -dsp) ----------
        eh_out: Dict[str, Any] = {
            "sigma_S_m": None,
            "window_start_ps": None,
            "window_end_ps": None,
            "r2": None,
            "note": None,
            "group": None,
            "reason": None,
        }
        try:
            trr = self.trr
            if trr is None:
                cand = self.tpr.parent / "md.trr"
                trr = cand if cand.exists() else None
            if trr is None or (not Path(trr).exists()):
                raise FileNotFoundError("No .trr trajectory found (EH requires velocities).")

            def _ndx_groups(ndx_path: Path) -> set[str]:
                names: set[str] = set()
                for raw in ndx_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    s = raw.strip()
                    if s.startswith("[") and s.endswith("]"):
                        names.add(s.strip("[]").strip())
                return names

            groups = _ndx_groups(self.ndx)
            group = "IONS" if "IONS" in groups else ("CATIONS" if "CATIONS" in groups else ("ANIONS" if "ANIONS" in groups else None))
            if group is None:
                raise KeyError("No IONS/CATIONS/ANIONS group in system.ndx; cannot run EH conductivity.")

            runner = GromacsRunner()
            outdir = self._analysis_dir() / "conductivity"
            outdir.mkdir(parents=True, exist_ok=True)
            main_xvg = outdir / f"current_{group}.xvg"
            dsp_xvg = outdir / f"current_dsp_{group}.xvg"

            proc = runner.current(
                tpr=self.tpr,
                traj=Path(trr),
                ndx=self.ndx,
                group=group,
                out_xvg=main_xvg,
                out_dsp=dsp_xvg,
                temp_k=float(temp_k),
                cwd=outdir,
            )
            try:
                fit: EHFit = conductivity_from_current_dsp(dsp_xvg)
            except Exception as _ehfit_err:
                # Fallback: some GROMACS versions may not write a dense -dsp file (or trajectory too short),
                # but still print conductivity to stdout. Try parsing stdout.
                parsed = parse_gmx_current_sigmas(proc.stdout.decode('utf-8', errors='replace'))
                sigma = parsed.get('eh_sigma_S_m')
                if sigma is None:
                    raise
                fit = EHFit(
                    sigma_S_m=float(sigma),
                    window_start_ps=float(parsed.get('fit_start_ps') or 0.0),
                    window_end_ps=float(parsed.get('fit_end_ps') or 0.0),
                    slope_per_ps=float('nan'),
                    intercept=float('nan'),
                    r2=float('nan'),
                    note=f"stdout-parse fallback: {_ehfit_err}",
                )
            try:
                eh_svg = plot_eh_fit_svg(dsp_xvg, fit, out_svg=outdir / f"eh_fit_{group}.svg", title=f"EH ({group})")
                eh_out["eh_fit_svg"] = str(eh_svg)
            except Exception as _pe:
                eh_out["eh_fit_plot_warning"] = str(_pe)

            eh_out = {
                "sigma_S_m": float(fit.sigma_S_m),
                "window_start_ps": float(fit.window_start_ps),
                "window_end_ps": float(fit.window_end_ps),
                "r2": float(fit.r2),
                "note": str(fit.note),
                "group": group,
                "reason": None,
            }
            try:
                eh_out["gmx_current_stdout_tail"] = proc.stdout.decode("utf-8", errors="replace").splitlines()[-20:]
            except Exception:
                pass
        except Exception as e:
            eh_out["reason"] = str(e)

        out: Dict[str, Any] = {"ne": ne_out}
        # Only include EH block if we attempted it (i.e., system has ions).
        out["eh"] = eh_out

        summary_path = self._analysis_dir() / "summary.json"
        summary_all: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary_all = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_all = {}
        summary_all["sigma"] = out
        summary_path.write_text(
            json.dumps(self._prune_raw_paths(summary_all), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Dedicated marker/output for resumable workflows
        (self._analysis_dir() / "sigma.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return out

    def density_distribution(
        self,
        mols: Sequence[object],
        *,
        axes: Sequence[str] = ("X", "Y", "Z"),
        begin_ps: Optional[float] = None,
        end_ps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Number density profile for each moltype in each direction."""
        topo = parse_system_top(self.top)
        runner = GromacsRunner()
        outdir = self._analysis_dir() / "number_density_distribution"
        outdir.mkdir(parents=True, exist_ok=True)
        plots_dir = outdir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        res: Dict[str, Any] = {}
        for moltype, _count in topo.molecules:
            for ax in axes:
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
                res.setdefault(moltype, {})[ax.upper()] = str(xvg)

                # --- auto plot
                try:
                    from ..gmx.analysis.plot import plot_xvg_svg

                    svg = plot_xvg_svg(xvg, out_svg=plots_dir / f"ndens_{moltype}_{ax.upper()}.svg", title=f"ndens {moltype} {ax.upper()}")
                    res.setdefault(moltype, {}).setdefault("plots", {})[ax.upper()] = str(svg)
                except Exception:
                    pass

        summary_path = self._analysis_dir() / "summary.json"
        summary_all: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary_all = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_all = {}
        summary_all["number_density_profile"] = res
        summary_path.write_text(
            json.dumps(self._prune_raw_paths(summary_all), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # Dedicated marker/output for resumable workflows
        (self._analysis_dir() / "number_density_profile.json").write_text(
            json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return res

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
