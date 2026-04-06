"""High-level workflow steps.

This module exists to keep user scripts clean.

Design principle
----------------
Each step function accepts a ``restart`` boolean. When ``restart=True``,
YadonPy will try to load/skip work that already exists in ``work_dir``.
When ``restart=False``, YadonPy will re-run the step (even if outputs exist).

All steps are implemented on top of :class:`yadonpy.workflow.Restart`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union, Literal

from rdkit import Chem
from rdkit import Geometry as Geom

from ..core import poly, utils
from ..sim import qm
from ..sim.analyzer import AnalyzeResult
from ..sim.preset import eq
from ..gmx.workflows._util import RunResources
from ..gmx.workflows.quick import QuickRelaxJob
from ..gmx.workflows.tg import TgJob
from ..gmx.workflows.elongation import ElongationJob
from .restart import Restart
from ..runtime import resolve_restart
from ..core.logging_utils import compact_path, format_elapsed


def _as_path(p: Union[str, Path]) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _stage_output_path(wd: Path, out_name: str, stage_dir: str) -> Path:
    """Resolve a default stage subdirectory for outputs.

    If `out_name` is just a file name (no parent components), place it under
    `wd/stage_dir/out_name`. If `out_name` already contains a parent path
    (e.g. "00_system/system.gro"), keep it relative to `wd` as-is.
    """
    p = Path(out_name)
    if p.is_absolute():
        return p
    if p.parent == Path('.'):
        return wd / stage_dir / p.name
    return wd / p


def _read_sdf_one(path: Path) -> Chem.Mol:
    sup = Chem.SDMolSupplier(str(path), removeHs=False)
    if not sup or sup[0] is None:
        raise ValueError(f"Cannot read molecule from SDF: {path}")
    return sup[0]


def _write_sdf_one(mol: Chem.Mol, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()


def resp_from_smiles(
    smiles: str,
    *,
    ff_obj: Any,
    work_dir: Union[str, Path],
    log_name: str,
    set_resname: Optional[str] = None,
    restart: Optional[bool] = None,
    charge: str = "RESP",
    nconf: int = 1000,
    dft_nconf: int = 4,
    psi4_omp: int = 16,
    memory: int = 2000,
) -> Chem.Mol:
    """Best-effort conformer search + RESP charge assignment.

    Cached outputs
    --------------
    - work_dir/qm_cache/{log_name}.opt.sdf
    - work_dir/charged_mol2_qm/{log_name}.charges.json

    Notes
    -----
    - The returned molecule is reconstructed from SMILES to preserve '*' linkers
      (stored as isotope-labeled H atoms internally).
    - Charges are loaded from the cached JSON when possible.
    - For monoatomic ions (e.g. [Li+]) we skip QM and just assign the formal
      charge, then run FF typing. QM/RESP for a single atom is unnecessary and
      can be fragile across toolchains.
    """

    wd = _as_path(work_dir)
    rst = Restart(wd, restart=restart)

    mol2_dir = wd / "charged_mol2_qm"
    cache_dir = wd / "qm_cache"
    opt_sdf = cache_dir / f"{log_name}.opt.sdf"
    charges_json = mol2_dir / f"{log_name}.charges.json"

    def _formal_charge(m: Chem.Mol) -> int:
        try:
            return int(sum(int(a.GetFormalCharge()) for a in m.GetAtoms()))
        except Exception:
            return 0

    def _ensure_has_conformer(m: Chem.Mol) -> None:
        try:
            if int(m.GetNumConformers()) > 0:
                return
        except Exception:
            return
        conf = Chem.Conformer(int(m.GetNumAtoms()))
        for i in range(int(m.GetNumAtoms())):
            conf.SetAtomPosition(i, Geom.Point3D(0.0, 0.0, 0.0))
        m.AddConformer(conf, assignId=True)

    def _load() -> Chem.Mol:
        # Prefer cached optimized geometry if available.
        if opt_sdf.exists():
            mol0 = _read_sdf_one(opt_sdf)
        else:
            mol0 = utils.mol_from_smiles(smiles)

        # Ensure linker information is present (rebuild from SMILES if missing).
        try:
            if "*" in smiles and all(a.GetIsotope() < 3 for a in mol0.GetAtoms() if a.GetSymbol() == "H"):
                mol0 = utils.mol_from_smiles(smiles)
        except Exception:
            pass

        # Load charges (best-effort).
        try:
            qm.load_atomic_charges_json(mol0, charges_json, strict=True)
        except Exception:
            pass

        # Always ensure FF typing exists for downstream steps, except for the
        # special hydrogen terminator fragment (no GAFF/GAFF2 parameters for H-H).
        if not qm.is_h_terminator_placeholder(mol0, smiles_hint=smiles):
            try:
                ok = bool(ff_obj.ff_assign(mol0))
                if not ok:
                    raise RuntimeError("ff_assign returned False")
            except Exception as e:
                raise RuntimeError(f"Force-field assignment failed in resp_from_smiles(load) for {log_name}: {e}") from e

        return mol0

    def _run() -> Chem.Mol:
        mol = utils.mol_from_smiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Monoatomic ion shortcut
        if int(mol.GetNumAtoms()) == 1 and _formal_charge(mol) != 0:
            _ensure_has_conformer(mol)
            q = float(_formal_charge(mol))
            a0 = mol.GetAtomWithIdx(0)
            a0.SetDoubleProp("AtomicCharge", q)
            a0.SetDoubleProp("AtomicCharge_raw", q)
            try:
                a0.SetDoubleProp("RESP", q)
                a0.SetDoubleProp("RESP_raw", q)
            except Exception:
                pass
            ok = bool(ff_obj.ff_assign(mol))
            if not ok:
                raise RuntimeError(f"Force-field assignment failed for monoatomic ion {smiles} (ff={ff_obj.__class__.__name__})")
            cache_dir.mkdir(parents=True, exist_ok=True)
            mol2_dir.mkdir(parents=True, exist_ok=True)
            _write_sdf_one(mol, opt_sdf)
            # Persist charges JSON for resumable workflow
            try:
                qm._save_atomic_charges_json(mol, charges_json, charge_label=str(charge), log_name=str(log_name))
            except Exception:
                # minimal JSON fallback
                import json
                charges_json.write_text(json.dumps({"meta": {"log_name": str(log_name), "charge": str(charge), "num_atoms": 1}, "charges": [q]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return mol

        # Hydrogen terminator shortcut: "[H][*]" / "[*][H]" becomes a 2-atom H/H fragment
        # (one H is isotope-labeled to represent the linker). This fragment is ONLY a linker token;
        # it does not need QM charges nor GAFF typing.
        if qm.is_h_terminator_placeholder(mol, smiles_hint=smiles):
            qm.apply_placeholder_zero_charges(mol, charge_label=str(charge))
            cache_dir.mkdir(parents=True, exist_ok=True)
            mol2_dir.mkdir(parents=True, exist_ok=True)
            _write_sdf_one(mol, opt_sdf)
            try:
                qm._save_atomic_charges_json(mol, charges_json, charge_label=str(charge), log_name=str(log_name))
            except Exception:
                import json
                charges_json.write_text(json.dumps({"meta": {"log_name": str(log_name), "charge": str(charge), "num_atoms": int(mol.GetNumAtoms())}, "charges": [0.0]*int(mol.GetNumAtoms())}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            return mol

        def _explicit_charge_in_smiles(s: str) -> bool:
            # Heuristic: explicit charge annotation in (p)SMILES.
            # We avoid treating linker '*' patterns as charged tokens.
            s = (s or '').strip()
            if not s:
                return False
            if '*' in s:
                # pSMILES contains linkers; still allow charge detection if present.
                pass
            return ('+' in s) or ('-' in s)

        is_charged = (_formal_charge(mol) != 0) and _explicit_charge_in_smiles(str(smiles))
        if is_charged:
            # OpenBabel+UFF relaxed initial geometry -> Psi4 OPT + RESP
            # If OpenBabel is unavailable, fall back to the standard conformer-search route.
            try:
                mol_ob = utils.deepcopy_mol(mol)
                mol_ob.RemoveAllConformers()
                ok3d = utils.ensure_3d_coords(mol_ob, smiles_hint=str(smiles), engine='openbabel')
            except Exception:
                ok3d = False
                mol_ob = mol

            if ok3d and mol_ob.GetNumConformers() > 0:
                # Charge assignment performs Psi4 geometry optimization when opt=True.
                cache_dir.mkdir(parents=True, exist_ok=True)
                mol2_dir.mkdir(parents=True, exist_ok=True)
                qm.assign_charges(
                    mol_ob,
                    charge=str(charge),
                    opt=True,
                    work_dir=str(wd),
                    omp=int(psi4_omp),
                    memory=int(memory),
                    log_name=str(log_name),
                )
                # Cache the post-QM geometry
                _write_sdf_one(mol_ob, opt_sdf)
                if not charges_json.exists():
                    raise RuntimeError(f"Charge assignment finished but charges JSON missing: {charges_json}")

                mol_out = utils.mol_from_smiles(smiles)
                if mol_out is None:
                    raise ValueError(f"Invalid SMILES: {smiles}")
                ok = qm.load_atomic_charges_json(mol_out, charges_json, strict=True)
                if not ok:
                    mol_out = mol_ob

                try:
                    ok_ff = bool(ff_obj.ff_assign(mol_out))
                    if not ok_ff:
                        raise RuntimeError("ff_assign returned False")
                except Exception:
                    mol_out = mol_ob
                    ok_ff = bool(ff_obj.ff_assign(mol_out))
                    if not ok_ff:
                        raise RuntimeError(f"Force-field assignment failed for {log_name} (ff={ff_obj.__class__.__name__})")

                # IMPORTANT: if a (m)Seminario bonded-params patch was generated during QM,
                # re-apply it after ff_assign so high-symmetry ions keep sufficient rigidity.
                try:
                    import json as _json
                    from pathlib import Path as _Path
                    from ..sim.qm import apply_mseminario_params_to_mol as _apply_mseminario

                    if mol_out.HasProp('_yadonpy_mseminario_json'):
                        jp = str(mol_out.GetProp('_yadonpy_mseminario_json')).strip()
                        if jp and _Path(jp).exists():
                            params = _json.loads(_Path(jp).read_text(encoding='utf-8'))
                            if isinstance(params, dict):
                                _apply_mseminario(mol_out, params, overwrite=True)
                except Exception:
                    pass

                return mol_out

            utils.radon_print(
                f"OpenBabel 3D build not available for charged SMILES; falling back to conformer search. purpose={log_name}",
                level=2,
            )

        # Standard route: MM conformer search + (optional) DFT, then RESP
        mol_opt, _ = qm.conformation_search(
            mol,
            ff=ff_obj,
            nconf=int(nconf),
            dft_nconf=int(dft_nconf),
            work_dir=str(wd),
            psi4_omp=int(psi4_omp),
            memory=int(memory),
            log_name=str(log_name),
        )
        # Cache optimized geometry (SDF) for later reuse.
        cache_dir.mkdir(parents=True, exist_ok=True)
        _write_sdf_one(mol_opt, opt_sdf)

        # Charge assignment (writes charged_mol2_qm + charges.json).
        qm.assign_charges(
            mol_opt,
            charge=str(charge),
            opt=False,
            work_dir=str(wd),
            omp=int(psi4_omp),
            memory=int(memory),
            log_name=str(log_name),
        )
        if not charges_json.exists():
            raise RuntimeError(f"Charge assignment finished but charges JSON missing: {charges_json}")

        # Return a fresh molecule reconstructed from SMILES (to preserve linkers),
        # with cached charges applied, and with FF parameters assigned.
        mol_out = utils.mol_from_smiles(smiles)
        if mol_out is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        ok = qm.load_atomic_charges_json(mol_out, charges_json, strict=True)
        if not ok:
            mol_out = mol_opt

        # Ensure FF typing exists on the returned object, except for the special
        # hydrogen terminator fragment.
        if not _is_h_terminator(mol_out, smiles):
            try:
                ok_ff = bool(ff_obj.ff_assign(mol_out))
                if not ok_ff:
                    raise RuntimeError("ff_assign returned False")
            except Exception:
                # fall back to the optimized mol (already typed during conformation_search)
                mol_out = mol_opt
                ok_ff = bool(ff_obj.ff_assign(mol_out))
                if not ok_ff:
                    raise RuntimeError(f"Force-field assignment failed for {log_name} (ff={ff_obj.__class__.__name__})")

        # Re-apply (m)Seminario bonded-params if present (prevents high-symm ions
        # from being softened by ff_assign defaults).
        try:
            import json as _json
            from pathlib import Path as _Path
            from ..sim.qm import apply_mseminario_params_to_mol as _apply_mseminario

            if mol_out.HasProp('_yadonpy_mseminario_json'):
                jp = str(mol_out.GetProp('_yadonpy_mseminario_json')).strip()
                if jp and _Path(jp).exists():
                    params = _json.loads(_Path(jp).read_text(encoding='utf-8'))
                    if isinstance(params, dict):
                        _apply_mseminario(mol_out, params, overwrite=True)
        except Exception:
            pass

        return mol_out

    return rst.step(
        name=f"resp_{log_name}",
        outputs=[opt_sdf, charges_json],
        load=_load,
        run=_run,
        inputs={
            "smiles": str(smiles),
            "charge": str(charge),
            "nconf": int(nconf),
            "dft_nconf": int(dft_nconf),
            "psi4_omp": int(psi4_omp),
            "memory": int(memory),
            "ff": ff_obj.__class__.__name__,
        },
        description="QM charges: conformer-search (neutral) or OpenBabel-UFF init + Psi4 OPT (charged) + RESP",
        verbose=True,
    )


def build_copolymer(
    monomers: Sequence[Chem.Mol],
    *,
    ratio: Sequence[float],
    reac_ratio: Optional[Sequence[float]] = None,
    terminal: Optional[Chem.Mol] = None,
    num_atoms: Optional[int] = 1000,
    target_mw: Optional[float] = None,
    polymer_name: str = "POLY",
    tacticity: str = "atactic",
    ff_obj: Optional[Any] = None,
    work_dir: Union[str, Path],
    restart: Optional[bool] = None,
    out_name: str = "polymer.sdf",
    out_mol2_name: str = "polymer.mol2",
) -> Chem.Mol:
    """Build a random copolymer chain and (optionally) assign FF parameters.

    Provide either ``num_atoms`` (default) or ``target_mw``.
    """

    wd = _as_path(work_dir)
    rst = Restart(wd, restart=restart)

    out_sdf = _stage_output_path(wd, out_name, "0_build")
    out_mol2 = _stage_output_path(wd, out_mol2_name, "0_build")

    def _load() -> Chem.Mol:
        mol = _read_sdf_one(out_sdf)
        if ff_obj is not None:
            ff_obj.ff_assign(mol)
        # Best-effort: ensure a charged mol2 exists for the polymer chain
        try:
            from ..io.mol2 import write_mol2
            if not out_mol2.exists():
                write_mol2(mol=mol, out_mol2=out_mol2, mol_name=str(polymer_name))
        except Exception:
            pass
        return mol

    def _run() -> Chem.Mol:
        import time as _time
        _t0 = _time.perf_counter()
        utils.radon_print("=" * 88, level=1)
        utils.radon_print("[SECTION] Polymer build workflow", level=1)
        utils.radon_print(f"[ITEM] out_sdf           : {compact_path(out_sdf)}", level=1)
        if terminal is None:
            raise ValueError("terminal must be provided for polymerization")
        rr = list(reac_ratio) if reac_ratio is not None else []
        if target_mw is not None:
            dp = poly.calc_n_from_mol_weight(list(monomers), float(target_mw), ratio=list(ratio), terminal1=terminal)
        elif num_atoms is not None:
            dp = poly.calc_n_from_num_atoms(list(monomers), int(num_atoms), ratio=list(ratio), terminal1=terminal)
        else:
            raise ValueError('Either num_atoms or target_mw must be provided')
        copoly = poly.random_copolymerize_rw(
            list(monomers),
            dp,
            ratio=list(ratio),
            reac_ratio=rr,
            tacticity=str(tacticity),
            name=str(polymer_name),
            work_dir=wd / ".yadonpy_polymer_rw",
        )
        copoly = poly.terminate_rw(copoly, terminal, name=str(polymer_name), work_dir=wd / ".yadonpy_polymer_term")
        utils.radon_print(f"[ITEM] polymer_name      : {polymer_name}", level=1)
        utils.radon_print(f"[ITEM] degree_of_polymer : {dp}", level=1)
        if ff_obj is not None:
            ok = ff_obj.ff_assign(copoly)
            if not ok:
                raise RuntimeError(f"Cannot assign force field ({ff_obj.__class__.__name__}) to polymer")
        _write_sdf_one(copoly, out_sdf)
        # Also export a charged MOL2 for inspection/debugging
        try:
            from ..io.mol2 import write_mol2
            write_mol2(mol=copoly, out_mol2=out_mol2, mol_name=str(polymer_name))
        except Exception:
            pass
        utils.radon_print(f"[DONE] Polymer build workflow | elapsed={format_elapsed(_time.perf_counter() - _t0)} | outputs={compact_path(out_sdf)}", level=1)
        utils.radon_print("=" * 88, level=1)
        return copoly

    return rst.step(
        name="build_polymer",
        outputs=[out_sdf, out_mol2],
        load=_load,
        run=_run,
        inputs={
            "ratio": list(map(float, ratio)),
            "reac_ratio": list(map(float, reac_ratio)) if reac_ratio is not None else [],
            "num_atoms": int(num_atoms) if num_atoms is not None else None,
            "target_mw": float(target_mw) if target_mw is not None else None,
            "polymer_name": str(polymer_name),
            "tacticity": str(tacticity),
            "ff": ff_obj.__class__.__name__ if ff_obj is not None else None,
        },
        description="Random-walk copolymerization",
        verbose=True,
    )


def pack_amorphous_cell(
    species: Sequence[Chem.Mol],
    counts: Sequence[int],
    *,
    density_g_cm3: float,
    charge_scale: Optional[Sequence[float]] = None,
    ions: Optional[Any] = None,
    neutralize: bool = True,
    neutralize_tol: float = 1e-4,
    work_dir: Union[str, Path],
    restart: Optional[bool] = None,
    out_name: str = "amorphous_cell.sdf",
) -> Chem.Mol:
    """Pack an amorphous cell (polymer + solvents + ions).

    Notes
    -----
    - If you need automatic neutralization (polyelectrolytes), you can either:
        (1) build IonPack objects with ``core.poly.ion(...)``, then
        (2) pass them explicitly via ``ions=``.
      Implicit global ion injection is no longer supported.
    """

    wd = _as_path(work_dir)
    rst = Restart(wd, restart=restart)

    out_sdf = _stage_output_path(wd, out_name, "0_build")

    def _load() -> Chem.Mol:
        return _read_sdf_one(out_sdf)

    def _run() -> Chem.Mol:
        import time as _time
        _t0 = _time.perf_counter()
        utils.radon_print("=" * 88, level=1)
        utils.radon_print("[SECTION] Amorphous-cell packing workflow", level=1)
        utils.radon_print(f"[ITEM] out_sdf           : {compact_path(out_sdf)}", level=1)
        utils.radon_print(f"[ITEM] density_g_cm3     : {float(density_g_cm3):.4f}", level=1)
        utils.radon_print(f"[ITEM] species_count     : {len(list(species))}", level=1)
        ac = poly.amorphous_cell(
            list(species),
            list(counts),
            density=float(density_g_cm3),
            charge_scale=list(charge_scale) if charge_scale is not None else None,
            ions=ions,
            neutralize=bool(neutralize),
            neutralize_tol=float(neutralize_tol),
        )
        _write_sdf_one(ac, out_sdf)
        utils.radon_print(f"[DONE] Amorphous-cell packing workflow | elapsed={format_elapsed(_time.perf_counter() - _t0)} | output={compact_path(out_sdf)}", level=1)
        utils.radon_print("=" * 88, level=1)
        return ac

    return rst.step(
        name="pack_cell",
        outputs=[out_sdf],
        load=_load,
        run=_run,
        inputs={
            "counts": list(map(int, counts)),
            "density_g_cm3": float(density_g_cm3),
            "charge_scale": list(map(float, charge_scale)) if charge_scale is not None else None,
            "neutralize": bool(neutralize),
            "neutralize_tol": float(neutralize_tol),
        },
        description="Pack amorphous cell",
        verbose=True,
    )



def equilibrate_until_ok(
    ac: Chem.Mol,
    *,
    work_dir: Union[str, Path],
    temp: float,
    press: float,
    mpi: int,
    omp: int,
    gpu: int = 1,
    gpu_id: Optional[int] = None,
    sim_time_ns: float = 5.0,
    max_additional: int = 4,
    charge_method: str = "RESP",
    ff_name: str = "gaff2_mod",
    restart: Optional[bool] = None,
    rdf_species: Optional[Sequence[Chem.Mol]] = None,
    rdf_center: Optional[Chem.Mol] = None,
) -> tuple[AnalyzeResult, bool]:
    """Run EQ21 and (optionally) additional rounds until equilibrium is reached."""

    wd = _as_path(work_dir)
    rst_flag = resolve_restart(restart)
    utils.radon_print("=" * 88, level=1)
    utils.radon_print("[SECTION] Equilibration loop workflow", level=1)
    utils.radon_print(f"[ITEM] work_dir          : {compact_path(wd)}", level=1)
    utils.radon_print(f"[ITEM] restart           : {bool(rst_flag)}", level=1)
    utils.radon_print(f"[ITEM] target_T_K        : {float(temp):.2f}", level=1)
    utils.radon_print(f"[ITEM] target_P_bar      : {float(press):.3f}", level=1)

    eqmd = eq.EQ21step(ac, work_dir=wd, ff_name=str(ff_name), charge_method=str(charge_method))
    eqmd.exec(
        temp=float(temp),
        press=float(press),
        mpi=int(mpi),
        omp=int(omp),
        gpu=int(gpu),
        gpu_id=int(gpu_id) if gpu_id is not None else None,
        sim_time=float(sim_time_ns),
        restart=rst_flag,
    )
    analy = eqmd.analyze()
    analy.get_all_prop(temp=float(temp), press=float(press), save=True)
    ok = analy.check_eq()

    # Make it obvious whether Additional rounds will run.
    utils.radon_print(f"[EQ-LOOP] EQ21 convergence: {'PASS' if ok else 'FAIL'}", level=1 if ok else 2)

    if rdf_species is not None and rdf_center is not None:
        try:
            analy.rdf(list(rdf_species), center_mol=rdf_center)
        except Exception:
            pass

    for _i in range(int(max_additional)):
        if ok:
            break
        utils.radon_print(f"[EQ-LOOP] Additional round {_i:02d}: starting (previous check=FAIL)", level=1)
        eqmd2 = eq.Additional(ac, work_dir=wd, ff_name=str(ff_name), charge_method=str(charge_method))
        eqmd2.exec(
            temp=float(temp),
            press=float(press),
            mpi=int(mpi),
            omp=int(omp),
            gpu=int(gpu),
            gpu_id=int(gpu_id) if gpu_id is not None else None,
            sim_time=float(sim_time_ns),
            restart=rst_flag,
        )
        analy = eqmd2.analyze()
        analy.get_all_prop(temp=float(temp), press=float(press), save=True)
        ok = analy.check_eq()
        utils.radon_print(f"[EQ-LOOP] Additional round {_i:02d}: convergence: {'PASS' if ok else 'FAIL'}", level=1 if ok else 2)
        if rdf_species is not None and rdf_center is not None:
            try:
                analy.rdf(list(rdf_species), center_mol=rdf_center)
            except Exception:
                pass

    utils.radon_print(f"[DONE] Equilibration loop workflow | converged={bool(ok)}", level=1 if ok else 2)
    utils.radon_print("=" * 88, level=1)
    return analy, bool(ok)


def quick_relax_gmx(
    *,
    gro: Union[str, Path],
    top: Union[str, Path],
    out_dir: Union[str, Path],
    temperature_k: float = 300.0,
    dt_ps: float = 0.002,
    nvt_ps: float = 50.0,
    mpi: int = 16,
    omp: int = 1,
    gpu: int = 1,
    gpu_id: Optional[int] = None,
    restart: Optional[bool] = None,
) -> Path:
    """Run a small NVT relaxation using GROMACS.

    This is a convenience wrapper around :class:`yadonpy.gmx.workflows.quick.QuickRelaxJob`.

    Returns
    -------
    Path
        Path to the produced `summary.json`.
    """

    out = _as_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rst = Restart(out, restart=restart)
    summary = out / "summary.json"

    def _run() -> Path:
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=bool(int(gpu) == 1),
            gpu_id=str(int(gpu_id)) if (int(gpu) == 1 and gpu_id is not None) else None,
        )
        job = QuickRelaxJob(
            gro=_as_path(gro),
            top=_as_path(top),
            out_dir=out,
            temperature_k=float(temperature_k),
            dt_ps=float(dt_ps),
            quick_md_ns=float(nvt_ps) / 1000.0,
            resources=res,
        )
        job.run(restart=bool(rst.restart))
        return summary

    return rst.step(
        name="quick_relax",
        outputs=[summary],
        run=_run,
        inputs={
            "temperature_k": float(temperature_k),
            "dt_ps": float(dt_ps),
            "nvt_ps": float(nvt_ps),
            "mpi": int(mpi),
            "omp": int(omp),
            "gpu": int(gpu),
            "gpu_id": int(gpu_id) if gpu_id is not None else None,
        },
        description="GROMACS quick relaxation",
        verbose=True,
    )


def tg_scan_gmx(
    *,
    gro: Union[str, Path],
    top: Union[str, Path],
    out_dir: Union[str, Path],
    temperatures_k: Sequence[float],
    pressure_bar: float = 1.0,
    npt_ns: float = 2.0,
    frac_last: float = 0.5,
    mpi: int = 16,
    omp: int = 1,
    gpu: int = 1,
    gpu_id: Optional[int] = None,
    restart: Optional[bool] = None,
    auto_plot: bool = True,
    fit_metric: str = "density",
) -> Path:
    """Run a Tg scan workflow using GROMACS."""

    out = _as_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rst = Restart(out, restart=restart)
    summary = out / "summary.json"
    csv = out / "density_vs_T.csv"

    temps = list(map(float, temperatures_k))

    def _run() -> Path:
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=bool(int(gpu) == 1),
            gpu_id=str(int(gpu_id)) if (int(gpu) == 1 and gpu_id is not None) else None,
        )
        job = TgJob(
            gro=_as_path(gro),
            top=_as_path(top),
            out_dir=out,
            temperatures_k=temps,
            pressure_bar=float(pressure_bar),
            npt_ns=float(npt_ns),
            frac_last=float(frac_last),
            fit_metric=str(fit_metric),
            resources=res,
            auto_plot=bool(auto_plot),
        )
        job.run(restart=bool(rst.restart))
        return summary

    return rst.step(
        name="tg_scan",
        outputs=[summary, csv],
        run=_run,
        inputs={
            "temperatures_k": temps,
            "pressure_bar": float(pressure_bar),
            "npt_ns": float(npt_ns),
            "frac_last": float(frac_last),
            "mpi": int(mpi),
            "omp": int(omp),
            "gpu": int(gpu),
            "gpu_id": int(gpu_id) if gpu_id is not None else None,
            "auto_plot": bool(auto_plot),
            "fit_metric": str(fit_metric),
        },
        description="GROMACS Tg scan",
        verbose=True,
    )


def elongation_gmx(
    *,
    gro: Union[str, Path],
    top: Union[str, Path],
    out_dir: Union[str, Path],
    temperature_k: float = 300.0,
    pressure_bar: float = 1.0,
    strain_rate_1_ps: float = 1e-6,
    total_strain: float = 0.5,
    dt_ps: float = 0.002,
    mpi: int = 16,
    omp: int = 1,
    gpu: int = 1,
    gpu_id: Optional[int] = None,
    restart: Optional[bool] = None,
    auto_plot: bool = True,
    direction: Literal["x", "y", "z"] = "x",
    modulus_fit_max_strain: float = 0.02,
    modulus_fit_min_points: int = 5,
) -> Path:
    """Run a uniaxial elongation workflow using GROMACS."""

    out = _as_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rst = Restart(out, restart=restart)
    summary = out / "summary.json"
    csv = out / "stress_strain.csv"

    def _run() -> Path:
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=bool(int(gpu) == 1),
            gpu_id=str(int(gpu_id)) if (int(gpu) == 1 and gpu_id is not None) else None,
        )
        # NOTE: elongation is NVT-deform; pressure_bar is kept for backward compatibility
        job = ElongationJob(
            gro=_as_path(gro),
            top=_as_path(top),
            out_dir=out,
            direction=str(direction),
            temperature_k=float(temperature_k),
            strain_rate_per_ps=float(strain_rate_1_ps),
            final_strain=float(total_strain),
            dt_ps=float(dt_ps),
            modulus_fit_max_strain=float(modulus_fit_max_strain),
            modulus_fit_min_points=int(modulus_fit_min_points),
            resources=res,
            auto_plot=bool(auto_plot),
        )
        job.run(restart=bool(rst.restart))
        return summary

    return rst.step(
        name="elongation",
        outputs=[summary, csv],
        run=_run,
        inputs={
            "temperature_k": float(temperature_k),
            "pressure_bar": float(pressure_bar),
            "strain_rate_1_ps": float(strain_rate_1_ps),
            "total_strain": float(total_strain),
            "dt_ps": float(dt_ps),
            "mpi": int(mpi),
            "omp": int(omp),
            "gpu": int(gpu),
            "gpu_id": int(gpu_id) if gpu_id is not None else None,
            "auto_plot": bool(auto_plot),
            "direction": str(direction),
            "modulus_fit_max_strain": float(modulus_fit_max_strain),
            "modulus_fit_min_points": int(modulus_fit_min_points),
        },
        description="GROMACS elongation",
        verbose=True,
    )
