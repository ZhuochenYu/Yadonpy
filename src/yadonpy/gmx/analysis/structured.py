"""Structured post-processing helpers for MSD / RDF / CN / conductivity."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..topology import MoleculeType, SystemTopology, parse_system_top


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def _moving_average_1d(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    w = int(max(1, min(int(window), int(y.size))))
    if w <= 1:
        return y.copy()
    if w % 2 == 0:
        w += 1
    pad = w // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y_pad, kernel, mode="valid")


@dataclass
class GroupSpec:
    group_id: str
    label: str
    moltype: str
    species_kind: str
    atom_indices_0: np.ndarray
    masses: np.ndarray
    formal_charge_e: float = 0.0
    component_key: Optional[str] = None
    component_label: Optional[str] = None


def _normalize_geometry_mode(mode: str | None) -> str:
    token = str(mode or "auto").strip().lower()
    if token in {"3d", "xy", "z", "auto"}:
        return token
    return "auto"


def _normalize_unwrap_mode(mode: str | None) -> str:
    token = str(mode or "auto").strip().lower()
    if token in {"on", "off", "auto"}:
        return token
    return "auto"


def _normalize_drift_mode(mode: str | None) -> str:
    token = str(mode or "auto").strip().lower()
    if token in {"auto", "mobile_phase", "system", "off"}:
        return token
    return "auto"


def _geometry_axes(mode: str) -> np.ndarray:
    token = _normalize_geometry_mode(mode)
    if token == "xy":
        return np.asarray([True, True, False], dtype=bool)
    if token == "z":
        return np.asarray([False, False, True], dtype=bool)
    return np.asarray([True, True, True], dtype=bool)


def _default_geometry_mode(*, system_dir: Path, requested: str | None = None) -> str:
    token = _normalize_geometry_mode(requested)
    if token != "auto":
        return token
    work_dir = Path(system_dir).parent
    markers = [
        work_dir / "05_sandwich" / "sandwich_manifest.json",
        work_dir / "05_sandwich" / "sandwich_progress.json",
        work_dir / "sandwich_manifest.json",
        work_dir / "sandwich_progress.json",
    ]
    if any(p.exists() for p in markers):
        return "xy"
    try:
        meta = load_export_metadata(system_dir)["system_meta"]
        species = list(meta.get("species", []) or [])
        labels = " ".join(
            str(sp.get("moltype") or sp.get("mol_name") or sp.get("mol_id") or "")
            for sp in species
        ).lower()
        if any(tok in labels for tok in ("graphite", "graphene", "substrate")):
            return "xy"
    except Exception:
        pass
    return "3d"


def _species_is_mobile(moltype: str, payload: dict[str, Any]) -> bool:
    kind = str(payload.get("kind") or "").strip().lower()
    label = " ".join(
        [
            str(moltype or ""),
            str(payload.get("moltype") or ""),
            str(payload.get("mol_name") or ""),
            str(payload.get("mol_id") or ""),
            str(payload.get("name") or ""),
            str(payload.get("smiles") or ""),
        ]
    ).lower()
    if any(tok in label for tok in ("graphite", "graphene", "substrate", "frozen", "wall")):
        return False
    if kind in {"substrate", "wall", "frozen"}:
        return False
    return True


def _build_mobile_atom_payload(top: SystemTopology, system_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    catalog = build_species_catalog(top, system_dir)
    mobile_indices: list[int] = []
    mobile_masses: list[float] = []
    for moltype, sp in catalog.items():
        if not _species_is_mobile(str(moltype), dict(sp)):
            continue
        masses = np.asarray(sp.get("masses"), dtype=float)
        natoms = int(sp.get("natoms") or 0)
        if masses.size != natoms or not np.isfinite(masses).all() or float(np.sum(masses)) <= 0.0:
            masses = np.ones(natoms, dtype=float)
        for inst in sp.get("instances", []) or []:
            atom_idx = np.asarray(inst.get("atom_indices_0"), dtype=int)
            if atom_idx.size != natoms:
                continue
            mobile_indices.extend(int(i) for i in atom_idx.tolist())
            mobile_masses.extend(float(m) for m in masses.tolist())
    return np.asarray(mobile_indices, dtype=int), np.asarray(mobile_masses, dtype=float)


_MOBILE_DRIFT_CACHE: dict[tuple[str, str, str, str, int], tuple[np.ndarray, np.ndarray]] = {}


def _unwrap_position_series(positions_nm: np.ndarray, box_lengths_nm: np.ndarray, *, geometry_mode: str) -> np.ndarray:
    pos = np.asarray(positions_nm, dtype=float)
    box = np.asarray(box_lengths_nm, dtype=float)
    if pos.ndim != 3 or pos.shape[0] == 0:
        return np.asarray(pos, dtype=float)
    out = np.array(pos, copy=True)
    if out.shape[0] < 2:
        return out
    axes = _geometry_axes(geometry_mode)
    if not np.any(axes):
        return out
    delta = pos[1:] - pos[:-1]
    box_mid = 0.5 * (box[1:, :3] + box[:-1, :3])
    for axis in range(3):
        if not bool(axes[axis]):
            continue
        ref = np.maximum(box_mid[:, None, axis], 1.0e-12)
        delta[:, :, axis] -= ref * np.round(delta[:, :, axis] / ref)
    out[1:] = out[0] + np.cumsum(delta, axis=0)
    return out


def _compute_mobile_drift_series(
    *,
    gro_path: Path,
    xtc_path: Path,
    top_path: Path,
    system_dir: Path,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import mdtraj as md
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"mdtraj is required for transport drift correction: {exc}") from exc

    cache_key = (
        str(Path(gro_path).resolve()),
        str(Path(xtc_path).resolve()),
        str(Path(top_path).resolve()),
        str(Path(system_dir).resolve()),
        int(max(1, chunk)),
    )
    cached = _MOBILE_DRIFT_CACHE.get(cache_key)
    if cached is not None:
        t_cached, drift_cached = cached
        return np.array(t_cached, copy=True), np.array(drift_cached, copy=True)

    top = parse_system_top(Path(top_path))
    atom_indices, masses = _build_mobile_atom_payload(top, system_dir)
    if atom_indices.size == 0:
        return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float)
    if masses.size != atom_indices.size or not np.isfinite(masses).all() or float(np.sum(masses)) <= 0.0:
        masses = np.ones(atom_indices.size, dtype=float)
    weights = masses / float(np.sum(masses))

    times: list[np.ndarray] = []
    drift_chunks: list[np.ndarray] = []
    box_chunks: list[np.ndarray] = []
    for trj in md.iterload(str(xtc_path), top=str(gro_path), chunk=int(max(1, chunk))):
        xyz = np.asarray(trj.xyz[:, atom_indices, :], dtype=float)
        box = np.asarray(getattr(trj, "unitcell_lengths", None), dtype=float)
        if box.ndim != 2 or box.shape[0] != xyz.shape[0] or box.shape[1] < 3:
            continue
        # For drift correction we only need the global mobile-phase COM trajectory.
        # Computing the wrapped COM first and unwrapping that 3-vector time series is
        # substantially cheaper than unwrapping every mobile atom in large systems.
        drift = np.tensordot(xyz, weights, axes=(1, 0))
        times.append(np.asarray(trj.time, dtype=float))
        drift_chunks.append(np.asarray(drift, dtype=float))
        box_chunks.append(np.asarray(box[:, :3], dtype=float))
    if not times:
        return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float)
    t_ps = np.concatenate(times, axis=0)
    drift_wrapped = np.concatenate(drift_chunks, axis=0)
    box_nm = np.concatenate(box_chunks, axis=0)
    drift = _unwrap_position_series(drift_wrapped[:, None, :], box_nm, geometry_mode="3d")[:, 0, :]
    _MOBILE_DRIFT_CACHE[cache_key] = (np.array(t_ps, copy=True), np.array(drift, copy=True))
    return t_ps, drift


def _canonicalize_smiles_like(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return ""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        pass
    return s


def _mol_to_smiles(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return _canonicalize_smiles_like(obj)
    try:
        from rdkit import Chem

        if hasattr(obj, "HasProp") and obj.HasProp("_yadonpy_smiles"):
            return _canonicalize_smiles_like(obj.GetProp("_yadonpy_smiles"))
        mol = obj if hasattr(obj, "GetAtoms") else None
        if mol is not None:
            return _canonicalize_smiles_like(Chem.MolToSmiles(mol, isomericSmiles=True))
    except Exception:
        pass
    return ""


def load_export_metadata(system_dir: Path) -> dict[str, Any]:
    system_dir = Path(system_dir)
    meta = _read_json(system_dir / "system_meta.json")
    residue_map_payload = _read_json(system_dir / "residue_map.json")
    charge_groups_payload = _read_json(system_dir / "charge_groups.json")
    return {
        "system_meta": meta,
        "residue_map_payload": residue_map_payload,
        "charge_groups_payload": charge_groups_payload,
    }


def resolve_moltypes_from_mols(system_dir: Path, mol_or_mols: Any) -> list[str]:
    meta = load_export_metadata(system_dir)["system_meta"]
    species = meta.get("species", []) or []
    if mol_or_mols is None:
        return []
    objs = list(mol_or_mols) if isinstance(mol_or_mols, (list, tuple, set)) else [mol_or_mols]
    want = {_mol_to_smiles(obj) for obj in objs}
    want = {s for s in want if s}
    if not want:
        return []
    out: list[str] = []
    for sp in species:
        sp_smi = _canonicalize_smiles_like(sp.get("smiles", ""))
        if sp_smi in want:
            mt = sp.get("moltype") or sp.get("mol_name") or sp.get("mol_id")
            if mt:
                out.append(str(mt))
    return sorted(set(out))


def _species_index_payload(system_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = load_export_metadata(system_dir)
    meta = payload["system_meta"]
    residue_map_payload = payload["residue_map_payload"]
    charge_groups_payload = payload["charge_groups_payload"]
    species_map: dict[str, dict[str, Any]] = {}
    for sp in meta.get("species", []) or []:
        mt = sp.get("moltype") or sp.get("mol_name") or sp.get("mol_id")
        if mt:
            species_map[str(mt)] = dict(sp)
    residue_map_map: dict[str, dict[str, Any]] = {}
    for sp in residue_map_payload.get("species", []) or []:
        mt = sp.get("moltype") or sp.get("mol_name")
        if mt:
            residue_map_map[str(mt)] = dict(sp)
    charge_group_map: dict[str, dict[str, Any]] = {}
    for sp in charge_groups_payload.get("species", []) or []:
        mt = sp.get("moltype") or sp.get("mol_name")
        if mt:
            charge_group_map[str(mt)] = dict(sp)
    return species_map, residue_map_map, charge_group_map


def build_species_catalog(top: SystemTopology, system_dir: Path) -> dict[str, dict[str, Any]]:
    system_dir = Path(system_dir)
    species_map, residue_map_map, charge_group_map = _species_index_payload(system_dir)
    catalog: dict[str, dict[str, Any]] = {}
    current = 0
    for moltype, count in top.molecules:
        mt = top.moleculetypes.get(moltype)
        if mt is None:
            continue
        natoms = int(mt.natoms)
        sp_meta = dict(species_map.get(str(moltype), {}))
        residue_template = residue_map_map.get(str(moltype), {}).get("residue_map")
        charge_group_template = charge_group_map.get(str(moltype), {}).get("charge_groups", [])
        kind = str(sp_meta.get("kind") or "").strip().lower()
        smiles = str(sp_meta.get("smiles") or "")
        if not kind:
            kind = "polymer" if "*" in smiles else "molecule"
        formal_charge = float(sp_meta.get("formal_charge", getattr(mt, "net_charge", 0.0)))
        masses = np.asarray(mt.masses, dtype=float) if mt.masses else np.ones(natoms, dtype=float)
        if masses.size != natoms or not np.isfinite(masses).all() or float(np.sum(masses)) <= 0.0:
            masses = np.ones(natoms, dtype=float)
        instances: list[dict[str, Any]] = []
        for inst_idx in range(int(count)):
            start = current
            atom_indices = np.arange(start, start + natoms, dtype=int)
            current += natoms
            instances.append({"instance_index": int(inst_idx), "atom_indices_0": atom_indices})
        catalog[str(moltype)] = {
            "moltype": str(moltype),
            "smiles": smiles,
            "kind": kind,
            "formal_charge_e": float(formal_charge),
            "natoms": natoms,
            "count": int(count),
            "masses": masses,
            "residue_map": residue_template,
            "charge_groups": list(charge_group_template or []),
            "instances": instances,
            "moleculetype": mt,
        }
    return catalog


def build_msd_metric_catalog(top: SystemTopology, system_dir: Path) -> dict[str, dict[str, Any]]:
    catalog = build_species_catalog(top, system_dir)
    out: dict[str, dict[str, Any]] = {}
    for moltype, sp in catalog.items():
        natoms = int(sp["natoms"])
        kind = str(sp["kind"])
        count = int(sp["count"])
        masses = np.asarray(sp["masses"], dtype=float)
        formal_charge = float(sp["formal_charge_e"])
        metrics: dict[str, dict[str, Any]] = {}
        default_metric = None

        if natoms == 1 and abs(formal_charge) > 1.0e-12:
            groups = []
            for inst in sp["instances"]:
                groups.append(
                    GroupSpec(
                        group_id=f"{moltype}:ion:{inst['instance_index']}",
                        label=str(moltype),
                        moltype=moltype,
                        species_kind=kind,
                        atom_indices_0=np.asarray(inst["atom_indices_0"], dtype=int),
                        masses=np.asarray([float(masses[0])], dtype=float),
                        formal_charge_e=float(formal_charge),
                    )
                )
            metrics["ion_atomic_msd"] = {"group_kind": "ion_atomic", "groups": groups}
            default_metric = "ion_atomic_msd"
        else:
            groups = []
            for inst in sp["instances"]:
                groups.append(
                    GroupSpec(
                        group_id=f"{moltype}:molecule:{inst['instance_index']}",
                        label=str(moltype),
                        moltype=moltype,
                        species_kind=kind,
                        atom_indices_0=np.asarray(inst["atom_indices_0"], dtype=int),
                        masses=masses,
                        formal_charge_e=float(formal_charge),
                    )
                )
            metrics["molecule_com_msd"] = {"group_kind": "molecule_com", "groups": groups}
            default_metric = "molecule_com_msd"

        if kind == "polymer":
            chain_groups = []
            for inst in sp["instances"]:
                chain_groups.append(
                    GroupSpec(
                        group_id=f"{moltype}:chain:{inst['instance_index']}",
                        label=str(moltype),
                        moltype=moltype,
                        species_kind=kind,
                        atom_indices_0=np.asarray(inst["atom_indices_0"], dtype=int),
                        masses=masses,
                        formal_charge_e=float(formal_charge),
                    )
                )
            metrics["chain_com_msd"] = {"group_kind": "chain_com", "groups": chain_groups}
            default_metric = "chain_com_msd"

            residue_map = sp.get("residue_map") or {}
            residue_groups = []
            for inst in sp["instances"]:
                atom_offset = int(inst["atom_indices_0"][0])
                for residue in (residue_map.get("residues", []) or []):
                    local = np.asarray(residue.get("atom_indices", []), dtype=int)
                    if local.size == 0:
                        continue
                    global_idx = atom_offset + local
                    residue_groups.append(
                        GroupSpec(
                            group_id=f"{moltype}:residue:{inst['instance_index']}:{residue.get('residue_number', 0)}",
                            label=str(residue.get("residue_name") or "RES"),
                            moltype=moltype,
                            species_kind=kind,
                            atom_indices_0=global_idx,
                            masses=masses[local],
                            formal_charge_e=0.0,
                            component_key=str(residue.get("residue_name") or "RES"),
                            component_label=str(residue.get("residue_name") or "RES"),
                        )
                    )
            if residue_groups:
                metrics["residue_com_msd"] = {"group_kind": "residue_com", "groups": residue_groups}

        charge_groups = list(sp.get("charge_groups") or [])
        if charge_groups:
            cg_specs = []
            for inst in sp["instances"]:
                atom_offset = int(inst["atom_indices_0"][0])
                for grp in charge_groups:
                    local = np.asarray(grp.get("atom_indices", []), dtype=int)
                    if local.size == 0:
                        continue
                    global_idx = atom_offset + local
                    formal_q = float(grp.get("formal_charge", 0.0))
                    sign = "cation" if formal_q > 0 else "anion" if formal_q < 0 else "neutral"
                    label = str(grp.get("label") or grp.get("group_id") or "charge_group")
                    comp_key = f"{sign}:{label}:q{int(formal_q):+d}"
                    cg_specs.append(
                        GroupSpec(
                            group_id=f"{moltype}:charged_group:{inst['instance_index']}:{grp.get('group_id', label)}",
                            label=label,
                            moltype=moltype,
                            species_kind=kind,
                            atom_indices_0=global_idx,
                            masses=masses[local],
                            formal_charge_e=formal_q,
                            component_key=comp_key,
                            component_label=f"{sign}:{label}",
                        )
                    )
            if cg_specs:
                metrics["charged_group_com_msd"] = {"group_kind": "charged_group_com", "groups": cg_specs}

        out[moltype] = {
            "moltype": moltype,
            "kind": kind,
            "smiles": sp.get("smiles", ""),
            "n_molecules": count,
            "natoms": natoms,
            "formal_charge_e": float(formal_charge),
            "default_metric": default_metric,
            "metrics": metrics,
        }
    return out


def _infer_element(atomname: str, atomtype: str) -> str:
    token = str(atomname or "").strip()
    if not token:
        token = str(atomtype or "").strip()
    token = re.sub(r"[^A-Za-z]", "", token)
    if not token:
        return ""
    two = token[:2].capitalize()
    if two in {"Li", "Na", "Mg", "Al", "Si", "Cl", "Ca", "Fe", "Co", "Ni", "Cu", "Zn", "Br", "Rb", "Sr", "Ag", "Cd", "Sn", "Cs", "Ba", "Hg", "Pb"}:
        return two
    return token[:1].upper()


def _build_adjacency(mt: MoleculeType) -> dict[int, set[int]]:
    adj: dict[int, set[int]] = {i: set() for i in range(mt.natoms)}
    for ai, aj in mt.bonds:
        i = int(ai) - 1
        j = int(aj) - 1
        if i < 0 or j < 0 or i >= mt.natoms or j >= mt.natoms:
            continue
        adj[i].add(j)
        adj[j].add(i)
    return adj


def _site_label_for_atom(
    mt: MoleculeType,
    atom_idx0: int,
    adj: dict[int, set[int]],
    charge_group_lookup: dict[int, dict[str, Any]],
    *,
    include_h: bool = False,
) -> Optional[str]:
    atomname = str(mt.atomnames[atom_idx0] if atom_idx0 < len(mt.atomnames) else "")
    atomtype = str(mt.atomtypes[atom_idx0] if atom_idx0 < len(mt.atomtypes) else "")
    elem = _infer_element(atomname, atomtype)
    if not include_h and elem == "H":
        return None
    neigh = sorted(adj.get(atom_idx0, set()))
    neigh_elems = [_infer_element(mt.atomnames[j], mt.atomtypes[j]) for j in neigh]
    charge_group = charge_group_lookup.get(atom_idx0)
    formal_q = float(charge_group.get("formal_charge", 0.0)) if charge_group else 0.0

    def _bonded_to(element: str) -> bool:
        return any(e == element for e in neigh_elems)

    if formal_q < 0.0:
        if elem == "O":
            if _bonded_to("S"):
                return "sulfonyl_oxygen"
            if _bonded_to("P") or _bonded_to("Cl") or _bonded_to("B"):
                return "oxo_anion_oxygen"
            if _bonded_to("C"):
                for cidx in neigh:
                    if _infer_element(mt.atomnames[cidx], mt.atomtypes[cidx]) != "C":
                        continue
                    hetero = 0
                    for nb in adj.get(cidx, set()):
                        if nb != atom_idx0 and _infer_element(mt.atomnames[nb], mt.atomtypes[nb]) == "O":
                            hetero += 1
                    if hetero >= 1:
                        return "carboxylate_oxygen"
                return "anionic_oxygen"
        if elem == "F":
            return "anion_fluorine"
        if elem in {"Cl", "Br", "I"}:
            return "halide_anion_site"
        if elem == "N":
            return "anion_nitrogen"

    if formal_q > 0.0:
        if elem == "N":
            return "cationic_nitrogen"
        return "cationic_site"

    if elem == "O":
        if _bonded_to("H"):
            return "hydroxyl_oxygen"
        if _bonded_to("S"):
            return "sulfonyl_oxygen"
        if _bonded_to("P") or _bonded_to("Cl") or _bonded_to("B"):
            return "phosphate_or_oxo_oxygen"
        if len(neigh) == 2 and all(e in {"C", "Si"} for e in neigh_elems):
            return "ether_oxygen"
        if _bonded_to("C"):
            for cidx in neigh:
                if _infer_element(mt.atomnames[cidx], mt.atomtypes[cidx]) != "C":
                    continue
                hetero = 0
                for nb in adj.get(cidx, set()):
                    if nb != atom_idx0 and _infer_element(mt.atomnames[nb], mt.atomtypes[nb]) == "O":
                        hetero += 1
                if hetero >= 1:
                    return "carbonyl_oxygen"
        return "oxygen_site"
    if elem == "F":
        if _bonded_to("P") or _bonded_to("B") or _bonded_to("S"):
            return "coordination_fluorine"
        return "fluorine_site"
    if elem == "N":
        return "nitrogen_site"
    if elem in {"Cl", "Br", "I"}:
        return "halogen_site"
    if elem in {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Zn"}:
        return "cation_center"
    return None


def _site_coordination_metadata(site_label: str) -> dict[str, Any]:
    label = str(site_label or "").strip().lower()
    if label in {
        "carboxylate_oxygen",
        "anionic_oxygen",
        "oxo_anion_oxygen",
        "sulfonyl_oxygen",
        "phosphate_or_oxo_oxygen",
        "carbonyl_oxygen",
        "ether_oxygen",
        "hydroxyl_oxygen",
        "oxygen_site",
    }:
        return {
            "coordination_priority": 0,
            "coordination_relevance": "primary",
            "coordination_note": "oxygen-donor site; prioritize for cation coordination interpretation",
        }
    if label in {"anion_nitrogen", "nitrogen_site", "cationic_nitrogen"}:
        return {
            "coordination_priority": 1,
            "coordination_relevance": "secondary",
            "coordination_note": "nitrogen-centered site; keep as secondary coordination evidence",
        }
    if label in {"halide_anion_site", "cation_center", "cationic_site"}:
        return {
            "coordination_priority": 2,
            "coordination_relevance": "contextual",
            "coordination_note": "contextual site; useful for ion pairing or clustering, not a preferred donor interpretation",
        }
    if label in {"anion_fluorine", "coordination_fluorine", "fluorine_site", "halogen_site"}:
        return {
            "coordination_priority": 3,
            "coordination_relevance": "weak",
            "coordination_note": "halogen/fluorine site; default summaries should de-emphasize it versus O/N donor sites",
        }
    return {
        "coordination_priority": 2,
        "coordination_relevance": "contextual",
        "coordination_note": "site has no stronger coordination heuristic; treat as contextual",
    }


def build_site_map(
    top: SystemTopology,
    system_dir: Path,
    *,
    include_h: bool = False,
    selected_moltypes: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    selected = set(str(x) for x in selected_moltypes) if selected_moltypes is not None else None
    catalog = build_species_catalog(top, system_dir)
    entries: list[dict[str, Any]] = []
    for moltype, sp in catalog.items():
        if selected is not None and moltype not in selected:
            continue
        mt = sp.get("moleculetype")
        if mt is None:
            continue
        adj = _build_adjacency(mt)
        charge_group_lookup: dict[int, dict[str, Any]] = {}
        for grp in sp.get("charge_groups", []) or []:
            for idx in grp.get("atom_indices", []) or []:
                charge_group_lookup[int(idx)] = dict(grp)
        local_site_map: dict[str, list[int]] = {}
        for idx0 in range(int(sp["natoms"])):
            site_label = _site_label_for_atom(mt, idx0, adj, charge_group_lookup, include_h=include_h)
            if not site_label:
                continue
            local_site_map.setdefault(site_label, []).append(int(idx0))
        for site_label, local_indices in sorted(local_site_map.items()):
            global_indices: list[int] = []
            for inst in sp["instances"]:
                atom_offset = int(inst["atom_indices_0"][0])
                global_indices.extend([atom_offset + int(i) for i in local_indices])
            coord_meta = _site_coordination_metadata(site_label)
            entries.append(
                {
                    "site_id": f"{moltype}:{site_label}",
                    "moltype": moltype,
                    "site_label": site_label,
                    "local_atom_indices": [int(i) for i in local_indices],
                    "atom_indices": global_indices,
                    "count": len(global_indices),
                    **coord_meta,
                }
            )
    return {"site_groups": entries}


def load_group_positions(
    *,
    gro_path: Path,
    xtc_path: Path,
    group_specs: Sequence[GroupSpec],
    chunk: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import mdtraj as md
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"mdtraj is required for structured post-processing: {exc}") from exc

    if not group_specs:
        return np.zeros(0, dtype=float), np.zeros((0, 0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    times: list[np.ndarray] = []
    pos_chunks: list[np.ndarray] = []
    box_chunks: list[np.ndarray] = []
    for trj in md.iterload(str(xtc_path), top=str(gro_path), chunk=int(max(1, chunk))):
        xyz = np.asarray(trj.xyz, dtype=float)
        box = np.asarray(getattr(trj, "unitcell_lengths", None), dtype=float)
        if box.ndim != 2 or box.shape[0] != xyz.shape[0] or box.shape[1] < 3:
            raise RuntimeError("Trajectory unit-cell lengths are required for transport post-processing.")
        chunk_pos = np.zeros((xyz.shape[0], len(group_specs), 3), dtype=float)
        for gi, spec in enumerate(group_specs):
            idx = np.asarray(spec.atom_indices_0, dtype=int)
            if idx.size == 1:
                chunk_pos[:, gi, :] = xyz[:, idx[0], :]
                continue
            w = np.asarray(spec.masses, dtype=float)
            if w.size != idx.size or not np.isfinite(w).all() or float(np.sum(w)) <= 0.0:
                w = np.ones(idx.size, dtype=float)
            w = w / float(np.sum(w))
            coords = xyz[:, idx, :]
            chunk_pos[:, gi, :] = np.tensordot(coords, w, axes=(1, 0))
        times.append(np.asarray(trj.time, dtype=float))
        pos_chunks.append(chunk_pos)
        box_chunks.append(np.asarray(box[:, :3], dtype=float))
    if not times:
        return np.zeros(0, dtype=float), np.zeros((0, 0, 3), dtype=float), np.zeros((0, 3), dtype=float)
    return np.concatenate(times, axis=0), np.concatenate(pos_chunks, axis=0), np.concatenate(box_chunks, axis=0)


def preprocess_group_positions(
    *,
    gro_path: Path,
    xtc_path: Path,
    top_path: Path,
    system_dir: Path,
    group_specs: Sequence[GroupSpec],
    chunk: int = 50,
    geometry_mode: str = "auto",
    unwrap: str = "auto",
    drift: str = "auto",
) -> dict[str, Any]:
    geometry = _default_geometry_mode(system_dir=Path(system_dir), requested=geometry_mode)
    unwrap_mode = _normalize_unwrap_mode(unwrap)
    drift_mode = _normalize_drift_mode(drift)

    t_ps, positions, box_nm = load_group_positions(
        gro_path=gro_path,
        xtc_path=xtc_path,
        group_specs=group_specs,
        chunk=chunk,
    )
    if positions.size == 0:
        return {
            "t_ps": t_ps,
            "positions_nm": positions,
            "box_lengths_nm": box_nm,
            "preprocessing": {
                "used_unwrapped_positions": False,
                "drift_correction_mode": "off",
                "drift_reference_group": None,
                "geometry_mode": geometry,
            },
        }

    use_unwrap = bool(unwrap_mode in {"auto", "on"})
    if use_unwrap:
        positions = _unwrap_position_series(positions, box_nm, geometry_mode=geometry)

    effective_drift = drift_mode
    if effective_drift == "auto":
        effective_drift = "mobile_phase"
    drift_reference = None
    if effective_drift != "off":
        drift_t, drift_series = _compute_mobile_drift_series(
            gro_path=gro_path,
            xtc_path=xtc_path,
            top_path=top_path,
            system_dir=system_dir,
            chunk=max(int(chunk), 25),
        )
        if drift_series.size and drift_t.size == t_ps.size and np.allclose(drift_t, t_ps):
            axes = _geometry_axes(geometry)
            for axis in range(3):
                if not bool(axes[axis]):
                    drift_series[:, axis] = 0.0
            positions = positions - drift_series[:, None, :]
            drift_reference = "mobile_phase"
        else:
            effective_drift = "off"

    return {
        "t_ps": np.asarray(t_ps, dtype=float),
        "positions_nm": np.asarray(positions, dtype=float),
        "box_lengths_nm": np.asarray(box_nm, dtype=float),
        "preprocessing": {
            "used_unwrapped_positions": bool(use_unwrap),
            "drift_correction_mode": str(effective_drift),
            "drift_reference_group": drift_reference,
            "geometry_mode": str(geometry),
        },
    }


def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    n = int(x.shape[0])
    if n == 0:
        return np.zeros_like(x, dtype=float)
    f = np.fft.fft(x, n=2 * n, axis=0)
    ac = np.fft.ifft(f * np.conjugate(f), axis=0)[:n].real
    norm = np.arange(n, 0, -1, dtype=float).reshape((-1,) + (1,) * (ac.ndim - 1))
    return ac / norm


def msd_from_positions_fft(positions_nm: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions_nm, dtype=float)
    if pos.ndim != 3 or pos.shape[0] == 0 or pos.shape[1] == 0:
        return np.zeros(0, dtype=float)
    n_frames = int(pos.shape[0])
    n_groups = int(pos.shape[1])
    d = np.sum(pos * pos, axis=2)
    d_app = np.concatenate([d, np.zeros((1, n_groups), dtype=float)], axis=0)
    s2 = np.zeros((n_frames, n_groups), dtype=float)
    for dim in range(int(pos.shape[2])):
        s2 += _autocorr_fft(pos[:, :, dim])
    q = 2.0 * np.sum(d_app, axis=0)
    s1 = np.zeros((n_frames, n_groups), dtype=float)
    for m in range(n_frames):
        q = q - d_app[m - 1, :] - d_app[n_frames - m, :]
        s1[m, :] = q / float(n_frames - m)
    msd_groups = s1 - 2.0 * s2
    msd_groups[0, :] = 0.0
    msd_groups = np.maximum(msd_groups, 0.0)
    return np.mean(msd_groups, axis=1)


def select_diffusive_window(
    t_ps: np.ndarray,
    msd_nm2: np.ndarray,
    *,
    alpha_target: float = 1.0,
    primary_tol: float = 0.15,
    secondary_tol: float = 0.25,
    geometry: str = "3d",
) -> dict[str, Any]:
    t = np.asarray(t_ps, dtype=float)
    y = np.asarray(msd_nm2, dtype=float)
    out: dict[str, Any] = {
        "fit_t_start_ps": None,
        "fit_t_end_ps": None,
        "fit_r2": None,
        "fit_slope_nm2_ps": None,
        "fit_intercept_nm2": None,
        "alpha_mean": None,
        "alpha_std": None,
        "alpha_deviation": None,
        "selection_basis": "loglog_slope_closest_to_one",
        "confidence": "failed",
        "status": "failed",
        "D_nm2_ps": None,
        "D_m2_s": None,
        "warning": None,
        "geometry": "3d" if _normalize_geometry_mode(geometry) == "auto" else _normalize_geometry_mode(geometry),
    }
    if t.size < 12 or y.size != t.size:
        out["warning"] = "too_few_points"
        return out
    mask = (t > 0.0) & (y > 0.0) & np.isfinite(t) & np.isfinite(y)
    if int(np.sum(mask)) < 12:
        out["warning"] = "insufficient_positive_points"
        return out
    tv = t[mask]
    yv = y[mask]
    y_s = _moving_average_1d(yv, min(21, max(7, yv.size // 25)))
    log_t = np.log(tv)
    log_y = np.log(np.maximum(y_s, 1.0e-30))
    alpha = np.gradient(log_y, log_t)
    min_run = max(8, int(round(0.1 * tv.size)))

    candidates: list[dict[str, Any]] = []
    for tol, confidence in ((primary_tol, "high"), (secondary_tol, "medium")):
        ok = np.isfinite(alpha) & (np.abs(alpha - float(alpha_target)) <= float(tol))
        start = None
        for idx, flag in enumerate(ok):
            if flag and start is None:
                start = idx
            end_here = (not flag or idx == ok.size - 1) and start is not None
            if end_here:
                end = idx if flag and idx == ok.size - 1 else idx - 1
                if end - start + 1 >= min_run:
                    tt = tv[start : end + 1]
                    yy = yv[start : end + 1]
                    slope, intercept = np.polyfit(tt, yy, 1)
                    if slope > 0.0:
                        yhat = slope * tt + intercept
                        ss_res = float(np.sum((yy - yhat) ** 2))
                        ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
                        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
                        alpha_slice = alpha[start : end + 1]
                        candidates.append(
                            {
                                "confidence": confidence,
                                "fit_t_start_ps": float(tt[0]),
                                "fit_t_end_ps": float(tt[-1]),
                                "fit_r2": float(r2),
                                "fit_slope_nm2_ps": float(slope),
                                "fit_intercept_nm2": float(intercept),
                                "alpha_mean": float(np.mean(alpha_slice)),
                                "alpha_std": float(np.std(alpha_slice)),
                                "alpha_deviation": float(abs(float(np.mean(alpha_slice)) - float(alpha_target))),
                                "duration_ps": float(tt[-1] - tt[0]),
                                "selection_basis": "loglog_slope_closest_to_one",
                            }
                        )
                start = None
    if not candidates:
        finite_alpha = np.isfinite(alpha)
        if int(np.sum(finite_alpha)) < min_run:
            out["warning"] = "no_diffusive_regime_detected"
            out["confidence"] = "low"
            out["status"] = "no_formal_diffusion"
            return out
        alpha_dev_all = np.where(finite_alpha, np.abs(alpha - float(alpha_target)), np.inf)
        best_center = int(np.argmin(alpha_dev_all))
        start = int(np.clip(best_center - min_run // 2, 0, max(tv.size - min_run, 0)))
        end = int(min(tv.size - 1, start + min_run - 1))
        tt = tv[start : end + 1]
        yy = yv[start : end + 1]
        slope, intercept = np.polyfit(tt, yy, 1)
        if not np.isfinite(slope) or float(slope) <= 0.0:
            out["warning"] = "no_positive_slope_window_detected"
            out["confidence"] = "low"
            out["status"] = "no_formal_diffusion"
            return out
        yhat = slope * tt + intercept
        ss_res = float(np.sum((yy - yhat) ** 2))
        ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
        alpha_slice = alpha[start : end + 1]
        candidates.append(
            {
                "confidence": "low",
                "fit_t_start_ps": float(tt[0]),
                "fit_t_end_ps": float(tt[-1]),
                "fit_r2": float(r2),
                "fit_slope_nm2_ps": float(slope),
                "fit_intercept_nm2": float(intercept),
                "alpha_mean": float(np.mean(alpha_slice)),
                "alpha_std": float(np.std(alpha_slice)),
                "alpha_deviation": float(abs(float(np.mean(alpha_slice)) - float(alpha_target))),
                "duration_ps": float(tt[-1] - tt[0]),
                "selection_basis": "loglog_slope_closest_to_one",
                "warning": "closest_loglog_slope_window",
            }
        )

    def _score(item: dict[str, Any]) -> tuple[float, int, float, float]:
        conf_rank = {"high": 0, "medium": 1, "low": 2}.get(str(item.get("confidence") or "low"), 3)
        return (
            float(item.get("alpha_deviation", float("inf"))),
            int(conf_rank),
            -float(item.get("duration_ps", 0.0)),
            -float(item.get("fit_r2", float("-inf"))),
        )

    best = sorted(candidates, key=_score)[0]
    best.pop("duration_ps", None)
    out.update(best)
    alpha_mean = float(best.get("alpha_mean")) if best.get("alpha_mean") is not None else float("nan")
    if best["confidence"] in {"high", "medium"}:
        out["status"] = "ok"
    elif np.isfinite(alpha_mean) and alpha_mean < float(alpha_target):
        out["status"] = "subdiffusive_risk"
    else:
        out["status"] = "superdiffusive_risk"

    geometry_token = _normalize_geometry_mode(out.get("geometry"))
    divisor = 6.0
    if geometry_token == "xy":
        divisor = 4.0
    elif geometry_token == "z":
        divisor = 2.0
    d_nm2_ps = float(best["fit_slope_nm2_ps"]) / float(divisor)
    out["D_nm2_ps"] = float(d_nm2_ps)
    out["D_m2_s"] = float(d_nm2_ps) * 1.0e-6
    return out


def compute_msd_series(
    *,
    gro_path: Path,
    xtc_path: Path,
    top_path: Path,
    system_dir: Path,
    group_specs: Sequence[GroupSpec],
    chunk: int = 50,
    geometry_mode: str = "auto",
    unwrap: str = "auto",
    drift: str = "auto",
    begin_ps: float | None = None,
    end_ps: float | None = None,
) -> dict[str, Any]:
    prepared = preprocess_group_positions(
        gro_path=gro_path,
        xtc_path=xtc_path,
        top_path=top_path,
        system_dir=system_dir,
        group_specs=group_specs,
        chunk=chunk,
        geometry_mode=geometry_mode,
        unwrap=unwrap,
        drift=drift,
    )
    t_ps = np.asarray(prepared["t_ps"], dtype=float)
    positions = np.asarray(prepared["positions_nm"], dtype=float)
    geometry = str((prepared.get("preprocessing") or {}).get("geometry_mode") or _normalize_geometry_mode(geometry_mode))
    if t_ps.size and positions.size:
        mask = np.ones(t_ps.shape, dtype=bool)
        if begin_ps is not None:
            mask &= t_ps >= float(begin_ps)
        if end_ps is not None:
            mask &= t_ps <= float(end_ps)
        if not np.all(mask):
            t_ps = t_ps[mask]
            positions = positions[mask, :, :]
    if t_ps.size == 0 or positions.size == 0:
        return {
            "t_ps": np.zeros(0, dtype=float),
            "msd_nm2": np.zeros(0, dtype=float),
            "fit": select_diffusive_window(np.zeros(0, dtype=float), np.zeros(0, dtype=float), geometry=geometry),
            "frame_interval_ps": None,
            "n_groups": int(len(group_specs)),
            "component_metrics": {},
            "preprocessing": prepared.get("preprocessing") or {},
            "geometry": geometry,
        }
    msd = msd_from_positions_fft(positions)
    fit = select_diffusive_window(t_ps, msd, geometry=geometry)
    frame_interval = float(np.median(np.diff(t_ps))) if t_ps.size >= 2 else None
    component_metrics: dict[str, Any] = {}
    component_map: dict[str, list[int]] = {}
    for idx, spec in enumerate(group_specs):
        if spec.component_key:
            component_map.setdefault(spec.component_key, []).append(idx)
    for key, idxs in component_map.items():
        comp_pos = positions[:, idxs, :]
        comp_msd = msd_from_positions_fft(comp_pos)
        comp_fit = select_diffusive_window(t_ps, comp_msd, geometry=geometry)
        first = group_specs[idxs[0]]
        component_metrics[key] = {
            "component_key": key,
            "component_label": first.component_label or first.label,
            "formal_charge_e": float(first.formal_charge_e),
            "charge_sign": "cation" if float(first.formal_charge_e) > 0.0 else "anion" if float(first.formal_charge_e) < 0.0 else "neutral",
            "n_groups": int(len(idxs)),
            "t_ps": t_ps,
            "msd_nm2": comp_msd,
            **comp_fit,
        }
    return {
        "t_ps": t_ps,
        "msd_nm2": msd,
        "fit": fit,
        "frame_interval_ps": frame_interval,
        "n_groups": int(len(group_specs)),
        "component_metrics": component_metrics,
        "preprocessing": prepared.get("preprocessing") or {},
        "geometry": geometry,
    }


def detect_first_shell(r_nm: np.ndarray, g_r: np.ndarray, cn_curve: np.ndarray) -> dict[str, Any]:
    r = np.asarray(r_nm, dtype=float)
    g = np.asarray(g_r, dtype=float)
    cn = np.asarray(cn_curve, dtype=float)
    out: dict[str, Any] = {
        "r_peak_nm": None,
        "g_peak": None,
        "r_shell_nm": None,
        "cn_shell": None,
        "confidence": "failed",
        "status": "failed",
        "note": None,
    }
    if r.size < 8 or g.size != r.size or cn.size != r.size:
        out["note"] = "too_few_points"
        return out
    g_s = _moving_average_1d(g, min(21, max(7, g.size // 80)))
    peak_idx = None
    for i in range(1, g_s.size - 1):
        if r[i] < 0.08:
            continue
        if g_s[i] > g_s[i - 1] and g_s[i] > g_s[i + 1] and g_s[i] > 1.05:
            peak_idx = i
            break
    if peak_idx is None:
        valid = np.where(r >= 0.08)[0]
        if valid.size == 0:
            out["note"] = "no_valid_peak_region"
            return out
        peak_idx = int(valid[np.argmax(g_s[valid])])
        out["confidence"] = "low"
        out["status"] = "candidate_only"
        out["note"] = "used_global_peak_fallback"
    else:
        out["confidence"] = "medium"
        out["status"] = "candidate_only"
    out["r_peak_nm"] = float(r[peak_idx])
    out["g_peak"] = float(g[peak_idx])
    min_idx = None
    for j in range(peak_idx + 1, g_s.size - 1):
        if g_s[j] <= g_s[j - 1] and g_s[j] <= g_s[j + 1]:
            min_idx = j
            break
    if min_idx is None:
        out["note"] = (str(out["note"]) + "; " if out["note"] else "") + "no_post_peak_minimum"
        return out
    out["r_shell_nm"] = float(r[min_idx])
    out["cn_shell"] = float(np.interp(float(r[min_idx]), r, cn))
    if out["confidence"] == "medium":
        out["confidence"] = "high"
        out["status"] = "ok"
    return out


def _integrate_cn(r_nm: np.ndarray, g_r: np.ndarray, rho_target_nm3: float) -> np.ndarray:
    r = np.asarray(r_nm, dtype=float)
    g = np.asarray(g_r, dtype=float)
    if r.size == 0:
        return np.zeros(0, dtype=float)
    dr = np.gradient(r)
    return 4.0 * np.pi * float(rho_target_nm3) * np.cumsum(g * r * r * dr)


def compute_site_rdf(
    *,
    gro_path: Path,
    xtc_path: Path,
    center_indices_0: Sequence[int],
    target_indices_0: Sequence[int],
    bin_nm: float,
    r_max_nm: float,
    chunk: int = 10,
    region: str = "global",
) -> dict[str, Any]:
    try:
        import mdtraj as md
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"mdtraj is required for site-level RDF: {exc}") from exc

    center = np.asarray(sorted(set(int(i) for i in center_indices_0)), dtype=int)
    target = np.asarray(sorted(set(int(i) for i in target_indices_0)), dtype=int)
    if center.size == 0 or target.size == 0:
        empty = np.zeros(0, dtype=float)
        return {
            "r_nm": empty,
            "g_r": empty,
            "cn_curve": empty,
            "rho_target_nm3": 0.0,
            "n_frames": 0,
            "n_ref": int(center.size),
            "n_target": int(target.size),
            "shell": detect_first_shell(empty, empty, empty),
        }

    region_token = str(region or "global").strip().lower()
    nbins = max(8, int(np.ceil(float(r_max_nm) / float(bin_nm))))
    edges = np.linspace(0.0, float(r_max_nm), nbins + 1)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    hist = np.zeros(nbins, dtype=float)
    frame_count = 0
    volume_samples: list[np.ndarray] = []
    density_samples: list[float] = []
    effective_ref_count = 0.0

    max_pairs = 200000
    block_size = max(1, int(max_pairs // max(1, center.size)))
    target_blocks = [target[i : i + block_size] for i in range(0, target.size, block_size)]

    for trj in md.iterload(str(xtc_path), top=str(gro_path), chunk=int(max(1, chunk))):
        frame_count += int(trj.n_frames)
        box_lengths = np.asarray(getattr(trj, "unitcell_lengths", None), dtype=float)
        if box_lengths.ndim == 2 and box_lengths.shape[0] == int(trj.n_frames) and box_lengths.shape[1] >= 3:
            try:
                volume_samples.append(np.prod(np.asarray(box_lengths[:, :3], dtype=float), axis=1))
            except Exception:
                pass
        if region_token == "global":
            for tgt_block in target_blocks:
                pairs = []
                for ci in center:
                    for tj in tgt_block:
                        if int(ci) != int(tj):
                            pairs.append((int(ci), int(tj)))
                if not pairs:
                    continue
                pair_arr = np.asarray(pairs, dtype=int)
                dist = md.compute_distances(trj, pair_arr, periodic=True)
                hist += np.histogram(dist.reshape(-1), bins=edges)[0]
            effective_ref_count += float(int(trj.n_frames) * int(center.size))
            continue

        xyz = np.asarray(trj.xyz, dtype=float)
        for fi in range(int(trj.n_frames)):
            box = np.asarray(box_lengths[fi, :3], dtype=float)
            if box.size < 3:
                continue
            z = xyz[fi, :, 2]
            if region_token == "bulk_core":
                lo = 0.25 * float(box[2])
                hi = 0.75 * float(box[2])
                mask = (z >= lo) & (z <= hi)
                region_volume = float(box[0] * box[1] * max(hi - lo, 1.0e-12))
            elif region_token == "interface_shell":
                lo = 0.15 * float(box[2])
                hi = 0.85 * float(box[2])
                mask = (z <= lo) | (z >= hi)
                region_volume = float(box[0] * box[1] * max(float(box[2]) - max(hi - lo, 0.0), 1.0e-12))
            else:
                mask = np.ones(xyz.shape[1], dtype=bool)
                region_volume = float(box[0] * box[1] * box[2])
            centers_f = center[mask[center]]
            targets_f = target[mask[target]]
            if centers_f.size == 0 or targets_f.size == 0:
                continue
            c = xyz[fi, centers_f, :]
            t = xyz[fi, targets_f, :]
            delta = c[:, None, :] - t[None, :, :]
            delta -= box[None, None, :] * np.round(delta / np.maximum(box[None, None, :], 1.0e-12))
            pair_mask = centers_f[:, None] != targets_f[None, :]
            if not np.any(pair_mask):
                continue
            dist = np.linalg.norm(delta[pair_mask], axis=1).reshape(-1)
            hist += np.histogram(dist, bins=edges)[0]
            effective_ref_count += float(centers_f.size)
            density_samples.append(float(targets_f.size) / max(region_volume, 1.0e-12))

    mean_vol = float(np.mean(np.concatenate(volume_samples))) if volume_samples else 1.0
    if region_token == "global":
        rho_target_nm3 = float(target.size) / max(mean_vol, 1.0e-12)
    else:
        rho_target_nm3 = float(np.mean(density_samples)) if density_samples else 0.0
    denom = max(float(effective_ref_count) * max(rho_target_nm3, 1.0e-12), 1.0e-12)
    g_r = hist / (denom * shell_vol)
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    cn_curve = _integrate_cn(r_mid, g_r, rho_target_nm3)
    shell = detect_first_shell(r_mid, g_r, cn_curve)
    return {
        "r_nm": r_mid,
        "g_r": g_r,
        "cn_curve": cn_curve,
        "rho_target_nm3": float(rho_target_nm3),
        "n_frames": int(frame_count),
        "n_ref": int(center.size),
        "n_target": int(target.size),
        "region": region_token,
        "shell": shell,
    }


def build_ne_conductivity_from_msd(*, msd_payload: dict[str, Any], volume_nm3: float, temp_k: float) -> dict[str, Any]:
    e_c = 1.602176634e-19
    k_b = 1.380649e-23
    vol_m3 = float(volume_nm3) * 1.0e-27
    components: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    risk_annotations: list[dict[str, Any]] = []

    def _append_mobile_ion_risk(*, moltype: str, status: Any, alpha_mean: Any, confidence: Any, reason: Any = None) -> None:
        status_s = str(status or "")
        alpha_val = None
        try:
            alpha_val = float(alpha_mean) if alpha_mean is not None else None
        except Exception:
            alpha_val = None
        if status_s not in {"subdiffusive_risk", "no_formal_diffusion"}:
            return
        risk_annotations.append(
            {
                "moltype": str(moltype),
                "status": status_s,
                "alpha_mean": alpha_val,
                "confidence": str(confidence or ""),
                "reason": str(reason or "mobile ions remain subdiffusive; N-E used the log-log MSD window whose slope is closest to 1"),
            }
        )

    for moltype, record in (msd_payload or {}).items():
        if str(moltype).startswith("_") or not isinstance(record, dict):
            continue
        kind = str(record.get("kind") or "")
        natoms = int(record.get("natoms") or 0)
        n_molecules = int(record.get("n_molecules") or 0)
        formal_q = float(record.get("formal_charge_e") or 0.0)
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
        charged_metric = metrics.get("charged_group_com_msd") if isinstance(metrics, dict) else None
        is_polymer = bool(kind == "polymer")
        has_poly_group_metric = bool(is_polymer and isinstance(charged_metric, dict))
        is_charged_polymer = bool(is_polymer and abs(formal_q) > 1.0e-12)
        if has_poly_group_metric:
            for comp_key, comp in (charged_metric.get("component_metrics") or {}).items():
                d_val = comp.get("D_m2_s")
                if d_val is None:
                    ignored.append(
                        {
                            "moltype": str(moltype),
                            "component_key": str(comp_key),
                            "component_kind": "polymer_charged_group",
                            "reason": "no_formal_diffusion_coefficient",
                        }
                    )
                    continue
                q_e = float(comp.get("formal_charge_e") or 0.0)
                count = int(comp.get("n_groups") or 0)
                if count <= 0 or abs(q_e) < 1.0e-12:
                    continue
                term = (float(count) * (q_e * e_c) ** 2 * float(d_val)) / (max(float(vol_m3), 1.0e-300) * k_b * float(temp_k))
                components.append(
                    {
                        "moltype": str(moltype),
                        "component_key": str(comp_key),
                        "component_label": str(comp.get("component_label") or comp_key),
                        "component_kind": "polymer_charged_group",
                        "component_semantics": "polymer_charged_group_self_ne_contribution",
                        "interpretation": "self_upper_bound",
                        "charge_sign": str(comp.get("charge_sign") or "neutral"),
                        "count": count,
                        "charge_e": q_e,
                        "D_m2_s": float(d_val),
                        "msd_status": comp.get("status"),
                        "msd_confidence": comp.get("confidence"),
                        "msd_alpha_mean": comp.get("alpha_mean"),
                        "sigma_component_S_m": float(term),
                        "sigma_component_upper_bound_S_m": float(term),
                    }
                )
        if has_poly_group_metric:
            continue
        if is_charged_polymer:
            ignored.append(
                {
                    "moltype": str(moltype),
                    "component_kind": "polymer_charged_group",
                    "reason": "charged polymer requires charged_group_com_msd; whole-chain conductivity is disabled",
                }
            )
            continue

        default_metric = str(record.get("default_metric") or "")
        default_rec = metrics.get(default_metric) if isinstance(metrics, dict) else None
        d_val = default_rec.get("D_m2_s") if isinstance(default_rec, dict) else record.get("D_m2_s")
        if abs(formal_q) < 1.0e-12:
            continue
        if d_val is None:
            ignored.append(
                {
                    "moltype": str(moltype),
                    "component_kind": "species_default",
                    "reason": "no_formal_diffusion_coefficient",
                    "status": default_rec.get("status") if isinstance(default_rec, dict) else None,
                    "confidence": default_rec.get("confidence") if isinstance(default_rec, dict) else None,
                    "alpha_mean": default_rec.get("alpha_mean") if isinstance(default_rec, dict) else None,
                }
            )
            if not is_polymer:
                _append_mobile_ion_risk(
                    moltype=str(moltype),
                    status=default_rec.get("status") if isinstance(default_rec, dict) else None,
                    alpha_mean=default_rec.get("alpha_mean") if isinstance(default_rec, dict) else None,
                    confidence=default_rec.get("confidence") if isinstance(default_rec, dict) else None,
                    reason="mobile ions remain subdiffusive; no formal MSD diffusion window was available for N-E",
                )
            continue
        comp = {
            "moltype": str(moltype),
            "component_kind": "species_default",
            "component_semantics": "species_self_ne_contribution",
            "interpretation": "self_upper_bound",
            "count": n_molecules,
            "charge_e": float(formal_q),
            "D_m2_s": float(d_val),
            "msd_status": default_rec.get("status") if isinstance(default_rec, dict) else None,
            "msd_confidence": default_rec.get("confidence") if isinstance(default_rec, dict) else None,
            "msd_alpha_mean": default_rec.get("alpha_mean") if isinstance(default_rec, dict) else None,
        }
        if not is_polymer:
            _append_mobile_ion_risk(
                moltype=str(moltype),
                status=comp.get("msd_status"),
                alpha_mean=comp.get("msd_alpha_mean"),
                confidence=comp.get("msd_confidence"),
            )
        if abs(formal_q) >= 5.0 and natoms >= 30:
            comp["ignored_as_polyionic_macromolecule"] = True
            comp["reason"] = "large net charge on a large molecule; no charged-group MSD available"
            ignored.append(comp)
            continue
        term = (float(n_molecules) * (formal_q * e_c) ** 2 * float(d_val)) / (max(float(vol_m3), 1.0e-300) * k_b * float(temp_k))
        comp["sigma_component_S_m"] = float(term)
        comp["sigma_component_upper_bound_S_m"] = float(term)
        components.append(comp)

    sigma = float(sum(float(c.get("sigma_component_S_m", 0.0)) for c in components))
    polymer_self_term = float(
        sum(
            float(c.get("sigma_component_S_m", 0.0))
            for c in components
            if str(c.get("component_kind") or "") == "polymer_charged_group"
        )
    )
    mobile_ion_subdiffusive_risk = bool(risk_annotations)
    risk_note = None
    if mobile_ion_subdiffusive_risk:
        risk_note = (
            "risk: mobile ions remain subdiffusive; N-E used the log-log MSD window whose slope is closest to 1"
        )
    sigma_display = f"{sigma:.3e} S/m"
    if risk_note:
        sigma_display += f" ({risk_note})"
    return {
        "sigma_S_m": sigma,
        "sigma_ne_upper_bound_S_m": sigma,
        "sigma_ne_upper_bound_display": sigma_display,
        "sigma_ne_upper_bound_note": risk_note,
        "NE_is_upper_bound": True,
        "mobile_ion_subdiffusive_risk": mobile_ion_subdiffusive_risk,
        "risk_annotations": risk_annotations,
        "polymer_charged_group_self_ne_contribution_S_m": polymer_self_term,
        "temperature_K": float(temp_k),
        "volume_nm3": float(volume_nm3),
        "components": components,
        "ignored_components": ignored,
    }
