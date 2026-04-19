"""Reference parsing and audit helpers for the YadonPy OPLS-AA implementation."""

from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from rdkit import Chem

from ..core.resources import ff_data_path
from ..io.gromacs_top import defaults_for_ff_name


_KCAL_TO_KJ = 4.184
_ANGSTROM_TO_NM = 0.1
_REVERSE_TYPE_MAP = {
    "C°": "C:",
    "C^": "C$",
    "N^": "N$",
    "O^": "O$",
    "C|": "C#",
    "N§": "N*",
    "C⟮": "C(O)",
    "??": "X",
}
_CURRENT_LOCAL_REFINES = (
    {
        "kind": "donor_h_lj_floor",
        "classification": "unsafe_heuristic",
        "details": "Hydroxyl donor hydrogen LJ floor is still materialized locally in OPLS-AA typing.",
        "source": "local_refine",
    },
    {
        "kind": "special_bonded_fallbacks",
        "classification": "local_refine",
        "details": "Token-based bonded fallbacks in oplsaa.py should be reviewed against Moltemplate/GROMACS references.",
        "source": "local_refine",
    },
)
_GROMACS_OPLSAA_DEFAULTS = {
    "nbfunc": 1,
    "comb_rule": 3,
    "gen_pairs": "yes",
    "fudge_lj": 0.5,
    "fudge_qq": 0.5,
    "source": "gromacs_oplsaa",
}
_GROMACS_IMPROPER_TEMPLATE_DEFAULTS = {
    "improper_O_C_X_Y": {
        "tag": "improper_O_C_X_Y",
        "phase_deg": 180.0,
        "k": 43.93200,
        "multiplicity": 2,
        "d": -1,
        "n": 2,
        "source": "gromacs_oplsaa",
    },
    "improper_Z_N_X_Y": {
        "tag": "improper_Z_N_X_Y",
        "phase_deg": 180.0,
        "k": 4.18400,
        "multiplicity": 2,
        "d": -1,
        "n": 2,
        "source": "gromacs_oplsaa",
    },
    "improper_Z_CA_X_Y": {
        "tag": "improper_Z_CA_X_Y",
        "phase_deg": 180.0,
        "k": 4.60240,
        "multiplicity": 2,
        "d": -1,
        "n": 2,
        "source": "gromacs_oplsaa",
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_btype(token: str) -> str:
    return _REVERSE_TYPE_MAP.get(str(token).strip(), str(token).strip())


def _strip_comment(line: str) -> str:
    return line.split(";", 1)[0].strip()


def _find_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _extract_source_from_row(row: dict[str, Any]) -> str | None:
    source = row.get("source")
    if source:
        return str(source)
    desc = str(row.get("desc") or "")
    match = re.search(r"source=([A-Za-z0-9_.+-]+)", desc)
    if match:
        return match.group(1)
    return None


def _load_current_oplsaa_db() -> dict[str, Any]:
    return json.loads(ff_data_path("ff_dat", "oplsaa.json").read_text(encoding="utf-8"))


def discover_gromacs_oplsaa_root() -> Path | None:
    env = (
        Path(v).expanduser().resolve()
        for v in (
            str(Path.cwd()),
            str(Path.home() / "GROMACS-2026" / "share" / "gromacs" / "top" / "oplsaa.ff"),
            str(Path.home() / "GROMACS-2025" / "share" / "gromacs" / "top" / "oplsaa.ff"),
            str(Path.home() / "gromacs" / "share" / "gromacs" / "top" / "oplsaa.ff"),
        )
    )
    for path in env:
        if path.is_dir() and path.name == "oplsaa.ff":
            return path
    for key in ("YADONPY_GROMACS_OPLSAA_ROOT", "GMXLIB", "GMXDATA"):
        raw = str(Path((__import__("os").environ.get(key) or "")).expanduser()).strip()
        if not raw:
            continue
        path = Path(raw).resolve()
        if path.is_dir() and path.name == "oplsaa.ff":
            return path
        cand = path / "oplsaa.ff"
        if cand.is_dir():
            return cand
        cand = path / "share" / "gromacs" / "top" / "oplsaa.ff"
        if cand.is_dir():
            return cand
    return None


def discover_moltemplate_reference_paths() -> tuple[Path | None, Path | None]:
    root = _repo_root().parent / "_external" / "moltemplate" / "moltemplate" / "force_fields"
    par = root / "oplsaa2024_original_format" / "Jorgensen_et_al-2024-The_Journal_of_Physical_Chemistry_B.sup-2.par"
    lt = root / "oplsaa2024.lt"
    return (par if par.is_file() else None, lt if lt.is_file() else None)


def _parse_gromacs_defaults(gromacs_root: Path | None) -> dict[str, Any]:
    if gromacs_root is None:
        return dict(_GROMACS_OPLSAA_DEFAULTS)

    forcefield_itp = _find_existing_path(
        [
            gromacs_root / "forcefield.itp",
            gromacs_root / "ffnonbonded.itp",
        ]
    )
    if forcefield_itp is None:
        return dict(_GROMACS_OPLSAA_DEFAULTS)

    section = None
    for raw in forcefield_itp.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip().lower()
            continue
        if section != "defaults":
            continue
        body = _strip_comment(raw)
        if not body:
            continue
        toks = body.split()
        if len(toks) < 5:
            continue
        return {
            "nbfunc": int(toks[0]),
            "comb_rule": int(toks[1]),
            "gen_pairs": str(toks[2]),
            "fudge_lj": float(toks[3]),
            "fudge_qq": float(toks[4]),
            "source": "gromacs_oplsaa",
        }

    return dict(_GROMACS_OPLSAA_DEFAULTS)


def _parse_gromacs_atomtypes(gromacs_root: Path | None) -> dict[str, dict[str, Any]]:
    if gromacs_root is None:
        return {}
    ffnonbonded = _find_existing_path([gromacs_root / "ffnonbonded.itp"])
    if ffnonbonded is None:
        return {}

    atomtypes: dict[str, dict[str, Any]] = {}
    section = None
    for raw in ffnonbonded.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]").strip().lower()
            continue
        if section != "atomtypes":
            continue
        body = _strip_comment(raw)
        if not body:
            continue
        toks = body.split()
        if len(toks) < 6:
            continue
        try:
            atomtypes[toks[0]] = {
                "tag": toks[0],
                "mass": float(toks[1]),
                "sigma": float(toks[4]),
                "epsilon": float(toks[5]),
                "source": "gromacs_oplsaa",
            }
        except Exception:
            continue
    return atomtypes


def _parse_gromacs_improper_templates(gromacs_root: Path | None) -> dict[str, dict[str, Any]]:
    templates = {key: dict(value) for key, value in _GROMACS_IMPROPER_TEMPLATE_DEFAULTS.items()}
    if gromacs_root is None:
        return templates

    ffbonded = _find_existing_path([gromacs_root / "ffbonded.itp"])
    if ffbonded is None:
        return templates

    patt = re.compile(r"^\s*#define\s+(improper_[A-Za-z0-9_]+)\s+([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)\s+(\d+)\s*$")
    for raw in ffbonded.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = patt.match(_strip_comment(raw))
        if not match:
            continue
        tag = match.group(1)
        phase = float(match.group(2))
        k = float(match.group(3))
        multiplicity = int(match.group(4))
        d = None
        if math.isclose(phase % 360.0, 180.0, abs_tol=1.0e-6):
            d = -1
        elif math.isclose(phase % 360.0, 0.0, abs_tol=1.0e-6):
            d = 1
        templates[tag] = {
            "tag": tag,
            "phase_deg": phase,
            "k": k,
            "multiplicity": multiplicity,
            "d": d,
            "n": multiplicity,
            "source": "gromacs_oplsaa",
        }
    return templates


def _parse_gromacs_template_usage(gromacs_root: Path | None) -> dict[str, list[dict[str, Any]]]:
    usage: dict[str, list[dict[str, Any]]] = {
        key: [] for key in _GROMACS_IMPROPER_TEMPLATE_DEFAULTS.keys()
    }
    if gromacs_root is None or not gromacs_root.is_dir():
        return usage

    for rtp in sorted(gromacs_root.glob("*.rtp")):
        residue = None
        section = None
        for raw in rtp.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                label = line.strip("[]").strip()
                low = label.lower()
                if low in {"atoms", "bonds", "angles", "dihedrals", "impropers", "cmap"}:
                    section = low
                else:
                    residue = label
                    section = None
                continue
            if section != "impropers":
                continue
            body = _strip_comment(raw)
            if not body:
                continue
            toks = body.split()
            if len(toks) < 5:
                continue
            macro = toks[-1]
            if macro not in usage:
                continue
            usage[macro].append(
                {
                    "residue": residue,
                    "atoms": toks[:4],
                    "source": "gromacs_oplsaa",
                }
            )
    return usage


def _parse_moltemplate_reference(par_path: Path | None, lt_path: Path | None) -> dict[str, dict[str, Any]]:
    refs = {"bond_types": {}, "angle_types": {}, "dihedral_types": {}}
    if lt_path is None or not lt_path.is_file():
        return refs

    text = lt_path.read_text(encoding="utf-8", errors="ignore")
    bond_re = re.compile(r"^\s*bond_coeff\s+@bond:([^\s]+)\s+([^\s]+)\s+([^\s]+)(?:\s+#\s*(.*))?$", re.M)
    angle_re = re.compile(r"^\s*angle_coeff\s+@angle:([^\s]+)\s+([^\s]+)\s+([^\s]+)(?:\s+#\s*(.*))?$", re.M)
    dihedral_re = re.compile(
        r"^\s*dihedral_coeff\s+@dihedral:([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)(?:\s+#\s*(.*))?$",
        re.M,
    )

    for match in bond_re.finditer(text):
        tokens = [_normalize_btype(tok) for tok in match.group(1).split("_")]
        if len(tokens) != 2:
            continue
        tag = ",".join(tokens)
        refs["bond_types"][tag] = {
            "tag": tag,
            "k": float(match.group(2)) * _KCAL_TO_KJ * 400.0,
            "r0": float(match.group(3)) * _ANGSTROM_TO_NM,
            "source": "moltemplate_oplsaa2024",
        }

    for match in angle_re.finditer(text):
        tokens = [_normalize_btype(tok) for tok in match.group(1).split("_")]
        if len(tokens) != 3:
            continue
        tag = ",".join(tokens)
        refs["angle_types"][tag] = {
            "tag": tag,
            "k": float(match.group(2)) * _KCAL_TO_KJ * 4.0,
            "theta0": float(match.group(3)),
            "source": "moltemplate_oplsaa2024",
        }

    for match in dihedral_re.finditer(text):
        tokens = [_normalize_btype(tok) for tok in match.group(1).split("_")]
        if len(tokens) != 4:
            continue
        tag = ",".join(tokens)
        coeffs = [float(match.group(i)) for i in range(2, 6)]
        ks = []
        ds = []
        ns = []
        for idx, coeff in enumerate(coeffs, start=1):
            if math.isclose(coeff, 0.0, abs_tol=1.0e-12):
                continue
            base_phase = 0 if idx % 2 == 1 else 180
            phase = base_phase if coeff >= 0.0 else (180 if base_phase == 0 else 0)
            ks.append(abs(coeff) * _KCAL_TO_KJ / 2.0)
            ds.append(phase)
            ns.append(idx)
        refs["dihedral_types"][tag] = {
            "tag": tag,
            "k": ks,
            "d": ds,
            "n": ns,
            "m": len(ks),
            "source": "moltemplate_oplsaa2024",
        }

    return refs


def get_oplsaa_improper_templates(gromacs_root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    root = Path(gromacs_root).expanduser().resolve() if gromacs_root else discover_gromacs_oplsaa_root()
    return _parse_gromacs_improper_templates(root)


def _canonical_improper_atoms(a: int, b: int, c: int, d: int) -> tuple[int, int, int, int]:
    outer = sorted((int(a), int(b), int(d)))
    return (outer[0], outer[1], int(c), outer[2])


def _atom_btype(atom) -> str:
    if atom.HasProp("ff_btype"):
        return str(atom.GetProp("ff_btype"))
    return atom.GetSymbol()


def _classify_trigonal_carbon(atom) -> str | None:
    if atom.GetSymbol() != "C" or atom.GetDegree() != 3:
        return None
    neigh = list(atom.GetNeighbors())
    o_like = [nb for nb in neigh if nb.GetSymbol() == "O" or _atom_btype(nb) in {"O", "O2", "O~", "OH", "OS"}]
    n_like = [nb for nb in neigh if nb.GetSymbol() == "N" or _atom_btype(nb).startswith("N")]
    if len(o_like) == 3:
        return "carbonate"
    if len(o_like) >= 2:
        return "carboxylate"
    if len(o_like) >= 1 and len(n_like) >= 1:
        return "amide_carbonyl"
    if len(n_like) >= 2:
        return "guanidinium_amidinium"
    return None


def identify_oplsaa_planar_motifs(mol) -> list[dict[str, Any]]:
    motifs: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for atom in mol.GetAtoms():
        trigonal = _classify_trigonal_carbon(atom)
        if trigonal is not None:
            neigh = sorted((nb.GetIdx() for nb in atom.GetNeighbors()))
            atoms = _canonical_improper_atoms(neigh[0], neigh[1], atom.GetIdx(), neigh[2])
            key = (*atoms, "improper_O_C_X_Y")
            if key not in seen:
                seen.add(key)
                motifs.append(
                    {
                        "motif": trigonal,
                        "template": "improper_O_C_X_Y",
                        "atoms": atoms,
                        "source_backed": True,
                    }
                )

        if atom.GetSymbol() != "N" or atom.GetDegree() != 3:
            continue
        neighbors = list(atom.GetNeighbors())
        carbon_neighbors = [nb for nb in neighbors if nb.GetSymbol() == "C" and _classify_trigonal_carbon(nb) is not None]
        if len(carbon_neighbors) != 1:
            continue
        others = sorted((nb.GetIdx() for nb in neighbors if nb.GetIdx() != carbon_neighbors[0].GetIdx()))
        atoms = _canonical_improper_atoms(carbon_neighbors[0].GetIdx(), others[0], atom.GetIdx(), others[1])
        key = (*atoms, "improper_Z_N_X_Y")
        if key in seen:
            continue
        seen.add(key)
        motifs.append(
            {
                "motif": "planar_nitrogen",
                "template": "improper_Z_N_X_Y",
                "atoms": atoms,
                "source_backed": True,
            }
        )

    return motifs


def iter_source_backed_oplsaa_improper_candidates(mol):
    for motif in identify_oplsaa_planar_motifs(mol):
        if motif.get("source_backed"):
            yield motif


def _current_zero_lj_types(current_db: dict[str, Any]) -> set[str]:
    zeros = set()
    for row in current_db.get("particle_types", []):
        sigma = float(row.get("sigma", 0.0))
        epsilon = float(row.get("epsilon", 0.0))
        if sigma <= 0.0 or epsilon <= 0.0:
            zeros.add(str(row.get("tag")))
    return zeros


def _gather_assignment_summary(mol) -> dict[str, Any]:
    current_db = _load_current_oplsaa_db()
    allowed_zero = _current_zero_lj_types(current_db)
    missing_nonbonded = []
    for atom in mol.GetAtoms():
        ff_type = atom.GetProp("ff_type") if atom.HasProp("ff_type") else None
        if not ff_type:
            missing_nonbonded.append({"atom_index": int(atom.GetIdx()), "reason": "missing_ff_type"})
            continue
        has_sigma = atom.HasProp("ff_sigma")
        has_epsilon = atom.HasProp("ff_epsilon")
        if (not has_sigma or not has_epsilon) and ff_type not in allowed_zero:
            missing_nonbonded.append(
                {
                    "atom_index": int(atom.GetIdx()),
                    "ff_type": str(ff_type),
                    "reason": "missing_lj_props",
                }
            )

    assigned_impropers = {
        tuple(int(x) for x in key.split(",")): imp
        for key, imp in (getattr(mol, "impropers", {}) or {}).items()
    }
    missing_motifs = []
    for motif in identify_oplsaa_planar_motifs(mol):
        atoms = tuple(int(x) for x in motif["atoms"])
        if atoms not in assigned_impropers:
            missing_motifs.append(motif)

    return {
        "missing_nonbonded": missing_nonbonded,
        "angles_count": len(getattr(mol, "angles", {}) or {}),
        "dihedrals_count": len(getattr(mol, "dihedrals", {}) or {}),
        "impropers_count": len(getattr(mol, "impropers", {}) or {}),
        "missing_planar_impropers": missing_motifs,
        "assignment_complete": not missing_nonbonded and not missing_motifs,
    }


def _gather_topology_summary(mol) -> dict[str, Any]:
    from ..io.gmx import write_gmx

    with tempfile.TemporaryDirectory(prefix="yadonpy_oplsaa_audit_") as tmpdir:
        _, itp_path, top_path = write_gmx(mol=mol, out_dir=Path(tmpdir), mol_name="AUDIT")
        txt = itp_path.read_text(encoding="utf-8")
        improper_lines = []
        section = None
        for raw in txt.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]").strip().lower()
                continue
            if section != "dihedrals":
                continue
            body = _strip_comment(raw)
            if not body:
                continue
            toks = body.split()
            if len(toks) >= 5 and toks[4] == "4":
                improper_lines.append(body)
        return {
            "itp": str(itp_path),
            "top": str(top_path),
            "improper_dihedral_lines": len(improper_lines),
            "topology_complete": len(improper_lines) >= len(getattr(mol, "impropers", {}) or {}),
        }


def audit_oplsaa_reference(
    *,
    smiles: str | None = None,
    mol=None,
    gromacs_root: str | Path | None = None,
    moltemplate_par_path: str | Path | None = None,
    moltemplate_lt_path: str | Path | None = None,
    charge: str = "opls",
    export_topology: bool = False,
) -> dict[str, Any]:
    """Audit YadonPy's current OPLS-AA data and, optionally, an assigned molecule."""

    current_db = _load_current_oplsaa_db()
    current_atomtypes = {str(row["tag"]): dict(row) for row in current_db.get("particle_types", [])}
    current_bonds = {str(row["tag"]): dict(row) for row in current_db.get("bond_types", [])}
    current_angles = {str(row["tag"]): dict(row) for row in current_db.get("angle_types", [])}
    current_dihedrals = {str(row["tag"]): dict(row) for row in current_db.get("dihedral_types", [])}

    gmx_root = Path(gromacs_root).expanduser().resolve() if gromacs_root else discover_gromacs_oplsaa_root()
    par_default, lt_default = discover_moltemplate_reference_paths()
    par_path = Path(moltemplate_par_path).expanduser().resolve() if moltemplate_par_path else par_default
    lt_path = Path(moltemplate_lt_path).expanduser().resolve() if moltemplate_lt_path else lt_default

    gmx_defaults = _parse_gromacs_defaults(gmx_root)
    gmx_atomtypes = _parse_gromacs_atomtypes(gmx_root)
    gmx_impropers = _parse_gromacs_improper_templates(gmx_root)
    gmx_template_usage = _parse_gromacs_template_usage(gmx_root)
    moltemplate_refs = _parse_moltemplate_reference(par_path, lt_path)

    expected_defaults = defaults_for_ff_name("oplsaa")
    defaults_parity = {
        "current": {
            "comb_rule": int(expected_defaults.comb_rule),
            "gen_pairs": str(expected_defaults.gen_pairs),
            "fudge_lj": float(expected_defaults.fudge_lj),
            "fudge_qq": float(expected_defaults.fudge_qq),
        },
        "reference": gmx_defaults,
        "matches": (
            int(expected_defaults.comb_rule) == int(gmx_defaults["comb_rule"])
            and str(expected_defaults.gen_pairs).lower() == str(gmx_defaults["gen_pairs"]).lower()
            and math.isclose(float(expected_defaults.fudge_lj), float(gmx_defaults["fudge_lj"]), abs_tol=1.0e-12)
            and math.isclose(float(expected_defaults.fudge_qq), float(gmx_defaults["fudge_qq"]), abs_tol=1.0e-12)
        ),
    }

    atomtype_mismatches = []
    for tag, gmx in sorted(gmx_atomtypes.items()):
        cur = current_atomtypes.get(tag)
        if cur is None:
            continue
        if (
            not math.isclose(float(cur.get("mass", 0.0)), float(gmx["mass"]), abs_tol=1.0e-6)
            or not math.isclose(float(cur.get("sigma", 0.0)), float(gmx["sigma"]), abs_tol=1.0e-6)
            or not math.isclose(float(cur.get("epsilon", 0.0)), float(gmx["epsilon"]), abs_tol=1.0e-6)
        ):
            atomtype_mismatches.append(
                {
                    "tag": tag,
                    "current": {
                        "mass": float(cur.get("mass", 0.0)),
                        "sigma": float(cur.get("sigma", 0.0)),
                        "epsilon": float(cur.get("epsilon", 0.0)),
                        "source": _extract_source_from_row(cur),
                    },
                    "reference": gmx,
                }
            )

    def _compare_bonded(current_map, ref_map, value_keys):
        mismatches = []
        for tag, ref in sorted(ref_map.items()):
            cur = current_map.get(tag)
            if cur is None:
                continue
            mismatch = False
            for key in value_keys:
                cval = cur.get(key)
                rval = ref.get(key)
                if isinstance(cval, list):
                    if [float(x) for x in cval] != [float(x) for x in rval]:
                        mismatch = True
                        break
                else:
                    if not math.isclose(float(cval), float(rval), abs_tol=1.0e-6):
                        mismatch = True
                        break
            if mismatch:
                mismatches.append(
                    {
                        "tag": tag,
                        "current": {key: cur.get(key) for key in value_keys},
                        "reference": {key: ref.get(key) for key in value_keys},
                        "source": _extract_source_from_row(cur),
                    }
                )
        return mismatches

    bond_angle_dihedral_parity = {
        "bond_mismatches": _compare_bonded(current_bonds, moltemplate_refs["bond_types"], ("r0", "k")),
        "angle_mismatches": _compare_bonded(current_angles, moltemplate_refs["angle_types"], ("theta0", "k")),
        "dihedral_mismatches": _compare_bonded(current_dihedrals, moltemplate_refs["dihedral_types"], ("k", "d", "n")),
        "source_priority": {
            "defaults_nonbonded": "gromacs_oplsaa",
            "bonded": "moltemplate_oplsaa2024",
            "impropers": "gromacs_oplsaa_templates",
        },
    }

    improper_template_parity = {
        "available_in_gromacs": sorted(gmx_impropers.keys()),
        "template_usage": gmx_template_usage,
        "current_runtime_templates": sorted(_GROMACS_IMPROPER_TEMPLATE_DEFAULTS.keys()),
        "matches": set(_GROMACS_IMPROPER_TEMPLATE_DEFAULTS.keys()).issubset(set(gmx_impropers.keys())),
    }

    report: dict[str, Any] = {
        "defaults_parity": defaults_parity,
        "atomtype_lj_parity": {
            "gromacs_atomtype_count": len(gmx_atomtypes),
            "current_atomtype_count": len(current_atomtypes),
            "mismatches": atomtype_mismatches,
        },
        "bond_angle_dihedral_parity": bond_angle_dihedral_parity,
        "improper_template_parity": improper_template_parity,
        "missing": {
            "current_improper_types_in_json": len(current_db.get("improper_types", [])),
            "gromacs_templates_without_runtime_support": sorted(
                set(gmx_impropers.keys()) - set(_GROMACS_IMPROPER_TEMPLATE_DEFAULTS.keys())
            ),
        },
        "overridden": [],
        "locally_patched": list(_CURRENT_LOCAL_REFINES),
    }

    assigned_mol = mol
    if assigned_mol is None and smiles:
        from .oplsaa import OPLSAA

        assigned_mol = OPLSAA().ff_assign(Chem.AddHs(Chem.MolFromSmiles(smiles)), charge=charge, report=False)
    if assigned_mol:
        report["assignment"] = _gather_assignment_summary(assigned_mol)
        if export_topology:
            report["topology"] = _gather_topology_summary(assigned_mol)
    return report


__all__ = [
    "audit_oplsaa_reference",
    "discover_gromacs_oplsaa_root",
    "discover_moltemplate_reference_paths",
    "get_oplsaa_improper_templates",
    "identify_oplsaa_planar_motifs",
    "iter_source_backed_oplsaa_improper_candidates",
]
