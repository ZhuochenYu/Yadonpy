"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple



from .units import angstrom_to_nm


def _gro_wrap_index(value: int) -> int:
    value = int(value)
    if value < 0:
        return -((-value) % 100000)
    return value % 100000


def _format_gro_atom_line(*, resnr: int, resname: str, atomname: str, atomnr: int, x: float, y: float, z: float) -> str:
    return (
        f"{_gro_wrap_index(resnr):5d}{str(resname)[:5]:<5}{str(atomname)[:5]:>5}{_gro_wrap_index(atomnr):5d}"
        f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
    )


def _canon_angle_key(i: int, j: int, k: int) -> tuple[int, int, int]:
    """Canonicalize an angle key so that i-k ordering does not matter."""
    return (min(i, k), j, max(i, k))


def _parse_itp_bonds_angles(
    path: Path,
) -> tuple[dict[tuple[int, int], tuple[float, float]], dict[tuple[int, int, int], tuple[float, float]]]:
    """Parse a small .itp fragment and extract bond/angle parameters.

    The fragment is expected to contain (optionally) `[ bonds ]` and `[ angles ]`
    sections with explicit i,j,... indices.

    Returns:
        bond_map: (i,j) -> (r0_nm, k_kj_per_nm2)
        angle_map: (i,j,k) -> (theta0_deg, k_kj_per_rad2)
    """

    bond_map: dict[tuple[int, int], tuple[float, float]] = {}
    angle_map: dict[tuple[int, int, int], tuple[float, float]] = {}

    try:
        txt = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return bond_map, angle_map

    sec = None
    for raw in txt:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line.strip("[]").strip().lower()
            continue

        # Strip inline comments
        if ";" in line:
            line = line.split(";", 1)[0].strip()
        if not line:
            continue
        toks = line.split()

        if sec == "bonds" and len(toks) >= 5:
            try:
                i = int(toks[0])
                j = int(toks[1])
                r0_nm = float(toks[3])
                k = float(toks[4])
                key = (min(i, j), max(i, j))
                bond_map[key] = (r0_nm, k)
            except Exception:
                continue

        elif sec == "angles" and len(toks) >= 6:
            try:
                i = int(toks[0])
                j = int(toks[1])
                kidx = int(toks[2])
                theta0 = float(toks[4])
                kk = float(toks[5])
                angle_map[_canon_angle_key(i, j, kidx)] = (theta0, kk)
            except Exception:
                continue

    return bond_map, angle_map


def _apply_bond_angle_patch_from_fragment(itp_path: Path, fragment_path: Path, *, method: str = "mseminario") -> bool:
    """Patch an existing molecule .itp's [ bonds ]/[ angles ] using a fragment.

    This is used to inject modified-Seminario bond/angle parameters for rigid
    inorganic polyatomic ions (e.g., PF6-), overriding the GAFF-family default
    force constants without duplicating bonds/angles.
    """

    bond_map, angle_map = _parse_itp_bonds_angles(fragment_path)
    if not bond_map and not angle_map:
        return False

    try:
        lines = itp_path.read_text(encoding="utf-8", errors="ignore").splitlines(True)
    except Exception:
        return False

    sec = None
    out: list[str] = []
    changed = False

    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            sec = stripped.strip("[]").strip().lower()
            out.append(raw)
            continue

        # Only patch data lines; keep comments and blank lines verbatim
        if sec in ("bonds", "angles"):
            # split inline comment
            comment = ""
            body = raw
            if ";" in raw:
                body, comment = raw.split(";", 1)
                comment = ";" + comment
            body_s = body.strip()
            if not body_s:
                out.append(raw)
                continue

            toks = body_s.split()
            try:
                if sec == "bonds" and len(toks) >= 5:
                    i = int(toks[0])
                    j = int(toks[1])
                    key = (min(i, j), max(i, j))
                    if key in bond_map:
                        r0_nm, k = bond_map[key]
                        toks[3] = f"{r0_nm:.6f}"
                        toks[4] = f"{k:.2f}"
                        changed = True
                        out.append(" ".join(toks) + ("  " if comment else "") + comment.rstrip("\n") + "\n")
                        continue

                elif sec == "angles" and len(toks) >= 6:
                    i = int(toks[0])
                    j = int(toks[1])
                    kidx = int(toks[2])
                    key = _canon_angle_key(i, j, kidx)
                    if key in angle_map:
                        th0, kk = angle_map[key]
                        toks[4] = f"{th0:.3f}"
                        toks[5] = f"{kk:.2f}"
                        changed = True
                        out.append(" ".join(toks) + ("  " if comment else "") + comment.rstrip("\n") + "\n")
                        continue
            except Exception:
                pass

            out.append(raw)
        else:
            out.append(raw)

    if changed:
        # Add a small marker comment at the top (non-invasive)
        try:
            marker = f"; patched by yadonpy {method} (bond+angle)\n"
            if out and not out[0].lstrip().startswith(f"; patched by yadonpy {method}"):
                out.insert(0, marker)
        except Exception:
            pass
        try:
            itp_path.write_text("".join(out), encoding="utf-8")
        except Exception:
            return False
    return changed


def _get_atom_charge(atom) -> float:
    if atom.HasProp("AtomicCharge"):
        return float(atom.GetDoubleProp("AtomicCharge"))
    if atom.HasProp("RESP"):
        return float(atom.GetDoubleProp("RESP"))
    return 0.0


def _get_atom_mass(atom) -> float:
    if atom.HasProp("ff_mass"):
        try:
            return float(atom.GetDoubleProp("ff_mass"))
        except Exception:
            pass
    # Fallback: periodic table
    try:
        from rdkit.Chem import GetPeriodicTable

        return float(GetPeriodicTable().GetAtomicWeight(atom.GetAtomicNum()))
    except Exception:
        # Last resort: guess by atomic number
        return float(atom.GetAtomicNum())


def _atom_residue_fields(atom, *, atom_index: int, default_resnr: int, default_resname: str):
    info = None
    try:
        info = atom.GetPDBResidueInfo()
    except Exception:
        info = None
    if info is not None:
        try:
            resnr = int(info.GetResidueNumber())
        except Exception:
            resnr = int(default_resnr)
        resname = str(info.GetResidueName()).strip() or str(default_resname)
        atomname = str(info.GetName()).strip() or f"{atom.GetSymbol()}{atom_index}"
        return resnr, resname, atomname
    return int(default_resnr), str(default_resname), f"{atom.GetSymbol()}{atom_index}"


def write_gro_from_rdkit(mol, out_gro: Path, mol_name: str) -> None:
    """Write a minimal .gro from RDKit conformer (expects Angstrom coords)."""

    conf = mol.GetConformer()
    n = mol.GetNumAtoms()

    lines: List[str] = []
    lines.append(f"{mol_name}\n")
    lines.append(f"{n:5d}\n")
    for i, atom in enumerate(mol.GetAtoms(), start=1):
        pos = conf.GetAtomPosition(i - 1)
        x = angstrom_to_nm(pos.x)
        y = angstrom_to_nm(pos.y)
        z = angstrom_to_nm(pos.z)
        resnr, resname, aname = _atom_residue_fields(
            atom,
            atom_index=i,
            default_resnr=1,
            default_resname=mol_name[:5],
        )
        lines.append(_format_gro_atom_line(resnr=resnr, resname=resname, atomname=aname, atomnr=i, x=x, y=y, z=z) + "\n")

    # A dummy box (1 nm cube) - caller may override for systems.
    lines.append("   1.00000   1.00000   1.00000\n")
    out_gro.write_text("".join(lines), encoding="utf-8")


def write_gromacs_single_molecule_topology(
    mol,
    out_dir: Path,
    *,
    mol_name: str,
) -> Tuple[Path, Path, Path]:
    """Write self-contained .itp/.top/.gro for a single molecule.

    The molecule must already have GAFF-family properties assigned:
      - atom: ff_type, ff_sigma, ff_epsilon
      - bond: ff_type, ff_k, ff_r0
      - mol.angles dict (Angle_harmonic)
      - mol.dihedrals dict (Dihedral_fourier/Dihedral_harmonic)

    Returns:
        (gro_path, itp_path, top_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    gro_path = out_dir / f"{mol_name}.gro"
    itp_path = out_dir / f"{mol_name}.itp"
    top_path = out_dir / f"{mol_name}.top"

    # 1) GRO
    write_gro_from_rdkit(mol, gro_path, mol_name)

    # 2) ITP
    atomtypes: Dict[str, Dict[str, float]] = {}
    atoms_lines: List[str] = []
    for idx, atom in enumerate(mol.GetAtoms(), start=1):
        atype = atom.GetProp("ff_type") if atom.HasProp("ff_type") else atom.GetSymbol()
        sigma = float(atom.GetDoubleProp("ff_sigma")) if atom.HasProp("ff_sigma") else 0.0
        eps = float(atom.GetDoubleProp("ff_epsilon")) if atom.HasProp("ff_epsilon") else 0.0
        mass = _get_atom_mass(atom)
        charge = _get_atom_charge(atom)

        atomtypes.setdefault(
            atype,
            {
                "mass": mass,
                "sigma_nm": float(sigma),
                "epsilon_kj": float(eps),
            },
        )

        resnr, resname, aname = _atom_residue_fields(
            atom,
            atom_index=idx,
            default_resnr=1,
            default_resname=mol_name,
        )
        cgnr = idx
        atoms_lines.append(
            f"{idx:5d} {atype:<6} {resnr:5d} {resname:<8} {aname:<8} {cgnr:5d} {charge: .6f} {mass: .4f}\n"
        )

    # Bonds
    bonds_lines: List[str] = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx() + 1
        j = b.GetEndAtomIdx() + 1
        # funct 1: harmonic
        r0 = float(b.GetDoubleProp("ff_r0")) if b.HasProp("ff_r0") else 0.0
        k = float(b.GetDoubleProp("ff_k")) if b.HasProp("ff_k") else 0.0
        bonds_lines.append(
            f"{i:5d} {j:5d}  1  {float(r0): .6f}  {float(k): .2f}\n"
        )


    # Bonded terms (angles/dihedrals) are expected to be present on `mol`.
    # For mixed-system exports, ensure they are rebuilt *before* calling this writer.

    # Angles
    angles_lines: List[str] = []
    ang_dict = getattr(mol, "angles", {}) or {}
    for ang in ang_dict.values():
        i = int(ang.a) + 1
        j = int(ang.b) + 1
        kidx = int(ang.c) + 1
        k_ang = float(ang.ff.k)
        angles_lines.append(
            f"{i:5d} {j:5d} {kidx:5d}  1  {ang.ff.theta0: .3f}  {k_ang: .2f}\n"
        )

    # Dihedrals
    dihedrals_lines: List[str] = []
    dih_dict = getattr(mol, "dihedrals", {}) or {}
    for dih in dih_dict.values():
        i = int(dih.a) + 1
        j = int(dih.b) + 1
        kidx = int(dih.c) + 1
        l = int(dih.d) + 1
        ff = dih.ff
        # Proper dihedral periodic in GROMACS: funct=1, phi0, kphi, mult
        if hasattr(ff, "k") and hasattr(ff, "n") and hasattr(ff, "d0"):
            # Dihedral_fourier stores arrays (multi-term). Dihedral_harmonic stores scalars.
            try:
                ks = list(ff.k)  # type: ignore
                ns = list(ff.n)  # type: ignore
                ds = list(ff.d0)  # type: ignore
                # If any of these are non-iterable scalars, list(...) will raise.
                if len(ks) == 0 or len(ns) == 0 or len(ds) == 0:
                    continue
                for kk, nn, dd in zip(ks, ns, ds):
                    dihedrals_lines.append(
                        f"{i:5d} {j:5d} {kidx:5d} {l:5d}  1  {float(dd): .1f}  {float(kk): .4f}  {int(nn)}\n"
                    )
            except Exception:
                # Scalar fallback
                try:
                    dihedrals_lines.append(
                        f"{i:5d} {j:5d} {kidx:5d} {l:5d}  1  {float(getattr(ff,'d0')): .1f}  {float(getattr(ff,'k')): .4f}  {int(getattr(ff,'n'))}\n"
                    )
                except Exception:
                    continue
        else:
            continue

    itp = []
    itp.append("[ atomtypes ]\n")
    itp.append("; name  mass    charge  ptype  sigma(nm)  epsilon(kJ/mol)\n")
    for name, prm in sorted(atomtypes.items()):
        # charge here is 0 for atomtype; per-atom charges are in [ atoms ]
        itp.append(
            f"{name:<6} {prm['mass']:8.4f}  0.0000  A  {prm['sigma_nm']: .6f}  {prm['epsilon_kj']: .6f}\n"
        )
    itp.append("\n")

    itp.append("[ moleculetype ]\n")
    itp.append("; Name  nrexcl\n")
    itp.append(f"{mol_name:<12}  3\n\n")

    itp.append("[ atoms ]\n")
    itp.append("; nr  type  resnr  residue  atom  cgnr  charge  mass\n")
    itp.extend(atoms_lines)
    itp.append("\n")

    itp.append("[ bonds ]\n")
    itp.append("; i  j  funct  r0(nm)  k(kJ/mol/nm^2)\n")
    itp.extend(bonds_lines)
    itp.append("\n")

    if angles_lines:
        itp.append("[ angles ]\n")
        itp.append("; i  j  k  funct  theta0(deg)  k(kJ/mol/rad^2)\n")
        itp.extend(angles_lines)
        itp.append("\n")

    if dihedrals_lines:
        itp.append("[ dihedrals ]\n")
        itp.append("; i  j  k  l  funct  phi0(deg)  k(kJ/mol)  mult\n")
        itp.extend(dihedrals_lines)
        itp.append("\n")

    itp_path.write_text("".join(itp), encoding="utf-8")

    # Optional: patch bonds/angles from QM-derived methods (mseminario / DRIH)
    explicit_bonded = False
    requested_bonded = None
    frag = None
    method = "mseminario"
    try:
        if hasattr(mol, "HasProp"):
            if mol.HasProp("_yadonpy_bonded_explicit"):
                explicit_bonded = str(mol.GetProp("_yadonpy_bonded_explicit")).strip().lower() in ("1", "true", "yes", "on")
            if mol.HasProp("_yadonpy_bonded_requested"):
                requested_bonded = str(mol.GetProp("_yadonpy_bonded_requested")).strip().lower() or None
            if mol.HasProp("_yadonpy_bonded_method"):
                method = str(mol.GetProp("_yadonpy_bonded_method")).strip() or method
            if mol.HasProp("_yadonpy_bonded_itp"):
                frag = Path(mol.GetProp("_yadonpy_bonded_itp"))
            elif mol.HasProp("_yadonpy_mseminario_itp"):
                frag = Path(mol.GetProp("_yadonpy_mseminario_itp"))
                if not requested_bonded:
                    requested_bonded = 'mseminario'
    except Exception:
        explicit_bonded = False
        requested_bonded = None
        frag = None

    patched = False
    if frag is not None and frag.is_file():
        patched = bool(_apply_bond_angle_patch_from_fragment(itp_path, frag, method=method))

    if explicit_bonded and requested_bonded and not patched:
        raise RuntimeError(
            f"Explicit bonded override '{requested_bonded}' was requested, but no valid bonded patch was applied. "
            f"fragment={frag}"
        )

    # 3) TOP
    from .gromacs_top import defaults_block

    top = []
    top.append(defaults_block())
    top.append(f'#include "{mol_name}.itp"\n\n')
    top.append("[ system ]\n")
    top.append(f"{mol_name}\n\n")
    top.append("[ molecules ]\n")
    top.append(f"{mol_name:<12}  1\n")
    top_path.write_text("".join(top), encoding="utf-8")

    return gro_path, itp_path, top_path
