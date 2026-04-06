from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MolSpec:
    """A lightweight molecule handle used for MolDB-backed workflows.

    A MolSpec intentionally does **not** carry coordinates/charges. It is a
    declarative description of what should be retrieved (or precomputed) from
    MolDB.

    Fields are chosen to make the MolDB key/label deterministic and explicit.
    """

    smiles: str
    name: Optional[str] = None

    # Charge workflow (currently used for MolDB "ready" variants)
    charge: str = "RESP"              # charge scheme label, e.g. RESP
    basis_set: Optional[str] = None   # RESP/ESP basis set (None = default)
    method: Optional[str] = None      # RESP/ESP method/functional (None = default)

    # Optional strictness controls
    require_ready: bool = True        # require charges present for the selected variant
    prefer_db: bool = True            # prefer DB geometry if present
    polyelectrolyte_mode: Optional[bool] = None
    polyelectrolyte_detection: Optional[str] = None

    # Populated lazily after ff_assign(spec) resolves the handle. This keeps old
    # scripts working when they keep using the original variable name.
    _resolved_mol: Any = field(default=None, init=False, repr=False, compare=False)

    def cache_resolved_mol(self, mol: Any) -> Any:
        self._resolved_mol = mol
        return mol

    @property
    def resolved_mol(self) -> Any:
        return self._resolved_mol

    def get_resolved_mol(self, strict: bool = False) -> Any:
        if self._resolved_mol is not None:
            return self._resolved_mol
        if strict:
            raise TypeError(
                "MolSpec has not been resolved into an RDKit Mol yet. "
                "Call ff.ff_assign(spec) first or use ff.mol_rdkit(...)."
            )
        return None

    def __getattr__(self, name: str):
        mol = object.__getattribute__(self, '_resolved_mol')
        if mol is not None:
            return getattr(mol, name)
        raise AttributeError(name)

    def resolved_label(self) -> str:
        """Human-readable label for logging."""
        b = self.basis_set or "Default"
        m = self.method or "Default"
        pe = ""
        if self.polyelectrolyte_mode is not None:
            pe = f" polyelectrolyte_mode={bool(self.polyelectrolyte_mode)}"
        if self.polyelectrolyte_detection:
            pe += f" polyelectrolyte_detection={self.polyelectrolyte_detection}"
        return f"charge={self.charge} basis={b} method={m}{pe}"


def as_rdkit_mol(mol: Any, strict: bool = False) -> Any:
    """Return an RDKit Mol when given either an RDKit Mol or a resolved MolSpec.

    This is needed for C++-level RDKit APIs, such as ``Descriptors.MolWt``,
    which cannot consume the lightweight MolSpec proxy directly.
    """
    if isinstance(mol, MolSpec):
        mol = mol.get_resolved_mol(strict=strict)
    if mol is not None:
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
    return mol


def molecular_weight(mol: Any, strict: bool = False) -> float:
    """Return a robust molecular weight for an RDKit Mol or resolved MolSpec.

    The primary path uses ``Descriptors.MolWt``. If RDKit descriptor preconditions
    are not satisfied yet (common for unsanitized hypervalent species or freshly
    constructed monoatomic ions), fall back to a property-cache-aware atom-wise
    mass sum instead of raising a low-level RDKit exception.
    """
    rdkit_mol = as_rdkit_mol(mol, strict=strict)
    if rdkit_mol is None:
        raise TypeError("molecular_weight() requires an RDKit Mol or resolved MolSpec")

    from rdkit.Chem import Descriptors

    try:
        return float(Descriptors.MolWt(rdkit_mol))
    except Exception:
        pass

    from rdkit import Chem

    working = Chem.Mol(rdkit_mol)
    try:
        working.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    ptable = Chem.GetPeriodicTable()
    h_avg_mass = float(ptable.GetAtomicWeight(1))
    total = 0.0
    for atom in working.GetAtoms():
        atomic_num = int(atom.GetAtomicNum())
        if atomic_num <= 0:
            continue
        atom_mass = float(atom.GetMass())
        if atom_mass <= 0.0:
            atom_mass = float(ptable.GetAtomicWeight(atomic_num))
        total += atom_mass
        try:
            total += float(atom.GetNumImplicitHs()) * h_avg_mass
        except Exception:
            pass
    return float(total)
