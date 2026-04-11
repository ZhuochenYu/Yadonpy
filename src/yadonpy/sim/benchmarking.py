from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from ..gmx.analysis.structured import build_site_map
from ..gmx.analysis.xvg import read_xvg
from ..gmx.topology import AtomTypeParam, MoleculeType, parse_system_atomtype_params, parse_system_top
from ..interface.charge_audit import summarize_cell_charge, summarize_charge_meta


_COULOMB_KJMOL_NM_E2 = 138.935456


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def literature_band_peo_litfsi_60c() -> dict[str, Any]:
    """Return a deliberately broad experimental conductivity band for PEO20-LiTFSI @ 60°C."""

    return {
        "system": "PEO/LiTFSI",
        "eo_li_ratio": "20:1",
        "temperature_C": 60.0,
        "sigma_band_S_m": {
            "min": 5.0e-3,
            "max": 5.0e-2,
            "target_order_of_magnitude": 1.0e-2,
        },
        "note": (
            "Broad literature band for dry PEO20-LiTFSI around 60°C; use this as an order-of-magnitude "
            "reference rather than a single-paper exact target."
        ),
    }


def summarize_rdkit_species_forcefield(
    mol,
    *,
    label: str,
    moltype_hint: str | None = None,
    charge_scale: float = 1.0,
) -> dict[str, Any]:
    charges: list[float] = []
    atomtypes: dict[str, int] = {}
    natoms = 0
    for atom in mol.GetAtoms():
        natoms += 1
        charge = float(atom.GetDoubleProp("AtomicCharge")) if atom.HasProp("AtomicCharge") else 0.0
        charges.append(charge)
        atype = atom.GetProp("ff_type") if atom.HasProp("ff_type") else atom.GetSymbol()
        atomtypes[str(atype)] = int(atomtypes.get(str(atype), 0)) + 1
    total_charge = float(sum(charges))
    return {
        "label": str(label),
        "moltype_hint": str(moltype_hint or label),
        "natoms": int(natoms),
        "net_charge_e": total_charge,
        "scaled_net_charge_e": float(total_charge) * float(charge_scale),
        "charge_scale": float(charge_scale),
        "atomtype_counts": atomtypes,
        "charge_min_e": float(min(charges)) if charges else 0.0,
        "charge_max_e": float(max(charges)) if charges else 0.0,
    }


def _representative_site_rows(
    *,
    top_path: Path,
    system_dir: Path,
    moltype_hints: Mapping[str, str],
) -> list[dict[str, Any]]:
    topo = parse_system_top(top_path)
    atomtypes = parse_system_atomtype_params(top_path)
    site_map = build_site_map(topo, system_dir)

    wanted = [
        ("cation", "cation_center"),
        ("cation", "cationic_site"),
        ("polymer", "ether_oxygen"),
        ("anion", "sulfonyl_oxygen"),
        ("anion", "nitrogen_site"),
        ("anion", "anion_nitrogen"),
        ("anion", "fluorine_site"),
        ("anion", "coordination_fluorine"),
    ]

    picked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for role, label in wanted:
        moltype = str(moltype_hints.get(role) or "").strip()
        if not moltype:
            continue
        for site in site_map.get("site_groups", []) or []:
            if str(site.get("moltype") or "") != moltype:
                continue
            if str(site.get("site_label") or "") != label:
                continue
            mt = topo.moleculetypes.get(moltype)
            if mt is None:
                continue
            local_indices = list(site.get("local_atom_indices") or [])
            if not local_indices:
                continue
            idx0 = int(local_indices[0])
            atomtype = str(mt.atomtypes[idx0]) if idx0 < len(mt.atomtypes) else ""
            atomname = str(mt.atomnames[idx0]) if idx0 < len(mt.atomnames) else ""
            charge_e = float(mt.charges[idx0]) if idx0 < len(mt.charges) else 0.0
            mass = float(mt.masses[idx0]) if idx0 < len(mt.masses) else 0.0
            lj = atomtypes.get(atomtype, AtomTypeParam(name=atomtype, mass=mass, sigma_nm=0.0, epsilon_kj=0.0))
            key = (moltype, label)
            if key in seen:
                break
            seen.add(key)
            picked.append(
                {
                    "role": role,
                    "moltype": moltype,
                    "site_label": label,
                    "atomtype": atomtype,
                    "atomname": atomname,
                    "charge_e": charge_e,
                    "mass": mass,
                    "sigma_nm": float(lj.sigma_nm),
                    "epsilon_kj_mol": float(lj.epsilon_kj),
                    "coordination_relevance": str(site.get("coordination_relevance") or ""),
                    "coordination_note": str(site.get("coordination_note") or ""),
                }
            )
            break
    return picked


def _find_site_row(rows: Sequence[Mapping[str, Any]], *, role: str, label: str) -> Optional[dict[str, Any]]:
    for row in rows:
        if str(row.get("role") or "") == role and str(row.get("site_label") or "") == label:
            return dict(row)
    return None


def _pair_energy_proxy(site_a: Mapping[str, Any], site_b: Mapping[str, Any], *, multiplicity: int = 1) -> dict[str, Any]:
    qa = float(site_a.get("charge_e") or 0.0)
    qb = float(site_b.get("charge_e") or 0.0)
    sigma_a = float(site_a.get("sigma_nm") or 0.0)
    sigma_b = float(site_b.get("sigma_nm") or 0.0)
    eps_a = float(site_a.get("epsilon_kj_mol") or 0.0)
    eps_b = float(site_b.get("epsilon_kj_mol") or 0.0)
    sigma_mix = 0.5 * (sigma_a + sigma_b)
    eps_mix = float(np.sqrt(max(eps_a, 0.0) * max(eps_b, 0.0)))
    r_min = max((2.0 ** (1.0 / 6.0)) * sigma_mix, 1.0e-6)
    coul = _COULOMB_KJMOL_NM_E2 * qa * qb / r_min
    lj_at_min = -eps_mix
    pair_energy = float(multiplicity) * (coul + lj_at_min)
    return {
        "multiplicity": int(multiplicity),
        "charge_product_e2": qa * qb,
        "sigma_mix_nm": sigma_mix,
        "epsilon_mix_kj_mol": eps_mix,
        "lj_min_distance_nm": r_min,
        "coulomb_at_lj_min_kj_mol": coul,
        "lj_at_lj_min_kj_mol": lj_at_min,
        "pair_energy_proxy_kj_mol": pair_energy,
        "note": "Lorentz-Berthelot + point-charge proxy at the LJ minimum; use only for relative ranking.",
    }


def collect_force_balance_report(
    *,
    system_dir: Path,
    top_path: Path,
    cell=None,
    species_pre_export: Optional[Sequence[Mapping[str, Any]]] = None,
    moltype_hints: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    system_dir = Path(system_dir)
    top_path = Path(top_path)
    meta_path = system_dir / "system_meta.json"
    topo = parse_system_top(top_path)
    meta = _load_json(meta_path) if meta_path.exists() else {}
    species_meta = list(meta.get("species", []) or [])

    post_export = []
    for sp in species_meta:
        moltype = str(sp.get("moltype") or sp.get("mol_name") or sp.get("mol_id") or "")
        mt = topo.moleculetypes.get(moltype)
        post_export.append(
            {
                "name": str(sp.get("name") or moltype),
                "moltype": moltype,
                "kind": str(sp.get("kind") or ""),
                "smiles": str(sp.get("smiles") or ""),
                "formal_charge_e": float(sp.get("formal_charge") or 0.0),
                "count": int(sp.get("n") or 0),
                "charge_scale": float(sp.get("charge_scale") or 1.0),
                "polyelectrolyte_mode": bool(sp.get("polyelectrolyte_mode", False)),
                "has_charge_groups": bool(sp.get("charge_groups")),
                "net_charge_itp_e": float(mt.net_charge) if mt is not None else None,
                "natoms": int(mt.natoms) if mt is not None else None,
                "total_mass": float(mt.total_mass) if mt is not None else None,
            }
        )

    role_hints = dict(moltype_hints or {})
    if not role_hints:
        for sp in species_meta:
            moltype = str(sp.get("moltype") or sp.get("mol_name") or sp.get("mol_id") or "")
            name = str(sp.get("name") or moltype).strip().lower()
            smiles = str(sp.get("smiles") or "").strip()
            kind = str(sp.get("kind") or "").strip().lower()
            if kind == "polymer" and "polymer" not in role_hints:
                role_hints["polymer"] = moltype
            elif smiles == "[Li+]" and "cation" not in role_hints:
                role_hints["cation"] = moltype
            elif "tfsi" in name and "anion" not in role_hints:
                role_hints["anion"] = moltype

    site_rows = _representative_site_rows(top_path=top_path, system_dir=system_dir, moltype_hints=role_hints)
    li_row = (
        _find_site_row(site_rows, role="cation", label="cation_center")
        or _find_site_row(site_rows, role="cation", label="cationic_site")
    )
    ether_row = _find_site_row(site_rows, role="polymer", label="ether_oxygen")
    sulfonyl_row = _find_site_row(site_rows, role="anion", label="sulfonyl_oxygen")
    nitrogen_row = (
        _find_site_row(site_rows, role="anion", label="anion_nitrogen")
        or _find_site_row(site_rows, role="anion", label="nitrogen_site")
    )
    fluorine_row = (
        _find_site_row(site_rows, role="anion", label="coordination_fluorine")
        or _find_site_row(site_rows, role="anion", label="fluorine_site")
    )

    pair_proxies: dict[str, Any] = {}
    if li_row and ether_row:
        pair_proxies["Li_ether_oxygen"] = _pair_energy_proxy(li_row, ether_row)
        pair_proxies["Li_ether_oxygen_x2"] = _pair_energy_proxy(li_row, ether_row, multiplicity=2)
    if li_row and sulfonyl_row:
        pair_proxies["Li_sulfonyl_oxygen"] = _pair_energy_proxy(li_row, sulfonyl_row)
    if li_row and nitrogen_row:
        pair_proxies["Li_tfsi_nitrogen"] = _pair_energy_proxy(li_row, nitrogen_row)
    if li_row and fluorine_row:
        pair_proxies["Li_tfsi_fluorine"] = _pair_energy_proxy(li_row, fluorine_row)

    diagnosis: dict[str, Any] = {"primary_force_balance_flag": "undetermined"}
    if "Li_sulfonyl_oxygen" in pair_proxies and "Li_ether_oxygen" in pair_proxies:
        s = float(pair_proxies["Li_sulfonyl_oxygen"]["pair_energy_proxy_kj_mol"])
        e = float(pair_proxies["Li_ether_oxygen"]["pair_energy_proxy_kj_mol"])
        diagnosis["li_sulfonyl_vs_ether_proxy_ratio"] = abs(s) / max(abs(e), 1.0e-12)
        diagnosis["primary_force_balance_flag"] = (
            "anion_proxy_stronger" if abs(s) > abs(e) else "polymer_proxy_stronger"
        )

    return {
        "cell_charge_audit": summarize_cell_charge(cell),
        "system_meta_charge_audit": summarize_charge_meta(meta_path) if meta_path.exists() else None,
        "species_pre_export": list(species_pre_export or []),
        "species_post_export": post_export,
        "site_charge_lj_table": site_rows,
        "pair_energy_proxies": pair_proxies,
        "diagnosis": diagnosis,
    }


def build_coordination_partition(
    rdf_summary: Mapping[str, Any],
    *,
    polymer_moltype: Optional[str] = None,
    anion_moltype: Optional[str] = None,
) -> dict[str, Any]:
    entries = []
    for site_id, site in (rdf_summary or {}).items():
        if str(site_id).startswith("_") or not isinstance(site, Mapping):
            continue
        entries.append(
            {
                "site_id": str(site_id),
                "moltype": str(site.get("moltype") or ""),
                "site_label": str(site.get("site_label") or ""),
                "coordination_priority": int(site.get("coordination_priority") or 0),
                "coordination_relevance": str(site.get("coordination_relevance") or ""),
                "formal_cn_shell": site.get("formal_cn_shell"),
                "cn_shell": site.get("cn_shell"),
                "r_shell_nm": site.get("r_shell_nm"),
            }
        )

    entries.sort(
        key=lambda item: (
            int(item.get("coordination_priority") or 0),
            -float(item.get("formal_cn_shell") if item.get("formal_cn_shell") is not None else item.get("cn_shell") or 0.0),
            str(item.get("site_id") or ""),
        )
    )

    polymer_cn = 0.0
    anion_cn = 0.0
    primary_sites = []
    for item in entries:
        cn = item.get("formal_cn_shell")
        if cn is None:
            cn = item.get("cn_shell")
        try:
            cn_val = float(cn) if cn is not None else 0.0
        except Exception:
            cn_val = 0.0
        moltype = str(item.get("moltype") or "")
        site_label = str(item.get("site_label") or "")
        if polymer_moltype and moltype == polymer_moltype and site_label in {"ether_oxygen", "hydroxyl_oxygen", "oxygen_site"}:
            polymer_cn += cn_val
        if anion_moltype and moltype == anion_moltype and site_label in {"sulfonyl_oxygen", "nitrogen_site", "anion_nitrogen"}:
            anion_cn += cn_val
        if str(item.get("coordination_relevance") or "") == "primary":
            primary_sites.append(item)

    ratio = anion_cn / max(polymer_cn, 1.0e-12)
    if anion_cn > polymer_cn * 1.25:
        bias = "anion_rich"
    elif polymer_cn > anion_cn * 1.25:
        bias = "polymer_rich"
    else:
        bias = "mixed"

    return {
        "primary_sites": primary_sites,
        "polymer_first_shell_cn": polymer_cn,
        "anion_first_shell_cn": anion_cn,
        "anion_to_polymer_cn_ratio": ratio,
        "coordination_bias": bias,
        "note": "Primary partition compares polymer oxygen donor CN against anion donor CN.",
    }


def estimate_density_drift_fraction(thermo_xvg: Path) -> Optional[float]:
    vals = None
    try:
        df = read_xvg(Path(thermo_xvg)).df
        density_cols = [c for c in df.columns if str(c).strip().lower() in {"density", "mass density"}]
        if not density_cols:
            fallback_cols = [c for c in df.columns if str(c) != "x"]
            if len(fallback_cols) == 1:
                density_cols = fallback_cols
        if density_cols:
            vals = df[density_cols[0]].to_numpy(dtype=float)
    except Exception:
        vals = None
    if vals is None:
        raw_vals: list[float] = []
        try:
            for line in Path(thermo_xvg).read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if not s or s.startswith("@") or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) < 2:
                    continue
                raw_vals.append(float(parts[-1]))
        except Exception:
            return None
        vals = np.asarray(raw_vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        if vals.size < 2:
            return None
    n = max(1, int(0.2 * vals.size))
    start = float(np.mean(vals[:n]))
    end = float(np.mean(vals[-n:]))
    denom = max(abs(start), 1.0e-12)
    return float((end - start) / denom)


def build_transport_summary(
    *,
    msd: Mapping[str, Any],
    sigma: Mapping[str, Any],
    rdf: Mapping[str, Any],
    polymer_moltype: Optional[str] = None,
    anion_moltype: Optional[str] = None,
    thermo_xvg: Optional[Path] = None,
    literature_band: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    li_record = None
    for moltype, rec in (msd or {}).items():
        if str(moltype).startswith("_") or not isinstance(rec, Mapping):
            continue
        if str(moltype).strip().lower() == "li":
            li_record = rec
            break
    li_metric = {}
    if isinstance(li_record, Mapping):
        default_metric = str(li_record.get("default_metric") or "")
        metrics = li_record.get("metrics") if isinstance(li_record.get("metrics"), Mapping) else {}
        if default_metric and isinstance(metrics.get(default_metric), Mapping):
            li_metric = dict(metrics.get(default_metric) or {})

    coordination = build_coordination_partition(rdf, polymer_moltype=polymer_moltype, anion_moltype=anion_moltype)
    density_drift_fraction = estimate_density_drift_fraction(thermo_xvg) if thermo_xvg is not None else None

    eh = sigma.get("eh") if isinstance(sigma.get("eh"), Mapping) else {}
    eh_value = sigma.get("sigma_eh_total_S_m")
    literature = dict(literature_band or literature_band_peo_litfsi_60c())
    band = dict(literature.get("sigma_band_S_m") or {})
    band_min = float(band.get("min") or 0.0)
    band_max = float(band.get("max") or 0.0)
    factor_below_min = None
    within_band = None
    if eh_value is not None and band_min > 0.0:
        factor_below_min = float(band_min) / max(float(eh_value), 1.0e-30)
        within_band = bool(float(eh_value) >= band_min and (band_max <= 0.0 or float(eh_value) <= band_max))

    li_alpha = li_metric.get("alpha_mean")
    try:
        li_alpha = float(li_alpha) if li_alpha is not None else None
    except Exception:
        li_alpha = None

    sampling_flags = {
        "density_drift_fraction": density_drift_fraction,
        "density_drift_exceeds_2pct": bool(density_drift_fraction is not None and abs(float(density_drift_fraction)) > 0.02),
        "li_alpha_mean": li_alpha,
        "li_subdiffusive": bool(li_alpha is not None and float(li_alpha) < 0.7),
        "eh_low_confidence": str(eh.get("confidence") or "") == "low",
        "haven_gt_one": bool((sigma.get("haven_ratio") or 0.0) > 1.0),
    }
    sampling_mature = not any(bool(v) for k, v in sampling_flags.items() if k != "li_alpha_mean" and k != "density_drift_fraction")

    return {
        "li_diffusion_default_m2_s": li_metric.get("D_m2_s"),
        "li_status": li_metric.get("status"),
        "li_alpha_mean": li_alpha,
        "sigma_ne_upper_bound_S_m": sigma.get("sigma_ne_upper_bound_S_m"),
        "sigma_ne_upper_bound_display": sigma.get("sigma_ne_upper_bound_display"),
        "sigma_eh_total_S_m": eh_value,
        "collective_conductivity_unavailable": bool(sigma.get("collective_conductivity_unavailable", False)),
        "eh_confidence": eh.get("confidence"),
        "eh_quality_note": eh.get("quality_note"),
        "eh_method": eh.get("method"),
        "haven_ratio": sigma.get("haven_ratio"),
        "coordination_partition": coordination,
        "sampling_flags": sampling_flags,
        "sampling_mature": sampling_mature,
        "literature_band": literature,
        "within_literature_band": within_band,
        "factor_below_literature_min": factor_below_min,
    }


def build_benchmark_compare(
    *,
    force_balance_report: Mapping[str, Any],
    coordination_partition: Mapping[str, Any],
    transport_summary: Mapping[str, Any],
    charge_scale_polymer: float,
    charge_scale_li: float,
    charge_scale_anion: float,
    production_ns: float,
) -> dict[str, Any]:
    sigma_eh = transport_summary.get("sigma_eh_total_S_m")
    factor_below = transport_summary.get("factor_below_literature_min")
    coord_bias = str(coordination_partition.get("coordination_bias") or "undetermined")
    force_flag = str((force_balance_report.get("diagnosis") or {}).get("primary_force_balance_flag") or "undetermined")
    sampling_flags = dict(transport_summary.get("sampling_flags") or {})

    primary_cause = "mixed"
    notes: list[str] = []
    if bool(sampling_flags.get("density_drift_exceeds_2pct")) or bool(sampling_flags.get("li_subdiffusive")):
        primary_cause = "sampling_not_mature"
        notes.append("Production trajectory is not yet mature: density drift and/or Li subdiffusion remain significant.")
    if coord_bias == "anion_rich" and force_flag == "anion_proxy_stronger":
        primary_cause = "force_balance_likely_anion_biased"
        notes.append("Li-TFSI coordination dominates Li-PEO coordination and the simple pair proxy also favors sulfonyl O.")
    if primary_cause == "sampling_not_mature" and coord_bias == "anion_rich" and force_flag == "anion_proxy_stronger":
        primary_cause = "mixed_sampling_and_force_balance"
        notes.append("Both sampling immaturity and anion-biased force balance appear to contribute.")
    if sigma_eh is None:
        notes.append("gmx current -dsp did not yield a usable EH conductivity; benchmark comparison should not use fallback EH.")
    if factor_below is not None:
        notes.append(f"EH conductivity is about {float(factor_below):.1f}x below the lower edge of the literature band.")

    return {
        "charge_scale_polymer": float(charge_scale_polymer),
        "charge_scale_li": float(charge_scale_li),
        "charge_scale_anion": float(charge_scale_anion),
        "production_ns": float(production_ns),
        "primary_cause": primary_cause,
        "notes": notes,
        "coordination_bias": coord_bias,
        "force_balance_flag": force_flag,
        "sampling_flags": sampling_flags,
        "sigma_eh_total_S_m": sigma_eh,
        "sigma_ne_upper_bound_S_m": transport_summary.get("sigma_ne_upper_bound_S_m"),
        "factor_below_literature_min": factor_below,
    }


def load_benchmark_analysis_dir(analysis_dir: Path | str) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir)

    def _maybe(name: str) -> dict[str, Any]:
        path = analysis_dir / f"{name}.json"
        return _load_json(path) if path.exists() else {}

    compare_payload = _maybe("benchmark_compare")
    compare_block = dict(compare_payload.get("compare") or compare_payload)
    metadata = dict(compare_payload.get("metadata") or _maybe("benchmark_metadata"))
    return {
        "analysis_dir": str(analysis_dir),
        "metadata": metadata,
        "benchmark_compare": compare_block,
        "coordination_partition": _maybe("coordination_partition"),
        "transport_summary": _maybe("transport_summary"),
        "force_balance_report": _maybe("force_balance_report"),
    }


def build_screening_compare(
    *,
    runs: Sequence[Mapping[str, Any]],
    literature_band: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for run in runs:
        compare = dict(run.get("benchmark_compare") or {})
        transport = dict(run.get("transport_summary") or {})
        coordination = dict(run.get("coordination_partition") or {})
        force_balance = dict(run.get("force_balance_report") or {})
        metadata = dict(run.get("metadata") or {})

        scale_li = _safe_float(
            compare.get("charge_scale_li")
            if compare.get("charge_scale_li") is not None
            else (metadata.get("charge_scale") or {}).get("li")
        )
        scale_polymer = _safe_float(
            compare.get("charge_scale_polymer")
            if compare.get("charge_scale_polymer") is not None
            else (metadata.get("charge_scale") or {}).get("polymer")
        )
        scale_anion = _safe_float(
            compare.get("charge_scale_anion")
            if compare.get("charge_scale_anion") is not None
            else (metadata.get("charge_scale") or {}).get("tfsi")
        )
        sigma_eh = _safe_float(compare.get("sigma_eh_total_S_m"))
        sigma_ne = _safe_float(compare.get("sigma_ne_upper_bound_S_m"))
        li_alpha = _safe_float(transport.get("li_alpha_mean") if transport else compare.get("li_alpha_mean"))
        coord_ratio = _safe_float(coordination.get("anion_to_polymer_cn_ratio"))
        normalized.append(
            {
                "analysis_dir": str(run.get("analysis_dir") or ""),
                "charge_scale_polymer": scale_polymer,
                "charge_scale_li": scale_li,
                "charge_scale_anion": scale_anion,
                "sigma_eh_total_S_m": sigma_eh,
                "sigma_ne_upper_bound_S_m": sigma_ne,
                "li_alpha_mean": li_alpha,
                "li_subdiffusive": bool((transport.get("sampling_flags") or {}).get("li_subdiffusive", compare.get("li_subdiffusive", False))),
                "coordination_bias": str(coordination.get("coordination_bias") or compare.get("coordination_bias") or "undetermined"),
                "anion_to_polymer_cn_ratio": coord_ratio,
                "force_balance_flag": str((force_balance.get("diagnosis") or {}).get("primary_force_balance_flag") or compare.get("force_balance_flag") or "undetermined"),
                "primary_cause": str(compare.get("primary_cause") or "undetermined"),
                "production_ns": _safe_float(compare.get("production_ns") if compare.get("production_ns") is not None else metadata.get("prod_ns")),
            }
        )

    normalized = [item for item in normalized if item.get("charge_scale_li") is not None]
    normalized.sort(key=lambda item: float(item["charge_scale_li"]), reverse=True)
    if not normalized:
        return {
            "runs": [],
            "diagnosis": {
                "primary_diagnosis": "no_runs",
                "notes": ["No valid screening runs were provided."],
            },
        }

    baseline = min(normalized, key=lambda item: abs(float(item["charge_scale_li"]) - 1.0))
    best = max(
        normalized,
        key=lambda item: (
            _safe_float(item.get("sigma_eh_total_S_m")) or -1.0,
            _safe_float(item.get("sigma_ne_upper_bound_S_m")) or -1.0,
            _safe_float(item.get("li_alpha_mean")) or -1.0,
        ),
    )
    base_sigma_eh = _safe_float(baseline.get("sigma_eh_total_S_m"))
    best_sigma_eh = _safe_float(best.get("sigma_eh_total_S_m"))
    base_sigma_ne = _safe_float(baseline.get("sigma_ne_upper_bound_S_m"))
    best_sigma_ne = _safe_float(best.get("sigma_ne_upper_bound_S_m"))
    base_alpha = _safe_float(baseline.get("li_alpha_mean"))
    best_alpha = _safe_float(best.get("li_alpha_mean"))
    base_ratio = _safe_float(baseline.get("anion_to_polymer_cn_ratio"))
    best_ratio = _safe_float(best.get("anion_to_polymer_cn_ratio"))

    gain_eh = None
    if base_sigma_eh not in (None, 0.0) and best_sigma_eh is not None:
        gain_eh = best_sigma_eh / base_sigma_eh
    gain_ne = None
    if base_sigma_ne not in (None, 0.0) and best_sigma_ne is not None:
        gain_ne = best_sigma_ne / base_sigma_ne
    alpha_gain = None if base_alpha is None or best_alpha is None else best_alpha - base_alpha
    ratio_change = None
    if base_ratio not in (None, 0.0) and best_ratio is not None:
        ratio_change = best_ratio / base_ratio

    diagnosis = "inconclusive"
    notes: list[str] = []
    if gain_ne is not None:
        notes.append(f"NE screening gain from baseline to best candidate: {gain_ne:.2f}x.")
    if gain_eh is not None:
        notes.append(f"EH gain from baseline to best candidate: {gain_eh:.2f}x.")
    if alpha_gain is not None:
        notes.append(f"Li alpha_mean changed by {alpha_gain:+.3f} between baseline and best candidate.")
    if ratio_change is not None:
        notes.append(f"Anion/polymer first-shell CN ratio changed by a factor of {ratio_change:.3f}.")

    strong_screening_gain = bool(gain_ne is not None and gain_ne >= 3.0)
    alpha_recovers = bool(best_alpha is not None and best_alpha >= 0.7 and alpha_gain is not None and alpha_gain >= 0.15)
    anion_bias_relaxes = bool(
        (baseline.get("coordination_bias") == "anion_rich" and best.get("coordination_bias") != "anion_rich")
        or (ratio_change is not None and ratio_change <= 0.8)
    )
    baseline_force_biased = (
        baseline.get("coordination_bias") == "anion_rich"
        and baseline.get("force_balance_flag") == "anion_proxy_stronger"
    )

    if strong_screening_gain and alpha_recovers and (anion_bias_relaxes or baseline_force_biased):
        diagnosis = "force_balance_overbinding_likely"
        notes.append("Lower charge scaling materially improves Li mobility; over-strong Coulomb binding is the leading diagnosis.")
    elif strong_screening_gain and (best.get("li_subdiffusive") or baseline.get("li_subdiffusive")):
        diagnosis = "mixed_sampling_and_force_balance"
        notes.append("Charge scaling helps strongly, but subdiffusion persists; both force balance and sampling maturity matter.")
    elif all(bool(item.get("li_subdiffusive")) for item in normalized):
        diagnosis = "sampling_limited"
        notes.append("All screening runs remain subdiffusive, so sampling maturity still limits firm transport conclusions.")

    literature = dict(literature_band or literature_band_peo_litfsi_60c())
    recommended_next = {
        "candidate_charge_scale_li": best.get("charge_scale_li"),
        "candidate_charge_scale_polymer": best.get("charge_scale_polymer"),
        "candidate_charge_scale_anion": best.get("charge_scale_anion"),
        "reason": diagnosis,
        "suggested_followup": "Run 3 replicas x 20 ns for baseline and best candidate once remote resources are available.",
    }
    return {
        "runs": normalized,
        "baseline_run": baseline,
        "best_candidate_run": best,
        "gains_vs_baseline": {
            "sigma_eh_gain": gain_eh,
            "sigma_ne_gain": gain_ne,
            "li_alpha_gain": alpha_gain,
            "anion_to_polymer_cn_ratio_factor": ratio_change,
        },
        "diagnosis": {
            "primary_diagnosis": diagnosis,
            "notes": notes,
        },
        "literature_band": literature,
        "recommended_next_step": recommended_next,
    }


__all__ = [
    "build_benchmark_compare",
    "build_coordination_partition",
    "build_transport_summary",
    "build_screening_compare",
    "collect_force_balance_report",
    "estimate_density_drift_fraction",
    "literature_band_peo_litfsi_60c",
    "load_benchmark_analysis_dir",
    "summarize_rdkit_species_forcefield",
    "_dump_json",
]
