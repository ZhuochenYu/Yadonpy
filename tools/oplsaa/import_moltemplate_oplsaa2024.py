from __future__ import annotations

import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FF_DAT = ROOT / "src" / "yadonpy" / "ff" / "ff_dat"
OPLS_JSON = FF_DAT / "oplsaa.json"
RULES_JSON = FF_DAT / "oplsaa_rules.json"
MOLTEMPLATE_ROOT = ROOT.parent / "_external" / "moltemplate" / "moltemplate" / "force_fields"
PAR_PATH = MOLTEMPLATE_ROOT / "oplsaa2024_original_format" / "Jorgensen_et_al-2024-The_Journal_of_Physical_Chemistry_B.sup-2.par"
LT_PATH = MOLTEMPLATE_ROOT / "oplsaa2024.lt"

KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1

REVERSE_TYPE_MAP = {
    "C°": "C:",
    "C^": "C$",
    "N^": "N$",
    "O^": "O$",
    "C|": "C#",
    "N§": "N*",
    "C⟮": "C(O)",
    "??": "X",
}

ATOMIC_DATA = {
    0: ("X", 0.0),
    1: ("H", 1.008),
    2: ("He", 4.0026),
    3: ("Li", 6.94),
    4: ("Be", 9.0122),
    5: ("B", 10.81),
    6: ("C", 12.011),
    7: ("N", 14.007),
    8: ("O", 15.999),
    9: ("F", 18.998403163),
    10: ("Ne", 20.1797),
    11: ("Na", 22.98976928),
    12: ("Mg", 24.305),
    13: ("Al", 26.9815385),
    14: ("Si", 28.085),
    15: ("P", 30.973761998),
    16: ("S", 32.06),
    17: ("Cl", 35.45),
    18: ("Ar", 39.948),
    19: ("K", 39.0983),
    20: ("Ca", 40.078),
    30: ("Zn", 65.38),
    35: ("Br", 79.904),
    36: ("Kr", 83.798),
    37: ("Rb", 85.4678),
    38: ("Sr", 87.62),
    53: ("I", 126.90447),
    54: ("Xe", 131.293),
    55: ("Cs", 132.90545196),
    56: ("Ba", 137.327),
    57: ("La", 138.90547),
    60: ("Nd", 144.242),
    63: ("Eu", 151.964),
    64: ("Gd", 157.25),
    70: ("Yb", 173.045),
    99: ("X", 0.0),
    89: ("Ac", 227.0),
    90: ("Th", 232.0377),
    92: ("U", 238.02891),
    95: ("Am", 243.0),
}

DUMMY_BTYPES = {"LP", "DM", "XC", "XB", "XI", "MW", "tipM", "tipL", "opcE"}

ION_RULE_REMAP = {
    "[F-]": ("opls_1100", "F-", -1.0, "Fluoride Ion F- (OPLS-AA 2024)"),
    "[Cl-]": ("opls_1101", "Cl-", -1.0, "Chloride Ion Cl- (OPLS-AA 2024)"),
    "[Br-]": ("opls_1102", "Br-", -1.0, "Bromide Ion Br- (OPLS-AA 2024)"),
    "[I-]": ("opls_1103", "I-", -1.0, "Iodide Ion I- (OPLS-AA 2024)"),
    "[Li+]": ("opls_1106", "Li+", 1.0, "Lithium Ion Li+ (OPLS-AA 2024)"),
    "[Na+]": ("opls_1107", "Na+", 1.0, "Sodium Ion Na+ (OPLS-AA 2024)"),
    "[K+]": ("opls_1108", "K+", 1.0, "Potassium Ion K+ (OPLS-AA 2024)"),
    "[Rb+]": ("opls_1109", "Rb+", 1.0, "Rubidium Ion Rb+ (OPLS-AA 2024)"),
    "[Cs+]": ("opls_1110", "Cs+", 1.0, "Cesium Ion Cs+ (OPLS-AA 2024)"),
    "[Mg+2]": ("opls_1111", "Mg2+", 2.0, "Magnesium Ion Mg2+ (OPLS-AA 2024)"),
    "[Ca+2]": ("opls_1112", "Ca2+", 2.0, "Calcium Ion Ca2+ (OPLS-AA 2024)"),
    "[Sr+2]": ("opls_1113", "Sr2+", 2.0, "Strontium Ion Sr2+ (OPLS-AA 2024)"),
    "[Ba+2]": ("opls_1114", "Ba2+", 2.0, "Barium Ion Ba2+ (OPLS-AA 2024)"),
}

NEW_RULES = [
    {"element": "Zn", "btype": "Zn2+", "opls": "opls_1005", "smarts": "[Zn+2]", "charge": 2.0, "desc": "Zinc Ion Zn2+ (OPLS-AA 2024)", "priority": 10000},
    {"element": "H", "btype": "H~", "opls": "opls_1064", "smarts": "[#1][Si]", "charge": -0.01, "desc": "H on Si in silane, silanol, or silyl ether", "priority": 10010},
    {"element": "H", "btype": "HO", "opls": "opls_1074", "smarts": "[#1][O][Si]", "charge": 0.35, "desc": "H in SiOH silanol", "priority": 10020},
    {"element": "O", "btype": "OH", "opls": "opls_1073", "smarts": "[O;H1][Si]", "charge": -0.50, "desc": "O in SiOH silanol", "priority": 10030},
    {"element": "O", "btype": "OS", "opls": "opls_1078", "smarts": "[O;H0;X2]([#6])[Si]", "charge": -0.35, "desc": "O in alkyl silyl ether", "priority": 10040},
    {"element": "C", "btype": "CT", "opls": "opls_1065", "smarts": "[CH3X4][Si]", "charge": -0.26, "desc": "CH3 on Si in silane family", "priority": 10050},
    {"element": "C", "btype": "CT", "opls": "opls_1066", "smarts": "[CH2X4]([#6])[Si]", "charge": -0.20, "desc": "CH2 on Si in silane family", "priority": 10060},
    {"element": "C", "btype": "CT", "opls": "opls_1067", "smarts": "[CHX4]([#6])([#6])[Si]", "charge": -0.14, "desc": "CH on Si in silane family", "priority": 10070},
    {"element": "C", "btype": "CT", "opls": "opls_1068", "smarts": "[CX4]([#6])([#6])([#6])[Si]", "charge": -0.08, "desc": "Quaternary carbon on Si in silane family", "priority": 10080},
    {"element": "C", "btype": "CA", "opls": "opls_1069", "smarts": "[cX3]([Si])[c]", "charge": -0.08, "desc": "Ipso aromatic carbon in phenyl silane", "priority": 10090},
    {"element": "Si", "btype": "Si", "opls": "opls_1083", "smarts": "[SiH4]", "charge": 0.04, "desc": "Si in SiH4", "priority": 10100},
    {"element": "Si", "btype": "Si", "opls": "opls_1084", "smarts": "[SiH3][O;H1]", "charge": 0.18, "desc": "Si in SiH3OH", "priority": 10110},
    {"element": "Si", "btype": "Si", "opls": "opls_1082", "smarts": "[SiH3][Si]", "charge": 0.03, "desc": "Si in H3Si-Si disilane", "priority": 10120},
    {"element": "Si", "btype": "Si", "opls": "opls_1063", "smarts": "[SiH3][#6]", "charge": 0.11, "desc": "Si in RSiH3", "priority": 10130},
    {"element": "Si", "btype": "Si", "opls": "opls_1072", "smarts": "[SiH2]([#6])[O;H1]", "charge": 0.25, "desc": "Si in RSiH2OH", "priority": 10140},
    {"element": "Si", "btype": "Si", "opls": "opls_1077", "smarts": "[SiH2]([#6])[O;H0][#6]", "charge": 0.25, "desc": "Si in RSiH2OR", "priority": 10150},
    {"element": "Si", "btype": "Si", "opls": "opls_1081", "smarts": "[SiH2]([#6])[Si]", "charge": 0.10, "desc": "Si in RSiH2Si disilane", "priority": 10160},
    {"element": "Si", "btype": "Si", "opls": "opls_1062", "smarts": "[SiH2]([#6])[#6]", "charge": 0.18, "desc": "Si in R2SiH2", "priority": 10170},
    {"element": "Si", "btype": "Si", "opls": "opls_1071", "smarts": "[SiH1]([#6])([#6])[O;H1]", "charge": 0.32, "desc": "Si in R2SiHOH", "priority": 10180},
    {"element": "Si", "btype": "Si", "opls": "opls_1076", "smarts": "[SiH1]([#6])([#6])[O;H0][#6]", "charge": 0.32, "desc": "Si in R2SiHOR", "priority": 10190},
    {"element": "Si", "btype": "Si", "opls": "opls_1080", "smarts": "[SiH1]([#6])([#6])[Si]", "charge": 0.17, "desc": "Si in R2SiHSi disilane", "priority": 10200},
    {"element": "Si", "btype": "Si", "opls": "opls_1061", "smarts": "[SiH1]([#6])([#6])[#6]", "charge": 0.25, "desc": "Si in R3SiH", "priority": 10210},
    {"element": "Si", "btype": "Si", "opls": "opls_1070", "smarts": "[SiH0]([#6])([#6])([#6])[O;H1]", "charge": 0.39, "desc": "Si in R3SiOH", "priority": 10220},
    {"element": "Si", "btype": "Si", "opls": "opls_1075", "smarts": "[SiH0]([#6])([#6])([#6])[O;H0][#6]", "charge": 0.39, "desc": "Si in R3SiOR", "priority": 10230},
    {"element": "Si", "btype": "Si", "opls": "opls_1079", "smarts": "[SiH0]([#6])([#6])([#6])[Si]", "charge": 0.24, "desc": "Si in R3SiSi disilane", "priority": 10240},
    {"element": "Si", "btype": "Si", "opls": "opls_1060", "smarts": "[SiH0]([#6])([#6])([#6])[#6]", "charge": 0.32, "desc": "Si in tetraalkylsilane R4Si", "priority": 10250},
    {"element": "O", "btype": "O~", "opls": "opls_1159", "smarts": "[O]=[C]=[C]", "charge": -0.25, "desc": "Ketene oxygen", "priority": 10260},
    {"element": "C", "btype": "C:", "opls": "opls_1158", "smarts": "[C;X2](=[O])=[C]", "charge": 0.20, "desc": "Ketene central carbon", "priority": 10270},
    {"element": "C", "btype": "C:", "opls": "opls_1157", "smarts": "[C;X2](=[C])=[C]", "charge": -0.10, "desc": "Allene central carbon", "priority": 10280},
    {"element": "C", "btype": "CM", "opls": "opls_1154", "smarts": "[CH2]=[C]=[C]", "charge": -0.25, "desc": "Allene terminal carbon CH2", "priority": 10290},
    {"element": "C", "btype": "CM", "opls": "opls_1155", "smarts": "[CH]=[C]=[C]", "charge": -0.10, "desc": "Allene terminal carbon CHR", "priority": 10300},
    {"element": "C", "btype": "CM", "opls": "opls_1156", "smarts": "[C;H0]=[C]=[C]", "charge": 0.05, "desc": "Allene terminal carbon CR2", "priority": 10310},
    {"element": "H", "btype": "HC", "opls": "opls_1153", "smarts": "[#1][C]=[C]=[C]", "charge": 0.15, "desc": "Hydrogen on allene terminal carbon", "priority": 10320},
    {"element": "C", "btype": "C:", "opls": "opls_1160", "smarts": "[C](=[O])=[O]", "charge": 0.70, "desc": "Carbon dioxide carbon", "priority": 10330},
    {"element": "O", "btype": "O~", "opls": "opls_1161", "smarts": "[O]=[C]=[O]", "charge": -0.35, "desc": "Carbon dioxide oxygen", "priority": 10340},
    {"element": "O", "btype": "O$", "opls": "opls_1025", "smarts": "[O;r3]([#6;r3])[#6;r3]", "charge": -0.40, "desc": "Epoxide oxygen", "priority": 10350},
    {"element": "C", "btype": "CY", "opls": "opls_1026", "smarts": "[CH2;r3]1[O;r3][#6;r3]1", "charge": 0.14, "desc": "Epoxide CH2 carbon", "priority": 10360},
    {"element": "C", "btype": "CY", "opls": "opls_1027", "smarts": "[CH;r3]1[O;r3][#6;r3]1", "charge": 0.17, "desc": "Epoxide CH carbon", "priority": 10370},
    {"element": "C", "btype": "CY", "opls": "opls_1028", "smarts": "[C;r3]1[O;r3][#6;r3]1", "charge": 0.20, "desc": "Epoxide quaternary carbon", "priority": 10380},
    {"element": "H", "btype": "HC", "opls": "opls_1029", "smarts": "[#1][C;r3]1[O;r3][#6;r3]1", "charge": 0.03, "desc": "Hydrogen alpha to epoxide oxygen", "priority": 10390},
]


def normalize_type(token: str) -> str:
    token = token.strip()
    token = REVERSE_TYPE_MAP.get(token, token)
    if token in {"Y", "Z", "??"}:
        return "X"
    return token


def opls_tag(num: int) -> str:
    return f"opls_{num:03d}"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def infer_elem_mass(atomic_number: int, btype: str) -> tuple[str, float]:
    if btype in DUMMY_BTYPES or atomic_number in (0, 99):
        return ("X", 0.0)
    if atomic_number not in ATOMIC_DATA:
        raise KeyError(f"Unsupported atomic number in OPLS source: {atomic_number}")
    return ATOMIC_DATA[atomic_number]


def parse_par_particles():
    rows: dict[str, dict] = {}
    for raw in PAR_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 6 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        type_id = int(parts[0])
        atomic_number = int(parts[1])
        btype = normalize_type(parts[2])
        charge = float(parts[3])
        sigma = float(parts[4]) * ANGSTROM_TO_NM
        epsilon = float(parts[5]) * KCAL_TO_KJ
        desc = " ".join(parts[6:]).strip()
        elem, mass = infer_elem_mass(atomic_number, btype)
        rows[opls_tag(type_id)] = {
            "tag": opls_tag(type_id),
            "name": opls_tag(type_id),
            "elem": elem,
            "mass": mass,
            "epsilon": epsilon,
            "sigma": sigma,
            "desc": f"bond_type={btype}; itp_charge={charge}; source=moltemplate_oplsaa2024",
        }
    return rows


def parse_lt_sections():
    bonds = {}
    angles = {}
    dihedrals = {}

    bond_re = re.compile(r"^\s*bond_coeff\s+@bond:([^\s]+)\s+([^\s]+)\s+([^\s]+)(?:\s+#\s*(.*))?$", re.M)
    angle_re = re.compile(r"^\s*angle_coeff\s+@angle:([^\s]+)\s+([^\s]+)\s+([^\s]+)(?:\s+#\s*(.*))?$", re.M)
    dihedral_re = re.compile(
        r"^\s*dihedral_coeff\s+@dihedral:([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)(?:\s+#\s*(.*))?$",
        re.M,
    )

    text = LT_PATH.read_text(encoding="utf-8")

    for match in bond_re.finditer(text):
        name = match.group(1)
        tokens = [normalize_type(tok) for tok in name.split("_")]
        if len(tokens) != 2:
            continue
        tag = ",".join(tokens)
        bonds[tag] = {
            "tag": tag,
            "name": tag,
            "rname": ",".join(reversed(tokens)),
            "k": float(match.group(2)) * KCAL_TO_KJ * 400.0,
            "r0": float(match.group(3)) * ANGSTROM_TO_NM,
        }

    for match in angle_re.finditer(text):
        name = match.group(1)
        tokens = [normalize_type(tok) for tok in name.split("_")]
        if len(tokens) != 3:
            continue
        tag = ",".join(tokens)
        angles[tag] = {
            "tag": tag,
            "name": tag,
            "rname": ",".join(reversed(tokens)),
            "k": float(match.group(2)) * KCAL_TO_KJ * 4.0,
            "theta0": float(match.group(3)),
        }

    for match in dihedral_re.finditer(text):
        name = match.group(1)
        tokens = [normalize_type(tok) for tok in name.split("_")]
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
            ks.append(abs(coeff) * KCAL_TO_KJ / 2.0)
            ds.append(phase)
            ns.append(idx)
        dihedrals[tag] = {
            "tag": tag,
            "name": tag,
            "rname": ",".join(reversed(tokens)),
            "d": ds,
            "k": ks,
            "m": len(ks),
            "n": ns,
            "source": "moltemplate_oplsaa2024",
        }

    if "CT,Si,CT" not in angles:
        angles["CT,Si,CT"] = {
            "tag": "CT,Si,CT",
            "name": "CT,Si,CT",
            "rname": "CT,Si,CT",
            "k": 37.0 * KCAL_TO_KJ * 4.0,
            "theta0": 112.5,
        }

    return bonds, angles, dihedrals


def merge_particle_types(current: list[dict], imported: dict[str, dict]) -> list[dict]:
    merged = {row["tag"]: row for row in current}
    merged.update(imported)
    def sort_key(tag: str):
        body = tag.removeprefix("opls_")
        if body.isdigit():
            return (0, int(body), "")
        m = re.match(r"(\d+)(.*)", body)
        if m:
            return (0, int(m.group(1)), m.group(2))
        return (1, 0, body)
    return [merged[key] for key in sorted(merged, key=sort_key)]


def merge_bonded(current: list[dict], imported: dict[str, dict]) -> list[dict]:
    merged = {row["tag"]: row for row in current}
    merged.update(imported)
    return [merged[key] for key in sorted(merged)]


def patch_rules(current_rules: list[dict]) -> list[dict]:
    rules = [dict(row) for row in current_rules]

    for smarts, (opls, btype, charge, desc) in ION_RULE_REMAP.items():
        for row in rules:
            if row.get("smarts") == smarts:
                row["opls"] = opls
                row["btype"] = btype
                row["charge"] = charge
                row["desc"] = desc
                row["priority"] = 9500
                break

    existing = {(row["smarts"], row["opls"]) for row in rules}
    for row in NEW_RULES:
        key = (row["smarts"], row["opls"])
        if key not in existing:
            rules.append(dict(row))

    return rules


def main():
    data = load_json(OPLS_JSON)
    rules = load_json(RULES_JSON)

    imported_particles = parse_par_particles()
    imported_bonds, imported_angles, imported_dihedrals = parse_lt_sections()

    data["particle_types"] = merge_particle_types(data["particle_types"], imported_particles)
    data["bond_types"] = merge_bonded(data["bond_types"], imported_bonds)
    data["angle_types"] = merge_bonded(data["angle_types"], imported_angles)
    data["dihedral_types"] = merge_bonded(data["dihedral_types"], imported_dihedrals)

    OPLS_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    patched_rules = patch_rules(rules)
    RULES_JSON.write_text(json.dumps(patched_rules, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("particle_types", len(data["particle_types"]))
    print("bond_types", len(data["bond_types"]))
    print("angle_types", len(data["angle_types"]))
    print("dihedral_types", len(data["dihedral_types"]))
    print("rules", len(patched_rules))


if __name__ == "__main__":
    main()
