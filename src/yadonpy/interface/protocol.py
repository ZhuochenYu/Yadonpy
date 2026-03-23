from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

from ..core.logging_utils import yadon_print
from ..core.workdir import WorkDir, workdir
from ..gmx.mdp_templates import (
    MINIM_STEEP_MDP,
    NPT_MDP,
    NPT_NO_CONSTRAINTS_MDP,
    NVT_MDP,
    NVT_NO_CONSTRAINTS_MDP,
    MdpSpec,
    default_mdp_params,
)
from ..gmx.workflows._util import RunResources
from ..gmx.workflows.eq import EqStage, EquilibrationJob
from ..runtime import resolve_restart
from ..workflow.resume import ResumeManager, StepSpec
from .builder import BuiltInterface, validate_topology_include_order
from .postprocess import read_ndx_groups

_AXIS_TO_INDEX = {"X": 0, "Y": 1, "Z": 2}


@dataclass(frozen=True)
class InterfaceStageSpec:
    name: str
    kind: str
    description: str
    mdp: MdpSpec


@dataclass(frozen=True)
class InterfaceProtocol:
    route: str
    axis: str = "Z"
    stage_mode: str = "simple"
    pre_contact_ps: float = 100.0
    pre_contact_dt_ps: float = 0.001
    density_relax_ps: float = 200.0
    contact_ps: float = 200.0
    release_ps: float = 200.0
    exchange_ns: float = 2.0
    production_ns: float = 5.0
    temperature_k: float = 300.0
    pressure_bar: float = 1.0
    semiisotropic: bool = True
    wall_mode: Optional[str] = None
    wall_atomtype: Optional[str] = None
    wall_density_nm3: Optional[float] = None
    freeze_cores_pre_contact: bool = True
    use_region_thermostat_early: bool = True
    density_relax_barostat: str = "Berendsen"
    contact_barostat: str = "Berendsen"
    release_barostat: str = "Berendsen"
    exchange_barostat: str = "Berendsen"
    production_barostat: str = "C-rescale"
    density_relax_tau_p: float = 16.0
    contact_tau_p: float = 12.0
    release_tau_p: float = 10.0
    exchange_tau_p: float = 8.0
    production_tau_p: float = 5.0
    berendsen_compressibility_scale: float = 0.5

    @staticmethod
    def _extra_mdp_lines(*lines: str) -> str:
        return "\n".join(str(line).rstrip() for line in lines if str(line).strip())

    def _axis(self) -> str:
        return str(self.axis or "Z").strip().upper()

    def _pressure_params(self, *, barostat: str, tau_p: float, base_compressibility: object) -> dict[str, object]:
        params: dict[str, object] = {
            "pcoupl": str(barostat),
            "tau_p": float(tau_p),
            "compressibility": base_compressibility,
        }
        if str(barostat).strip().lower() == "berendsen":
            if isinstance(base_compressibility, str):
                parts = [float(x) for x in str(base_compressibility).split()]
                scaled = [self.berendsen_compressibility_scale * x for x in parts]
                params["compressibility"] = " ".join(f"{x:.6g}" for x in scaled)
            else:
                params["compressibility"] = float(base_compressibility) * float(self.berendsen_compressibility_scale)
        return params

    def _temperature_params(self, *, split_regions: bool) -> dict[str, object]:
        base_tau = float(default_mdp_params()["tau_t"])
        if split_regions and self.use_region_thermostat_early:
            ref_t = f"{float(self.temperature_k)} {float(self.temperature_k)}"
            tau_t = f"{base_tau:.6g} {base_tau:.6g}"
            return {"tc_grps": "BOTTOM TOP", "tau_t": tau_t, "ref_t": ref_t}
        return {
            "tc_grps": "System",
            "tau_t": base_tau,
            "ref_t": float(self.temperature_k),
        }

    def _freeze_block(self, *, mode: str) -> str:
        if not self.freeze_cores_pre_contact:
            return ""
        groups = ("BOTTOM_CORE", "TOP_CORE")
        dims: tuple[bool, bool, bool]
        if mode == "all":
            dims = (True, True, True)
        elif mode == "normal":
            axis_idx = _AXIS_TO_INDEX.get(self._axis(), 2)
            dims = tuple(idx == axis_idx for idx in range(3))
        else:
            return ""
        flags = " ".join(("Y" if flag else "N") for _group in groups for flag in dims)
        if mode == "all":
            note = "; keep slab cores fixed while the gap geometry settles"
        else:
            note = "; keep slab cores pinned only along the interface normal while each phase densifies"
        return self._extra_mdp_lines(
            note,
            f"freezegrps               = {' '.join(groups)}",
            f"freezedim                = {flags}",
        )

    def _base_params(self) -> tuple[dict[str, object], object]:
        p = default_mdp_params()
        p["ref_t"] = float(self.temperature_k)
        p["gen_temp"] = float(self.temperature_k)
        p["gen_seed"] = -1
        p["extra_mdp"] = ""
        p["periodic-molecules"] = "yes"
        if self.semiisotropic:
            p["pcoupltype"] = "semiisotropic"
            p["compressibility"] = "4.5e-5 4.5e-5"
            p["ref_p"] = f"{float(self.pressure_bar)} {float(self.pressure_bar)}"
        else:
            p["ref_p"] = float(self.pressure_bar)
        if self.route == "route_b" and self.wall_mode:
            p["pbc"] = "xy"
            p["nwall"] = 2
            p["wall_type"] = str(self.wall_mode)
            if self.wall_atomtype is not None:
                p["wall_atomtype"] = str(self.wall_atomtype)
            if self.wall_density_nm3 is not None:
                p["wall_density"] = float(self.wall_density_nm3)
        return p, p["compressibility"]

    def _simple_stages(self) -> list[InterfaceStageSpec]:
        p, base_compressibility = self._base_params()
        freeze_core_block = self._freeze_block(mode="all")
        pre_contact_dt = min(float(p["dt"]), float(self.pre_contact_dt_ps))
        pre_contact_nvt_params = {
            **p,
            **self._temperature_params(split_regions=False),
            "dt": pre_contact_dt,
            "nsteps": max(int(float(self.pre_contact_ps) / float(pre_contact_dt)), 2000),
            "gen_vel": "yes",
            "extra_mdp": freeze_core_block,
        }
        return [
            InterfaceStageSpec(
                "01_pre_contact_em",
                "minim",
                "Remove severe overlaps while keeping the intentionally separated interface geometry with a fully unconstrained minimization.",
                MdpSpec(
                    MINIM_STEEP_MDP,
                    {
                        **p,
                        "nsteps": 100000,
                        "emtol": 1000.0,
                        "emstep": 0.0005,
                        "extra_mdp": freeze_core_block,
                    },
                ),
            ),
            InterfaceStageSpec(
                "02_pre_contact_nvt",
                "nvt",
                "Short no-constraints NVT hold before the two slabs are allowed to close the initial gap.",
                MdpSpec(NVT_NO_CONSTRAINTS_MDP, pre_contact_nvt_params),
            ),
            InterfaceStageSpec(
                "03_contact",
                "npt",
                "Mild semiisotropic contact stage using a damped barostat for robust early densification.",
                MdpSpec(
                    NPT_MDP,
                    {
                        **p,
                        **self._temperature_params(split_regions=False),
                        **self._pressure_params(
                            barostat=self.contact_barostat,
                            tau_p=self.contact_tau_p,
                            base_compressibility=base_compressibility,
                        ),
                        "nsteps": max(int(float(self.contact_ps) / float(p["dt"])), 1000),
                        "gen_vel": "no",
                    },
                ),
            ),
            InterfaceStageSpec(
                "04_exchange",
                "md",
                "Semiisotropic exchange equilibration after contact, kept conservative for robust interfacial relaxation.",
                MdpSpec(
                    NPT_MDP,
                    {
                        **p,
                        **self._temperature_params(split_regions=False),
                        **self._pressure_params(
                            barostat=self.exchange_barostat,
                            tau_p=self.exchange_tau_p,
                            base_compressibility=base_compressibility,
                        ),
                        "nsteps": max(int(float(self.exchange_ns) * 1000.0 / float(p["dt"])), 1000),
                        "gen_vel": "no",
                    },
                ),
            ),
            InterfaceStageSpec(
                "05_production",
                "md",
                "Production interface sampling.",
                MdpSpec(
                    NPT_MDP,
                    {
                        **p,
                        **self._temperature_params(split_regions=False),
                        **self._pressure_params(
                            barostat=self.production_barostat,
                            tau_p=self.production_tau_p,
                            base_compressibility=base_compressibility,
                        ),
                        "nsteps": max(int(float(self.production_ns) * 1000.0 / float(p["dt"])), 1000),
                        "gen_vel": "no",
                    },
                ),
            ),
        ]

    def _diffusion_stages(self) -> list[InterfaceStageSpec]:
        p, base_compressibility = self._base_params()
        pre_contact_dt = min(float(p["dt"]), float(self.pre_contact_dt_ps))
        freeze_all = self._freeze_block(mode="all")
        freeze_normal = self._freeze_block(mode="normal")
        early_temp = self._temperature_params(split_regions=True)
        late_temp = self._temperature_params(split_regions=False)

        pre_contact_nvt_params = {
            **p,
            **early_temp,
            "dt": pre_contact_dt,
            "nsteps": max(int(float(self.pre_contact_ps) / float(pre_contact_dt)), 2000),
            "gen_vel": "yes",
            "extra_mdp": freeze_all,
        }
        density_relax_params = {
            **p,
            **early_temp,
            **self._pressure_params(
                barostat=self.density_relax_barostat,
                tau_p=self.density_relax_tau_p,
                base_compressibility=base_compressibility,
            ),
            "dt": pre_contact_dt,
            "nsteps": max(int(float(self.density_relax_ps) / float(pre_contact_dt)), 4000),
            "gen_vel": "no",
            "extra_mdp": freeze_normal,
        }
        contact_params = {
            **p,
            **early_temp,
            **self._pressure_params(
                barostat=self.contact_barostat,
                tau_p=self.contact_tau_p,
                base_compressibility=base_compressibility,
            ),
            "nsteps": max(int(float(self.contact_ps) / float(p["dt"])), 2000),
            "gen_vel": "no",
            "extra_mdp": freeze_normal,
        }
        release_params = {
            **p,
            **early_temp,
            **self._pressure_params(
                barostat=self.release_barostat,
                tau_p=self.release_tau_p,
                base_compressibility=base_compressibility,
            ),
            "nsteps": max(int(float(self.release_ps) / float(p["dt"])), 2000),
            "gen_vel": "no",
            "extra_mdp": "",
        }
        exchange_params = {
            **p,
            **late_temp,
            **self._pressure_params(
                barostat=self.exchange_barostat,
                tau_p=self.exchange_tau_p,
                base_compressibility=base_compressibility,
            ),
            "nsteps": max(int(float(self.exchange_ns) * 1000.0 / float(p["dt"])), 1000),
            "gen_vel": "no",
            "extra_mdp": "",
        }
        production_params = {
            **p,
            **late_temp,
            **self._pressure_params(
                barostat=self.production_barostat,
                tau_p=self.production_tau_p,
                base_compressibility=base_compressibility,
            ),
            "nsteps": max(int(float(self.production_ns) * 1000.0 / float(p["dt"])), 1000),
            "gen_vel": "no",
            "extra_mdp": "",
        }
        return [
            InterfaceStageSpec(
                "01_pre_contact_em",
                "minim",
                "Remove severe overlaps while holding the slab cores fixed so the stitched vacuum-gap geometry does not collapse immediately.",
                MdpSpec(
                    MINIM_STEEP_MDP,
                    {
                        **p,
                        "nsteps": 100000,
                        "emtol": 1000.0,
                        "emstep": 0.0005,
                        "extra_mdp": freeze_all,
                    },
                ),
            ),
            InterfaceStageSpec(
                "02_gap_hold_nvt",
                "nvt",
                "Short no-constraints NVT hold with split BOTTOM/TOP thermal coupling while the two phases stay separated by the initial vacuum gap.",
                MdpSpec(NVT_NO_CONSTRAINTS_MDP, pre_contact_nvt_params),
            ),
            InterfaceStageSpec(
                "03_density_relax",
                "npt",
                "No-constraints semiisotropic NPT stage that keeps slab cores pinned only along the interface normal so each phase can relax density without premature interpenetration.",
                MdpSpec(NPT_NO_CONSTRAINTS_MDP, density_relax_params),
            ),
            InterfaceStageSpec(
                "04_contact",
                "npt",
                "Constrained contact stage with only normal-direction core support, allowing the gap to close gently while preserving the rectangular interface box.",
                MdpSpec(NPT_MDP, contact_params),
            ),
            InterfaceStageSpec(
                "05_release",
                "npt",
                "Remove the remaining core hold and keep split BOTTOM/TOP thermostatting briefly so both phases can finish local densification before unrestricted exchange.",
                MdpSpec(NPT_MDP, release_params),
            ),
            InterfaceStageSpec(
                "06_exchange",
                "md",
                "Unrestricted semiisotropic exchange stage where electrolyte can diffuse gradually into the polymer phase.",
                MdpSpec(NPT_MDP, exchange_params),
            ),
            InterfaceStageSpec(
                "07_production",
                "md",
                "Production interface sampling after the staged release protocol has finished.",
                MdpSpec(NPT_MDP, production_params),
            ),
        ]

    def stages(self) -> list[InterfaceStageSpec]:
        mode = str(self.stage_mode or "simple").strip().lower()
        if mode == "diffusion":
            return self._diffusion_stages()
        return self._simple_stages()

    @staticmethod
    def route_a(
        *,
        temperature_k: float = 300.0,
        pressure_bar: float = 1.0,
        pre_contact_ps: float = 100.0,
        contact_ps: float = 200.0,
        exchange_ns: float = 2.0,
        production_ns: float = 5.0,
        contact_barostat: str = "Berendsen",
        exchange_barostat: str = "Berendsen",
        production_barostat: str = "C-rescale",
        axis: str = "Z",
    ) -> "InterfaceProtocol":
        return InterfaceProtocol(
            route="route_a",
            axis=axis,
            stage_mode="simple",
            pre_contact_ps=pre_contact_ps,
            contact_ps=contact_ps,
            exchange_ns=exchange_ns,
            production_ns=production_ns,
            temperature_k=temperature_k,
            pressure_bar=pressure_bar,
            semiisotropic=True,
            contact_barostat=contact_barostat,
            exchange_barostat=exchange_barostat,
            production_barostat=production_barostat,
        )

    @staticmethod
    def route_a_diffusion(
        *,
        temperature_k: float = 300.0,
        pressure_bar: float = 1.0,
        pre_contact_ps: float = 100.0,
        pre_contact_dt_ps: float = 0.001,
        density_relax_ps: float = 250.0,
        contact_ps: float = 250.0,
        release_ps: float = 250.0,
        exchange_ns: float = 2.0,
        production_ns: float = 5.0,
        axis: str = "Z",
        freeze_cores_pre_contact: bool = True,
        use_region_thermostat_early: bool = True,
    ) -> "InterfaceProtocol":
        return InterfaceProtocol(
            route="route_a",
            axis=axis,
            stage_mode="diffusion",
            pre_contact_ps=pre_contact_ps,
            pre_contact_dt_ps=pre_contact_dt_ps,
            density_relax_ps=density_relax_ps,
            contact_ps=contact_ps,
            release_ps=release_ps,
            exchange_ns=exchange_ns,
            production_ns=production_ns,
            temperature_k=temperature_k,
            pressure_bar=pressure_bar,
            semiisotropic=True,
            freeze_cores_pre_contact=freeze_cores_pre_contact,
            use_region_thermostat_early=use_region_thermostat_early,
        )

    @staticmethod
    def route_b_wall(
        *,
        temperature_k: float = 300.0,
        pressure_bar: float = 1.0,
        pre_contact_ps: float = 100.0,
        contact_ps: float = 200.0,
        exchange_ns: float = 2.0,
        production_ns: float = 5.0,
        wall_mode: str = "12-6",
        wall_atomtype: Optional[str] = None,
        wall_density_nm3: Optional[float] = None,
        contact_barostat: str = "Berendsen",
        exchange_barostat: str = "Berendsen",
        production_barostat: str = "C-rescale",
        axis: str = "Z",
    ) -> "InterfaceProtocol":
        return InterfaceProtocol(
            route="route_b",
            axis=axis,
            stage_mode="simple",
            pre_contact_ps=pre_contact_ps,
            contact_ps=contact_ps,
            exchange_ns=exchange_ns,
            production_ns=production_ns,
            temperature_k=temperature_k,
            pressure_bar=pressure_bar,
            semiisotropic=True,
            wall_mode=wall_mode,
            wall_atomtype=wall_atomtype,
            wall_density_nm3=wall_density_nm3,
            contact_barostat=contact_barostat,
            exchange_barostat=exchange_barostat,
            production_barostat=production_barostat,
        )

    @staticmethod
    def route_b_wall_diffusion(
        *,
        temperature_k: float = 300.0,
        pressure_bar: float = 1.0,
        pre_contact_ps: float = 100.0,
        pre_contact_dt_ps: float = 0.001,
        density_relax_ps: float = 250.0,
        contact_ps: float = 250.0,
        release_ps: float = 250.0,
        exchange_ns: float = 2.0,
        production_ns: float = 5.0,
        wall_mode: str = "12-6",
        wall_atomtype: Optional[str] = None,
        wall_density_nm3: Optional[float] = None,
        axis: str = "Z",
        freeze_cores_pre_contact: bool = True,
        use_region_thermostat_early: bool = True,
    ) -> "InterfaceProtocol":
        return InterfaceProtocol(
            route="route_b",
            axis=axis,
            stage_mode="diffusion",
            pre_contact_ps=pre_contact_ps,
            pre_contact_dt_ps=pre_contact_dt_ps,
            density_relax_ps=density_relax_ps,
            contact_ps=contact_ps,
            release_ps=release_ps,
            exchange_ns=exchange_ns,
            production_ns=production_ns,
            temperature_k=temperature_k,
            pressure_bar=pressure_bar,
            semiisotropic=True,
            wall_mode=wall_mode,
            wall_atomtype=wall_atomtype,
            wall_density_nm3=wall_density_nm3,
            freeze_cores_pre_contact=freeze_cores_pre_contact,
            use_region_thermostat_early=use_region_thermostat_early,
        )


class InterfaceDynamics:
    def __init__(self, *, built: BuiltInterface, work_dir: str | Path | WorkDir, restart: Optional[bool] = None):
        self.built = built
        self.work_dir = workdir(work_dir, restart=restart)
        self.restart = bool(self.work_dir.restart)
        self._resume = ResumeManager(Path(self.work_dir), enabled=self.restart, strict_inputs=True)

    def _preflight_protocol(self, protocol: InterfaceProtocol) -> tuple[InterfaceProtocol, dict[str, object]]:
        ndx_path = Path(self.built.system_ndx)
        if not ndx_path.exists():
            raise FileNotFoundError(f"Interface index file not found: {ndx_path}")
        top_path = Path(self.built.system_top)
        if not top_path.exists():
            raise FileNotFoundError(f"Interface topology file not found: {top_path}")

        topo_issues = validate_topology_include_order(top_path)
        if topo_issues:
            raise ValueError(
                "Interface system.top has an invalid include order: "
                + "; ".join(topo_issues)
                + f". Rebuild the interface before running dynamics: {top_path}"
            )

        groups = read_ndx_groups(ndx_path)
        missing_required = [name for name in ("System", "BOTTOM", "TOP") if not groups.get(name)]
        if missing_required:
            raise ValueError(
                "Interface system.ndx is missing required non-empty groups: "
                + ", ".join(missing_required)
                + f". Rebuild the interface before running dynamics: {ndx_path}"
            )

        notes: list[str] = []
        effective = protocol
        if str(protocol.axis or "Z").strip().upper() != str(self.built.axis or "Z").strip().upper():
            effective = replace(effective, axis=str(self.built.axis or "Z").strip().upper())
            notes.append(
                f"Aligned interface protocol axis to built interface axis: {protocol.axis} -> {effective.axis}"
            )
        try:
            system_meta = json.loads(Path(self.built.system_meta).read_text(encoding="utf-8"))
        except Exception:
            system_meta = {}
        interface_charge = system_meta.get("net_charge_e")
        if interface_charge is not None and abs(float(interface_charge)) > 1.0:
            raise ValueError(
                f"Interface system carries a large net charge ({float(interface_charge):.6f} e) before MD. "
                "This usually means slab construction discarded counter-ions or other charge-compensating fragments. "
                f"Rebuild the interface and inspect {self.built.system_meta}."
            )
        core_group_sizes = {
            "BOTTOM_CORE": len(groups.get("BOTTOM_CORE") or []),
            "TOP_CORE": len(groups.get("TOP_CORE") or []),
        }
        if effective.freeze_cores_pre_contact:
            missing_freeze = [name for name, size in core_group_sizes.items() if size <= 0]
            if missing_freeze:
                effective = replace(effective, freeze_cores_pre_contact=False)
                notes.append(
                    "Disabled early core freezing because these ndx groups were missing or empty: "
                    + ", ".join(missing_freeze)
                )

        if len(groups.get("INTERFACE_ZONE") or []) <= 0:
            notes.append(
                "INTERFACE_ZONE is empty; interface MD can still run, but downstream interface-specific analysis groups will be limited."
            )

        return effective, {
            "topology_issues": topo_issues,
            "interface_charge_e": float(interface_charge) if interface_charge is not None else None,
            "ndx_group_sizes": {name: len(idxs) for name, idxs in groups.items()},
            "core_group_sizes": core_group_sizes,
            "notes": notes,
        }

    def run(
        self,
        *,
        protocol: InterfaceProtocol,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        restart: Optional[bool] = None,
    ) -> Path:
        rst_flag = resolve_restart(restart if restart is not None else self.restart)
        run_root = Path(self.work_dir)
        if (not rst_flag) and run_root.exists():
            for p in run_root.glob("0*_*"):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
        run_root.mkdir(parents=True, exist_ok=True)
        effective_protocol, preflight = self._preflight_protocol(protocol)
        manifest = run_root / "protocol.json"
        manifest.write_text(
            json.dumps(
                {
                    "requested_protocol": asdict(protocol),
                    "effective_protocol": asdict(effective_protocol),
                    "preflight": preflight,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        for note in preflight.get("notes", []):
            yadon_print(f"[WARN] {note}", level=1)
        stages = [EqStage(s.name, s.kind, s.mdp) for s in effective_protocol.stages()]
        res = RunResources(
            ntmpi=int(mpi),
            ntomp=int(omp),
            use_gpu=bool(gpu),
            gpu_id=str(gpu_id) if gpu_id is not None else None,
        )
        job = EquilibrationJob(
            gro=self.built.system_gro,
            top=self.built.system_top,
            ndx=self.built.system_ndx,
            out_dir=run_root,
            stages=stages,
            resources=res,
        )
        final_stage = run_root / stages[-1].name / "md.gro"
        spec = StepSpec(
            name=f"interface_dynamics_{Path(self.work_dir).name}",
            outputs=[final_stage, manifest],
            inputs={
                "protocol": asdict(effective_protocol),
                "gro": str(self.built.system_gro),
                "top": str(self.built.system_top),
                "ndx": str(self.built.system_ndx),
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="Run the staged interface MD protocol",
        )
        self._resume.run(spec, lambda: job.run(restart=bool(rst_flag)))
        return final_stage


__all__ = ["InterfaceDynamics", "InterfaceProtocol", "InterfaceStageSpec"]
