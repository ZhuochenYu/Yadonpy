"""Readable facade for layer-stack interface post-processing.

The facade mirrors the script style used in Example 02: construct an analyzer,
then call small methods named after the physical question.  Heavy numerical
work remains in :mod:`yadonpy.gmx.analysis.interface_profile`; this class mainly
keeps parameter handling, output locations, and summary extraction tidy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


class InterfaceAnalysis:
    """Layer-stack interface analysis helper returned by ``AnalyzeResult.interface``."""

    def __init__(
        self,
        analyzer,
        *,
        manifest_path: str | Path | None = None,
        analysis_profile: str = "interface_fast",
        bin_nm: float = 0.05,
        frame_stride: int | str = "auto",
        surface_distance_nm: float = 0.50,
        region_width_nm: float = 0.75,
        surface_grid_nm: float = 0.5,
        penetration_threshold_nm: float = 0.20,
        adsorption_min_residence_ps: float = 10.0,
        potential_reference: str = "zero_mean",
        split_electrodes: bool = False,
        report_potential_drop: bool = False,
        penetration_species: Sequence[str] | None = None,
        adsorption_species: Sequence[str] | None = None,
        phase_groups: Sequence[str] | None = None,
        out_dir: str | Path | None = None,
        compute_transport: bool = True,
        time_series_sample_count: int = 10,
        time_series_fps: float = 1.0,
        time_series_rdf: bool = True,
        time_series_concentration: bool = True,
        time_series_angles: bool = True,
        time_series_rdf_rmax_nm: float = 1.2,
        time_series_rdf_bin_nm: float = 0.02,
        resume: bool = False,
    ) -> None:
        self.analyzer = analyzer
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.analysis_profile = str(analysis_profile)
        self.bin_nm = float(bin_nm)
        self.frame_stride = frame_stride
        self.surface_distance_nm = float(surface_distance_nm)
        self.region_width_nm = float(region_width_nm)
        self.surface_grid_nm = float(surface_grid_nm)
        self.penetration_threshold_nm = float(penetration_threshold_nm)
        self.adsorption_min_residence_ps = float(adsorption_min_residence_ps)
        self.potential_reference = str(potential_reference)
        self.split_electrodes = bool(split_electrodes)
        self.report_potential_drop = bool(report_potential_drop)
        self.penetration_species = (
            tuple(str(x) for x in penetration_species)
            if penetration_species is not None
            else None
        )
        self.adsorption_species = tuple(str(x) for x in adsorption_species) if adsorption_species is not None else None
        self.phase_groups = (
            tuple(str(x) for x in phase_groups)
            if phase_groups is not None
            else self._phase_groups_from_manifest()
        )
        self.out_dir = Path(out_dir) if out_dir is not None else self._default_out_dir()
        self.compute_transport = bool(compute_transport)
        self.time_series_sample_count = int(max(1, time_series_sample_count))
        self.time_series_fps = float(time_series_fps)
        self.time_series_rdf = bool(time_series_rdf)
        self.time_series_concentration = bool(time_series_concentration)
        self.time_series_angles = bool(time_series_angles)
        self.time_series_rdf_rmax_nm = float(time_series_rdf_rmax_nm)
        self.time_series_rdf_bin_nm = float(time_series_rdf_bin_nm)
        self.resume = bool(resume)
        self._cache: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}

    def _default_out_dir(self) -> Path:
        try:
            return Path(self.analyzer._analysis_dir()) / "layer_stack_interface"
        except Exception:
            return Path("06_analysis") / "layer_stack_interface"

    def _phase_groups_from_manifest(self) -> tuple[str, ...]:
        if self.manifest_path is None or not self.manifest_path.exists():
            return ("GRAPHITE", "POLYMER", "ELECTROLYTE")
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return ("GRAPHITE", "POLYMER", "ELECTROLYTE")
        names = [
            str(layer.get("name"))
            for layer in payload.get("layers", [])
            if isinstance(layer, dict)
            and str(layer.get("kind", "")).lower() != "vacuum"
            and layer.get("name")
        ]
        return tuple(names) if names else ("GRAPHITE", "POLYMER", "ELECTROLYTE")

    def _run(self, **overrides: Any) -> dict[str, Any]:
        params = {
            "manifest_path": self.manifest_path,
            "analysis_profile": self.analysis_profile,
            "bin_nm": self.bin_nm,
            "frame_stride": self.frame_stride,
            "surface_distance_nm": self.surface_distance_nm,
            "region_width_nm": self.region_width_nm,
            "surface_grid_nm": self.surface_grid_nm,
            "penetration_threshold_nm": self.penetration_threshold_nm,
            "adsorption_min_residence_ps": self.adsorption_min_residence_ps,
            "potential_reference": self.potential_reference,
            "split_electrodes": self.split_electrodes,
            "report_potential_drop": self.report_potential_drop,
            "penetration_species": self.penetration_species,
            "adsorption_species": self.adsorption_species,
            "phase_groups": self.phase_groups,
            "out_dir": self.out_dir,
            "compute_transport": self.compute_transport,
            "time_series_analysis": False,
            "time_series_sample_count": self.time_series_sample_count,
            "time_series_fps": self.time_series_fps,
            "time_series_rdf": self.time_series_rdf,
            "time_series_concentration": self.time_series_concentration,
            "time_series_angles": self.time_series_angles,
            "time_series_rdf_rmax_nm": self.time_series_rdf_rmax_nm,
            "time_series_rdf_bin_nm": self.time_series_rdf_bin_nm,
            "resume": self.resume,
        }
        params.update({key: value for key, value in overrides.items() if value is not None})
        cache_key = tuple(sorted((str(key), repr(value)) for key, value in params.items()))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        result = self.analyzer.interface_profile(**params)
        self._cache[cache_key] = result
        return result

    def geometry_health(self, *, time_series_analysis: bool = False, **kwargs: Any) -> dict[str, Any]:
        return dict(self._run(time_series_analysis=bool(time_series_analysis), **kwargs).get("geometry_health") or {})

    def z_profiles(self, *, time_series_analysis: bool = False, **kwargs: Any) -> dict[str, Any]:
        result = self._run(time_series_analysis=bool(time_series_analysis), **kwargs)
        return {
            "outputs": {
                key: value
                for key, value in (result.get("outputs") or {}).items()
                if key in (
                    "z_density_profiles_csv",
                    "charge_density_profiles_csv",
                    "phase_z_quantiles_csv",
                    "z_profiles_svg",
                )
            },
            "phase_stats": (result.get("region_summary") or {}).get("phase_stats") or {},
        }

    def edl_profiles(
        self,
        *,
        split_electrodes: bool | None = None,
        potential_reference: str | None = None,
        report_potential_drop: bool | None = None,
        time_series_analysis: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = self._run(
            split_electrodes=split_electrodes,
            potential_reference=potential_reference,
            report_potential_drop=report_potential_drop,
            time_series_analysis=bool(time_series_analysis),
            **kwargs,
        )
        return dict(result.get("edl_profiles") or {})

    def penetration(
        self,
        *,
        species: Sequence[str] | None = None,
        time_series_analysis: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = self._run(penetration_species=species, time_series_analysis=bool(time_series_analysis), **kwargs)
        return dict(result.get("penetration") or {})

    def graphite_adsorption(
        self,
        *,
        species: Sequence[str] | None = None,
        surface_distance_nm: float | None = None,
        adsorption_min_residence_ps: float | None = None,
        time_series_analysis: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = self._run(
            adsorption_species=species,
            surface_distance_nm=surface_distance_nm,
            adsorption_min_residence_ps=adsorption_min_residence_ps,
            time_series_analysis=bool(time_series_analysis),
            **kwargs,
        )
        return dict(result.get("graphite_adsorption") or {})

    def region_transport(self, *, time_series_analysis: bool = False, **kwargs: Any) -> dict[str, Any]:
        return dict(
            self._run(time_series_analysis=bool(time_series_analysis), **kwargs).get("region_transport_summary") or {}
        )

    def coordination_by_region(self, *, time_series_analysis: bool = False, **kwargs: Any) -> dict[str, Any]:
        return dict(
            self._run(time_series_analysis=bool(time_series_analysis), **kwargs).get("coordination_by_region")
            or {}
        )

    def time_series(self, *, time_series_analysis: bool = False, **kwargs: Any) -> dict[str, Any]:
        return dict(self._run(time_series_analysis=bool(time_series_analysis), **kwargs).get("time_series") or {})

    def summary(self, *, time_series_analysis: bool = False, **kwargs: Any) -> dict[str, Any]:
        return self._run(time_series_analysis=bool(time_series_analysis), **kwargs)


__all__ = ["InterfaceAnalysis"]
