"""Controlled MRO orbit-fit experiments; no fits run unless ``--run`` is given.

See MRO_CAMPAIGN_SOL_INSTRUCTIONS.md for the experiment sequence and acceptance
checks. Each arc runs in a separate Python process (SPICE/MCD are not thread-safe).
"""

import argparse
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "mro_orbit_campaign"
ORIGINAL_NOTEBOOK = "https://github.com/tudat-team/tudatpy-examples/blob/master/estimation/mro_tnf_estimation.ipynb"
ARCS = [
    ("2012-01-01 03:18:01.965", "2012-01-04 01:58:15.132"),
    ("2012-01-04 02:25:09.706", "2012-01-07 02:55:23.122"),
    ("2012-01-07 03:23:14.407", "2012-01-10 02:03:27.113"),
    ("2012-01-10 02:22:44.539", "2012-01-13 02:52:58.104"),
    ("2012-01-13 03:15:38.112", "2012-01-16 02:05:51.095"),
    ("2012-01-16 02:22:44.352", "2012-01-19 02:52:57.085"),
    ("2012-01-19 03:17:17.831", "2012-01-22 01:57:31.076"),
]
CONDITION_NUMBER_LIMIT = 5.0e15
CONDITION_POLL_SECONDS = 0.1
MIN_POSITION_SIGMA_M = 100.0
MIN_VELOCITY_SIGMA_M_S = 0.1
REQUESTED_POSITIVE_SEED_OFFSETS_M = (1.0, 2.5, 10.0, 25.0, 100.0)
ADAPTIVE_CASE_IDS = {
    "020", "021", "035", "038", "040", "043", "044", "045", "047",
    "058", "059", "060", "061", "062", "063", "064", "065", "066",
    "067", "068", "069", "070", "071", "072", "073", "074",
    "075", "076", "077",
    "078", "079", "080", "081", "082", "083", "084", "085",
}
FINAL_SHADOW_CASE_IDS = {
    "078", "079", "080", "081", "082", "083", "084", "085",
}
QUEUE_PRIORITY_CASE_IDS = (
    "007", "046", "047", "043", "044", "045", "048", "049", "050",
    "008", "009", "051", "011", "013", "035", "038", "052", "053",
    "054", "056", "057", "055", "033", "019", "010", "018", "022",
    "023", "024", "025", "012", "014", "015", "040", "034", "020", "021",
    "058", "059", "060", "061", "062", "063", "064", "065", "066",
    "067", "068", "069", "070", "071", "072", "073", "074",
    "075", "076", "077",
    "078", "079", "080", "081", "082", "083", "084", "085",
)
DEFERRED_CASE_REASONS = {
    "026": "covered by the exact frozen-variational repeat case053",
    "027": "covered by the reintegrated-variational comparison case054",
    "028": "covered by the exact five-evaluated-iterate repeat case053",
    "029": "covered by the 15-second numerical comparison case046",
    "030": "covered by the 60-second numerical comparison case006",
}
CANCELLED_CASE_REASONS = {
    "079": (
        "terminated and deleted at explicit user request on 2026-09-28; "
        "partial outputs are intentionally unrecoverable and must not be relaunched"
    ),
    **{
        case_id: (
            "cancelled by user before launch on 2026-09-28 after ending the "
            "self-shadowing phase; do not replace or relaunch"
        )
        for case_id in ("080", "081", "082", "083", "084", "085")
    },
}
CONDITION_PATTERN = re.compile(
    r"condition number is\s*"
    r"(?P<value>[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan))",
    re.IGNORECASE,
)


class ConditionLimitError(RuntimeError):
    """Legacy exception retained only for reading/testing archived hard-gate attempts."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def seed_offset_variant(control, offset_m):
    """Return one requested +[d,d,d] m variant without changing other settings."""
    if tuple(float(value) for value in control.initial_position_offset_m) != (0.0, 0.0, 0.0):
        raise ValueError("Seed-matrix controls must use the unperturbed zero-offset seed.")
    value = float(offset_m)
    if value not in REQUESTED_POSITIVE_SEED_OFFSETS_M:
        raise ValueError(
            f"Seed offset must be one of {REQUESTED_POSITIVE_SEED_OFFSETS_M}."
        )
    return replace(
        control,
        description=(f"Seed/prior-centre robustness variant of selected control: "
                     f"+[{value:g},{value:g},{value:g}] m"),
        initial_position_offset_m=(value, value, value),
    )


class MissingConditionDiagnosticsError(RuntimeError):
    """Legacy exception retained only for archived hard-gate diagnostics."""


class ConditionLogScanner:
    """Incrementally parse complete log lines without truncating split exponents."""

    def __init__(self, threshold=CONDITION_NUMBER_LIMIT):
        self.threshold = float(threshold)
        self.offset = 0
        self.buffer = ""
        self.records = []

    def feed(self, chunk, final=False):
        self.buffer += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        lines = self.buffer.splitlines(keepends=True)
        self.buffer = ""
        if lines and not lines[-1].endswith(("\n", "\r")) and not final:
            self.buffer = lines.pop()
        elif final and self.buffer:
            lines.append(self.buffer)
            self.buffer = ""
        new = []
        for line in lines:
            for match in CONDITION_PATTERN.finditer(line):
                raw = match.group("value")
                value = float(raw)
                finite = math.isfinite(value)
                record = {
                    "iteration": len(self.records),
                    "raw": raw,
                    "value": value if finite else None,
                    "finite": finite,
                    "within_reference_ceiling": finite and value <= self.threshold,
                    # Compatibility field for the archived hard-gate attempts.
                    # Subsequent runs treat this only as a quality flag.
                    "passed": finite and value <= self.threshold,
                }
                self.records.append(record)
                new.append(record)
        return new

    def read_path(self, path, final=False):
        path = Path(path)
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read()
                self.offset = stream.tell()
            new = self.feed(chunk, final=final)
        else:
            new = self.feed(b"", final=final)
        return new

    def diagnostics(self, arc_index):
        finite = [record["value"] for record in self.records if record["finite"]]
        invalid = [record for record in self.records if not record["passed"]]
        reason = ""
        if not self.records:
            reason = "missing condition-number report"
        elif invalid:
            reason = ("non-finite condition-number diagnostic"
                      if not invalid[0]["finite"] else "reference condition-number ceiling exceeded")
        return {
            "arc_index": int(arc_index),
            "threshold": self.threshold,
            "values": self.records,
            "maximum_finite": max(finite) if finite else None,
            "within_reference_ceiling": bool(self.records) and not invalid,
            "passed": bool(self.records) and not invalid,
            "diagnostic_only": True,
            "reason": reason,
        }


@dataclass(frozen=True)
class Case:
    """One reproducible model/parameter choice, independent of inherited MRO variables."""

    description: str = "Control: projected-area aerodynamics, global scales, T/N empirical terms"
    lift: bool = True
    aerodynamic_model: str = "variable_cross_section"
    reduced_solar_arrays: bool = False
    mars_radiation_target: str = "cannonball"
    sun_radiation_shadowing_pixels: int = 0
    aerodynamic_shadowing_pixels: int = 0
    mars_radiation_shadowing_pixels: int = 0
    mcd_scenario: int = 1
    mcd_high_resolution: int = 0
    mcd_data_path: str = str(HERE.parents[2] / "third_parties" / "mcd" / "data")
    drag_scale: str = "global"
    lift_scale: str = "global"
    sun_scale: str = "global"
    empirical_components: str = "TN"
    empirical_shapes: tuple = ("constant", "sine", "cosine")
    empirical_periods: float = 1.0
    empirical_edge_policy: str = "uniform"
    step_seconds: float = 30.0
    integrator: str = "rkf78"
    iterations: int = 5
    reintegrate_variational: bool = False
    apply_apriori_parameter_deviation: bool = True
    constrain_initial_state_prior: bool = True
    observation_sigma_hz: float = 0.003
    position_sigma_m: float = 1000.0
    velocity_sigma_m_s: float = 0.1
    scale_sigma: float = 2.0
    # Optional drag-only override.  ``None`` preserves every historical case,
    # where drag, lift, and Sun scaling shared ``scale_sigma``.  This field is
    # needed for matched drag-prior studies that must leave the Sun prior fixed.
    drag_scale_sigma: float | None = None
    empirical_sigma_m_s2: float = 1.0e-6
    constant_empirical_sigma_m_s2: float = 1.0e-6
    radial_empirical_sigma_m_s2: float | None = None
    normal_empirical_sigma_m_s2: float | None = None
    residual_cutoff_hz: float = 0.008
    score_step_seconds: float = 60.0
    condition_number_limit: float = CONDITION_NUMBER_LIMIT
    initial_position_offset_m: tuple = (0.0, 0.0, 0.0)

    def validate(self):
        """Reject ambiguous or physically inconsistent parameterizations before loading data."""
        for mode in (self.drag_scale, self.lift_scale, self.sun_scale):
            if mode not in {"fixed", "global", "arcwise"}:
                raise ValueError(f"Unknown scale mode: {mode}")
        if not self.lift and self.lift_scale != "fixed":
            raise ValueError("Zero lift requires fixed lift scaling (no unobservable parameter).")
        if self.aerodynamic_model not in {"variable_cross_section", "storch"}:
            raise ValueError("Only projected-area and Storch aerodynamic models are supported.")
        if self.aerodynamic_model == "storch" and not self.lift:
            raise ValueError("Storch's physical transverse force cannot be removed with C_L=0.")
        if self.mars_radiation_target not in {"cannonball", "panelled"}:
            raise ValueError("Unknown Mars radiation target.")
        for name in ("sun_radiation_shadowing_pixels",
                     "aerodynamic_shadowing_pixels",
                     "mars_radiation_shadowing_pixels"):
            if getattr(self, name) not in {0, 20}:
                raise ValueError(f"{name} must be disabled (0) or enabled at 20 pixels.")
        if (self.mars_radiation_target != "panelled"
                and self.mars_radiation_shadowing_pixels):
            raise ValueError("Mars target shadowing requires the panelled Mars target.")
        if ("arcwise" in (self.drag_scale, self.lift_scale)
                and "T" in self.empirical_components):
            raise ValueError("Arc-wise aerodynamic scaling must not estimate T empirical terms.")
        if any(c not in "RTN" for c in self.empirical_components) or len(set(self.empirical_components)) != len(self.empirical_components):
            raise ValueError("Empirical components must be distinct R, T, N entries.")
        if not set(self.empirical_shapes) <= {"constant", "sine", "cosine"} or len(set(self.empirical_shapes)) != len(self.empirical_shapes):
            raise ValueError("Unknown or repeated empirical shape.")
        if bool(self.empirical_components) != bool(self.empirical_shapes):
            raise ValueError("Supply both empirical components and shapes, or neither.")
        if self.empirical_edge_policy not in {
                "uniform", "merge_case001_zero_edges",
                "merge_one_orbit_conservative_edges",
                "merge_one_orbit_h_zero_edges"}:
            raise ValueError("Unknown empirical edge policy.")
        if self.integrator not in {"rkf56", "rkf78"}:
            raise ValueError("Integrator must be 'rkf56' or 'rkf78'.")
        if (self.empirical_edge_policy == "merge_case001_zero_edges"
                and self.empirical_periods != 2.0):
            raise ValueError("The audited edge merge is defined only for two-orbit intervals.")
        if (self.empirical_edge_policy in {
                "merge_one_orbit_conservative_edges",
                "merge_one_orbit_h_zero_edges"}
                and self.empirical_periods != 1.0):
            raise ValueError(
                "The audited one-orbit edge merge is defined only for one-orbit intervals."
            )
        for name in ("empirical_periods", "step_seconds", "observation_sigma_hz",
                     "position_sigma_m", "velocity_sigma_m_s", "scale_sigma",
                     "empirical_sigma_m_s2", "constant_empirical_sigma_m_s2",
                     "residual_cutoff_hz", "score_step_seconds", "condition_number_limit"):
            value = getattr(self, name)
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("drag_scale_sigma", "radial_empirical_sigma_m_s2",
                     "normal_empirical_sigma_m_s2"):
            value = getattr(self, name)
            if value is not None and not 0 < value < float("inf"):
                raise ValueError(f"{name} must be None or finite and positive.")
        if self.position_sigma_m < MIN_POSITION_SIGMA_M:
            raise ValueError(
                f"position_sigma_m may not be tighter than {MIN_POSITION_SIGMA_M} m."
            )
        if self.velocity_sigma_m_s < MIN_VELOCITY_SIGMA_M_S:
            raise ValueError(
                f"velocity_sigma_m_s may not be tighter than {MIN_VELOCITY_SIGMA_M_S} m/s."
            )
        if self.iterations < 2:
            raise ValueError("Use at least two iterations to assess convergence.")
        if not isinstance(self.apply_apriori_parameter_deviation, bool):
            raise ValueError("apply_apriori_parameter_deviation must be a boolean.")
        if not isinstance(self.constrain_initial_state_prior, bool):
            raise ValueError("constrain_initial_state_prior must be a boolean.")
        try:
            position_offset = tuple(
                float(value) for value in self.initial_position_offset_m
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "initial_position_offset_m must contain three finite Cartesian metres."
            ) from error
        if (len(position_offset) != 3
                or not all(math.isfinite(value) for value in position_offset)):
            raise ValueError(
                "initial_position_offset_m must contain three finite Cartesian metres."
            )
        if self.mcd_scenario not in (*range(1, 9), *range(24, 36)):
            raise ValueError("Unsupported MCD scenario.")
        if self.mcd_high_resolution not in (0, 1):
            raise ValueError("MCD high-resolution flag must be 0 or 1.")

    @property
    def blocked_reason(self):
        """Explain native-kernel limitations without substituting a different force model."""
        if self.sun_scale == "arcwise":
            return ("No arc-wise panelled Sun radiation scaling in this kernel; "
                    "arcwise_radiation_pressure_coefficient affects the cannonball target.")
        directory = Path(self.mcd_data_path)
        folders = {1: ["clim_aveEUV"], 2: ["clim_aveEUV", "clim_minEUV"],
                   3: ["clim_aveEUV", "clim_maxEUV"], 4: ["strm"], 5: ["strm"],
                   6: ["strm"], 7: ["warm"], 8: ["cold"]}.get(
                       self.mcd_scenario, [f"MY{self.mcd_scenario}"])
        missing = [folder for folder in folders if not (directory / folder).is_dir()]
        if missing:
            return f"MCD scenario {self.mcd_scenario} data unavailable in {directory}: {missing}"
        return None

    def environment(self):
        """Return all MRO environment overrides used by the shared interactive example."""
        return {
            "MRO_ATMOSPHERE_MODEL": "mcd", "MRO_MCD_DUST_SCENARIO": str(self.mcd_scenario),
            "MRO_MCD_HIGH_RESOLUTION": str(self.mcd_high_resolution),
            "MRO_MCD_DATA_PATH": self.mcd_data_path,
            # Keep the two legacy fallbacks disabled and record each actual
            # target/source shadowing control independently.
            "MRO_SELF_SHADOWING_PIXELS": "0",
            "MRO_RADIATION_SELF_SHADOWING_PIXELS": "0",
            "MRO_AERODYNAMIC_SELF_SHADOWING_PIXELS": str(
                self.aerodynamic_shadowing_pixels
            ),
            "MRO_SUN_RADIATION_SELF_SHADOWING_PIXELS": str(
                self.sun_radiation_shadowing_pixels
            ),
            "MRO_MARS_RADIATION_SELF_SHADOWING_PIXELS": str(
                self.mars_radiation_shadowing_pixels
            ),
            "MRO_LIFT_COEFFICIENT": "0.01" if self.lift else "0",
            "MRO_AERODYNAMIC_COEFFICIENT_MODEL": self.aerodynamic_model,
            "MRO_REDUCED_SOLAR_ARRAY_MACROMODEL": str(int(self.reduced_solar_arrays)),
            "MRO_MARS_RADIATION_TARGET": self.mars_radiation_target,
            "MRO_INTEGRATION_STEP_SIZE": str(self.step_seconds),
            "MRO_INTEGRATOR_COEFFICIENT_SET": self.integrator,
            "MRO_MAXIMUM_ITERATIONS": str(self.iterations),
            "MRO_REINTEGRATE_VARIATIONAL_EQUATIONS": str(int(self.reintegrate_variational)),
            "MRO_APPLY_APRIORI_PARAMETER_DEVIATION": str(
                int(self.apply_apriori_parameter_deviation)
            ),
            "MRO_OBSERVATION_SIGMA_HZ": str(self.observation_sigma_hz),
            "MRO_PREFIT_RESIDUAL_CUTOFF_HZ": str(self.residual_cutoff_hz),
            "MRO_PRINT_ESTIMATION_OUTPUT": "1", "MRO_PROPAGATION_PRINT_INTERVAL": "7200",
            "MRO_MARS_GRAVITY_DEGREE": "120", "MRO_PREFIT_ONLY": "0",
            "MRO_CONDITION_NUMBER_WARNING_LIMIT": "1.0",
            "MRO_INITIAL_POSITION_OFFSET_M": ",".join(
                str(float(value)) for value in self.initial_position_offset_m
            ),
        }


def cases():
    """Initial paired comparisons; later slots should be adapted to the measured leader."""
    initial = Case()
    guarded_baseline = replace(
        initial,
        description=("Guarded seven-arc baseline with fixed aerodynamic scales; "
                     "recommended forces and TN empiricals retained"),
        observation_sigma_hz=1.0, position_sigma_m=100.0,
        velocity_sigma_m_s=0.1, scale_sigma=0.2,
        drag_scale="fixed", lift_scale="fixed", empirical_periods=2.0,
        apply_apriori_parameter_deviation=False,
    )
    edge_baseline = replace(
        guarded_baseline,
        description=("Edge-merged comparison to case001: remove only empirically "
                     "proven zero two-orbit TN boundaries"),
        empirical_edge_policy="merge_case001_zero_edges",
    )
    anchored_edge_baseline = replace(
        edge_baseline,
        apply_apriori_parameter_deviation=True,
    )
    step_sensitivity = replace(
        anchored_edge_baseline,
        description=("Numerical sensitivity: anchored edge-merged case004 "
                     "with 60-second integration step"),
        step_seconds=60.0,
    )
    # Every unrun/adaptive definition starts from the corrected edge topology
    # and anchored-prior semantics. Cases 001--003 remain explicit legacy
    # evidence and are never mutated in place.
    base = anchored_edge_baseline
    no_lift = replace(base, lift=False, lift_scale="fixed")
    arc_drag = replace(no_lift, drag_scale="arcwise", empirical_components="N")
    radial = replace(no_lift, drag_scale="fixed", empirical_components="RN")
    definitions = [
        guarded_baseline,
        edge_baseline,
        replace(
            edge_baseline,
            description="Convergence check: case002 edge-merged baseline, eight iterations",
            iterations=8,
        ),
        replace(
            anchored_edge_baseline,
            description=("Anchored-prior counterpart of corrected case002: edge-merged "
                         "two-orbit TN baseline, five iterations"),
        ),
        replace(
            anchored_edge_baseline,
            description=("Anchored-prior convergence counterpart of case003: edge-merged "
                         "two-orbit TN baseline, eight iterations"),
            iterations=8,
        ),
        step_sensitivity,
        replace(
            anchored_edge_baseline,
            description=("Full-geometry Storch aerodynamic-law comparison; "
                         "exact case004 control except aerodynamic law"),
            aerodynamic_model="storch",
            reduced_solar_arrays=False,
        ),
        replace(
            base,
            description=("Empirical-shape diagnostic: exact case044 with TN "
                         "sine and cosine terms only"),
            empirical_shapes=("sine", "cosine"), integrator="rkf56",
            empirical_sigma_m_s2=3.0e-6,
            constant_empirical_sigma_m_s2=3.0e-6,
        ),
        replace(
            base,
            description=("Empirical-shape diagnostic: exact case044 with TN "
                         "constant terms only"),
            empirical_shapes=("constant",), integrator="rkf56",
            empirical_sigma_m_s2=3.0e-6,
            constant_empirical_sigma_m_s2=3.0e-6,
        ),
        replace(no_lift, description="No empirical terms", empirical_components="", empirical_shapes=()),
        replace(arc_drag, description="Arc-wise drag replaces T empirical estimation"),
        replace(base, description="Arc-wise drag/lift replaces T empirical estimation",
                drag_scale="arcwise", lift_scale="arcwise", empirical_components="N"),
        replace(arc_drag, description="Arc-wise drag, R/N empirical terms", empirical_components="RN"),
        replace(arc_drag, description="Arc-wise drag, N sine/cosine only", empirical_shapes=("sine", "cosine")),
        replace(arc_drag, description="Arc-wise drag, no empirical terms", empirical_components="", empirical_shapes=()),
        replace(radial, description="Fixed aerodynamic scales, R/N constant/sine/cosine"),
        replace(radial, description="Fixed aerodynamic scales, R/N sine/cosine", empirical_shapes=("sine", "cosine")),
        replace(no_lift, description="Fixed aerodynamic scales, R/T/N empirical terms",
                drag_scale="fixed", empirical_components="RTN"),
        replace(no_lift, description="Fixed Sun scale", sun_scale="fixed"),
        replace(
            base,
            description=("Approved adaptive case019 derivative: one-orbit TN "
                         "intervals with independently audited minimal edge merges"),
            sun_scale="fixed", integrator="rkf56",
            empirical_periods=1.0,
            empirical_edge_policy="merge_one_orbit_conservative_edges",
            empirical_sigma_m_s2=3.0e-6,
            constant_empirical_sigma_m_s2=3.0e-6,
        ),
        replace(
            base,
            description=("Approved adaptive case012 derivative: one-orbit "
                         "arc-wise drag/lift and N empirical intervals"),
            drag_scale="arcwise", lift_scale="arcwise", sun_scale="global",
            empirical_components="N", integrator="rkf56",
            empirical_periods=1.0,
            empirical_edge_policy="merge_one_orbit_conservative_edges",
            empirical_sigma_m_s2=3.0e-6,
            constant_empirical_sigma_m_s2=3.0e-6,
        ),
        replace(no_lift, description="Tighter empirical priors 1e-7", empirical_sigma_m_s2=1e-7, constant_empirical_sigma_m_s2=1e-7),
        replace(no_lift, description="Looser empirical priors 1e-5", empirical_sigma_m_s2=1e-5, constant_empirical_sigma_m_s2=1e-5),
        replace(
            base,
            description=("Arc-wise drag-prior comparison: exact case011 topology "
                         "with drag sigma 0.2 to 0.5"),
            drag_scale="arcwise", lift_scale="fixed", empirical_components="N",
            integrator="rkf56", scale_sigma=0.2, drag_scale_sigma=0.5,
            empirical_sigma_m_s2=3.0e-6,
            constant_empirical_sigma_m_s2=3.0e-6,
        ),
        replace(
            base,
            description=("Arc-wise drag-prior comparison: exact case011 topology "
                         "with drag sigma 0.2 to 5"),
            drag_scale="arcwise", lift_scale="fixed", empirical_components="N",
            integrator="rkf56", scale_sigma=0.2, drag_scale_sigma=5.0,
            empirical_sigma_m_s2=3.0e-6,
            constant_empirical_sigma_m_s2=3.0e-6,
        ),
        replace(no_lift, description="Frozen-partial control repeated", reintegrate_variational=False),
        replace(no_lift, description="Reintegrated variational equations", reintegrate_variational=True),
        replace(no_lift, description="Repeated five-iteration anchored control"),
        replace(no_lift, description="Integration check 15 s (rebase onto selected leader)", step_seconds=15),
        replace(no_lift, description="Integration check 60 s (rebase onto same leader)", step_seconds=60),
        replace(no_lift, description="BLOCKED: arc-wise panelled Sun scale", sun_scale="arcwise"),
        replace(arc_drag, description="BLOCKED: arc-wise aerodynamic and Sun scales", sun_scale="arcwise"),
        replace(no_lift, description="LAST: panelled Mars radiation target", mars_radiation_target="panelled"),
        replace(base, description="LAST: panelled Mars radiation and Storch",
                mars_radiation_target="panelled", aerodynamic_model="storch"),
        replace(
            base,
            description=("ADAPTIVE MCD: high-resolution scenario1 on selected "
                         "fixed-aero numerical control"),
            mcd_high_resolution=1,
        ),
        replace(no_lift, description="UNSUPPORTED DATA: original MCD maximum-EUV concept", mcd_scenario=3),
        replace(no_lift, description="UNSUPPORTED DATA: original MCD MY31 global-aero concept", mcd_scenario=31),
        replace(
            base,
            description=("ADAPTIVE MCD: high-resolution scenario1 on selected "
                         "global-drag/lift numerical control"),
            drag_scale="global", lift_scale="global", mcd_high_resolution=1,
        ),
        replace(arc_drag, description="UNSUPPORTED DATA: original MCD MY31 arc-drag concept", mcd_scenario=31),
        replace(
            arc_drag,
            description=("ADAPTIVE MCD: high-resolution scenario1 on selected "
                         "arc-wise-drag numerical control"),
            mcd_high_resolution=1,
        ),
        replace(no_lift, description="UNSUPPORTED DATA: original MCD warm-extreme concept", mcd_scenario=7),
        replace(no_lift, description="UNSUPPORTED DATA: original MCD cold-extreme concept", mcd_scenario=8),
    ]
    for case in definitions:
        case.validate()
    numbered = {f"{i:03d}": case for i, case in enumerate(definitions, 1)}
    # User-cancelled cases retain their historical catalogue numbers so that
    # every later ID remains stable, but are absent from the runnable plan.
    for cancelled in ("016", "017"):
        numbered.pop(cancelled)
    numbered.update({
        "043": replace(
            anchored_edge_baseline,
            description=("PENDING NUMERICAL REBASE: restore global drag and "
                         "lift scales on the selected numerical control"),
            drag_scale="global", lift_scale="global",
        ),
        "044": replace(
            anchored_edge_baseline,
            description=("PENDING NUMERICAL REBASE: loosen TN constant/periodic "
                         "empirical sigmas to 3e-6 m/s^2 on selected control"),
            empirical_sigma_m_s2=3.0e-6,
            constant_empirical_sigma_m_s2=3.0e-6,
        ),
        "045": replace(
            anchored_edge_baseline,
            description=("PENDING NUMERICAL REBASE: turn lift coefficient/force "
                         "off on the selected numerical control"),
            lift=False, lift_scale="fixed",
        ),
        "046": replace(
            anchored_edge_baseline,
            description=("Numerical sensitivity: exact case004 RKF78 with "
                         "15-second fixed step"),
            step_seconds=15.0,
        ),
        "047": replace(
            anchored_edge_baseline,
            description=("PENDING CASE046 DECISION: RKF56 at the selected matched "
                         "15- or 30-second step"),
            integrator="rkf56",
        ),
    })
    return numbered


def local_inputs(arc_index):
    """Select local January kernels/media and overlapping TNF days, without downloads."""
    start, end = map(datetime.fromisoformat, ARCS[arc_index])
    directory = HERE / "mro_kernels"
    lower, upper = start - timedelta(days=1), end + timedelta(days=1)
    orientation = []
    for prefix in ("sc", "hga", "sa"):
        selected = []
        for path in sorted(directory.glob(f"mro_{prefix}_psp_*.bc")):
            dates = re.search(r"(\d{6})_(\d{6})\.bc$", path.name)
            first, last = [datetime.strptime(d, "%y%m%d") for d in dates.groups()]
            if first <= upper and last + timedelta(days=1) >= lower:
                selected.append(str(path))
        if not selected:
            raise FileNotFoundError(f"No {prefix} attitude kernels for arc {arc_index}.")
        orientation.extend(selected)
    media = []
    for suffix in ("tro", "ion"):
        selected = []
        for path in sorted(directory.glob(f"mromagr*.{suffix}")):
            dates = re.search(r"(\d{4}_\d{3})_(\d{4}_\d{3})", path.name)
            first, last = [datetime.strptime(d, "%Y_%j") for d in dates.groups()]
            if first <= upper and last >= lower:
                selected.append(str(path))
        if not selected:
            raise FileNotFoundError(f"No {suffix} corrections for arc {arc_index}.")
        media.append(selected)
    tnf = []
    for path in sorted(directory.glob("*.tnf")):
        stamp = re.search(r"mromagr(\d{4}_\d{3})", path.name).group(1)
        day = datetime.strptime(stamp, "%Y_%j").date()
        if lower.date() <= day <= end.date():
            tnf.append(str(path))
    if not tnf:
        raise FileNotFoundError(f"No TNF data for arc {arc_index}.")
    result = [arc_index, start, end, tnf,
              [str(directory / "mro_sclkscet_00112_65536.tsc")], orientation,
              *media, [str(directory / "mro_psp22.bsp")],
              str(directory / "mro_v16.tf"), str(directory / "mro_struct_v10.bsp")]
    for item in result[3:]:
        for path in item if isinstance(item, list) else [item]:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    return result


def write_json(path, value):
    """Write human-readable metadata; reject NaN rather than publishing invalid metrics."""
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_case(path):
    """Load a complete case configuration, rejecting misspelled or unsupported fields."""
    data = json.loads(Path(path).read_text())
    case = Case(**data)
    case.validate()
    return case


EDGE_MERGE_BOUNDARY_INDICES = {
    # case001 selected-iteration H: all TN columns at both starts were exactly zero.
    0: (18, 19),
    # Retain index 0 at propagation start; merge its unsupported span through index 1.
    5: (1,),
}

# Independent conservative receive-time/light-time/interpolation audit of the
# original one-orbit arrays.  Arc 05 index 0 remains the lookup anchor and its
# unsupported leading span is extended through the removed starts, matching
# the established no-extrapolation edge policy rather than deleting it.
ONE_ORBIT_EDGE_MERGE_BOUNDARY_INDICES = {
    0: (37, 38),
    5: (1, 2),
}

# Selected-iteration H audit of completed cases 020/021.  Keep the conservative
# policy above unchanged so their saved settings retain exact source semantics.
# Arc 05 start 0 remains the pre-lookup anchor; deleting starts 1--3 extends its
# coefficient through the first observation-sensitive interval.
ONE_ORBIT_H_ZERO_EDGE_MERGE_BOUNDARY_INDICES = {
    0: (36, 37, 38),
    1: (39,),
    3: (39,),
    5: (1, 2, 3, 39),
}


def empirical_arc_times(case, original_arc_times, arc_index=None):
    """Return the configured starts, optionally merging audited zero edge blocks."""
    import numpy as np
    first = original_arc_times[0]
    period = original_arc_times[1] - first
    end = original_arc_times[-1] + period
    arc_times = list(np.arange(
        first, end - period + 1.0e-6, period * case.empirical_periods
    ))
    if case.empirical_edge_policy in {
            "merge_case001_zero_edges", "merge_one_orbit_conservative_edges",
            "merge_one_orbit_h_zero_edges"}:
        if arc_index is None:
            raise ValueError("The audited empirical edge merge requires an explicit arc index.")
        remove_map = {
            "merge_case001_zero_edges": EDGE_MERGE_BOUNDARY_INDICES,
            "merge_one_orbit_conservative_edges":
                ONE_ORBIT_EDGE_MERGE_BOUNDARY_INDICES,
            "merge_one_orbit_h_zero_edges":
                ONE_ORBIT_H_ZERO_EDGE_MERGE_BOUNDARY_INDICES,
        }[case.empirical_edge_policy]
        remove = remove_map.get(int(arc_index), ())
        if any(index >= len(arc_times) for index in remove):
            raise ValueError(
                f"Arc {arc_index} has {len(arc_times)} empirical starts; "
                f"cannot remove {remove}."
            )
        arc_times = [epoch for index, epoch in enumerate(arc_times) if index not in remove]
    return arc_times


class ParameterPlan:
    """Bind priors and plot labels to actual Tudat parameter indices, not list order."""

    def __init__(self, case, arc_index=None):
        self.case = case
        self.arc_index = arc_index
        self.specs = []
        self.rows = []
        self.arc_times = []
        self.unconstrained_parameter_types = set()

    def settings(self, propagator, bodies, original_arc_times):
        """Build only the parameters requested by this case on shared subarc boundaries."""
        from tudatpy.dynamics import parameters_setup as p
        if self.case.blocked_reason:
            raise NotImplementedError(self.case.blocked_reason)
        self.specs = []
        self.unconstrained_parameter_types = set()
        self.arc_times = empirical_arc_times(
            self.case, original_arc_times, self.arc_index
        )
        settings = p.initial_states(propagator, bodies)
        self.specs.append((p.initial_body_state_type,
                           [(name, "m" if i < 3 else "m/s", None,
                             self.case.position_sigma_m if i < 3 else self.case.velocity_sigma_m_s)
                            for i, name in enumerate(("x", "y", "z", "vx", "vy", "vz"))]))
        if not self.case.constrain_initial_state_prior:
            self.unconstrained_parameter_types.add(p.initial_body_state_type)
        for component, mode in (("drag", self.case.drag_scale), ("lift", self.case.lift_scale)):
            if mode == "fixed":
                continue
            arcwise = mode == "arcwise"
            function = getattr(p, f"{'arcwise_' if arcwise else ''}{component}_component_scaling")
            settings.append(function("MRO", self.arc_times) if arcwise else function("MRO"))
            kind = getattr(p, f"{'arc_wise_' if arcwise else ''}{component}_component_scaling_factor_type")
            prior_sigma = (
                self.case.drag_scale_sigma
                if component == "drag" and self.case.drag_scale_sigma is not None
                else self.case.scale_sigma
            )
            self.specs.append((kind, [(f"{component}_scale", "1", epoch, prior_sigma)
                                     for epoch in self.arc_times if arcwise] if arcwise
                               else [(f"{component}_scale", "1", None, prior_sigma)]))
        if self.case.sun_scale == "global":
            settings.append(p.radiation_pressure_target_direction_scaling("MRO", "Sun"))
            self.specs.append((p.radiation_pressure_target_direction_scaling_factor_type,
                               [("sun_scale", "1", None, self.case.scale_sigma)]))
        if self.case.empirical_components:
            components = dict(zip("RTN", (
                p.EmpiricalAccelerationComponents.radial_empirical_acceleration_component,
                p.EmpiricalAccelerationComponents.along_track_empirical_acceleration_component,
                p.EmpiricalAccelerationComponents.across_track_empirical_acceleration_component)))
            shapes = {name: getattr(p.EmpiricalAccelerationFunctionalShapes, f"{name}_empirical")
                      for name in ("constant", "sine", "cosine")}
            selection = {components[c]: [shapes[s] for s in self.case.empirical_shapes]
                         for c in "RTN" if c in self.case.empirical_components}
            settings.append(p.arcwise_empirical_accelerations("MRO", "Mars", selection, self.arc_times))
            # C++ packs subarc first, then functional shape, then R/T/N, NOT component first.
            component_sigmas = {
                "R": self.case.radial_empirical_sigma_m_s2,
                "N": self.case.normal_empirical_sigma_m_s2,
            }
            labels = [(f"empirical_{c}_{s}", "m/s^2", epoch,
                       component_sigmas.get(c)
                       if component_sigmas.get(c) is not None
                       else (self.case.constant_empirical_sigma_m_s2
                             if s == "constant" else self.case.empirical_sigma_m_s2))
                      for epoch in self.arc_times
                      for s in sorted(self.case.empirical_shapes, key=lambda s: int(shapes[s]))
                      for c in "RTN" if c in self.case.empirical_components]
            self.specs.append((p.arc_wise_empirical_acceleration_coefficients_type, labels))
        return settings

    def priors(self, parameter_set):
        """Check complete, unique index coverage and create inverse covariance in SI units."""
        import numpy as np
        information = np.full(parameter_set.parameter_set_size, np.nan)
        assigned = np.zeros(parameter_set.parameter_set_size, dtype=bool)
        self.rows = []
        for kind, labels in self.specs:
            identifier = (kind, ("", ""))
            blocks = parameter_set.indices_for_parameter_type(identifier)
            parameters = parameter_set.parameters_for_parameter_type(identifier)
            if len(blocks) != 1 or len(parameters) != 1:
                raise ValueError(f"Expected exactly one parameter block for {kind}; got {blocks}.")
            start, size = blocks[0]
            if size != len(labels) or assigned[start:start + size].any():
                raise ValueError(f"Prior labels overlap or have wrong size for {kind}.")
            constrained = kind not in self.unconstrained_parameter_types
            for i, (name, unit, epoch, sigma) in enumerate(labels, start):
                assigned[i] = True
                information[i] = sigma ** -2 if constrained else 0.0
                self.rows.append(dict(index=i, name=name, unit=unit, subarc_start_tdb=epoch,
                                      prior_sigma=sigma if constrained else None,
                                      prior_constrained=constrained,
                                      prior_information_diagonal=information[i],
                                      parameter_type=str(kind)))
        if (not assigned.all() or not np.isfinite(information).all()
                or (information < 0.0).any()):
            raise ValueError("Unassigned, non-finite, or negative prior information.")
        self.rows.sort(key=lambda row: row["index"])
        return np.diag(information)


def orbit_comparison(history, start, end, step):
    """Compare to independent SPICE states, in the REFERENCE RTN basis, on a common grid."""
    import numpy as np
    import pandas as pd
    import spiceypy
    from tudatpy.math import interpolators
    epochs = np.arange(start, end + 1e-7, step)
    if start - min(history) < 8 * step or max(history) - end < 8 * step:
        raise ValueError("Scoring grid is too close to a propagated ephemeris boundary.")
    interpolation = interpolators.create_one_dimensional_vector_interpolator(
        history, interpolators.lagrange_interpolation(8))
    states = np.array([interpolation.interpolate(float(epoch)).reshape(6) for epoch in epochs])
    reference = np.array([spiceypy.spkezr("-74", float(epoch), "J2000", "NONE", "499")[0]
                          for epoch in epochs]) * 1000.0
    radial = reference[:, :3] / np.linalg.norm(reference[:, :3], axis=1)[:, None]
    normal = np.cross(reference[:, :3], reference[:, 3:])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    transverse = np.cross(normal, radial)
    delta = states[:, :3] - reference[:, :3]
    rtn = np.column_stack([np.einsum("ij,ij->i", delta, direction)
                           for direction in (radial, transverse, normal)])
    np.testing.assert_allclose(np.linalg.norm(rtn, axis=1), np.linalg.norm(delta, axis=1), atol=1e-9)
    return pd.DataFrame(dict(t=epochs, R=rtn[:, 0], T=rtn[:, 1], N=rtn[:, 2],
                             dx=delta[:, 0], dy=delta[:, 1], dz=delta[:, 2]))


def metrics(residuals, orbit):
    """Compute pooled (not averaged-RMS) statistics from samples in physical units."""
    import numpy as np
    if residuals.empty or orbit.empty:
        raise ValueError("Cannot score an empty arc.")
    position = orbit[["R", "T", "N"]].to_numpy()
    if not np.isfinite(position).all() or not np.isfinite(residuals[["spice", "postfit"]]).all(axis=None):
        raise ValueError("Non-finite residuals or orbit states.")
    return {
        "observations": len(residuals), "orbit_samples": len(orbit),
        "spice_rms_mhz": float(np.sqrt(np.mean(residuals.spice ** 2)) * 1e3),
        "residual_rms_mhz": float(np.sqrt(np.mean(residuals.postfit ** 2)) * 1e3),
        "residual_max_mhz": float(np.max(np.abs(residuals.postfit)) * 1e3),
        **{f"{c}_rms_m": float(np.sqrt(np.mean(orbit[c] ** 2))) for c in "RTN"},
        "position_rms_m": float(np.sqrt(np.mean(np.sum(position ** 2, axis=1)))),
        "position_max_m": float(np.max(np.linalg.norm(position, axis=1))),
    }


def orbit_only_metrics(orbit, prefix="full"):
    """Score one orbit frame without conflating it with a residual iteration."""
    import numpy as np
    if orbit.empty:
        raise ValueError("Cannot score an empty orbit frame.")
    position = orbit[["R", "T", "N"]].to_numpy()
    if not np.isfinite(position).all():
        raise ValueError("Non-finite RTN orbit state.")
    return {
        f"{prefix}_orbit_samples": len(orbit),
        **{
            f"{prefix}_{component}_rms_m": float(
                np.sqrt(np.mean(orbit[component] ** 2))
            )
            for component in "RTN"
        },
        f"{prefix}_position_rms_m": float(
            np.sqrt(np.mean(np.sum(position ** 2, axis=1)))
        ),
        f"{prefix}_position_max_m": float(
            np.max(np.linalg.norm(position, axis=1))
        ),
    }


def orbit_span_metrics(orbit, observation_min_tdb, observation_max_tdb):
    """Score observation-bracketed and outside-edge samples without changing primary scoring."""
    import numpy as np

    bracketed = orbit[
        (orbit.t >= observation_min_tdb) & (orbit.t <= observation_max_tdb)
    ].copy()
    edge = orbit[
        (orbit.t < observation_min_tdb) | (orbit.t > observation_max_tdb)
    ].copy()
    if bracketed.empty or edge.empty:
        raise ValueError("Both observation-bracketed and outside-edge scoring spans must be nonempty.")

    def score(frame, prefix):
        position = frame[["R", "T", "N"]].to_numpy()
        return {
            f"{prefix}_orbit_samples": len(frame),
            **{
                f"{prefix}_{component}_rms_m": float(np.sqrt(np.mean(frame[component] ** 2)))
                for component in "RTN"
            },
            f"{prefix}_position_rms_m": float(
                np.sqrt(np.mean(np.sum(position ** 2, axis=1)))
            ),
            f"{prefix}_position_max_m": float(np.max(np.linalg.norm(position, axis=1))),
        }

    summary = {
        "observation_bracket_min_tdb": float(observation_min_tdb),
        "observation_bracket_max_tdb": float(observation_max_tdb),
        **score(bracketed, "bracketed"),
        **score(edge, "outside_edge"),
    }
    return summary, bracketed, edge


def plot_results(directory, residuals, orbit, parameter_table):
    """One PDF: residuals, RTN errors, and readable groups of parameter histories."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    origin = orbit.t.min()
    with PdfPages(Path(directory) / "results.pdf") as pdf:
        for columns, data, units, title in (
                (["spice", "prefit", "postfit"], residuals, "mHz", "Observed minus computed Doppler"),
                (["R", "T", "N"], orbit, "m", "Estimated minus SPICE position (SPICE RTN basis)")):
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            for ax, column in zip(axes, columns):
                for arc, frame in data.groupby("arc_index", sort=True):
                    x = frame["time" if column in {"spice", "prefit", "postfit"} else "t"]
                    values = frame[column] * (1000 if units == "mHz" else 1)
                    ax.plot((x - origin) / 86400, values, ".", ms=1.5, label=f"arc {arc}")
                ax.set_ylabel(f"{column} [{units}]")
                ax.grid(alpha=.3)
            axes[0].legend(ncol=7, fontsize=8)
            axes[-1].set_xlabel("TDB days since first nominal arc start")
            fig.suptitle(title)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        names = list(parameter_table.name.unique())
        for offset in range(0, len(names), 6):
            fig, axes = plt.subplots(3, 2, figsize=(12, 9), squeeze=False)
            for ax, name in zip(axes.flat, names[offset:offset + 6]):
                selected = parameter_table[parameter_table.name == name]
                for arc, frame in selected.groupby("arc_index", sort=True):
                    frame = frame.sort_values("plot_start_tdb")
                    values = frame["delta" if name in {"x", "y", "z", "vx", "vy", "vz"} else "value"]
                    x = np.r_[frame.plot_start_tdb, frame.plot_end_tdb.iloc[-1]]
                    ax.step((x - origin) / 86400, np.r_[values, values.iloc[-1]], where="post", label=f"arc {arc}")
                suffix = " correction at midpoint" if name in {"x", "y", "z", "vx", "vy", "vz"} else ""
                ax.set_title(f"{name}{suffix} [{selected.unit.iloc[0]}]", fontsize=9)
                ax.grid(alpha=.3)
                ax.set_xlabel("TDB days since first nominal arc start")
            for ax in list(axes.flat)[len(names[offset:offset + 6]):]:
                ax.set_visible(False)
            fig.suptitle("Estimated coefficients (steps denote parameter validity, not instantaneous force)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def parameter_diagnostics(table):
    """Duration-weighted fitted scale/empirical magnitudes, excluding propagation padding.

    A scale of one is nominal. Report both its absolute magnitude and its departure
    from one. Missing coefficients are absent, never interpreted as fitted zeros.
    """
    import numpy as np
    rows = []
    for name, group in table.groupby("name", sort=False):
        if not (name.endswith("_scale") or name.startswith("empirical_")):
            continue
        durations = np.maximum(0., group.plot_end_tdb.to_numpy() - group.plot_start_tdb.to_numpy())
        if durations.sum() == 0:
            continue
        values = group.value.to_numpy()
        nominal = 1.0 if name.endswith("_scale") else 0.0
        rows.append(dict(name=str(name), unit=str(group.unit.iloc[0]),
                         rms_absolute=float(np.sqrt(np.average(values ** 2, weights=durations))),
                         mean_absolute=float(np.average(np.abs(values), weights=durations)),
                         rms_correction=float(np.sqrt(np.average((values - nominal) ** 2, weights=durations))),
                         maximum_absolute=float(np.max(np.abs(values))),
                         mean_signed=float(np.average(values, weights=durations))))
    return rows


def prepare_prefit(frame, directory, reference_directory):
    """Save the diagnostic before propagation and enforce an identical data mask across cases."""
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from mro_tnf_estimation import plot_spice_residual_diagnostics
    if frame.empty or not np.isfinite(frame.spice).all():
        raise ValueError("No finite prefit data; do not propagate.")
    frame.to_csv(directory / "spice_residuals.csv", index=False)
    write_json(directory / "prefit_summary.json", dict(frame.attrs, retained_count=len(frame),
                                                        rejected_count=frame.attrs.get("unfiltered_count", len(frame)) - len(frame)))
    plot_spice_residual_diagnostics(frame)
    plt.gcf().savefig(directory / "prefit.pdf")
    plt.close("all")
    if reference_directory is not None:
        reference = pd.read_csv(reference_directory / "spice_residuals.csv")
        if not frame[["time", "link_ends", "msrType"]].reset_index(drop=True).equals(
                reference[["time", "link_ends", "msrType"]]):
            # CSV round trips can move the last bit of a floating-point epoch.
            if (len(frame) != len(reference)
                    or not frame[["link_ends", "msrType"]].reset_index(drop=True).equals(reference[["link_ends", "msrType"]])
                    or not np.allclose(frame.time, reference.time, rtol=0, atol=1e-6)):
                raise ValueError("Observation identity/order differs from reference; comparison invalid.")
        np.testing.assert_allclose(frame.spice, reference.spice, rtol=0, atol=1e-7)


def _prior_cost_group(name):
    """Return a stable scientific block label for per-iteration prior costs."""
    if name in {"x", "y", "z"}:
        return "initial_position"
    if name in {"vx", "vy", "vz"}:
        return "initial_velocity"
    if name.startswith("empirical_"):
        return name
    return name


def iteration_objective_history(output, parameter_rows, weights,
                                inverse_apriori, anchored_priors):
    """Reconstruct Tudat's per-iteration data, prior, and total objective.

    The a-priori reference is the frozen parameter vector in history column 0.
    In anchored mode this mirrors the native objective used for
    ``best_iteration`` (there are no inter-arc constraints in this campaign).
    """
    import numpy as np
    import pandas as pd

    residuals = np.asarray(output.residual_history, dtype=float)
    if residuals.ndim == 1:
        residuals = residuals[:, None]
    parameters = np.asarray(output.parameter_history, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    inverse_apriori = np.asarray(inverse_apriori, dtype=float)
    metadata = pd.DataFrame(parameter_rows).sort_values("index").reset_index(drop=True)
    count = len(metadata)
    if (residuals.ndim != 2 or parameters.ndim != 2
            or residuals.shape[0] != weights.size
            or parameters.shape[0] != count
            or parameters.shape[1] < residuals.shape[1]
            or inverse_apriori.shape != (count, count)):
        raise ValueError(
            "Cannot reconstruct iteration objective from inconsistent residual, "
            "parameter, weight, prior, or metadata dimensions."
        )
    if not isinstance(anchored_priors, bool):
        raise ValueError("anchored_priors must be a boolean.")

    reference = parameters[:, 0].copy()
    group_labels = [_prior_cost_group(str(name)) for name in metadata.name]
    groups = list(dict.fromkeys(group_labels))
    rows = []
    for iteration in range(residuals.shape[1]):
        residual = residuals[:, iteration]
        deviation = parameters[:, iteration] - reference
        weighted_residual_sum_squares = float(np.dot(weights * residual, residual))
        prior_vector = inverse_apriori @ deviation
        absolute_prior_cost = float(0.5 * np.dot(deviation, prior_vector))
        data_cost = 0.5 * weighted_residual_sum_squares
        objective_prior_cost = absolute_prior_cost if anchored_priors else 0.0
        total_cost = data_cost + objective_prior_cost
        element_costs = 0.5 * deviation * prior_vector
        group_costs = {}
        for group in groups:
            mask = np.asarray([label == group for label in group_labels])
            group_costs[group] = float(np.sum(element_costs[mask]))
        finite = bool(np.isfinite([
            weighted_residual_sum_squares, data_cost, absolute_prior_cost,
            objective_prior_cost, total_cost, *group_costs.values(),
        ]).all())
        rows.append({
            "iteration": iteration,
            "data_cost_half_rWr": data_cost if np.isfinite(data_cost) else None,
            "weighted_residual_sum_squares": (
                weighted_residual_sum_squares
                if np.isfinite(weighted_residual_sum_squares) else None
            ),
            "absolute_prior_cost_half_delta_Pinv_delta": (
                absolute_prior_cost if np.isfinite(absolute_prior_cost) else None
            ),
            "objective_prior_cost": (
                objective_prior_cost if np.isfinite(objective_prior_cost) else None
            ),
            "total_cost": total_cost if np.isfinite(total_cost) else None,
            "prior_cost_by_parameter_group": {
                group: value if np.isfinite(value) else None
                for group, value in group_costs.items()
            },
            "finite": finite,
        })

    finite_total = np.asarray([
        np.inf if row["total_cost"] is None else row["total_cost"] for row in rows
    ])
    finite_data = np.asarray([
        np.inf if row["data_cost_half_rWr"] is None else row["data_cost_half_rWr"]
        for row in rows
    ])
    reconstructed_best = (
        int(np.argmin(finite_total)) if np.isfinite(finite_total).any() else None
    )
    data_only_best = (
        int(np.argmin(finite_data)) if np.isfinite(finite_data).any() else None
    )
    native_best = int(output.best_iteration)
    selected_total = (
        finite_total[native_best]
        if 0 <= native_best < len(finite_total) else np.inf
    )
    minimum_total = float(np.min(finite_total)) if finite_total.size else np.inf
    selection_tolerance = 1.0e-12 * max(
        1.0, abs(minimum_total), abs(float(selected_total))
    )
    selection_matches = bool(
        np.isfinite(selected_total)
        and selected_total <= minimum_total + selection_tolerance
    )
    return {
        "apply_apriori_parameter_deviation": anchored_priors,
        "reference_parameter_history_column": 0,
        "reference_semantics": "parameter vector frozen at estimation start",
        "data_cost_definition": "0.5 * residual.T @ diag(weight_matrix_diagonal) @ residual",
        "absolute_prior_cost_definition": (
            "0.5 * (parameter_history[:,i]-parameter_history[:,0]).T @ "
            "inverse_apriori_covariance_physical @ "
            "(parameter_history[:,i]-parameter_history[:,0])"
        ),
        "total_cost_definition": (
            "data_cost + absolute_prior_cost" if anchored_priors else
            "data_cost (legacy correction-only prior contributes no absolute prior cost)"
        ),
        "native_best_iteration": native_best,
        "reconstructed_total_cost_best_iteration": reconstructed_best,
        "data_only_best_iteration": data_only_best,
        "native_best_matches_reconstructed_total_cost": selection_matches,
        "native_best_exactly_equals_reconstructed_argmin": native_best == reconstructed_best,
        "best_selection_cost_absolute_tolerance": selection_tolerance,
        "native_best_differs_from_data_only_best": native_best != data_only_best,
        "per_iteration": rows,
    }


def save_estimation_diagnostics(output, observation_table, parameter_rows,
                                weights, inverse_apriori, directory,
                                anchored_priors=False):
    """Persist residual history and the selected-iteration normal-matrix audit.

    Tudat's public ``EstimationOutput`` retains the design matrix and normal
    matrix for ``best_iteration`` only, while ``residual_history`` retains every
    evaluated iteration. The filenames and metadata below make that distinction
    explicit. The normalized prior is reconstructed directly from the exact
    physical inverse prior and Tudat normalization terms; it is never inferred
    by subtracting two large matrices.
    """
    import numpy as np
    import pandas as pd

    directory = Path(directory)
    parameters = pd.DataFrame(parameter_rows).sort_values("index").reset_index(drop=True)
    parameters.to_csv(directory / "parameter_prior_metadata.csv", index=False)
    observations = observation_table[
        ["time", "link_id", "link_ends", "msrType", "spice"]
    ].reset_index(drop=True).rename(columns={"spice": "spice_prefit_residual_hz"})
    residual_history = np.asarray(output.residual_history, dtype=float)
    if residual_history.ndim == 1 and residual_history.size:
        residual_history = residual_history[:, None]
    if residual_history.ndim != 2 or residual_history.shape[0] != len(observations):
        write_json(directory / "estimation_output_availability.json", {
            "available": False,
            "reason": (f"residual history shape {residual_history.shape} does not match "
                       f"{len(observations)} retained observations"),
            "exception_during_inversion": bool(output.exception_during_inversion),
            "exception_during_propagation": bool(output.exception_during_propagation),
        })
        return None

    # Retain the raw history and each column before any all-history finiteness
    # decision. A bad later correction must not erase a finite iteration 0/1.
    np.save(directory / "residual_history.npy", residual_history)
    residual_iteration_flags = []
    for iteration in range(residual_history.shape[1]):
        iteration_finite = np.isfinite(residual_history[:, iteration])
        residuals = observations.copy()
        residuals["residual_hz"] = residual_history[:, iteration]
        residuals["estimation_iteration"] = iteration
        residuals["residual_stage"] = (
            "initial_propagated_before_any_differential_correction"
            if iteration == 0 else f"post_correction_evaluation_{iteration}"
        )
        residuals.to_csv(
            directory / f"propagated_residual_iteration_{iteration:02d}.csv", index=False
        )
        residual_iteration_flags.append({
            "iteration": iteration,
            "stage": residuals["residual_stage"].iloc[0],
            "finite": bool(iteration_finite.all()),
            "nonfinite_count": int((~iteration_finite).sum()),
        })
    write_json(directory / "residual_history_diagnostics.json", {
        "iteration_count": int(residual_history.shape[1]),
        "all_finite": all(item["finite"] for item in residual_iteration_flags),
        "per_iteration": residual_iteration_flags,
    })

    # Save objective evidence before selected-matrix validation. A malformed
    # later matrix must not erase finite iteration histories and their costs.
    objective_history = iteration_objective_history(
        output, parameter_rows, weights, inverse_apriori, anchored_priors
    )
    write_json(directory / "iteration_objective_history.json", objective_history)
    objective_rows = []
    for row in objective_history["per_iteration"]:
        flat = {
            key: value for key, value in row.items()
            if key != "prior_cost_by_parameter_group"
        }
        flat.update({
            f"prior_cost_{group}": value
            for group, value in row["prior_cost_by_parameter_group"].items()
        })
        objective_rows.append(flat)
    pd.DataFrame(objective_rows).to_csv(
        directory / "iteration_objective_history.csv", index=False
    )

    design = np.asarray(output.normalized_design_matrix, dtype=float)
    normalization = np.asarray(output.normalization_terms, dtype=float).reshape(-1)
    native_total = np.asarray(output.inverse_normalized_covariance, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    inverse_apriori = np.asarray(inverse_apriori, dtype=float)
    count = len(parameters)
    matrix_iteration = int(output.best_iteration)
    # These are the unmodified public/input arrays. Save them before dimension or
    # finiteness validation so a failing native output remains diagnosable.
    np.savez_compressed(
        directory / "normal_matrix_inputs_raw.npz",
        normalized_design_matrix=design,
        weight_matrix_diagonal=weights,
        normalization_terms=normalization,
        inverse_apriori_covariance_physical=inverse_apriori,
        inverse_normalized_covariance_native=native_total,
    )
    matrix_input_flags = {
        "matrix_iteration": matrix_iteration,
        "normalized_design_shape": list(design.shape),
        "normalization_shape": list(normalization.shape),
        "native_total_shape": list(native_total.shape),
        "inverse_apriori_shape": list(inverse_apriori.shape),
        "weights_shape": list(weights.shape),
        "normalized_design_finite": bool(np.isfinite(design).all()),
        "normalization_finite": bool(np.isfinite(normalization).all()),
        "native_total_finite": bool(np.isfinite(native_total).all()),
        "inverse_apriori_finite": bool(np.isfinite(inverse_apriori).all()),
        "weights_finite": bool(np.isfinite(weights).all()),
    }
    write_json(directory / "normal_matrix_input_diagnostics.json", matrix_input_flags)
    if (design.ndim != 2 or design.shape != (len(observations), count)
            or normalization.shape != (count,)
            or native_total.shape != (count, count)
            or inverse_apriori.shape != (count, count)
            or weights.shape != (len(observations),)):
        write_json(directory / "estimation_output_availability.json", {
            "available": False,
            "reason": "selected-iteration matrix dimensions do not match observation/parameter metadata",
            **matrix_input_flags,
        })
        return None
    if not all(np.isfinite(item).all() for item in
               (design, normalization, native_total, weights, inverse_apriori)):
        raise ValueError("Non-finite selected-iteration matrix diagnostic returned by estimation.")
    if np.any(normalization == 0.0) or np.any(weights < 0.0):
        raise ValueError("Invalid zero normalization term or negative observation weight.")

    prior_constrained = (
        parameters.prior_constrained.to_numpy(dtype=bool)
        if "prior_constrained" in parameters else np.ones(count, dtype=bool)
    )
    unconstrained_indices = np.flatnonzero(~prior_constrained)
    unconstrained_prior_exact_zero = bool(
        not unconstrained_indices.size
        or (np.all(inverse_apriori[unconstrained_indices, :] == 0.0)
            and np.all(inverse_apriori[:, unconstrained_indices] == 0.0))
    )
    if not unconstrained_prior_exact_zero:
        raise ValueError(
            "Unconstrained parameters have nonzero physical inverse-prior rows or columns."
        )

    data_normal = design.T @ (weights[:, None] * design)
    prior_normal = inverse_apriori / np.outer(normalization, normalization)
    reconstructed_total = data_normal + prior_normal
    reconstruction_error = reconstructed_total - native_total
    max_abs_error = float(np.max(np.abs(reconstruction_error), initial=0.0))
    fro_relative_error = float(
        np.linalg.norm(reconstruction_error)
        / max(np.linalg.norm(native_total), np.finfo(float).tiny)
    )
    reconstruction_matches = bool(np.allclose(
        reconstructed_total, native_total, rtol=1.0e-10, atol=1.0e-10
    ))

    singular_values = np.linalg.svd(reconstructed_total, compute_uv=False)
    _, _, right_singular_vectors = np.linalg.svd(reconstructed_total, full_matrices=False)
    symmetric_total = 0.5 * (reconstructed_total + reconstructed_total.T)
    eigenvalues = np.linalg.eigvalsh(symmetric_total)
    leading_singular = float(singular_values[0]) if singular_values.size else 0.0
    trailing_singular = float(singular_values[-1]) if singular_values.size else 0.0
    independent_condition = (
        float(leading_singular / trailing_singular)
        if trailing_singular > 0.0 else None
    )
    numpy_relative_cutoff = float(np.finfo(float).eps * max(reconstructed_total.shape))
    quality_relative_cutoff = 1.0 / CONDITION_NUMBER_LIMIT
    numpy_rank = int(np.sum(singular_values > leading_singular * numpy_relative_cutoff))
    quality_rank = int(np.sum(singular_values >= leading_singular * quality_relative_cutoff))

    labels = [
        f"{int(row['index'])}:{row['name']}@{row['subarc_start_tdb']}"
        for _, row in parameters.iterrows()
    ]
    weak_modes = []
    for rank_from_weakest, vector_index in enumerate(
            range(len(singular_values) - 1, max(-1, len(singular_values) - 6), -1), start=1):
        vector = right_singular_vectors[vector_index]
        largest = np.argsort(np.abs(vector))[::-1][:10]
        weak_modes.append({
            "rank_from_weakest": rank_from_weakest,
            "singular_value": float(singular_values[vector_index]),
            "relative_to_largest": (
                float(singular_values[vector_index] / leading_singular)
                if leading_singular else None
            ),
            "parameter_loadings": [
                {"index": int(index), "parameter": labels[index],
                 "loading": float(vector[index])}
                for index in largest
            ],
        })

    data_diagonal = np.diag(data_normal)
    denominators = np.sqrt(np.outer(data_diagonal, data_diagonal))
    data_correlations = np.divide(
        data_normal, denominators, out=np.zeros_like(data_normal), where=denominators > 0.0
    )
    pair_indices = np.triu_indices(count, 1)
    strongest = np.argsort(np.abs(data_correlations[pair_indices]))[::-1][:20]
    top_data_correlations = []
    for flat_index in strongest:
        first = int(pair_indices[0][flat_index])
        second = int(pair_indices[1][flat_index])
        correlation = float(data_correlations[first, second])
        if correlation == 0.0:
            continue
        top_data_correlations.append({
            "first_index": first, "first_parameter": labels[first],
            "second_index": second, "second_parameter": labels[second],
            "weighted_design_column_correlation": correlation,
        })

    column_norm = np.linalg.norm(design, axis=0)
    largest_column = float(column_norm.max(initial=0.0))
    nearly_zero_limit = largest_column * 1.0e-12
    zero = column_norm == 0.0
    nearly_zero = (~zero) & (column_norm <= nearly_zero_limit)
    prior_diagonal = np.diag(prior_normal)
    prior_dominates = prior_diagonal > data_diagonal
    prior_ratio = np.divide(
        prior_diagonal, data_diagonal,
        out=np.full(count, np.inf), where=data_diagonal > 0.0,
    )
    metadata_columns = [
        "index", "name", "unit", "subarc_start_tdb", "prior_sigma",
    ]
    for optional in ("prior_constrained", "prior_information_diagonal"):
        if optional in parameters:
            metadata_columns.append(optional)
    columns = parameters[metadata_columns].copy()
    columns["normalization_term"] = normalization
    columns["normalized_design_l2"] = column_norm
    columns["normalized_data_information"] = data_diagonal
    columns["normalized_prior_information_exact"] = prior_diagonal
    columns["normalized_total_information_native"] = np.diag(native_total)
    columns["prior_to_data_information"] = prior_ratio
    columns["zero_sensitivity"] = zero
    columns["nearly_zero_sensitivity"] = nearly_zero
    columns["prior_dominates"] = prior_dominates
    columns.to_csv(directory / "conditioning_columns.csv", index=False)

    np.savez_compressed(
        directory / "normal_matrix_diagnostics.npz",
        normalized_design_matrix=design,
        weight_matrix_diagonal=weights,
        normalization_terms=normalization,
        inverse_apriori_covariance_physical=inverse_apriori,
        data_only_normal_matrix_normalized=data_normal,
        prior_information_normalized=prior_normal,
        total_normal_matrix_reconstructed=reconstructed_total,
        inverse_normalized_covariance_native=native_total,
        reconstruction_error=reconstruction_error,
        singular_values=singular_values,
        eigenvalues=eigenvalues,
    )

    def selected_labels(mask):
        return [labels[index] for index in np.flatnonzero(mask)]

    trustworthy_inverse = bool(
        independent_condition is not None
        and independent_condition <= CONDITION_NUMBER_LIMIT
        and numpy_rank == count
        and eigenvalues[0] > 0.0
    )
    summary = {
        "available": True,
        "matrix_iteration": matrix_iteration,
        "matrix_iteration_semantics": (
            "Tudat EstimationOutput retains H and inverse_normalized_covariance for best_iteration only"
        ),
        "residual_iterations_saved": int(residual_history.shape[1]),
        "parameter_count": count,
        "observation_count": int(design.shape[0]),
        "normalization_definition": "Hn[:,j] = H[:,j] / normalization_terms[j]",
        "prior_normalization_definition": "Pinv_normalized[i,j] = Pinv_physical[i,j] / (n[i]*n[j])",
        "data_normal_definition": "Hn.T @ diag(weight_matrix_diagonal) @ Hn",
        "total_normal_definition": "data_only_normal_matrix_normalized + prior_information_normalized",
        "apply_apriori_parameter_deviation": anchored_priors,
        "prior_rhs_semantics": (
            "inverse prior constrains total deviation from frozen initial parameter vector"
            if anchored_priors else
            "legacy correction-only regularization: inverse prior is on each correction LHS "
            "without an accumulated-deviation RHS"
        ),
        "unconstrained_parameter_count": int(unconstrained_indices.size),
        "unconstrained_parameters": selected_labels(~prior_constrained),
        "unconstrained_prior_information_semantics": (
            "physical inverse-prior rows and columns are exactly zero; state remains estimated"
            if unconstrained_indices.size else "all estimated parameters are prior-constrained"
        ),
        "unconstrained_prior_rows_and_columns_exact_zero": (
            unconstrained_prior_exact_zero
        ),
        "iteration_objective_history": objective_history,
        "reconstruction_rtol": 1.0e-10,
        "reconstruction_atol": 1.0e-10,
        "reconstruction_matches_native": reconstruction_matches,
        "reconstruction_max_abs_error": max_abs_error,
        "reconstruction_fro_relative_error": fro_relative_error,
        "independent_svd_condition_number": independent_condition,
        "reference_condition_ceiling": CONDITION_NUMBER_LIMIT,
        "within_reference_condition_ceiling": (
            independent_condition is not None and independent_condition <= CONDITION_NUMBER_LIMIT
        ),
        "numpy_rank_relative_cutoff": numpy_relative_cutoff,
        "numpy_rank": numpy_rank,
        "quality_rank_relative_cutoff": quality_relative_cutoff,
        "quality_rank": quality_rank,
        "minimum_eigenvalue_symmetric_total": float(eigenvalues[0]),
        "negative_eigenvalue_count": int(np.sum(eigenvalues < 0.0)),
        "inverse_correlation_diagnostics_trustworthy": trustworthy_inverse,
        "inverse_correlation_policy": (
            "native inverse correlations may be reported only when this flag is true; otherwise use "
            "weighted-design correlations and weak singular modes below"
        ),
        "zero_sensitivity_count": int(zero.sum()),
        "zero_sensitivity_columns": selected_labels(zero),
        "nearly_zero_relative_threshold": 1.0e-12,
        "nearly_zero_l2_threshold": nearly_zero_limit,
        "nearly_zero_sensitivity_count": int(nearly_zero.sum()),
        "nearly_zero_sensitivity_columns": selected_labels(nearly_zero),
        "prior_dominated_count": int(prior_dominates.sum()),
        "prior_dominated_columns": selected_labels(prior_dominates),
        "weak_singular_modes": weak_modes,
        "top_weighted_design_column_correlations": top_data_correlations,
    }
    if trustworthy_inverse:
        correlations = np.asarray(output.correlations, dtype=float)
        np.save(directory / "posterior_correlations_trusted.npy", correlations)
    write_json(directory / "conditioning_diagnostics.json", summary)
    write_json(directory / "estimation_output_availability.json", {
        "available": True,
        "matrix_iteration": matrix_iteration,
        "residual_iterations_saved": int(residual_history.shape[1]),
        "residual_history_all_finite": all(
            item["finite"] for item in residual_iteration_flags
        ),
        "exception_during_inversion": bool(output.exception_during_inversion),
        "exception_during_propagation": bool(output.exception_during_propagation),
    })
    nonfinite_residual_iterations = [
        item["iteration"] for item in residual_iteration_flags if not item["finite"]
    ]
    if nonfinite_residual_iterations:
        raise ValueError(
            "Non-finite propagated residual output in iterations "
            f"{nonfinite_residual_iterations}; raw diagnostics were retained."
        )
    if not objective_history["native_best_matches_reconstructed_total_cost"]:
        raise ValueError(
            "Native best_iteration does not match independently reconstructed total cost."
        )
    return summary


def save_iteration_orbit_diagnostics(output, parameter_history, parameter_rows,
                                     estimation_epoch, lower, upper, score_step,
                                     observation_min_tdb, observation_max_tdb,
                                     selected_orbit, directory,
                                     objective_history=None):
    """Score and retain every propagated estimation iteration on fixed orbit grids.

    Tudat stores one additional, not-yet-propagated parameter column after the
    final residual evaluation. The per-iteration rows therefore label both the
    update calculated from an evaluation and whether its target was subsequently
    propagated. Simulation index alignment is checked at the estimation epoch
    against the parameter vector used for that propagation.
    """
    import numpy as np
    import pandas as pd

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    residual_history = np.asarray(output.residual_history, dtype=float)
    if residual_history.ndim == 1:
        residual_history = residual_history[:, None]
    simulations = list(output.simulation_results_per_iteration)
    iteration_count = residual_history.shape[1]
    if len(simulations) != iteration_count:
        raise ValueError(
            f"Simulation-result count {len(simulations)} does not match "
            f"residual iteration count {iteration_count}."
        )
    parameter_history = np.asarray(parameter_history, dtype=float)
    parameters = pd.DataFrame(parameter_rows).sort_values("index").reset_index(drop=True)
    if (parameter_history.ndim != 2
            or parameter_history.shape[0] != len(parameters)
            or parameter_history.shape[1] < iteration_count):
        raise ValueError(
            f"Parameter history shape {parameter_history.shape} cannot label "
            f"{iteration_count} propagated iterations and {len(parameters)} parameters."
        )
    best = int(output.best_iteration)
    if not 0 <= best < iteration_count:
        raise ValueError(f"Best iteration {best} is outside {iteration_count} evaluations.")
    if objective_history is not None:
        objective_rows = objective_history.get("per_iteration", [])
        if (len(objective_rows) != iteration_count
                or objective_history.get("native_best_iteration") != best):
            raise ValueError("Objective history does not align with propagated iterations.")
    else:
        objective_rows = [None] * iteration_count
    prior_constrained = (
        parameters.prior_constrained.to_numpy(dtype=bool)
        if "prior_constrained" in parameters else np.ones(len(parameters), dtype=bool)
    )
    prior_sigmas = parameters.prior_sigma.to_numpy(dtype=float)
    if (not np.isfinite(prior_sigmas[prior_constrained]).all()
            or np.any(prior_sigmas[prior_constrained] <= 0.0)):
        raise ValueError(
            "Iteration diagnostics require finite positive sigmas for constrained parameters."
        )
    if ((~prior_constrained).any()
            and ("prior_information_diagonal" not in parameters
                 or not np.all(parameters.loc[
                     ~prior_constrained, "prior_information_diagonal"
                 ].to_numpy(dtype=float) == 0.0))):
        raise ValueError(
            "Unconstrained parameters require explicit zero physical prior information."
        )

    summaries = []
    update_rows = []
    orbit_values = []
    orbit_epochs = None
    bracketed_mask = None
    state_index_checks = []
    for iteration, simulation in enumerate(simulations):
        history = simulation.dynamics_results.state_history_float
        epochs = np.asarray(sorted(history), dtype=float)
        states = np.asarray([history[float(epoch)] for epoch in epochs], dtype=float)
        if states.ndim != 2 or states.shape[1] < 6 or not np.isfinite(states).all():
            raise ValueError(f"Invalid propagated state history for iteration {iteration}.")
        if estimation_epoch not in history:
            raise ValueError(
                f"Iteration {iteration} state history lacks estimation epoch {estimation_epoch}."
            )
        epoch_state = np.asarray(history[estimation_epoch], dtype=float).reshape(-1)[:6]
        expected_state = parameter_history[:6, iteration]
        np.testing.assert_allclose(epoch_state, expected_state, rtol=0.0, atol=1.0e-9)
        state_index_checks.append({
            "iteration": iteration,
            "maximum_abs_state_index_error": float(
                np.max(np.abs(epoch_state - expected_state), initial=0.0)
            ),
            "matches_parameter_history_column": True,
        })
        np.savez_compressed(
            directory / f"propagated_state_iteration_{iteration:02d}.npz",
            epochs=epochs,
            states=states,
            estimation_iteration=iteration,
            parameter_history_column=iteration,
            is_best_iteration=(iteration == best),
        )

        scored_orbit = orbit_comparison(history, lower, upper, score_step)
        if iteration == best:
            np.testing.assert_allclose(
                scored_orbit[["t", "R", "T", "N", "dx", "dy", "dz"]].to_numpy(),
                selected_orbit[["t", "R", "T", "N", "dx", "dy", "dz"]].to_numpy(),
                rtol=0.0,
                atol=1.0e-12,
            )
        current_epochs = scored_orbit.t.to_numpy(dtype=float)
        if orbit_epochs is None:
            orbit_epochs = current_epochs
            bracketed_mask = (
                (orbit_epochs >= observation_min_tdb)
                & (orbit_epochs <= observation_max_tdb)
            )
            if not bracketed_mask.any() or bracketed_mask.all():
                raise ValueError("Iteration scoring requires nonempty bracketed and edge spans.")
        else:
            np.testing.assert_array_equal(current_epochs, orbit_epochs)
        orbit_values.append(
            scored_orbit[["R", "T", "N", "dx", "dy", "dz"]].to_numpy(dtype=float)
        )
        span_summary, _, _ = orbit_span_metrics(
            scored_orbit, observation_min_tdb, observation_max_tdb
        )

        update = (
            parameter_history[:, iteration + 1] - parameter_history[:, iteration]
            if iteration + 1 < parameter_history.shape[1] else None
        )
        update_target_evaluated = iteration + 1 < iteration_count
        if update is not None:
            normalized_update = update[prior_constrained] / prior_sigmas[prior_constrained]
            update_l2 = float(np.linalg.norm(normalized_update))
            update_max = float(np.max(np.abs(normalized_update), initial=0.0))
            normalized_update_full = np.full(len(parameters), np.nan)
            normalized_update_full[prior_constrained] = normalized_update
            for row_index, parameter in parameters.iterrows():
                update_rows.append({
                    "source_estimation_iteration": iteration,
                    "target_parameter_history_column": iteration + 1,
                    "target_was_propagated_and_evaluated": update_target_evaluated,
                    "parameter_index": int(parameter["index"]),
                    "name": parameter["name"],
                    "unit": parameter["unit"],
                    "subarc_start_tdb": parameter["subarc_start_tdb"],
                    "prior_sigma": parameter["prior_sigma"],
                    "prior_constrained": bool(prior_constrained[row_index]),
                    "source_value": parameter_history[row_index, iteration],
                    "target_value": parameter_history[row_index, iteration + 1],
                    "update": update[row_index],
                    "update_over_prior_sigma": (
                        normalized_update_full[row_index]
                        if prior_constrained[row_index] else None
                    ),
                })
        else:
            update_l2 = None
            update_max = None

        residual = residual_history[:, iteration]
        objective = objective_rows[iteration]
        summaries.append({
            "iteration": iteration,
            "stage": (
                "initial_propagated_before_any_differential_correction"
                if iteration == 0 else f"post_correction_evaluation_{iteration}"
            ),
            "is_best_iteration": iteration == best,
            "parameter_history_column": iteration,
            "residual_rms_mhz": float(np.sqrt(np.mean(residual ** 2)) * 1.0e3),
            "residual_max_mhz": float(np.max(np.abs(residual)) * 1.0e3),
            "update_target_parameter_history_column": (
                iteration + 1 if update is not None else None
            ),
            "update_target_was_propagated_and_evaluated": update_target_evaluated,
            "parameter_update_to_next_normalized_l2": update_l2,
            "parameter_update_to_next_normalized_max_abs": update_max,
            "parameter_update_normalization_scope": (
                "constrained parameters only; unconstrained state updates are retained "
                "in SI units and excluded from prior-normalized aggregates"
            ),
            "constrained_parameter_count": int(prior_constrained.sum()),
            "unconstrained_parameter_count": int((~prior_constrained).sum()),
            **({
                "data_cost_half_rWr": objective["data_cost_half_rWr"],
                "absolute_prior_cost_half_delta_Pinv_delta": objective[
                    "absolute_prior_cost_half_delta_Pinv_delta"
                ],
                "objective_prior_cost": objective["objective_prior_cost"],
                "total_cost": objective["total_cost"],
            } if objective is not None else {}),
            **orbit_only_metrics(scored_orbit, "full"),
            **span_summary,
        })

    np.savez_compressed(
        directory / "iteration_orbits.npz",
        epochs=orbit_epochs,
        values=np.asarray(orbit_values),
        columns=np.asarray(["R", "T", "N", "dx", "dy", "dz"]),
        bracketed_mask=bracketed_mask,
        outside_edge_mask=~bracketed_mask,
        observation_bracket_min_tdb=float(observation_min_tdb),
        observation_bracket_max_tdb=float(observation_max_tdb),
        best_iteration=best,
    )
    write_json(directory / "iteration_orbit_metrics.json", {
        "iteration_count": iteration_count,
        "best_iteration": best,
        "orbit_grid_semantics": "identical full nominal score grid; propagation padding excluded",
        "observation_span_semantics": "fixed retained-observation receive-time min/max",
        "parameter_update_semantics": (
            "update from propagated parameter_history column i to column i+1; "
            "the final target can be unpropagated; prior-normalized aggregates "
            "cover constrained parameters only"
        ),
        "state_index_checks": state_index_checks,
        "per_iteration": summaries,
    })
    pd.DataFrame(summaries).to_csv(directory / "iteration_orbit_metrics.csv", index=False)
    pd.DataFrame(update_rows).to_csv(directory / "iteration_parameter_updates.csv", index=False)
    return summaries


def run_arc(case, directory, arc_index, reference_directory=None):
    """Run one isolated fit and persist enough evidence to audit every reported metric."""
    import numpy as np
    import pandas as pd
    import spiceypy
    import mro_tnf_estimation as example
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plan = ParameterPlan(case, arc_index)
    captured_diagnostics = {}
    captured_seed = {}

    def capture_estimation_output(output, observations, weights, inverse_apriori):
        captured_diagnostics["summary"] = save_estimation_diagnostics(
            output, observations, plan.rows, weights, inverse_apriori, directory,
            anchored_priors=case.apply_apriori_parameter_deviation,
        )

    def verify_initial_state(unperturbed_seed, estimation_seed, epoch):
        offset = np.asarray(case.initial_position_offset_m, dtype=float)
        direct = np.asarray(
            spiceypy.spkezr("-74", epoch, "J2000", "NONE", "499")[0]
        ) * 1000.0
        unperturbed_position_error = float(
            np.linalg.norm(unperturbed_seed[:3] - direct[:3])
        )
        unperturbed_velocity_error = float(
            np.linalg.norm(unperturbed_seed[3:6] - direct[3:6])
        )
        applied_offset_error = float(np.linalg.norm(
            estimation_seed[:3] - unperturbed_seed[:3] - offset
        ))
        applied_velocity_change = float(np.linalg.norm(
            estimation_seed[3:6] - unperturbed_seed[3:6]
        ))
        if unperturbed_position_error > 0.002:
            raise ValueError(
                "Buffered midpoint SPICE seed is "
                f"{unperturbed_position_error} m from direct SPICE."
            )
        if unperturbed_velocity_error > 2.0e-6:
            raise ValueError(
                "Buffered midpoint SPICE velocity seed is "
                f"{unperturbed_velocity_error} m/s from direct SPICE."
            )
        if applied_offset_error > 1.0e-12 or applied_velocity_change > 1.0e-12:
            raise ValueError(
                "Actual estimation seed does not equal the verified SPICE seed "
                "plus the declared position-only offset."
            )
        captured_seed.update(
            estimation_epoch_tdb=float(epoch),
            declared_position_offset_m=offset.tolist(),
            unperturbed_seed=unperturbed_seed.tolist(),
            actual_estimation_seed=estimation_seed.tolist(),
            direct_spice_state=direct.tolist(),
            unperturbed_spice_position_error_m=unperturbed_position_error,
            unperturbed_spice_velocity_error_m_s=unperturbed_velocity_error,
            applied_position_offset_error_m=applied_offset_error,
            applied_velocity_change_m_s=applied_velocity_change,
            anchored_prior_center_follows_estimation_seed=bool(
                case.apply_apriori_parameter_deviation
            ),
            initial_state_prior_constrained=bool(
                case.constrain_initial_state_prior
            ),
            initial_state_prior_information_semantics=(
                "standard finite state prior"
                if case.constrain_initial_state_prior else
                "all six physical inverse-prior rows and columns exactly zero"
            ),
        )
        write_json(directory / "initial_state_seed.json", captured_seed)

    start = time.perf_counter()
    result = example.process_arc(
        local_inputs(arc_index), parameter_builder=plan.settings, prior_builder=plan.priors,
        prefit_callback=lambda frame: prepare_prefit(frame, directory, reference_directory),
        estimation_output_callback=capture_estimation_output,
        initial_state_callback=verify_initial_state,
        interactive=False, pad_propagation=True)
    output = result["estimation_output"]
    best = output.best_iteration
    parameter_history = np.asarray(output.parameter_history)
    best_parameters = parameter_history[:, best]
    np.testing.assert_allclose(output.final_parameters, best_parameters, rtol=0, atol=1e-12)
    epoch = result["estimation_epoch"]
    np.testing.assert_allclose(result["postfit_state_history"][epoch], best_parameters[:6], rtol=0, atol=1e-9)
    offset = np.asarray(case.initial_position_offset_m, dtype=float)
    initial_error = np.linalg.norm(
        result["nominal_parameters"][:3] - offset
        - np.asarray(captured_seed["direct_spice_state"][:3])
    )
    lower, upper = result["arc_bounds"]
    orbit = orbit_comparison(result["postfit_state_history"], lower, upper, case.score_step_seconds)
    residuals = result["residuals"].assign(arc_index=arc_index)
    orbit["arc_index"] = arc_index
    if (not np.isfinite(best_parameters).all()
            or not np.isfinite(residuals[["prefit", "postfit"]].to_numpy()).all()
            or not np.isfinite(orbit[["R", "T", "N"]].to_numpy()).all()):
        raise ValueError("Non-finite fitted parameter, propagated residual, or RTN orbit output.")
    parameter_table = pd.DataFrame(plan.rows)
    parameter_table["nominal"] = result["nominal_parameters"]
    parameter_table["value"] = best_parameters
    parameter_table["delta"] = best_parameters - result["nominal_parameters"]
    conditioning = captured_diagnostics.get("summary")
    if not conditioning or not conditioning.get("available"):
        raise RuntimeError("Completed estimation did not provide auditable matrix diagnostics.")
    parameter_table["native_formal_error"] = output.formal_errors
    parameter_table["native_inverse_diagnostics_trustworthy"] = conditioning[
        "inverse_correlation_diagnostics_trustworthy"
    ]
    constrained = parameter_table.prior_constrained.astype(bool)
    parameter_table["prior_pull_applicable"] = constrained
    parameter_table["prior_pull"] = np.nan
    parameter_table.loc[constrained, "prior_pull"] = (
        parameter_table.loc[constrained, "delta"]
        / parameter_table.loc[constrained, "prior_sigma"]
    )
    parameter_table["arc_index"] = arc_index
    parameter_table["plot_start_tdb"] = parameter_table.subarc_start_tdb.fillna(lower).clip(lower=lower)
    endpoints = dict(zip(plan.arc_times, plan.arc_times[1:] + [upper]))
    parameter_table["plot_end_tdb"] = parameter_table.subarc_start_tdb.map(endpoints).fillna(upper).clip(upper=upper)
    iteration_orbit_metrics = save_iteration_orbit_diagnostics(
        output,
        parameter_history,
        plan.rows,
        epoch,
        lower,
        upper,
        case.score_step_seconds,
        float(residuals.time.min()),
        float(residuals.time.max()),
        orbit,
        directory,
        objective_history=conditioning["iteration_objective_history"],
    )
    residuals.to_csv(directory / "residuals.csv", index=False)
    orbit.to_csv(directory / "orbit.csv", index=False)
    parameter_table.to_csv(directory / "parameters.csv", index=False)
    np.savez_compressed(directory / "iteration_history.npz", parameters=parameter_history,
                        residuals=output.residual_history)
    for label in ("prefit", "postfit"):
        history = result[f"{label}_state_history"]
        epochs = sorted(history)
        np.savez_compressed(directory / f"{label}_states.npz", epochs=epochs,
                            states=np.array([history[t] for t in epochs]))
    summary = metrics(residuals, orbit)
    write_json(directory / "parameter_diagnostics.json", parameter_diagnostics(parameter_table))
    summary.update(arc_index=arc_index, best_iteration=int(best), parameter_count=len(parameter_table),
                   apply_apriori_parameter_deviation=case.apply_apriori_parameter_deviation,
                   iteration_objective_history=conditioning["iteration_objective_history"],
                   nominal_spice_position_error_m=float(initial_error),
                   initial_position_offset_m=list(map(float, case.initial_position_offset_m)),
                   initial_state_seed_file="initial_state_seed.json",
                   anchored_prior_center_follows_estimation_seed=bool(
                       case.apply_apriori_parameter_deviation),
                   iteration_rms_mhz=(np.sqrt(np.mean(np.asarray(output.residual_history) ** 2, axis=0)) * 1e3).tolist(),
                   iteration_orbit_metrics=iteration_orbit_metrics,
                   max_abs_prior_pull=float(
                       parameter_table.loc[constrained, "prior_pull"].abs().max()
                   ),
                   unconstrained_parameter_count=int((~constrained).sum()),
                   unconstrained_parameter_names=parameter_table.loc[
                       ~constrained, "name"
                   ].tolist(),
                   zero_sensitivity_count=conditioning["zero_sensitivity_count"],
                   nearly_zero_sensitivity_count=conditioning["nearly_zero_sensitivity_count"],
                   prior_dominated_count=conditioning["prior_dominated_count"])
    plot_results(directory, residuals, orbit, parameter_table)
    summary["wall_seconds"] = time.perf_counter() - start
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def _condition_case_summary(diagnostics):
    """Combine JSON-safe condition quality flags; never accept/reject a fit here."""
    diagnostics = sorted(diagnostics, key=lambda item: item["arc_index"])
    maxima = [item["maximum_finite"] for item in diagnostics
              if item["maximum_finite"] is not None]
    invalid = [item for item in diagnostics if item["values"] and not item["passed"]]
    missing = [item["arc_index"] for item in diagnostics if not item["values"]]
    threshold = diagnostics[0]["threshold"] if diagnostics else CONDITION_NUMBER_LIMIT
    complete = len(diagnostics) == len(ARCS) and not missing
    reason = ""
    if invalid:
        reason = "; ".join(f"arc {item['arc_index']:02d}: {item['reason']}" for item in invalid)
    elif missing:
        reason = f"missing condition-number reports for arcs {missing}"
    elif not complete:
        reason = "monitoring in progress"
    return {
        "threshold": threshold,
        "condition_number_max": max(maxima) if maxima else None,
        "all_within_reference_ceiling": complete and not invalid,
        "passed": complete and not invalid,
        "diagnostic_only": True,
        "reason": reason,
        "expected_arc_count": len(ARCS),
        "reported_arc_count": sum(bool(item["values"]) for item in diagnostics),
        "violating_arcs": [item["arc_index"] for item in invalid],
        "missing_arcs": missing,
        "per_arc": diagnostics,
    }


def _write_condition_diagnostics(directory, diagnostics):
    """Persist per-arc and case condition reports, including partial/failed batches."""
    directory = Path(directory)
    supplied = {item["arc_index"]: item for item in diagnostics}
    for arc, item in supplied.items():
        arc_directory = directory / "arcs" / f"arc_{arc:02d}"
        arc_directory.mkdir(parents=True, exist_ok=True)
        write_json(arc_directory / "condition_numbers.json", item)
    combined = dict(supplied)
    for path in sorted((directory / "arcs").glob("arc_*/condition_numbers.json")):
        item = json.loads(path.read_text())
        combined.setdefault(int(item["arc_index"]), item)
    summary = _condition_case_summary(list(combined.values()))
    write_json(directory / "condition_numbers.json", summary)
    return summary


def scan_condition_logs(directory, arc_indices=range(len(ARCS)), threshold=CONDITION_NUMBER_LIMIT,
                        require_reports=True):
    """Final-scan condition diagnostics without accepting/rejecting fit outputs."""
    diagnostics = []
    for arc in arc_indices:
        scanner = ConditionLogScanner(threshold)
        scanner.read_path(Path(directory) / "arcs" / f"arc_{arc:02d}" / "fit.log", final=True)
        diagnostics.append(scanner.diagnostics(arc))
    summary = _write_condition_diagnostics(directory, diagnostics)
    missing = [item["arc_index"] for item in diagnostics if not item["values"]]
    summary["reports_required_for_diagnostics"] = bool(require_reports)
    summary["diagnostic_complete"] = not missing
    return summary


def _stop_children(entries):
    """Terminate all live siblings, escalating to kill only after a bounded wait."""
    for _, child, _ in entries:
        if child.poll() is None:
            child.terminate()
    for _, child, _ in entries:
        if child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def monitor_children(directory, entries, threshold=CONDITION_NUMBER_LIMIT,
                     poll_seconds=CONDITION_POLL_SECONDS):
    """Tail all live logs and record conditioning without interrupting estimation."""
    scanners = {arc: ConditionLogScanner(threshold) for arc, _, _ in entries}
    while True:
        for arc, _, path in entries:
            scanners[arc].read_path(path)
        codes = [child.poll() for _, child, _ in entries]
        if all(code is not None for code in codes):
            break
        time.sleep(poll_seconds)
    # A worker can print its last warning and exit between the final live read and
    # poll. This final read keeps the diagnostics complete.
    for arc, _, path in entries:
        scanners[arc].read_path(path, final=True)
    diagnostics = [scanners[arc].diagnostics(arc) for arc, _, _ in entries]
    _write_condition_diagnostics(directory, diagnostics)
    return codes, diagnostics


def aggregate_iteration_orbit_diagnostics(directory):
    """Pool per-iteration orbit evidence when every arc provides it."""
    import numpy as np
    import pandas as pd

    paths = [
        Path(directory) / "arcs" / f"arc_{arc:02d}" / "iteration_orbits.npz"
        for arc in range(len(ARCS))
    ]
    if not all(path.is_file() for path in paths):
        return None
    data = [np.load(path) for path in paths]
    iteration_counts = {item["values"].shape[0] for item in data}
    if len(iteration_counts) != 1:
        raise ValueError(f"Per-arc iteration counts differ: {sorted(iteration_counts)}")
    iteration_count = iteration_counts.pop()
    if any(item["values"].ndim != 3 or item["values"].shape[2] != 6 for item in data):
        raise ValueError("Invalid per-iteration orbit artifact dimensions.")
    values = np.concatenate([item["values"] for item in data], axis=1)
    epochs = np.concatenate([item["epochs"] for item in data])
    bracketed_mask = np.concatenate([item["bracketed_mask"] for item in data]).astype(bool)
    arc_indices = np.concatenate([
        np.full(item["epochs"].shape, arc, dtype=int) for arc, item in enumerate(data)
    ])
    if (len(epochs) != values.shape[1] or bracketed_mask.shape != epochs.shape
            or not bracketed_mask.any() or bracketed_mask.all()):
        raise ValueError("Invalid pooled fixed-span iteration orbit artifacts.")

    residual_histories = [
        np.load(Path(directory) / "arcs" / f"arc_{arc:02d}" / "residual_history.npy")
        for arc in range(len(ARCS))
    ]
    if any(history.ndim != 2 or history.shape[1] != iteration_count
           for history in residual_histories):
        raise ValueError("Residual histories do not align with iteration orbit artifacts.")
    update_tables = [
        pd.read_csv(
            Path(directory) / "arcs" / f"arc_{arc:02d}" / "iteration_parameter_updates.csv"
        )
        for arc in range(len(ARCS))
    ]
    objective_histories = [
        json.loads((
            Path(directory) / "arcs" / f"arc_{arc:02d}"
            / "iteration_objective_history.json"
        ).read_text())
        for arc in range(len(ARCS))
    ]
    if any(len(item.get("per_iteration", [])) != iteration_count
           for item in objective_histories):
        raise ValueError("Objective histories do not align with iteration orbit artifacts.")

    rows = []
    columns = ["R", "T", "N", "dx", "dy", "dz"]
    for iteration in range(iteration_count):
        orbit = pd.DataFrame(values[iteration], columns=columns)
        orbit.insert(0, "t", epochs)
        residual = np.concatenate([history[:, iteration] for history in residual_histories])
        updates = pd.concat([
            table[table.source_estimation_iteration == iteration]
            for table in update_tables
        ], ignore_index=True)
        if "prior_constrained" in updates:
            constrained_updates = updates[updates.prior_constrained.astype(bool)]
        else:
            constrained_updates = updates
        normalized_update = constrained_updates.update_over_prior_sigma.to_numpy(dtype=float)
        if not np.isfinite(normalized_update).all():
            raise ValueError(
                "Prior-normalized aggregate contains a non-finite constrained update."
            )
        objectives = [item["per_iteration"][iteration] for item in objective_histories]
        if not all(item["finite"] for item in objectives):
            raise ValueError(f"Non-finite objective in pooled iteration {iteration}.")
        rows.append({
            "iteration": iteration,
            "stage": (
                "initial_propagated_before_any_differential_correction"
                if iteration == 0 else f"post_correction_evaluation_{iteration}"
            ),
            "residual_rms_mhz": float(np.sqrt(np.mean(residual ** 2)) * 1.0e3),
            "residual_max_mhz": float(np.max(np.abs(residual)) * 1.0e3),
            "update_target_parameter_history_column": (
                iteration + 1 if not updates.empty else None
            ),
            "update_target_was_propagated_and_evaluated": iteration + 1 < iteration_count,
            "parameter_update_to_next_normalized_l2": (
                float(np.linalg.norm(normalized_update)) if normalized_update.size else None
            ),
            "parameter_update_to_next_normalized_max_abs": (
                float(np.max(np.abs(normalized_update), initial=0.0))
                if normalized_update.size else None
            ),
            "parameter_update_normalization_scope": (
                "constrained parameters only; unconstrained state updates excluded"
            ),
            "data_cost_half_rWr": float(sum(
                item["data_cost_half_rWr"] for item in objectives
            )),
            "absolute_prior_cost_half_delta_Pinv_delta": float(sum(
                item["absolute_prior_cost_half_delta_Pinv_delta"]
                for item in objectives
            )),
            "objective_prior_cost": float(sum(
                item["objective_prior_cost"] for item in objectives
            )),
            "total_cost": float(sum(item["total_cost"] for item in objectives)),
            **orbit_only_metrics(orbit, "full"),
            **orbit_only_metrics(orbit[bracketed_mask], "bracketed"),
            **orbit_only_metrics(orbit[~bracketed_mask], "outside_edge"),
        })

    best_iterations = [int(item["best_iteration"]) for item in data]
    np.savez_compressed(
        Path(directory) / "iteration_orbits.npz",
        epochs=epochs,
        arc_index=arc_indices,
        values=values,
        columns=np.asarray(columns),
        bracketed_mask=bracketed_mask,
        outside_edge_mask=~bracketed_mask,
        best_iteration_per_arc=np.asarray(best_iterations, dtype=int),
    )
    result = {
        "iteration_count": iteration_count,
        "best_iteration_per_arc": best_iterations,
        "all_arcs_share_best_iteration": len(set(best_iterations)) == 1,
        "apply_apriori_parameter_deviation": all(
            item["apply_apriori_parameter_deviation"] for item in objective_histories
        ),
        "native_best_iterations_match_reconstructed_total_cost": all(
            item["native_best_matches_reconstructed_total_cost"]
            for item in objective_histories
        ),
        "orbit_grid_semantics": "pooled identical per-arc full nominal score grids; padding excluded",
        "observation_span_semantics": "fixed per-arc retained-observation receive-time min/max",
        "parameter_update_semantics": (
            "pooled normalized updates from propagated parameter column i to i+1; "
            "the final target can be unpropagated; unconstrained state updates "
            "remain in SI tables and are excluded from normalized aggregates"
        ),
        "per_iteration": rows,
    }
    write_json(Path(directory) / "iteration_orbit_metrics.json", result)
    pd.DataFrame(rows).to_csv(Path(directory) / "iteration_orbit_metrics.csv", index=False)
    return result


def aggregate(directory):
    """Publish metrics for seven completed arcs, retaining conditioning flags."""
    import numpy as np
    import pandas as pd
    case = load_case(Path(directory) / "settings.json")
    condition_summary = scan_condition_logs(
        directory, threshold=case.condition_number_limit)
    condition_by_arc = {item["arc_index"]: item for item in condition_summary["per_arc"]}
    frames = {}
    summaries = []
    for arc in range(len(ARCS)):
        subdir = directory / "arcs" / f"arc_{arc:02d}"
        arc_summary = json.loads((subdir / "summary.json").read_text())
        condition = condition_by_arc[arc]
        selected_iteration = int(arc_summary["best_iteration"])
        selected_native_condition = (
            condition["values"][selected_iteration]["value"]
            if selected_iteration < len(condition["values"]) else None
        )
        arc_summary.update(
            condition_numbers=[item["value"] for item in condition["values"]],
            condition_number_max=condition["maximum_finite"],
            condition_number_threshold=condition["threshold"],
            condition_number_within_reference_ceiling=condition["within_reference_ceiling"],
            condition_number_passed=condition["passed"],
            selected_iteration_native_condition_number=selected_native_condition,
        )
        arc_orbit = pd.read_csv(subdir / "orbit.csv")
        common_observations = pd.read_csv(subdir / "spice_residuals.csv")
        span_summary, bracketed_orbit, outside_edge_orbit = orbit_span_metrics(
            arc_orbit,
            float(common_observations.time.min()),
            float(common_observations.time.max()),
        )
        arc_summary.update(span_summary)
        write_json(subdir / "orbit_span_metrics.json", span_summary)
        frames.setdefault("orbit_bracketed", []).append(bracketed_orbit)
        frames.setdefault("orbit_outside_edge", []).append(outside_edge_orbit)
        write_json(subdir / "summary.json", arc_summary)
        summaries.append(arc_summary)
        for name in ("residuals", "orbit", "parameters"):
            frames.setdefault(name, []).append(pd.read_csv(subdir / f"{name}.csv"))
    frames = {name: pd.concat(items, ignore_index=True) for name, items in frames.items()}
    for name, frame in frames.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    summary = metrics(frames["residuals"], frames["orbit"])
    per_arc_position_rms = [s["position_rms_m"] for s in summaries]
    summary["mean_arc_position_rms_m"] = float(np.mean(per_arc_position_rms))
    summary["median_arc_position_rms_m"] = float(np.median(per_arc_position_rms))
    summary["worst_arc_position_rms_m"] = max(per_arc_position_rms)
    summary["parameters_total"] = sum(s["parameter_count"] for s in summaries)
    summary["per_arc"] = summaries
    summary["condition_number_max"] = condition_summary["condition_number_max"]
    summary["condition_number_threshold"] = condition_summary["threshold"]
    summary["condition_number_all_within_reference_ceiling"] = condition_summary[
        "all_within_reference_ceiling"
    ]
    summary["condition_number_passed"] = condition_summary["passed"]
    for name, prefix in (("orbit_bracketed", "bracketed"),
                         ("orbit_outside_edge", "outside_edge")):
        position = frames[name][["R", "T", "N"]].to_numpy()
        summary[f"{prefix}_orbit_samples"] = len(frames[name])
        for component in "RTN":
            summary[f"{prefix}_{component}_rms_m"] = float(
                np.sqrt(np.mean(frames[name][component] ** 2))
            )
        summary[f"{prefix}_position_rms_m"] = float(
            np.sqrt(np.mean(np.sum(position ** 2, axis=1)))
        )
        summary[f"{prefix}_position_max_m"] = float(
            np.max(np.linalg.norm(position, axis=1))
        )
    write_json(directory / "parameter_diagnostics.json", parameter_diagnostics(frames["parameters"]))
    iteration_diagnostics = aggregate_iteration_orbit_diagnostics(directory)
    if iteration_diagnostics is not None:
        summary["iteration_orbit_metrics"] = iteration_diagnostics
    plot_results(directory, frames["residuals"], frames["orbit"], frames["parameters"])
    return summary


def update_register(root):
    """Regenerate the human-readable and CSV case register, including failed cases."""
    import csv
    columns = ["case", "status", "description", "residual_rms_mhz", "R_rms_m", "T_rms_m", "N_rms_m",
               "position_rms_m", "mean_arc_position_rms_m", "median_arc_position_rms_m",
               "worst_arc_position_rms_m", "parameters_total",
               "condition_number_max", "condition_number_threshold",
               "condition_number_all_within_reference_ceiling", "wall_seconds", "reason"]
    rows = {}
    catalogue = cases()
    for number in sorted(ADAPTIVE_CASE_IDS):
        # Collision-safe adaptive IDs may exist only as reviewed planned JSONs;
        # the planned-file pass below supplies their exact descriptions.
        if number not in catalogue:
            continue
        case = catalogue[number]
        rows[f"case_{number}"] = dict(
            case=f"case_{number}", status="adaptive",
            description=(case.description +
                         " — exact control/topology must be documented and supplied explicitly before launch"),
        )
    for path in sorted((root / "planned").glob("case_*.json")):
        # Display is intentionally tolerant of fields added by a newer runner
        # while an older fit parent is still alive. Strict Case construction
        # remains mandatory at validation/launch time.
        raw = json.loads(path.read_text())
        known = {field.name for field in fields(Case)}
        case = Case(**{key: value for key, value in raw.items() if key in known})
        case_id = path.stem.removeprefix("case_")
        status = ("cancelled_by_user" if case_id in CANCELLED_CASE_REASONS else
                  "deferred" if case_id in DEFERRED_CASE_REASONS else
                  "blocked" if case.blocked_reason else "planned")
        reason = CANCELLED_CASE_REASONS.get(
            case_id, DEFERRED_CASE_REASONS.get(case_id, case.blocked_reason or "")
        )
        rows[path.stem] = dict(case=path.stem, status=status,
                               description=str(raw.get("description", case.description)) +
                               (f" — {reason}" if reason else ""))
    for directory in sorted(root.glob("case_*")):
        if (not directory.is_dir() or not (directory / "settings.json").is_file()
                or not (directory / "status.json").is_file()):
            continue
        settings = json.loads((directory / "settings.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        row = dict(case=directory.name, description=settings["description"], **status)
        structural_flag_path = directory / "STRUCTURAL_FLAG.json"
        if structural_flag_path.exists():
            structural_flag = json.loads(structural_flag_path.read_text())
            row["status"] = structural_flag.get(
                "status", "completed_but_structurally_flagged"
            )
            row["reason"] = structural_flag.get("reason", "")
        summary_path = directory / "summary.json"
        if summary_path.exists() and not row.get("median_arc_position_rms_m"):
            summary = json.loads(summary_path.read_text())
            per_arc_rms = [item["position_rms_m"] for item in summary.get("per_arc", [])]
            if per_arc_rms:
                row.setdefault(
                    "mean_arc_position_rms_m", float(sum(per_arc_rms) / len(per_arc_rms)))
                ordered = sorted(per_arc_rms)
                middle = len(ordered) // 2
                row["median_arc_position_rms_m"] = float(
                    ordered[middle] if len(ordered) % 2 else
                    (ordered[middle - 1] + ordered[middle]) / 2)
        rows[directory.name] = row
    names = sorted(rows)
    through_current = [name for name in names if name <= "case_007"]
    queue_priority = [f"case_{number}" for number in QUEUE_PRIORITY_CASE_IDS
                      if f"case_{number}" in rows and f"case_{number}" not in through_current]
    remaining = [name for name in names
                 if name not in through_current and name not in queue_priority]
    ordered_names = through_current + queue_priority + remaining
    priority_status = " -> ".join(
        f"{number} ({rows[f'case_{number}'].get('status', 'unknown')})"
        for number in QUEUE_PRIORITY_CASE_IDS if f"case_{number}" in rows
    )
    rows = [{key: rows[name].get(key, "") for key in columns} for name in ordered_names]
    with (root / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    text = (
        "# MRO orbit-fit campaign\n\n"
        f"**Queue priority/status: {priority_status}.**\n\n"
        "**Phases: defined non-shadowing comparisons, requested +[d,d,d] m "
        "matrices and same-objective unconstrained-state diagnostics complete; "
        "Sun-shadow case078 complete; case079 terminated and its partial result "
        "tree deleted at explicit user request; cases080--085 cancelled before "
        "launch. The durable queue is empty and no replacement run is "
        "authorized.**\n\n"
        "R/T/N and 3D errors are against direct SPICE over nominal arcs, excluding padding.\n\n"
    )
    text += "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n"
    for row in rows:
        text += "| " + " | ".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in columns) + " |\n"
    (root / "CASES.md").write_text(text)


def record_reporting_error(directory, phase, error):
    """Persist a non-scientific reporting error without changing fit status."""
    directory = Path(directory)
    path = directory / "reporting_errors.json"
    records = json.loads(path.read_text()) if path.exists() else []
    records.append({
        "phase": phase,
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        "fit_status_preserved": True,
    })
    write_json(path, records)
    try:
        with (directory / "case.log").open("a") as log:
            log.write(f"Reporting warning during {phase}: {type(error).__name__}: {error}\n")
    except Exception:
        pass


def update_register_safely(root, directory, phase):
    """Refresh catalogue as reporting only; never alter scientific status."""
    try:
        update_register(root)
        return True
    except Exception as error:
        record_reporting_error(directory, phase, error)
        print(
            f"WARNING: catalogue refresh failed during {phase}: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        return False


def run_workers(directory, environment, reference, workers,
                condition_limit=CONDITION_NUMBER_LIMIT):
    """Run bounded batches; on suspected OOM retry failed arcs at ceil(workers/2).

    Exit -9/137 is labelled *suspected* OOM, since SIGKILL alone cannot prove its
    cause. Ordinary modelling errors are never retried or hidden.
    """
    pending = list(range(len(ARCS)))
    attempts = {arc: 0 for arc in pending}
    retry_history = []
    while pending:
        batch, pending = pending[:workers], pending[workers:]
        entries = []
        try:
            for arc in batch:
                arc_directory = directory / "arcs" / f"arc_{arc:02d}"
                if arc_directory.exists():
                    archive = directory / "failed_attempts" / f"arc_{arc:02d}_attempt_{attempts[arc]:02d}"
                    archive.parent.mkdir(exist_ok=True)
                    shutil.move(str(arc_directory), str(archive))
                arc_directory.mkdir(parents=True)
                attempts[arc] += 1
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(arc),
                           "--directory", str(directory)]
                if reference:
                    command += ["--reference", str(reference.resolve())]
                log_path = arc_directory / "fit.log"
                with log_path.open("wb", buffering=0) as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             env=environment, cwd=HERE)
                entries.append((arc, child, log_path))
            codes, _ = monitor_children(
                directory, entries, threshold=condition_limit)
        finally:
            _stop_children(entries)
        memory_failures = []
        for arc, code in zip(batch, codes):
            if code == 0:
                continue
            log = (directory / "arcs" / f"arc_{arc:02d}" / "fit.log").read_text(errors="replace")
            memory_error = code in (-9, 137) or any(term in log for term in ("MemoryError", "std::bad_alloc", "Cannot allocate memory"))
            if not memory_error:
                raise RuntimeError(f"Arc {arc} exited {code}; inspect its fit.log (no automatic model changes).")
            memory_failures.append(arc)
        if memory_failures:
            if workers == 1:
                raise RuntimeError(f"Suspected OOM even at one worker: arcs {memory_failures}.")
            reduced = (workers + 1) // 2
            message = dict(previous_workers=workers, workers=reduced, arcs=memory_failures,
                           reason="Memory allocation error or SIGKILL (suspected OOM)")
            retry_history.append(message)
            write_json(directory / "memory_retries.json", retry_history)
            print(f"Suspected OOM: {workers} -> {reduced} workers; retaining failed logs.", flush=True)
            workers = reduced
            pending = memory_failures + pending
    return workers


def launch(case, case_id, root, reference=None, workers=None):
    """Start one isolated process per arc, with bounded concurrency and live native output."""
    case.validate()
    if case.blocked_reason:
        raise NotImplementedError(case.blocked_reason)
    if not re.fullmatch(r"\d{3}", case_id):
        raise ValueError("Case ID must be three digits.")
    workers = len(ARCS) if workers is None else workers
    if not 1 <= workers <= len(ARCS):
        raise ValueError("Workers must be between 1 and the number of arcs.")
    directory = root / f"case_{case_id}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "settings.json", asdict(case))
    write_json(directory / "status.json", {"status": "running"})
    start = time.perf_counter()
    try:
        (directory / "case.log").write_text(f"Starting {case_id}: {case.description}\nWorkers: {workers}\n")
        manifest = {}
        inputs = [local_inputs(i) for i in range(len(ARCS))]
        paths = {p for data in inputs for item in data[3:] for p in (item if isinstance(item, list) else [item])}
        for path in sorted(paths):
            with open(path, "rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest[path] = {"sha256": digest, "bytes": Path(path).stat().st_size}
        write_json(directory / "input_manifest.json", manifest)
        write_json(directory / "arcs.json", ARCS)
        snapshot = directory / "source"
        snapshot.mkdir()
        import shutil
        for source in [Path(__file__), HERE / "mro_tnf_estimation.py", HERE / "mro_utils.py"]:
            shutil.copy2(source, snapshot / source.name)
        for mesh in (HERE / "mro_macromodel").glob("*.dae"):
            shutil.copy2(mesh, snapshot / mesh.name)
        environment = {k: v for k, v in os.environ.items() if not k.startswith("MRO_")}
        environment.update(case.environment())
        environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
        write_json(directory / "runtime.json", dict(python=sys.executable, workers=workers,
                                                     environment=case.environment(), reference=str(reference)))
        update_register_safely(root, directory, "pre-launch catalogue refresh")
        print(f"{directory.name}: {workers} arc workers; logs in {directory}/arcs/arc_*/fit.log", flush=True)
        final_workers = run_workers(directory, environment, reference, workers,
                                    condition_limit=case.condition_number_limit)
        summary = aggregate(directory)
        summary.update(status="complete", wall_seconds=time.perf_counter() - start,
                       initial_workers=workers, final_workers=final_workers)
        write_json(directory / "summary.json", summary)
        write_json(directory / "status.json", {k: v for k, v in summary.items() if k != "per_arc"})
        try:
            with (directory / "case.log").open("a") as log:
                log.write(f"Complete in {summary['wall_seconds']:.3f} seconds; final worker cap {final_workers}.\n")
        except Exception as error:
            record_reporting_error(directory, "completion log write", error)
    except BaseException as error:
        failure = dict(status="failed", reason=str(error), wall_seconds=time.perf_counter() - start)
        if isinstance(error, ConditionLimitError):
            failure.update(
                condition_number_max=error.diagnostics.get("condition_number_max"),
                condition_number_threshold=error.diagnostics.get("threshold"),
                violating_arcs=error.diagnostics.get("violating_arcs", []),
            )
        write_json(directory / "status.json", failure)
        with (directory / "case.log").open("a") as log:
            log.write(f"Failed: {error}\n")
        raise
    finally:
        update_register_safely(root, directory, "post-run catalogue refresh")


def archive_failed_run(root, case_id, reason_slug):
    """Move one whole rejected attempt out of the active namespace without data loss."""
    root = Path(root)
    source = root / f"case_{case_id}"
    if not source.is_dir():
        raise FileNotFoundError(f"No failed active directory to archive: {source}")
    failed_root = root / "failed_runs"
    failed_root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in failed_root.glob(f"case_{case_id}_attempt_*_*"):
        match = re.fullmatch(rf"case_{re.escape(case_id)}_attempt_(\d+)_.*", path.name)
        if match:
            numbers.append(int(match.group(1)))
    attempt = max(numbers, default=0) + 1
    safe_reason = re.sub(r"[^a-z0-9]+", "_", reason_slug.lower()).strip("_") or "failed"
    target = failed_root / f"case_{case_id}_attempt_{attempt:02d}_{safe_reason}"
    shutil.move(str(source), str(target))
    status_path = target / "status.json"
    settings_path = target / "settings.json"
    metadata = {
        "case_id": case_id,
        "attempt": attempt,
        "reason_slug": safe_reason,
        "status": json.loads(status_path.read_text()) if status_path.exists() else {"status": "failed"},
        "settings": json.loads(settings_path.read_text()) if settings_path.exists() else None,
        "active_result_disposition": "whole attempt archived; not eligible for aggregation or scoring",
    }
    write_json(target / "failed_run.json", metadata)
    update_register(root)
    return target


def launch_guarded(case, case_id, root, reference=None, workers=None,
                   max_condition_prior_reductions=0):
    """Launch once; condition numbers are diagnostics and never trigger retries."""
    if max_condition_prior_reductions != 0:
        raise ValueError("Automatic condition-prior reductions are disabled by campaign policy.")
    root = Path(root)
    planned = root / "planned"
    planned.mkdir(parents=True, exist_ok=True)
    active = root / f"case_{case_id}"
    if active.exists():
        raise FileExistsError(f"Active case directory already exists: {active}")
    write_json(planned / f"case_{case_id}.json", asdict(case))
    try:
        launch(case, case_id, root, reference, workers)
        return case
    except BaseException as error:
        status_path = active / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("status") == "complete":
            record_reporting_error(
                active, "post-completion launch wrapper", error
            )
            print(
                f"WARNING: preserving complete case_{case_id} after "
                f"post-completion {type(error).__name__}: {error}",
                flush=True,
            )
            return case
        archive_failed_run(root, case_id, type(error).__name__)
        raise


def main():
    """Safe default: list cases and validate local filenames, never start an estimation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--validate-setup", metavar="CASE_ID", help="Load real data and construct the estimator; no propagation/fit")
    parser.add_argument("--arc", type=int, default=0, choices=range(len(ARCS)), help="Arc used by --validate-setup")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run", metavar="CASE_ID")
    parser.add_argument("--config", type=Path, help="Complete JSON Case override; requires --run")
    parser.add_argument("--reference", type=Path, help="Completed control case whose observation mask must match")
    parser.add_argument("--workers", type=int, default=len(ARCS), help="Concurrent arc processes (default: number of arcs)")
    parser.add_argument("--condition-prior-retries", type=int, choices=[0],
                        help="Compatibility option; conditioning is diagnostic-only and never retried")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate_setup:
        case = load_case(args.config) if args.config else cases()[args.validate_setup]
        if case.blocked_reason:
            raise NotImplementedError(case.blocked_reason)
        for key in list(os.environ):
            if key.startswith("MRO_"):
                del os.environ[key]
        os.environ.update(case.environment())
        os.environ["MPLBACKEND"] = "Agg"
        import mro_tnf_estimation as example
        plan = ParameterPlan(case, args.arc)
        directory = args.root / "validation" / f"case_{args.validate_setup}_arc_{args.arc:02d}"
        if directory.exists():
            directory = directory.with_name(directory.name + f"_{time.time_ns()}")
        directory.mkdir(parents=True, exist_ok=False)
        result = example.process_arc(
            local_inputs(args.arc), parameter_builder=plan.settings, prior_builder=plan.priors,
            prefit_callback=lambda frame: prepare_prefit(frame, directory, None),
            interactive=False, pad_propagation=True, validate_only=True)
        import pandas as pd
        prefit = pd.read_csv(directory / "spice_residuals.csv")
        result.update(
            empirical_arc_start_times=[float(epoch) for epoch in plan.arc_times],
            observation_epoch_min_tdb=float(prefit.time.min()),
            observation_epoch_max_tdb=float(prefit.time.max()),
            propagation_bounds=(float(result["arc_bounds"][0] - 3600.0),
                                float(result["arc_bounds"][1] + 3600.0)),
        )
        write_json(directory / "setup.json", result)
        print("Setup validated without propagation or estimation:", result, flush=True)
        return
    if args.worker is not None:
        directory = args.directory / "arcs" / f"arc_{args.worker:02d}"
        try:
            case = load_case(args.directory / "settings.json")
            run_arc(case, directory, args.worker,
                    args.reference / "arcs" / f"arc_{args.worker:02d}" if args.reference else None)
        except BaseException as error:
            write_json(directory / "failure.json", {"type": type(error).__name__, "message": str(error)})
            traceback.print_exc()
            sys.exit(1)
        return
    if args.check_inputs:
        for i in range(len(ARCS)):
            inputs = local_inputs(i)
            print(f"arc {i}: {ARCS[i]}, {len(inputs[3])} TNF, {len(inputs[5])} CK files")
    if args.write_plan:
        directory = args.root / "planned"
        directory.mkdir(parents=True, exist_ok=True)
        for number, case in cases().items():
            if number in ADAPTIVE_CASE_IDS:
                # Their exact matched control depends on preceding scientific
                # outcomes, so no executable catalogue default is emitted.
                continue
            path = directory / f"case_{number}.json"
            # Refresh only genuinely unrun plans. A completed/active case keeps
            # the exact settings file and planned provenance it launched with.
            if not (args.root / f"case_{number}").exists():
                write_json(path, asdict(case))
        update_register(args.root)
    if args.run:
        if args.run in ADAPTIVE_CASE_IDS and args.config is None:
            parser.error(
                f"case {args.run} is adaptive; supply the reviewed exact --config"
            )
        case = load_case(args.config) if args.config else cases()[args.run]
        args.root.mkdir(parents=True, exist_ok=True)
        reductions = 0 if args.condition_prior_retries is None else args.condition_prior_retries
        launch_guarded(case, args.run, args.root.resolve(), args.reference, args.workers,
                       max_condition_prior_reductions=reductions)
    elif args.list or not (args.check_inputs or args.write_plan):
        for number, case in cases().items():
            print(f"{number}: {case.description}" + (f" [{case.blocked_reason}]" if case.blocked_reason else ""))


if __name__ == "__main__":
    main()
