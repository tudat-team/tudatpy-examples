"""Promote retained-observation-tag brackets to the primary MRO orbit score.

This is a reporting-only backfill. It never changes observations, propagation,
estimated parameters, residuals, raw orbit/state artifacts, or fit selection.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

import mro_tnf_estimation_test as campaign


MASK_VERSION = "retained_observation_tag_bracket_v1"
MASK_SEMANTICS = (
    "Per arc, include score-grid epochs inside the inclusive interval from the "
    "earliest to latest retained observation tag; retain internal observation gaps."
)
ORBIT_KEYS = (
    "orbit_samples", "R_rms_m", "T_rms_m", "N_rms_m",
    "position_rms_m", "position_max_m",
)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(Path(path), value.encode())


def _atomic_json(path: Path, value) -> None:
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _backup_once(source: Path, target: Path) -> None:
    if target.exists():
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def orbit_metrics(orbit: pd.DataFrame) -> dict:
    """Return pooled sample statistics for exactly the supplied orbit rows."""
    if orbit.empty:
        raise ValueError("Primary orbit mask selected no score-grid samples.")
    position = orbit[["R", "T", "N"]].to_numpy(dtype=float)
    if not np.isfinite(position).all():
        raise ValueError("Primary orbit mask contains nonfinite RTN values.")
    return {
        "orbit_samples": int(len(orbit)),
        **{
            f"{component}_rms_m": float(np.sqrt(np.mean(orbit[component] ** 2)))
            for component in "RTN"
        },
        "position_rms_m": float(np.sqrt(np.mean(np.sum(position ** 2, axis=1)))),
        "position_max_m": float(np.max(np.linalg.norm(position, axis=1))),
    }


def retained_tag_mask(orbit: pd.DataFrame, retained: pd.DataFrame) -> tuple:
    """Return the inclusive first/last-retained-tag mask and audit metadata."""
    if orbit.empty or retained.empty:
        raise ValueError("Orbit and retained-observation tables must be nonempty.")
    lower = float(retained.time.min())
    upper = float(retained.time.max())
    mask = (orbit.t.to_numpy(dtype=float) >= lower) & (
        orbit.t.to_numpy(dtype=float) <= upper
    )
    if not mask.any() or mask.all():
        raise ValueError("Expected nonempty primary bracket and excluded edge samples.")
    selected = orbit.loc[mask]
    metadata = {
        "observation_tag_min_tdb": lower,
        "observation_tag_max_tdb": upper,
        "inclusive_bounds": True,
        "retained_observation_count": int(len(retained)),
        "full_nominal_grid_min_tdb": float(orbit.t.min()),
        "full_nominal_grid_max_tdb": float(orbit.t.max()),
        "full_nominal_grid_samples": int(len(orbit)),
        "primary_grid_first_tdb": float(selected.t.min()),
        "primary_grid_last_tdb": float(selected.t.max()),
        "primary_grid_samples": int(mask.sum()),
        "excluded_edge_grid_samples": int((~mask).sum()),
    }
    return mask, metadata


def promote_iteration_metrics(value: dict) -> dict:
    """Make existing bracketed columns primary and retain full nominal explicitly."""
    promoted = json.loads(json.dumps(value))
    for row in promoted.get("per_iteration", []):
        for suffix in (
            "orbit_samples", "R_rms_m", "T_rms_m", "N_rms_m",
            "position_rms_m", "position_max_m",
        ):
            full_key = f"full_{suffix}"
            bracket_key = f"bracketed_{suffix}"
            if full_key in row:
                row[f"full_nominal_{suffix}"] = row[full_key]
                del row[full_key]
            if bracket_key in row:
                row[suffix] = row[bracket_key]
    promoted["primary_orbit_mask_version"] = MASK_VERSION
    promoted["orbit_grid_semantics"] = (
        "primary pooled retained-observation-tag bracket score grids; inclusive bounds; "
        "internal observation gaps retained"
    )
    promoted["full_nominal_orbit_semantics"] = (
        "secondary original nominal-arc score grids; propagation padding excluded"
    )
    return promoted


def _write_iteration_metrics(directory: Path) -> dict | None:
    path = directory / "iteration_orbit_metrics.json"
    if not path.is_file():
        return None
    legacy = json.loads(path.read_text())
    _backup_once(path, directory / "iteration_orbit_metrics_full_nominal_legacy.json")
    promoted = promote_iteration_metrics(legacy)
    _atomic_json(path, promoted)
    _atomic_text(
        directory / "iteration_orbit_metrics.csv",
        pd.DataFrame(promoted.get("per_iteration", [])).to_csv(index=False),
    )
    return promoted


def _plot_results_atomic(directory: Path, residuals: pd.DataFrame,
                         primary_orbit: pd.DataFrame,
                         parameters: pd.DataFrame,
                         origin_tdb: float) -> None:
    """Regenerate the combined PDF with only primary-mask orbit samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output = directory / "results.pdf"
    _backup_once(output, directory / "results_full_nominal.pdf")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with PdfPages(temporary) as pdf:
        blocks = (
            (["spice", "prefit", "postfit"], residuals, "mHz",
             "Observed minus computed Doppler (fit observations unchanged)"),
            (["R", "T", "N"], primary_orbit, "m",
             "PRIMARY estimated minus SPICE position: inclusive retained-tag bracket"),
        )
        for columns, data, units, title in blocks:
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            for axis, column in zip(axes, columns):
                for arc, frame in data.groupby("arc_index", sort=True):
                    x_name = "time" if column in {"spice", "prefit", "postfit"} else "t"
                    values = frame[column] * (1000.0 if units == "mHz" else 1.0)
                    axis.plot(
                        (frame[x_name] - origin_tdb) / 86400.0,
                        values, ".", ms=1.5, label=f"arc {arc}",
                    )
                axis.set_ylabel(f"{column} [{units}]")
                axis.grid(alpha=.3)
            axes[0].legend(ncol=7, fontsize=8)
            axes[-1].set_xlabel("TDB days since nominal arc-grid origin")
            fig.suptitle(title)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        names = list(parameters.name.unique())
        for offset in range(0, len(names), 6):
            fig, axes = plt.subplots(3, 2, figsize=(12, 9), squeeze=False)
            for axis, name in zip(axes.flat, names[offset:offset + 6]):
                selected = parameters[parameters.name == name]
                for arc, frame in selected.groupby("arc_index", sort=True):
                    frame = frame.sort_values("plot_start_tdb")
                    values = frame[
                        "delta" if name in {"x", "y", "z", "vx", "vy", "vz"}
                        else "value"
                    ]
                    x = np.r_[frame.plot_start_tdb, frame.plot_end_tdb.iloc[-1]]
                    axis.step(
                        (x - origin_tdb) / 86400.0,
                        np.r_[values, values.iloc[-1]], where="post", label=f"arc {arc}",
                    )
                suffix = (
                    " correction at midpoint"
                    if name in {"x", "y", "z", "vx", "vy", "vz"} else ""
                )
                axis.set_title(f"{name}{suffix} [{selected.unit.iloc[0]}]", fontsize=9)
                axis.grid(alpha=.3)
                axis.set_xlabel("TDB days since nominal arc-grid origin")
            for axis in list(axes.flat)[len(names[offset:offset + 6]):]:
                axis.set_visible(False)
            fig.suptitle(
                "Estimated coefficients (full validity coverage; orbit-score mask does not alter fit)"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    os.replace(temporary, output)


def _prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _reference_cutoffs(root: Path, reference_case: str = "001") -> list:
    reference = root / f"case_{reference_case}" / "arcs"
    values = []
    for arc in range(len(campaign.ARCS)):
        retained = pd.read_csv(reference / f"arc_{arc:02d}" / "spice_residuals.csv")
        values.append((float(retained.time.min()), float(retained.time.max())))
    return values


def backfill_case(root: Path, case_id: str, reference_cutoffs: list) -> dict:
    """Atomically backfill one scientifically complete case from saved artifacts."""
    root = Path(root)
    directory = root / f"case_{case_id}"
    status_path = directory / "status.json"
    status = json.loads(status_path.read_text())
    if status.get("status") != "complete":
        raise ValueError(f"case_{case_id} is not complete.")
    required = [directory / name for name in ("orbit.csv", "residuals.csv", "parameters.csv",
                                                "summary.json", "status.json", "results.pdf")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"case_{case_id} missing root artifacts: {missing}")

    legacy_summary_path = directory / "summary_legacy_before_primary_mask.json"
    legacy_status_path = directory / "status_legacy_before_primary_mask.json"
    _backup_once(directory / "summary.json", legacy_summary_path)
    _backup_once(status_path, legacy_status_path)

    root_residuals = pd.read_csv(directory / "residuals.csv")
    root_parameters = pd.read_csv(directory / "parameters.csv")
    full_root_orbit = pd.read_csv(directory / "orbit.csv")
    old_summary = json.loads(legacy_summary_path.read_text())
    arc_primary_orbits = []
    arc_primary_summaries = []
    arc_full_summaries = []
    mask_rows = []
    per_arc_metadata = []

    for arc in range(len(campaign.ARCS)):
        arc_directory = directory / "arcs" / f"arc_{arc:02d}"
        arc_required = [arc_directory / name for name in (
            "orbit.csv", "spice_residuals.csv", "residuals.csv", "parameters.csv",
            "summary.json", "results.pdf",
        )]
        arc_missing = [str(path) for path in arc_required if not path.is_file()]
        if arc_missing:
            raise FileNotFoundError(f"case_{case_id} arc {arc:02d} missing: {arc_missing}")
        arc_orbit = pd.read_csv(arc_directory / "orbit.csv")
        retained = pd.read_csv(arc_directory / "spice_residuals.csv")
        mask, metadata = retained_tag_mask(arc_orbit, retained)
        expected = reference_cutoffs[arc]
        if (metadata["observation_tag_min_tdb"], metadata["observation_tag_max_tdb"]) != expected:
            raise ValueError(
                f"case_{case_id} arc {arc:02d} retained cutoffs differ from case_001: "
                f"{metadata['observation_tag_min_tdb'], metadata['observation_tag_max_tdb']} "
                f"!= {expected}"
            )
        primary_orbit = arc_orbit.loc[mask].copy()
        edge_orbit = arc_orbit.loc[~mask].copy()
        primary_metrics = orbit_metrics(primary_orbit)
        full_metrics = orbit_metrics(arc_orbit)
        edge_metrics = orbit_metrics(edge_orbit)
        arc_old_path = arc_directory / "summary_legacy_before_primary_mask.json"
        _backup_once(arc_directory / "summary.json", arc_old_path)
        arc_old = json.loads(arc_old_path.read_text())
        full_summary = dict(arc_old)
        full_summary.update(full_metrics)
        full_summary.update(
            orbit_mask="full_nominal_secondary",
            orbit_mask_version=MASK_VERSION,
            primary_summary_file="summary.json",
        )
        _atomic_json(arc_directory / "full_nominal_summary.json", full_summary)

        primary_summary = dict(arc_old)
        primary_summary.update(primary_metrics)
        primary_summary.update(
            primary_orbit_mask_version=MASK_VERSION,
            primary_orbit_mask_semantics=MASK_SEMANTICS,
            full_nominal_summary_file="full_nominal_summary.json",
            **_prefix_metrics(full_metrics, "full_nominal"),
            **_prefix_metrics(primary_metrics, "bracketed"),
            **_prefix_metrics(edge_metrics, "outside_edge"),
            observation_bracket_min_tdb=metadata["observation_tag_min_tdb"],
            observation_bracket_max_tdb=metadata["observation_tag_max_tdb"],
        )
        promoted_iterations = _write_iteration_metrics(arc_directory)
        if promoted_iterations is not None:
            primary_summary["iteration_orbit_metrics"] = promoted_iterations["per_iteration"]
        _atomic_json(arc_directory / "summary.json", primary_summary)
        _atomic_text(
            arc_directory / "orbit_primary_retained_tag_bracket.csv",
            primary_orbit.to_csv(index=False),
        )
        arc_residuals = pd.read_csv(arc_directory / "residuals.csv")
        arc_parameters = pd.read_csv(arc_directory / "parameters.csv")
        _plot_results_atomic(
            arc_directory, arc_residuals, primary_orbit, arc_parameters,
            float(arc_orbit.t.min()),
        )
        metadata.update(arc_index=arc)
        per_arc_metadata.append(metadata)
        mask_rows.append(metadata)
        arc_primary_orbits.append(primary_orbit)
        arc_primary_summaries.append(primary_summary)
        arc_full_summaries.append(full_summary)

    primary_orbit = pd.concat(arc_primary_orbits, ignore_index=True)
    primary_metrics = orbit_metrics(primary_orbit)
    full_metrics = orbit_metrics(full_root_orbit)
    primary_arc_rms = [item["position_rms_m"] for item in arc_primary_summaries]
    full_arc_rms = [item["position_rms_m"] for item in arc_full_summaries]
    primary_summary = dict(old_summary)
    primary_summary.update(primary_metrics)
    primary_summary.update(
        primary_orbit_mask_version=MASK_VERSION,
        primary_orbit_mask_semantics=MASK_SEMANTICS,
        primary_orbit_mask_file="primary_orbit_mask.json",
        full_nominal_summary_file="full_nominal_summary.json",
        mean_arc_position_rms_m=float(np.mean(primary_arc_rms)),
        median_arc_position_rms_m=float(np.median(primary_arc_rms)),
        worst_arc_position_rms_m=float(np.max(primary_arc_rms)),
        **_prefix_metrics(full_metrics, "full_nominal"),
        full_nominal_mean_arc_position_rms_m=float(np.mean(full_arc_rms)),
        full_nominal_median_arc_position_rms_m=float(np.median(full_arc_rms)),
        full_nominal_worst_arc_position_rms_m=float(np.max(full_arc_rms)),
        **_prefix_metrics(primary_metrics, "bracketed"),
        per_arc=arc_primary_summaries,
    )
    root_iterations = _write_iteration_metrics(directory)
    if root_iterations is not None:
        primary_summary["iteration_orbit_metrics"] = root_iterations

    full_summary = dict(old_summary)
    full_summary.update(full_metrics)
    full_summary.update(
        orbit_mask="full_nominal_secondary",
        orbit_mask_version=MASK_VERSION,
        mean_arc_position_rms_m=float(np.mean(full_arc_rms)),
        median_arc_position_rms_m=float(np.median(full_arc_rms)),
        worst_arc_position_rms_m=float(np.max(full_arc_rms)),
        per_arc=arc_full_summaries,
    )
    _atomic_json(directory / "full_nominal_summary.json", full_summary)

    _atomic_text(
        directory / "orbit_primary_retained_tag_bracket.csv",
        primary_orbit.to_csv(index=False),
    )
    mask_document = {
        "version": MASK_VERSION,
        "primary": True,
        "semantics": MASK_SEMANTICS,
        "time_scale": "TDB",
        "score_grid_step_seconds": 60.0,
        "internal_observation_gaps_retained": True,
        "light_time_or_count_interval_expansion": False,
        "raw_full_orbit_file": "orbit.csv",
        "primary_orbit_file": "orbit_primary_retained_tag_bracket.csv",
        "per_arc": per_arc_metadata,
    }
    canonical = json.dumps(mask_document, sort_keys=True, separators=(",", ":"))
    mask_document["cutoff_and_count_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    _atomic_json(directory / "primary_orbit_mask.json", mask_document)
    _atomic_text(directory / "primary_orbit_mask.csv", pd.DataFrame(mask_rows).to_csv(index=False))

    _plot_results_atomic(
        directory, root_residuals, primary_orbit, root_parameters,
        float(full_root_orbit.t.min()),
    )
    _atomic_json(directory / "summary.json", primary_summary)
    primary_status = {key: value for key, value in primary_summary.items() if key != "per_arc"}
    _atomic_json(status_path, primary_status)
    return {
        "case_id": case_id,
        "primary_position_rms_m": primary_metrics["position_rms_m"],
        "primary_worst_arc_position_rms_m": max(primary_arc_rms),
        "primary_mean_arc_position_rms_m": float(np.mean(primary_arc_rms)),
        "primary_median_arc_position_rms_m": float(np.median(primary_arc_rms)),
        "full_nominal_position_rms_m": full_metrics["position_rms_m"],
        "iteration_metrics_available": root_iterations is not None,
        "plots_regenerated": 1 + len(campaign.ARCS),
        "mask_sha256": mask_document["cutoff_and_count_sha256"],
    }


def write_register(root: Path) -> None:
    """Atomically publish a mask-explicit catalogue; never mix unlabeled orbit scores."""
    root = Path(root)
    columns = [
        "case", "status", "orbit_mask_version", "description", "residual_rms_mhz",
        "R_rms_m", "T_rms_m", "N_rms_m", "position_rms_m",
        "mean_arc_position_rms_m", "median_arc_position_rms_m",
        "worst_arc_position_rms_m", "parameters_total", "condition_number_max",
        "condition_number_threshold", "condition_number_all_within_reference_ceiling",
        "wall_seconds", "reason",
    ]
    rows = {}
    catalogue = campaign.cases()
    for number in sorted(campaign.ADAPTIVE_CASE_IDS):
        if number not in catalogue:
            continue
        case = catalogue[number]
        rows[f"case_{number}"] = {
            "case": f"case_{number}", "status": "adaptive", "description": case.description,
        }
    for path in sorted((root / "planned").glob("case_*.json")):
        raw = json.loads(path.read_text())
        known = {field.name for field in fields(campaign.Case)}
        case = campaign.Case(**{key: value for key, value in raw.items() if key in known})
        case_id = path.stem.removeprefix("case_")
        status = (
            "cancelled_by_user" if case_id in campaign.CANCELLED_CASE_REASONS else
            "deferred" if case_id in campaign.DEFERRED_CASE_REASONS else
            "blocked" if case.blocked_reason else "planned"
        )
        reason = campaign.CANCELLED_CASE_REASONS.get(
            case_id, campaign.DEFERRED_CASE_REASONS.get(
                case_id, case.blocked_reason or ""
            )
        )
        rows[path.stem] = {
            "case": path.stem,
            "status": status,
            "description": str(raw.get("description", case.description)) +
            (f" — {reason}" if reason else ""),
        }
    orbit_columns = {
        "R_rms_m", "T_rms_m", "N_rms_m", "position_rms_m",
        "mean_arc_position_rms_m", "median_arc_position_rms_m",
        "worst_arc_position_rms_m",
    }
    for directory in sorted(root.glob("case_*")):
        if not (directory / "settings.json").is_file() or not (directory / "status.json").is_file():
            continue
        settings = json.loads((directory / "settings.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        row = {"case": directory.name, "description": settings["description"], **status}
        structural_flag_path = directory / "STRUCTURAL_FLAG.json"
        if structural_flag_path.exists():
            structural_flag = json.loads(structural_flag_path.read_text())
            row["status"] = structural_flag.get(
                "status", "completed_but_structurally_flagged"
            )
            row["reason"] = structural_flag.get("reason", "")
        mask_version = status.get("primary_orbit_mask_version", "")
        row["orbit_mask_version"] = mask_version or (
            "legacy_full_nominal_pending_backfill"
            if status.get("status") == "complete" else "pending_fit"
        )
        if status.get("status") == "complete" and mask_version != MASK_VERSION:
            for key in orbit_columns:
                row[key] = ""
            reason = row.get("reason", "")
            row["reason"] = (reason + "; " if reason else "") + "primary orbit-mask backfill pending"
        rows[directory.name] = row
    names = sorted(rows)
    through_current = [name for name in names if name <= "case_007"]
    queue_priority = [
        f"case_{number}" for number in campaign.QUEUE_PRIORITY_CASE_IDS
        if f"case_{number}" in rows and f"case_{number}" not in through_current
    ]
    remaining = [name for name in names if name not in through_current and name not in queue_priority]
    ordered = through_current + queue_priority + remaining
    output_rows = [{key: rows[name].get(key, "") for key in columns} for name in ordered]
    csv_lines = []
    from io import StringIO
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(output_rows)
    _atomic_text(root / "cases.csv", stream.getvalue())
    completed_statuses = {"complete", "completed_but_structurally_flagged"}
    non_pending_statuses = completed_statuses | {
        "cancelled_by_user", "deferred", "blocked",
    }
    converted = sum(row["orbit_mask_version"] == MASK_VERSION for row in output_rows)
    complete = sum(row["status"] in completed_statuses for row in output_rows)
    next_execution = [
        f"{number} ({rows[f'case_{number}'].get('status', 'unknown')})"
        for number in campaign.QUEUE_PRIORITY_CASE_IDS
        if (f"case_{number}" in rows
            and rows[f"case_{number}"].get("status") not in non_pending_statuses)
    ]
    text = (
        "# MRO orbit-fit campaign\n\n"
        f"**Primary orbit mask: `{MASK_VERSION}`; converted complete cases: "
        f"{converted}/{complete}. Orbit columns are blank for complete legacy rows pending backfill.**\n\n"
        f"**Next execution: {' -> '.join(next_execution) if next_execution else 'needs scientific decision'}.**\n\n"
        "**Phases: defined non-shadowing comparisons, requested +[d,d,d] m "
        "matrices and same-objective unconstrained-state diagnostics complete; "
        "Sun-shadow case078 complete; case079 terminated and its partial result "
        "tree deleted at explicit user request; cases080--085 cancelled before "
        "launch. The durable queue is empty and no replacement run is "
        "authorized.**\n\n"
        "Primary R/T/N and 3D errors use each arc's inclusive first/last retained-observation-tag bracket; "
        "internal gaps remain. Full nominal metrics are retained inside each case.\n\n"
        "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"] * len(columns)) + " |\n"
    )
    for row in output_rows:
        text += "| " + " | ".join(
            f"{row[key]:.6g}" if isinstance(row[key], float) else str(row[key])
            for key in columns
        ) + " |\n"
    _atomic_text(root / "CASES.md", text)


def complete_case_ids(root: Path) -> list:
    values = []
    for directory in sorted(Path(root).glob("case_[0-9][0-9][0-9]")):
        status_path = directory / "status.json"
        if status_path.is_file() and json.loads(status_path.read_text()).get("status") == "complete":
            values.append(directory.name.removeprefix("case_"))
    return values


def write_coverage_inventory(root: Path, last_batch=None) -> dict:
    """Publish cumulative coverage for every case complete at inspection time."""
    root = Path(root)
    records = []
    for case_id in complete_case_ids(root):
        directory = root / f"case_{case_id}"
        status = json.loads((directory / "status.json").read_text())
        mask_path = directory / "primary_orbit_mask.json"
        mask = json.loads(mask_path.read_text()) if mask_path.is_file() else {}
        primary_pdfs = [directory / "results.pdf"] + [
            directory / "arcs" / f"arc_{arc:02d}" / "results.pdf"
            for arc in range(len(campaign.ARCS))
        ]
        full_pdfs = [directory / "results_full_nominal.pdf"] + [
            directory / "arcs" / f"arc_{arc:02d}" / "results_full_nominal.pdf"
            for arc in range(len(campaign.ARCS))
        ]
        iteration_source = (
            directory / "iteration_orbit_metrics_full_nominal_legacy.json"
        ).is_file()
        primary_iterations = (directory / "iteration_orbit_metrics.json").is_file()
        per_arc_iterations = sum(
            (directory / "arcs" / f"arc_{arc:02d}"
             / "iteration_orbit_metrics.json").is_file()
            for arc in range(len(campaign.ARCS))
        )
        required = {
            "summary_and_status_primary": (
                status.get("primary_orbit_mask_version") == MASK_VERSION
                and json.loads((directory / "summary.json").read_text()).get(
                    "primary_orbit_mask_version"
                ) == MASK_VERSION
            ),
            "mask_document": mask.get("version") == MASK_VERSION,
            "primary_orbit_csv": (
                directory / "orbit_primary_retained_tag_bracket.csv"
            ).is_file(),
            "raw_full_orbit_preserved": (directory / "orbit.csv").is_file(),
            "full_nominal_summary": (directory / "full_nominal_summary.json").is_file(),
            "primary_pdfs": sum(path.is_file() for path in primary_pdfs),
            "full_nominal_pdfs": sum(path.is_file() for path in full_pdfs),
            "iteration_source_available": iteration_source,
            "primary_iteration_metrics": primary_iterations,
            "per_arc_iteration_metrics": per_arc_iterations,
        }
        converted = (
            all(required[key] for key in (
                "summary_and_status_primary", "mask_document", "primary_orbit_csv",
                "raw_full_orbit_preserved", "full_nominal_summary",
            ))
            and required["primary_pdfs"] == 1 + len(campaign.ARCS)
            and required["full_nominal_pdfs"] == 1 + len(campaign.ARCS)
            and (not iteration_source or primary_iterations)
        )
        records.append({
            "case_id": case_id,
            "converted": bool(converted),
            "cutoff_and_count_sha256": mask.get("cutoff_and_count_sha256"),
            **required,
        })
    converted_ids = [item["case_id"] for item in records if item["converted"]]
    pending_ids = [item["case_id"] for item in records if not item["converted"]]
    inventory = {
        "mask_version": MASK_VERSION,
        "state": (
            "complete_for_all_currently_completed_cases"
            if not pending_ids else "backfill_pending"
        ),
        "completed_case_count": len(records),
        "converted_case_count": len(converted_ids),
        "converted_case_ids": converted_ids,
        "pending_case_ids": pending_ids,
        "missing_iteration_source_case_ids": [
            item["case_id"] for item in records
            if not item["iteration_source_available"]
        ],
        "per_case": records,
    }
    if last_batch is not None:
        inventory["last_batch_manifest"] = "PRIMARY_MASK_BACKFILL_LAST_BATCH.json"
    _atomic_json(root / "PRIMARY_MASK_BACKFILL_STATUS.json", inventory)
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=campaign.DEFAULT_ROOT)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--all-complete", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.register_only:
        write_register(root)
        write_coverage_inventory(root)
        return
    case_ids = complete_case_ids(root) if args.all_complete else (args.cases or [])
    if not case_ids:
        parser.error("Supply --all-complete or at least one --case ID.")
    write_register(root)
    cutoffs = _reference_cutoffs(root)
    report = []
    for case_id in case_ids:
        result = backfill_case(root, str(case_id).zfill(3), cutoffs)
        report.append(result)
        batch = {
            "mask_version": MASK_VERSION,
            "state": "running",
            "converted_case_ids": [item["case_id"] for item in report],
            "latest": result,
        }
        _atomic_json(root / "PRIMARY_MASK_BACKFILL_LAST_BATCH.json", batch)
        write_coverage_inventory(root, last_batch=batch)
        write_register(root)
        print(json.dumps(result, sort_keys=True), flush=True)
    batch = {
        "mask_version": MASK_VERSION,
        "state": "complete_for_requested_case_set",
        "converted_case_ids": [item["case_id"] for item in report],
        "results": report,
    }
    _atomic_json(root / "PRIMARY_MASK_BACKFILL_LAST_BATCH.json", batch)
    write_coverage_inventory(root, last_batch=batch)
    write_register(root)


if __name__ == "__main__":
    main()
