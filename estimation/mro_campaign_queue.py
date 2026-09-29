"""Persistent sequential dispatcher for the approved MRO campaign queue.

This module deliberately delegates each fit to the existing campaign launch,
monitor, OOM-retry, aggregation, and failure-archive implementation.  It adds
only durable queue state, exact matched-config validation, and a process lock.
"""

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import mro_tnf_estimation_test as campaign


class QueueLockedError(RuntimeError):
    """Raised when another dispatcher already owns the campaign queue."""


class QueueValidationError(RuntimeError):
    """Raised before launch when queue provenance or a result is inconsistent."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _append_log(root, message):
    with (Path(root) / "QUEUE_DRIVER.log").open("a") as stream:
        stream.write(f"{_now()} {message}\n")


@contextmanager
def queue_lock(root):
    path = Path(root) / "QUEUE.lock"
    stream = path.open("a+")
    try:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise QueueLockedError(f"Another dispatcher owns {path}") from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} started={_now()}\n")
        stream.flush()
        yield path
    finally:
        try:
            fcntl.flock(stream, fcntl.LOCK_UN)
        finally:
            stream.close()


def _load_json(path):
    return json.loads(Path(path).read_text())


def _normalized_case(path):
    # JSON settings represent sequence-valued fields as lists, whereas missing
    # historical fields receive tuple dataclass defaults.  Compare their JSON
    # forms so an explicit [0, 0, 0] zero offset is not a scientific diff.
    return json.loads(json.dumps(asdict(campaign.load_case(Path(path)))))


def validate_job(root, job):
    """Return the validated Case and reference path for one matched job."""
    root = Path(root).resolve()
    required = {"case_id", "control_case_id", "config", "reference_case_id",
                "changed_fields", "expected_values", "purpose"}
    missing = required - set(job)
    if missing:
        raise QueueValidationError(f"Queue job missing keys: {sorted(missing)}")
    case_id = str(job["case_id"])
    if not case_id.isdigit() or len(case_id) != 3:
        raise QueueValidationError(f"Invalid case ID {case_id!r}")
    config_path = (root / job["config"]).resolve()
    if root not in config_path.parents:
        raise QueueValidationError(f"Config escapes campaign root: {config_path}")
    if not config_path.is_file():
        raise QueueValidationError(f"Missing approved config: {config_path}")
    control_path = root / f"case_{job['control_case_id']}" / "settings.json"
    reference = root / f"case_{job['reference_case_id']}"
    if not control_path.is_file():
        raise QueueValidationError(f"Missing named control settings: {control_path}")
    if not (reference / "status.json").is_file():
        raise QueueValidationError(f"Missing observation reference: {reference}")
    reference_status = _load_json(reference / "status.json")
    if reference_status.get("status") != "complete":
        raise QueueValidationError(f"Reference {reference.name} is not complete")

    control = _normalized_case(control_path)
    candidate = _normalized_case(config_path)
    changed = set(job["changed_fields"])
    actual = {
        key for key in control
        if key != "description" and control[key] != candidate[key]
    }
    if actual != changed:
        raise QueueValidationError(
            f"case_{case_id} differs from case_{job['control_case_id']} in "
            f"{sorted(actual)}, expected exactly {sorted(changed)}"
        )
    for key, expected in job["expected_values"].items():
        if key not in changed or candidate.get(key) != expected:
            raise QueueValidationError(
                f"case_{case_id} expected {key}={expected!r}, got {candidate.get(key)!r}"
            )

    case = campaign.load_case(config_path)
    case.validate()
    if not case.apply_apriori_parameter_deviation:
        raise QueueValidationError("Future queued fits must use anchored priors")
    if case.position_sigma_m < campaign.MIN_POSITION_SIGMA_M:
        raise QueueValidationError("Position prior violates the 100 m floor")
    if case.velocity_sigma_m_s < campaign.MIN_VELOCITY_SIGMA_M_S:
        raise QueueValidationError("Velocity prior violates the 0.1 m/s floor")
    if case.iterations > 5:
        raise QueueValidationError("Queued fits may use at most five iterations")
    if case.empirical_edge_policy not in {
            "merge_case001_zero_edges", "merge_one_orbit_conservative_edges",
            "merge_one_orbit_h_zero_edges"}:
        raise QueueValidationError("Queued fit does not retain audited edge handling")
    environment = case.environment()
    if (environment["MRO_SELF_SHADOWING_PIXELS"] != "0"
            or environment["MRO_RADIATION_SELF_SHADOWING_PIXELS"] != "0"):
        raise QueueValidationError("Legacy coupled shadowing fallbacks must stay disabled")
    expected_shadowing = {
        "MRO_SUN_RADIATION_SELF_SHADOWING_PIXELS":
            str(case.sun_radiation_shadowing_pixels),
        "MRO_AERODYNAMIC_SELF_SHADOWING_PIXELS":
            str(case.aerodynamic_shadowing_pixels),
        "MRO_MARS_RADIATION_SELF_SHADOWING_PIXELS":
            str(case.mars_radiation_shadowing_pixels),
    }
    if any(environment[key] != value for key, value in expected_shadowing.items()):
        raise QueueValidationError("Source-specific shadowing environment is inconsistent")
    if (any(value != "0" for value in expected_shadowing.values())
            and case_id not in campaign.FINAL_SHADOW_CASE_IDS):
        raise QueueValidationError(
            "Only explicitly approved final-phase case IDs may enable self-shadowing"
        )

    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return case, reference, config_path, digest


def validate_completed_result(root, job, expected_config=None):
    """Perform the lightweight checks required before the next launch."""
    root = Path(root)
    case_id = str(job["case_id"])
    directory = root / f"case_{case_id}"
    status = _load_json(directory / "status.json")
    if status.get("status") != "complete":
        raise QueueValidationError(
            f"case_{case_id} terminal status is {status.get('status')!r}, not complete"
        )
    settings = _normalized_case(directory / "settings.json")
    expected = expected_config or _normalized_case(root / job["config"])
    if settings != expected:
        raise QueueValidationError(f"case_{case_id} settings do not match approved config")
    summary = _load_json(directory / "summary.json")
    if summary.get("status") != "complete" or len(summary.get("per_arc", [])) != len(campaign.ARCS):
        raise QueueValidationError(f"case_{case_id} summary is incomplete")
    required_finite = (
        "residual_rms_mhz", "R_rms_m", "T_rms_m", "N_rms_m",
        "position_rms_m", "worst_arc_position_rms_m",
    )
    for key in required_finite:
        value = summary.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise QueueValidationError(f"case_{case_id} has invalid summary {key}={value!r}")
    if summary.get("observations") != _load_json(
        root / f"case_{job['reference_case_id']}" / "summary.json"
    ).get("observations"):
        raise QueueValidationError(f"case_{case_id} observation count changed")
    return summary


def _write_status(root, **updates):
    root = Path(root)
    path = root / "QUEUE_STATUS.json"
    state = _load_json(path) if path.exists() else {
        "completed_case_ids": [], "history": [],
    }
    state.update(updates, updated_at=_now(), dispatcher_pid=os.getpid())
    _atomic_json(path, state)
    return state


def _case_processes(directory, case_id):
    """Identify a case parent/workers by command and immutable proc start time."""
    directory = str(Path(directory).resolve())
    matches = []
    for process in Path("/proc").glob("[0-9]*"):
        try:
            command = (process / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            fields = (process / "stat").read_text().split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        is_runner = "mro_tnf_estimation_test.py" in command
        owns_parent = f"--run {case_id}" in command
        owns_worker = "--directory" in command and directory in command
        if is_runner and (owns_parent or owns_worker):
            matches.append({
                "pid": int(process.name), "proc_start_ticks": int(fields[21]),
                "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "role": "parent" if owns_parent else "worker",
            })
    return sorted(matches, key=lambda item: item["pid"])


def _wait_for_existing(root, job, poll_seconds, next_case_id):
    directory = Path(root) / f"case_{job['case_id']}"
    missing_process_checks = 0
    while True:
        if not (directory / "status.json").is_file():
            raise QueueValidationError(
                f"Existing {directory.name} has no status.json; refusing duplicate launch"
            )
        state = _load_json(directory / "status.json").get("status")
        if state != "running":
            return state
        processes = _case_processes(directory, str(job["case_id"]))
        if processes:
            missing_process_checks = 0
        else:
            missing_process_checks += 1
        _write_status(
            root, state="waiting_existing", current_case_id=job["case_id"],
            next_case_id=next_case_id,
            reason="waiting for already-running case to finish",
            verified_existing_processes=processes,
            existing_process_verified_at=_now(),
            missing_existing_process_checks=missing_process_checks,
        )
        if missing_process_checks >= 2:
            raise QueueValidationError(
                f"case_{job['case_id']} still says running but no matching "
                "parent/worker process survived two checks"
            )
        time.sleep(poll_seconds)


def dispatch_queue(root, poll_seconds=5.0, launch_function=campaign.launch_guarded,
                   wait_for_scientific_decision=True):
    """Run approved jobs sequentially, re-reading the atomic queue between jobs."""
    root = Path(root).resolve()
    queue_path = root / "RUN_QUEUE.json"
    with queue_lock(root):
        _append_log(root, f"dispatcher pid={os.getpid()} acquired lock")
        try:
            while True:
                queue = _load_json(queue_path)
                jobs = queue.get("jobs", [])
                ids = [str(item.get("case_id")) for item in jobs]
                if len(ids) != len(set(ids)):
                    raise QueueValidationError("RUN_QUEUE.json contains duplicate case IDs")
                previous = _load_json(root / "QUEUE_STATUS.json") if (
                    root / "QUEUE_STATUS.json"
                ).exists() else {"completed_case_ids": [], "history": []}
                completed = list(previous.get("completed_case_ids", []))
                worker_cap = int(previous.get("worker_cap", len(campaign.ARCS)))
                if not 1 <= worker_cap <= len(campaign.ARCS):
                    raise QueueValidationError(f"Invalid carried worker cap {worker_cap}")
                pending = [job for job in jobs if str(job["case_id"]) not in completed]
                if not pending:
                    _write_status(
                        root, state="needs_scientific_decision", current_case_id=None,
                        next_case_id=None, completed_case_ids=completed,
                        history=previous.get("history", []),
                        worker_cap=worker_cap,
                        reason=("approved queue exhausted while the campaign remains unfinished; "
                                "append reviewed jobs atomically"),
                    )
                    _append_log(root, "approved queue exhausted; scientific decision required")
                    if not wait_for_scientific_decision:
                        return
                    time.sleep(poll_seconds)
                    continue

                job = pending[0]
                next_id = str(pending[1]["case_id"]) if len(pending) > 1 else None
                approval = job.get("approval_state", "approved")
                if approval != "approved":
                    _write_status(
                        root, state="needs_scientific_decision",
                        current_case_id=None, next_case_id=str(job["case_id"]),
                        completed_case_ids=completed,
                        history=previous.get("history", []),
                        worker_cap=worker_cap,
                        reason=(f"case_{job['case_id']} approval_state={approval}; "
                                "dispatcher is waiting and will reread RUN_QUEUE.json"),
                    )
                    if not wait_for_scientific_decision:
                        return
                    time.sleep(poll_seconds)
                    continue
                case, reference, config_path, digest = validate_job(root, job)
                # Freeze the same normalized JSON representation used for the
                # completed settings check.  Keeping a separate dataclass
                # representation here made the post-run gate needlessly
                # sensitive to sequence container types and to any in-process
                # Case normalization performed by the launcher.
                expected_config = _normalized_case(config_path)
                case_id = str(job["case_id"])
                directory = root / f"case_{case_id}"
                _write_status(
                    root, state="validating", current_case_id=case_id,
                    next_case_id=next_id, completed_case_ids=completed,
                    history=previous.get("history", []), reason="",
                    current_config_sha256=digest, worker_cap=worker_cap,
                )
                if directory.exists():
                    terminal = _wait_for_existing(root, job, poll_seconds, next_id)
                    if terminal != "complete":
                        raise QueueValidationError(
                            f"Existing case_{case_id} ended with status {terminal!r}"
                        )
                else:
                    _write_status(
                        root, state="running", current_case_id=case_id,
                        next_case_id=next_id, completed_case_ids=completed,
                        history=previous.get("history", []), reason="",
                        current_config_sha256=digest, started_at=_now(),
                        worker_cap=worker_cap,
                    )
                    _append_log(
                        root, f"launch case_{case_id} control=case_{job['control_case_id']} "
                              f"changes={job['changed_fields']} config_sha256={digest}"
                    )
                    launch_function(case, case_id, root, reference, worker_cap)

                summary = validate_completed_result(root, job, expected_config)
                worker_cap = int(summary.get("final_workers", worker_cap))
                if not 1 <= worker_cap <= len(campaign.ARCS):
                    raise QueueValidationError(
                        f"case_{case_id} reported invalid final worker cap {worker_cap}"
                    )
                completed.append(case_id)
                history = list(previous.get("history", []))
                history.append({
                    "case_id": case_id, "completed_at": _now(),
                    "config_sha256": digest,
                    "residual_rms_mhz": summary["residual_rms_mhz"],
                    "position_rms_m": summary["position_rms_m"],
                    "worst_arc_position_rms_m": summary["worst_arc_position_rms_m"],
                    "final_workers": worker_cap,
                })
                _write_status(
                    root, state="completed_job", current_case_id=case_id,
                    next_case_id=next_id, completed_case_ids=completed,
                    history=history, reason="",
                    worker_cap=worker_cap,
                )
                _append_log(root, f"validated complete case_{case_id}; advancing")
        except BaseException as error:
            _write_status(
                root, state="paused_error", reason=f"{type(error).__name__}: {error}",
            )
            _append_log(root, f"PAUSED {type(error).__name__}: {error}")
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=campaign.DEFAULT_ROOT)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    dispatch_queue(args.root, args.poll_seconds)


if __name__ == "__main__":
    main()
