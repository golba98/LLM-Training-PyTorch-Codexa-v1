"""Typed checks and storage estimates for long training-run preflight."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in {"pass", "warning", "fail"}:
            raise ValueError("Preflight status must be pass, warning, or fail.")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointStorageEstimate:
    checkpoint_bytes: int
    milestone_count: int
    retained_checkpoint_count: int
    projected_bytes: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_checkpoint_storage(
    *,
    parameter_count: int,
    max_steps: int,
    checkpoint_interval: int,
) -> CheckpointStorageEstimate:
    """Estimate AdamW checkpoint retention plus serialization overhead."""

    for name, value in (
        ("parameter_count", parameter_count),
        ("max_steps", max_steps),
        ("checkpoint_interval", checkpoint_interval),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer.")
    checkpoint_bytes = parameter_count * 12 + 1024 * 1024
    milestone_count = max_steps // checkpoint_interval
    if max_steps % checkpoint_interval:
        milestone_count += 1
    retained_count = milestone_count + 3
    return CheckpointStorageEstimate(
        checkpoint_bytes=checkpoint_bytes,
        milestone_count=milestone_count,
        retained_checkpoint_count=retained_count,
        projected_bytes=checkpoint_bytes * retained_count,
    )


def independent_filesystems(source: Path, destination: Path) -> bool:
    """Return whether existing source and destination use different devices."""

    if not source.exists() or not destination.exists():
        raise FileNotFoundError("Both filesystem paths must already exist.")
    return source.stat().st_dev != destination.stat().st_dev


def disk_capacity_check(
    path: Path,
    *,
    required_bytes: int,
    name: str = "disk_capacity",
) -> PreflightCheck:
    if required_bytes <= 0:
        raise ValueError("required_bytes must be positive.")
    if not name:
        raise ValueError("name must be non-empty.")
    if not path.exists():
        return PreflightCheck(
            name,
            "fail",
            f"Path does not exist: {path}",
        )
    free = shutil.disk_usage(path).free
    status = "pass" if free >= required_bytes else "fail"
    return PreflightCheck(
        name,
        status,
        f"free={free} required={required_bytes} path={path}",
    )


def preflight_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_component_storage(*, model_bytes: int, optimizer_bytes: int,
                               updated_generations: int, initial_models: int = 1,
                               metadata_snapshots: int | None = None) -> dict:
    """Conservative no-GC capacity estimate; counts unique complete states.

    Caller budgets off-schedule saves as additional generations, not free aliases.
    No cross-run reuse is assumed except explicitly verified initial models.
    """
    values = (model_bytes, optimizer_bytes, updated_generations, initial_models)
    if any(type(v) is not int or v < 0 for v in values) or model_bytes == 0 or initial_models == 0:
        raise ValueError('Storage sizes/counts must be valid nonnegative integers.')
    snapshots = metadata_snapshots if metadata_snapshots is not None else updated_generations + initial_models + 1
    if type(snapshots) is not int or snapshots < updated_generations + initial_models:
        raise ValueError('Metadata snapshot count is below unique state count.')
    metadata_bytes = snapshots * 1024**2
    payload_bytes = initial_models * model_bytes + updated_generations * (model_bytes + optimizer_bytes)
    temporary_bytes = model_bytes + optimizer_bytes + 8 * 1024**2
    peak = payload_bytes + metadata_bytes + temporary_bytes
    return dict(backend='optimized', garbage_collection_enabled=False,
                model_bytes=model_bytes, optimizer_bytes=optimizer_bytes,
                physical_model_objects=updated_generations + initial_models,
                physical_optimizer_objects=updated_generations,
                metadata_snapshots=snapshots, retained_payload_bytes=payload_bytes,
                metadata_budget_bytes=metadata_bytes, temporary_bytes=temporary_bytes,
                projected_peak_bytes=peak,
                required_free_bytes=max(180 * 1024**3, 100 * 1024**3 + peak))


def checkpoint_publication_bound(*, max_steps: int, checkpoint_interval: int,
                                 evaluation_interval: int) -> int:
    """Budget interval saves, every evaluation improvement, and four boundaries.

    Counts independently even when intervals coincide. Boundary allowance covers
    initialization, baseline, controlled interruption and finalization. A resumed
    job must recheck capacity; no reuse credit is assumed.
    """
    if any(type(v) is not int or v <= 0 for v in (max_steps, checkpoint_interval, evaluation_interval)):
        raise ValueError('Checkpoint schedule values must be positive integers.')
    return ((max_steps + checkpoint_interval - 1) // checkpoint_interval
            + (max_steps + evaluation_interval - 1) // evaluation_interval + 4)


def estimate_bounded_component_storage(*, model_bytes: int, optimizer_bytes: int,
                                       retention: dict) -> dict:
    """Account for the entire hard quota, including orphans, plus staging margin."""
    from .checkpoint_retention import RetentionPolicy
    if any(type(v) is not int or v < 0 for v in (model_bytes, optimizer_bytes)) or not model_bytes:
        raise ValueError('Invalid bounded checkpoint tensor size estimate.')
    payload_bound = model_bytes + optimizer_bytes + 1024**2
    options = dict(retention)
    options.setdefault('max_checkpoint_bytes', payload_bound)
    policy = RetentionPolicy(**options)
    if policy.max_checkpoint_bytes < payload_bound:
        raise ValueError('Checkpoint byte bound is below the conservative optimizer estimate.')
    return dict(backend='optimized', retention=policy.document(),
        garbage_collection_enabled=False, retained_snapshot_limit=policy.max_snapshots,
        projected_peak_bytes=policy.max_bytes, temporary_bytes=payload_bound + 8 * 1024**2,
        required_free_bytes=max(180 * 1024**3, 100 * 1024**3 + policy.max_bytes),
        completion_requires_schedule_fit=True)


def project_continuous_storage(*, checkpoint_bytes: int, retained_states: int = 6,
                               active_runs: int = 1, checkpoint_minutes: int = 10) -> dict:
    """Explicit schedule projections, never an admission credit for hypothetical deletion."""
    if any(type(v) is not int or v <= 0 for v in
           (checkpoint_bytes, retained_states, active_runs, checkpoint_minutes)):
        raise ValueError('Projection sizes and schedule values must be positive integers.')
    if retained_states < 2 * active_runs:
        raise ValueError('Quota cannot preserve latest and previous for every active run.')
    periods = []
    for hours in (2, 5, 10):
        publications = (hours * 60 + checkpoint_minutes - 1) // checkpoint_minutes + 2
        periods.append(dict(hours=hours, publications_per_run=publications,
            legacy_six_slot_upper_bytes=6*checkpoint_bytes*active_runs,
            v1_append_only_upper_bytes=publications*checkpoint_bytes*active_runs,
            v2_bounded_upper_bytes=retained_states*checkpoint_bytes,
            v2_continuous=False,
            v3_simulated_retained_upper_bytes=retained_states*checkpoint_bytes,
            v3_continuous_conditional=True))
    # 10% representation margin and 16 MiB container/recovery allowance.
    mib = 1024**2
    bound = ((checkpoint_bytes*110 + 99)//100 + 16*mib + mib-1)//mib*mib
    quota = (retained_states + 2*active_runs + 2)*bound + 8*mib
    return dict(measured_checkpoint_bytes=checkpoint_bytes, periods=periods,
        retained_states=retained_states, active_runs=active_runs,
        declared_checkpoint_bound=bound, hard_quota_bytes=quota,
        temporary_and_final_bounds=2*active_runs+2,
        required_free_bytes=max(180*1024**3, 100*1024**3+quota),
        hypothetical_reclamation_is_available_space=False,
        historical_migration=False, production_gc_enabled=False)


def verified_optimizer_storage_evidence(path: Path, *, model_config: dict,
                                        optimizer_name: str) -> dict:
    """Reverify explicit native evidence; absence keeps the existing FP32 estimate.

    Evidence is a bound for a matching full-model 8-bit representation, never proof
    of recovery/GPU qualification. Checkpoint precision and optimizer math are unchanged.
    """
    import json
    from llm_architecture.checkpoint_inspection import inspect_checkpoint_storage
    document=json.loads(Path(path).read_text())
    supplied=document.get('evidence',document)
    if supplied.get('format')!='codexa-storage-evidence-v1':
        raise ValueError('Unsupported checkpoint storage evidence.')
    actual=inspect_checkpoint_storage(Path(supplied['path']))
    if (actual['sha256']!=supplied['sha256'] or actual['config']['model']!=model_config
            or optimizer_name!='adamw8bit' or not actual['complete_optimizer_state']
            or not actual['components']['optimizer_state_dict']['by_dtype'].get('torch.uint8')):
        raise ValueError('Optimizer storage evidence does not match this complete model/optimizer.')
    return dict(model_bytes=actual['components']['model_state_dict']['bytes'],
                optimizer_bytes=actual['components']['optimizer_state_dict']['bytes'],
                checkpoint_sha256=actual['sha256'],
                declared_checkpoint_bound=project_continuous_storage(
                    checkpoint_bytes=actual['file_bytes'])['declared_checkpoint_bound'])
