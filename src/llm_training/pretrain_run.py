"""Recoverable indexed-window continued pretraining with a phase-aware heartbeat.

The integration repository supplies the frozen job specification and optional
generation evaluator; this module owns optimization, guards and recovery.
"""

from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Callable
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
from llm_architecture.checkpointing import read_native_checkpoint, file_sha256
from llm_data.token_data import MemmapTokenDataset
from llm_data.token_views import IndexedTokenWindowDataset
from llm_training.config import ProjectConfig, TrainingConfig
from llm_training.checkpointing import build_checkpoint_payload, load_checkpoint, SchedulerState
from llm_training.recovery import RecoveryStore
from llm_training.session import SessionBudget, StopRequest, atomic_document
from llm_training.training import (TrainingState, JsonlRunLogger, create_adamw_optimizer,
    resolve_precision, set_deterministic_seed, evaluate, train_model)

PHASES = ('STARTING', 'BASELINE', 'TRAINING', 'VALIDATING', 'GENERATION_EVAL', 'CHECKPOINTING', 'FINALIZING')
TERMINAL = ('COMPLETED', 'EARLY_STOPPED', 'FAILED', 'INTERRUPTED')
WAKEUP_ENVIRONMENT = 'CODEXA_WAKEUP_AGENT'


def identity(spec: dict) -> str:
    """Fingerprint the complete immutable run specification."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


class FixedOrderSampler(Sampler[int]):
    """Replay one frozen no-replacement order; epochs never reshuffle it."""

    def __init__(self, positions: np.ndarray) -> None:
        self.positions = [int(value) for value in positions]

    def __iter__(self):
        return iter(self.positions)

    def __len__(self) -> int:
        return len(self.positions)


class SourcedWindows(Dataset):
    """Attach a declared source label to each held-out validation window."""

    def __init__(self, view: IndexedTokenWindowDataset, sources: list[str]) -> None:
        if len(sources) != len(view):
            raise ValueError('Every validation window requires one source label.')
        self.view, self.sources = view, sources

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, index: int):
        inputs, labels = self.view[index]
        return inputs, labels, self.sources[index]


def collate_sourced(rows):
    inputs = torch.stack([row[0] for row in rows])
    return dict(input_ids=inputs, labels=torch.stack([row[1] for row in rows]),
                attention_mask=torch.ones_like(inputs, dtype=torch.bool),
                sources=[row[2] for row in rows])


class Heartbeat:
    """Liveness plus current activity; written by a daemon thread, never by signal handlers."""

    def __init__(self, path: Path, *, interval: float = 15.0) -> None:
        self.path, self.interval = Path(path), interval
        self.lock = threading.Lock()
        self.value = dict(pid=os.getpid(), phase='STARTING', phase_started_utc=time.time(),
                          optimizer_step=0, last_step_utc=None, run_id=None,
                          boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._loop, name='codexa-heartbeat', daemon=True)

    def start(self) -> 'Heartbeat':
        self.write()
        self.thread.start()
        return self

    def set(self, phase: str | None = None, **values) -> None:
        with self.lock:
            if phase is not None:
                if phase not in PHASES:
                    raise ValueError(f'Unknown heartbeat phase {phase!r}.')
                if phase != self.value['phase']:
                    self.value['phase_started_utc'] = time.time()
                self.value['phase'] = phase
            self.value.update(values)
        self.write()

    def write(self) -> None:
        with self.lock:
            value = dict(self.value, ts_utc=time.time())
        atomic_document(self.path, value)

    def _loop(self) -> None:
        while not self.stopped.wait(self.interval):
            try:
                self.write()
            except OSError:
                pass

    def stop(self) -> None:
        self.stopped.set()


def load_order(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique windows and the frozen traversal as view positions."""
    data = spec['data']
    path = Path(data['train_window_order'])
    if file_sha256(path) != data['train_window_order_sha256']:
        raise ValueError('Training window order fingerprint drift.')
    order = np.load(path, allow_pickle=False)
    if order.ndim != 1 or order.dtype.kind not in 'iu' or len(np.unique(order)) != len(order):
        raise ValueError('Training order must be a unique integer vector.')
    required = spec['training']['max_steps'] * spec['training']['gradient_accumulation_steps'] * spec['training']['micro_batch_size']
    if len(order) != required:
        raise ValueError('Training order length differs from the frozen update budget.')
    windows = np.sort(order)
    return windows, np.searchsorted(windows, order)


def load_validation(spec: dict) -> tuple[np.ndarray, list[str]]:
    data = spec['data']
    for key in ('validation_window_ids', 'validation_sources'):
        if file_sha256(Path(data[key])) != data[key + '_sha256']:
            raise ValueError(f'{key} fingerprint drift.')
    windows = np.load(data['validation_window_ids'], allow_pickle=False)
    mapping = {int(key): value for key, value in json.loads(Path(data['validation_sources']).read_text()).items()}
    sources = [mapping[int(window)] for window in windows]
    return windows, sources


def weighted_validation(details: dict, weights: dict) -> float:
    """Combine per-source CE with predeclared corpus token shares."""
    if set(details['sources']) != set(weights):
        raise ValueError('Validation sources differ from declared loss weights.')
    return sum(weights[name] * details['sources'][name]['loss'] for name in weights) / sum(weights.values())


def run_pretraining(spec: dict, directory: Path, budget: SessionBudget, *, resume: bool = False,
                    generation_evaluator: Callable | None = None,
                    heartbeat_interval: float = 15.0) -> dict:
    """Run one frozen phase. Weights-only initialization is never treated as resume."""
    if os.environ.get(WAKEUP_ENVIRONMENT):
        raise RuntimeError('A wakeup analysis agent may not start training.')
    directory = Path(directory)
    metrics_dir = directory / 'metrics'
    evaluations = directory / 'evaluations'
    if not resume and (metrics_dir / 'run_metadata.json').exists():
        raise FileExistsError('Fresh run namespace required.')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluations.mkdir(parents=True, exist_ok=True)
    device = torch.device(spec.get('device', 'cuda'))
    training, evaluation, guards = spec['training'], spec['evaluation'], spec['guards']
    checkpoint_policy = spec['checkpoint']
    if checkpoint_policy.get('backend', 'legacy') != 'legacy':
        raise ValueError('This phase uses the legacy recovery backend only.')
    status_path = metrics_dir / 'progress.json'
    heartbeat = Heartbeat(metrics_dir / 'heartbeat.json', interval=heartbeat_interval).start()
    progress = dict(phase='foundational', activity='STARTING', run_name=spec['run_name'],
        lineage=spec['initialization']['path'], state='STARTING', step=0,
        total_steps=training['max_steps'], pid=os.getpid(), kind='934M learned-position decoder',
        total_tokens_seen=0, context_tokens=0, effective_epochs=0.0)
    atomic_document(status_path, progress)
    atomic_document(metrics_dir / 'budget.json', budget.document())
    config_hash = identity(spec)
    store = RecoveryStore(directory / 'checkpoints', reserve_bytes=checkpoint_policy['reserve_bytes'])
    report = dict(run_name=spec['run_name'], configuration_sha256=config_hash,
                  initialized_from=spec['initialization'], exact_resume=resume, state='STARTING',
                  checkpoints=[], budget=budget.document(), final_checkpoint_verified=False)
    model = optimizer = viewer = logger = None
    state = TrainingState()
    counters = dict(gradient_bad=0, source_bad=0, validations=[], generations=[],
                    baseline_sources=None, baseline_validation_loss=None, best_validation_loss=None)
    stop_reason = None
    project = ProjectConfig(ModelConfig(**spec['model']), TrainingConfig(**training))

    def activity(name: str, **values) -> None:
        heartbeat.set(name, **values)
        progress['activity'] = name
        atomic_document(status_path, progress)

    try:
        if spec.get('viewer_required', True):
            from llm_training.conversational_training import start_chat_viewer
            viewer = start_chat_viewer(status_path, title=f"Codexa foundational: {spec['run_name']}")
        if device.type == 'cuda':
            if spec['optimization'].get('allocator', 'default') != os.environ.get('PYTORCH_ALLOC_CONF', 'default'):
                raise ValueError('CUDA allocator differs from the frozen run configuration.')
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError('This full model workload requires CUDA BF16.')
            torch.backends.cuda.matmul.allow_tf32 = spec['optimization'].get('tf32', False)
        set_deterministic_seed(project.training.seed)
        tokenizer_path = Path(spec['tokenizer']['path'])
        if file_sha256(tokenizer_path) != spec['tokenizer']['sha256']:
            raise ValueError('Tokenizer fingerprint drift.')
        data = spec['data']
        source = MemmapTokenDataset(data['corpus_path'], dtype=data['dtype'],
                                    context_length=project.model.context_length)
        windows, positions = load_order(spec)
        train_view = IndexedTokenWindowDataset(source, windows, vocab_size=project.model.vocab_size)
        loader = DataLoader(train_view, batch_size=project.training.micro_batch_size,
                            sampler=FixedOrderSampler(positions), pin_memory=device.type == 'cuda')
        validation_windows, validation_sources = load_validation(spec)
        validation_view = SourcedWindows(IndexedTokenWindowDataset(source, validation_windows,
            vocab_size=project.model.vocab_size), validation_sources)
        validation_loader = DataLoader(validation_view, batch_size=1, collate_fn=collate_sourced,
                                       pin_memory=device.type == 'cuda')
        model = LanguageModel(project.model).to(device)
        model.set_gradient_checkpointing(spec['optimization'].get('gradient_checkpointing', True))
        expected_parameters = spec.get('expected_parameters')
        if expected_parameters is not None and count_parameters(model) != expected_parameters:
            raise ValueError('Architecture parameter count drift.')
        if model.lm_head.weight is not model.token_embeddings.weight:
            raise ValueError('Tied embedding identity drift.')
        optimization = spec['optimization']
        optimizer = create_adamw_optimizer(model, learning_rate=project.training.learning_rate,
            weight_decay=project.training.weight_decay, optimizer_name=optimization['optimizer'],
            betas=tuple(optimization['betas']), epsilon=optimization['epsilon'])
        scheduler = SchedulerState(project.training.warmup_steps, project.training.max_steps,
                                   project.training.learning_rate, optimization['minimum_learning_rate'])
        run_id = uuid.uuid4().hex
        if resume:
            restored = load_checkpoint(store.resolve(), model=model, optimizer=optimizer, scaler=None,
                expected_config=project, expected_data_manifest_sha256=data['selection_sha256'])
            if restored.resume_context['configuration_sha256'] != config_hash:
                raise ValueError('Run configuration drift on resume.')
            state, scheduler, run_id = restored.state, restored.scheduler, restored.run_id
            counters = restored.resume_context['pretraining_counters']
        else:
            initial = Path(spec['initialization']['path'])
            if file_sha256(initial) != spec['initialization']['sha256']:
                raise ValueError('Initialization checksum drift.')
            payload = read_native_checkpoint(initial)
            if payload['config']['model'] != asdict(project.model):
                raise ValueError('Initialization architecture mismatch.')
            model.load_state_dict(payload['model_state_dict'], strict=True)
            del payload
            gc.collect()
        report['run_id'] = run_id
        heartbeat.set(run_id=run_id, optimizer_step=state.optimizer_step)
        precision = resolve_precision(training['precision'], device)
        logger = JsonlRunLogger(directory, 'metrics', resume=resume, overwrite=False,
            expected_run_id=run_id if resume else None,
            expected_optimizer_step=state.optimizer_step if resume else None)
        logger.write_metadata(dict(run_id=run_id, run_name=spec['run_name'], spec=spec,
                                   configuration_sha256=config_hash))

        def free_disk() -> int:
            return shutil.disk_usage(directory / 'checkpoints' if (directory / 'checkpoints').exists() else directory).free

        def save(*, final: bool = False, best_validation: bool = False) -> dict | None:
            """Publish a verified state, or stop gracefully when the reserve would be violated."""
            nonlocal stop_reason
            needed = checkpoint_policy['estimated_checkpoint_bytes'] + checkpoint_policy['reserve_bytes']
            if free_disk() < needed:
                stop_reason = 'DISK_HEADROOM'
                report['disk_headroom'] = dict(free_bytes=free_disk(), required_bytes=needed)
                return None
            activity('CHECKPOINTING')
            context = dict(configuration_sha256=config_hash, data_manifest_sha256=data['selection_sha256'],
                data_cursor=dict(completed_epochs=state.completed_epochs, batches_in_epoch=state.batches_in_epoch),
                microstep_within_gradient_accumulation=0, pretraining_counters=counters,
                budget=budget.document(), initialization=spec['initialization'],
                optimizer=optimization['optimizer'])
            payload = build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None, state=state,
                scheduler=scheduler, config=project, run_name=spec['run_name'], run_id=run_id,
                tokenizer_reference=str(tokenizer_path), tokenizer_sha256=spec['tokenizer']['sha256'],
                resume_context=context)
            try:
                record = store.save(payload, final=final, best_validation=best_validation)
            except Exception as error:
                stop_reason = 'CHECKPOINT_FAILURE'
                raise CheckpointPublicationError(str(error)) from error
            finally:
                del payload
            report['checkpoints'].append(record)
            progress['checkpoint_path'] = str(store.resolve())
            progress['last_verified_checkpoint'] = dict(path=progress['checkpoint_path'],
                optimizer_step=record['optimizer_step'], sha256=record['sha256'])
            atomic_document(metrics_dir / 'budget.json', budget.document())
            activity('TRAINING')
            return record

        def validate(step: int) -> float | None:
            activity('VALIDATING')
            details = {}
            try:
                evaluate(model, validation_loader, device=device, precision=precision,
                         max_batches=evaluation.get('validation_batches'),
                         deadline_monotonic=budget.training_deadline + evaluation['validation_overrun_seconds'],
                         non_blocking=device.type == 'cuda', details=details)
            except TimeoutError:
                return None
            finally:
                model.train()
            sources = {name: value['loss'] for name, value in details['sources'].items()}
            value = weighted_validation(details, data['source_loss_weights'])
            record = dict(optimizer_step=step, weighted_loss=value, sources=sources, tokens=details['tokens'])
            counters['validations'].append(record)
            with (metrics_dir / 'validation-strata.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            return value

        def generate(step: int, label: str) -> None:
            if generation_evaluator is None:
                return
            if budget.remaining <= budget.reserve_seconds + evaluation['generation_reserve_seconds'] and label != 'final':
                counters['generations'].append(dict(optimizer_step=step, label=label, skipped='budget'))
                return
            activity('GENERATION_EVAL')
            target = evaluations / f'generation-{label}-step-{step:05d}'
            target.mkdir(parents=True, exist_ok=True)
            try:
                summary = generation_evaluator(model, target, budget.deadline_monotonic - 30)
                counters['generations'].append(dict(optimizer_step=step, label=label, **summary))
            except TimeoutError as error:
                counters['generations'].append(dict(optimizer_step=step, label=label, skipped=str(error)))
            except (FloatingPointError, torch.cuda.OutOfMemoryError):
                raise
            except Exception as error:
                # Diagnostic generations never gate optimization; numerical/OOM failures above do.
                counters['generations'].append(dict(optimizer_step=step, label=label,
                                                    error=f'{type(error).__name__}: {error}'[:500]))
            finally:
                model.train()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            activity('TRAINING')

        if not resume:
            activity('BASELINE')
            baseline = validate(0)
            if baseline is None:
                raise RuntimeError('Baseline validation could not complete; no optimizer update permitted.')
            counters['baseline_validation_loss'] = counters['best_validation_loss'] = baseline
            counters['baseline_sources'] = counters['validations'][-1]['sources']
            state.best_validation_loss = baseline
            report['baseline_validation'] = counters['validations'][-1]
            progress['validation_loss'] = baseline
            generate(0, 'baseline')
        mid_steps = set(evaluation.get('generation_steps', []))

        def update(current: TrainingState, metrics) -> None:
            nonlocal stop_reason
            if not math.isfinite(metrics.training_loss):
                raise FloatingPointError('Nonfinite training loss.')
            counters['gradient_bad'] = counters['gradient_bad'] + 1 if metrics.gradient_norm > guards['gradient_norm_threshold'] else 0
            if counters['gradient_bad'] >= guards['gradient_norm_patience']:
                stop_reason = 'GRADIENT_NORM_GUARD'
            if device.type == 'cuda':
                free_bytes, _ = torch.cuda.mem_get_info()
                progress['free_gpu_bytes'] = free_bytes
                if free_bytes < optimization['minimum_headroom_bytes']:
                    stop_reason = 'MEMORY_HEADROOM_GATE'
            progress.update(state='RUNNING', step=current.optimizer_step, training_loss=metrics.training_loss,
                learning_rate=metrics.learning_rate, tokens_per_second=metrics.tokens_per_second,
                gradient_norm=metrics.gradient_norm, total_tokens_seen=current.tokens_seen,
                context_tokens=current.tokens_seen, effective_epochs=current.optimizer_step / training['max_steps'],
                peak_vram_bytes=metrics.peak_reserved_vram_bytes,
                remaining_seconds=budget.remaining, elapsed_seconds=budget.seconds - budget.remaining,
                eta_seconds=min(budget.remaining, metrics.step_time_seconds * (training['max_steps'] - current.optimizer_step)),
                viewer_closed=viewer is not None and viewer.poll() is not None,
                next_checkpoint_step=(current.optimizer_step // checkpoint_policy['interval'] + 1) * checkpoint_policy['interval'])
            progress['deadline_warning'] = budget.remaining < budget.reserve_seconds + 300
            progress['loss_history'] = (progress.get('loss_history', []) + [metrics.training_loss])[-30:]
            progress['speed_history'] = (progress.get('speed_history', []) + [metrics.tokens_per_second])[-30:]
            heartbeat.set(optimizer_step=current.optimizer_step, last_step_utc=time.time())
            atomic_document(status_path, progress)
            atomic_document(metrics_dir / 'budget.json', budget.document())
            best = False
            if current.optimizer_step % evaluation['cadence_updates'] == 0 and budget.remaining > budget.reserve_seconds:
                value = validate(current.optimizer_step)
                if value is not None:
                    progress['validation_loss'] = value
                    best = value < counters['best_validation_loss']
                    if best:
                        counters['best_validation_loss'] = current.best_validation_loss = value
                    progress['best_validation_loss'] = counters['best_validation_loss']
                    worse = any(loss - counters['baseline_sources'][name] > guards['source_validation_CE_increase']
                                for name, loss in counters['validations'][-1]['sources'].items())
                    counters['source_bad'] = counters['source_bad'] + 1 if worse else 0
                    if counters['source_bad'] >= guards['validation_patience']:
                        stop_reason = 'SOURCE_CE_DETERIORATION'
            if current.optimizer_step in mid_steps:
                generate(current.optimizer_step, 'mid')
            if current.optimizer_step % checkpoint_policy['interval'] == 0 and current.optimizer_step < training['max_steps']:
                if save(best_validation=best) is None and best:
                    counters['pending_best'] = True
            elif best:
                counters['pending_best'] = True
            activity('TRAINING')

        with StopRequest() as requested:
            def stop_before_update() -> bool:
                return requested() or stop_reason is not None
            activity('TRAINING')
            progress['state'] = 'RUNNING'
            try:
                state, _ = train_model(model, loader, optimizer, device=device, precision=precision,
                    max_steps=project.training.max_steps,
                    gradient_accumulation_steps=project.training.gradient_accumulation_steps,
                    gradient_clip=project.training.gradient_clip, warmup_steps=project.training.warmup_steps,
                    peak_learning_rate=project.training.learning_rate, state=state, seed=project.training.seed,
                    minimum_learning_rate=scheduler.minimum_learning_rate, run_name=spec['run_name'],
                    run_id=run_id, logger=logger, deadline_monotonic=budget.training_deadline,
                    loss_normalization=optimization['loss_normalization'],
                    on_optimizer_step=update, stop_requested=stop_before_update, max_epochs=1)
            except TimeoutError:
                stop_reason = stop_reason or 'TRAINING_DEADLINE'
            if requested():
                stop_reason = f'INTERRUPTED_SIGNAL_{requested.signal_number}'
            elif time.monotonic() >= budget.training_deadline and stop_reason is None:
                stop_reason = 'TRAINING_DEADLINE'
        activity('FINALIZING')
        last = counters['validations'][-1]['optimizer_step'] if counters['validations'] else -1
        best = counters.pop('pending_best', False)
        if last != state.optimizer_step and budget.remaining > budget.reserve_seconds / 2:
            value = validate(state.optimizer_step)
            if value is not None and value < counters['best_validation_loss']:
                counters['best_validation_loss'] = state.best_validation_loss = value
                best = True
        if stop_reason == 'DISK_HEADROOM' or budget.remaining <= checkpoint_policy['minimum_save_seconds']:
            stop_reason = stop_reason or 'FINAL_SAVE_RESERVE_EXHAUSTED'
        else:
            record = save(final=True, best_validation=best)
            if record is not None:
                report['final_checkpoint'] = verify_final(store, record, run_id=run_id,
                                                          step=state.optimizer_step, config_hash=config_hash)
                report['final_checkpoint_verified'] = True
                generate(state.optimizer_step, 'final')
        if stop_reason is None or stop_reason == 'TRAINING_DEADLINE':
            final_state = 'COMPLETED'
        elif stop_reason.startswith('INTERRUPTED'):
            final_state = 'INTERRUPTED'
        elif stop_reason == 'FINAL_SAVE_RESERVE_EXHAUSTED':
            final_state = 'FAILED'
        else:
            final_state = 'EARLY_STOPPED'
        report.update(state=final_state,
            stop_reason=stop_reason or ('STEP_CAP' if state.optimizer_step == training['max_steps'] else 'DATA_COVERAGE_CAP'),
            training_state=state.to_dict(), counters=counters, recovery_index=store.index())
    except BaseException as error:
        # Optimizer/OOM failures can leave uncertain weights. Never save them.
        interrupted = isinstance(error, KeyboardInterrupt)
        report.update(state='INTERRUPTED' if interrupted else 'FAILED', error_type=type(error).__name__,
                      error=str(error)[:2000], stop_reason=stop_reason or type(error).__name__,
                      training_state=state.to_dict(), counters=counters,
                      recovery_index=store.index(), recovery_required=True)
        raise
    finally:
        report['budget'] = budget.document()
        progress.update(state=report['state'], stop_reason=report.get('stop_reason'),
                        remaining_seconds=budget.remaining, elapsed_seconds=budget.seconds - budget.remaining,
                        error=report.get('error'))
        heartbeat.set('FINALIZING', terminal_state=report['state'])
        heartbeat.stop()
        atomic_document(metrics_dir / 'report.json', report)
        atomic_document(status_path, progress)
        atomic_document(metrics_dir / 'budget.json', budget.document())
        if logger is not None:
            logger.close()
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


class CheckpointPublicationError(RuntimeError):
    """A checkpoint could not be published; the prior verified state is retained."""


def verify_final(store: RecoveryStore, record: dict, *, run_id: str, step: int, config_hash: str) -> dict:
    """Independent reload of the published final role; exit status alone proves nothing."""
    path = store.resolve('final')
    if file_sha256(path) != record['sha256']:
        raise ValueError('Final checkpoint digest mismatch.')
    loaded = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    try:
        if (loaded['run_id'] != run_id or loaded['training_state']['optimizer_step'] != step
                or loaded['resume_context']['configuration_sha256'] != config_hash):
            raise ValueError('Final checkpoint identity mismatch.')
    finally:
        del loaded
    return dict(path=str(path), sha256=record['sha256'], bytes=record['bytes'],
                optimizer_step=step, verified=True)
