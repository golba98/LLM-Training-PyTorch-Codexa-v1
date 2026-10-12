"""Recoverable conversational training; integration supplies behavioral evaluation."""

from collections import Counter
from dataclasses import asdict
from functools import partial
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Callable
import uuid

import torch
from torch.utils.data import DataLoader, Subset

from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
from llm_architecture.checkpointing import read_native_checkpoint, file_sha256
from llm_data.chat_dataset import IndexedChatDataset, ConversationSampler, collate_chat
from llm_tokenizer.tokenizer import load_tokenizer
from llm_training.config import ProjectConfig, TrainingConfig
from llm_training.checkpointing import build_checkpoint_payload, load_checkpoint, SchedulerState
from llm_training.conversational_training import start_chat_viewer
from llm_training.recovery import RecoveryStore
from llm_training.quality_guard import behavioral_regression, validate_regression_policy
from llm_training.session import SessionBudget, StopRequest, atomic_document
from llm_training.training import (TrainingState, JsonlRunLogger, create_adamw_optimizer,
    resolve_precision, set_deterministic_seed, evaluate, train_model)


def identity(spec: dict) -> str:
    """Fingerprint the complete immutable run specification."""
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def run_chat(spec: dict, directory: Path, budget: SessionBudget, *, resume: bool = False,
             quality_evaluator: Callable | None = None, reclamation_authorization=None) -> dict:
    """Run one validated phase; never infer weights initialization as resume."""
    if spec.get('purpose') == 'preparation' and (spec['training']['max_steps'] > 128 or budget.seconds > 1200):
        raise ValueError('Preparation pilots are capped at 128 steps and 20 minutes.')
    if spec.get('checkpoint', {}).get('retention') is not None and spec['checkpoint'].get('storage_backend', 'legacy') != 'optimized':
        raise ValueError('Retention requires the optimized backend.')
    validate_regression_policy(spec['evaluation'].get('regression_policy'))
    patience = spec['evaluation'].get('regression_patience', 2)
    if type(patience) is not int or patience < 1:
        raise ValueError('Behavioral regression patience must be a positive integer.')
    directory = Path(directory)
    if spec['checkpoint'].get('storage_backend', 'legacy') == 'optimized':
        import shutil
        from llm_training.preflight import (estimate_component_storage, checkpoint_publication_bound,
            estimate_bounded_component_storage)
        if spec['checkpoint'].get('reserve_bytes', 100 * 1024**3) < 100 * 1024**3:
            raise ValueError('Optimized trainer cannot reduce the production recovery reserve.')
        with torch.device('meta'):
            parameters = count_parameters(LanguageModel(ModelConfig(**spec['model'])))
        plan = estimate_component_storage(model_bytes=parameters * 4,
            optimizer_bytes=parameters * 8 + 1024**2,
            updated_generations=checkpoint_publication_bound(max_steps=spec['training']['max_steps'],
                checkpoint_interval=spec['training']['checkpoint_interval'],
                evaluation_interval=spec['training']['evaluation_interval']))
        if spec['checkpoint'].get('retention') is not None:
            model_bytes,optimizer_bytes=parameters*4,parameters*8+1024**2
            options=dict(spec['checkpoint']['retention'])
            if spec['checkpoint'].get('optimizer_storage_evidence'):
                from llm_training.preflight import verified_optimizer_storage_evidence
                evidence=verified_optimizer_storage_evidence(
                    Path(spec['checkpoint']['optimizer_storage_evidence']),
                    model_config=spec['model'],optimizer_name=spec['optimization']['optimizer'])
                model_bytes,optimizer_bytes=evidence['model_bytes'],evidence['optimizer_bytes']
                options.setdefault('max_checkpoint_bytes',evidence['declared_checkpoint_bound'])
            plan = estimate_bounded_component_storage(model_bytes=model_bytes,
                optimizer_bytes=optimizer_bytes, retention=options)
        target = Path(spec['checkpoint'].get('store_root') or directory)
        while not target.exists():
            target = target.parent
        free = shutil.disk_usage(target).free
        if free < plan['required_free_bytes']:
            raise RuntimeError(f"Optimized checkpoint capacity refused: free={free}, required={plan['required_free_bytes']}; no pruning enabled.")
    if not resume and (directory / "run_metadata.json").exists():
        raise FileExistsError("Fresh run namespace required.")
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / 'progress.json'
    progress = dict(phase='conversational', run_name=directory.name, lineage=spec['initialization']['path'],
        state='STARTING', step=0, total_steps=spec['training']['max_steps'], pid=os.getpid(),
        kind='934M learned-position decoder', total_tokens_seen=0)
    atomic_document(status_path, progress)
    atomic_document(directory / 'budget.json', budget.document())
    # A visible viewer is a launch gate, never a dependency after attachment.
    viewer = start_chat_viewer(status_path, title=f'Codexa conversational: {directory.name}')
    project = ProjectConfig(ModelConfig(**spec['model']), TrainingConfig(**spec['training']))
    config_hash = identity(spec)
    storage_backend = spec['checkpoint'].get('storage_backend', 'legacy')
    store = RecoveryStore(directory / 'checkpoints', backend=storage_backend,
        store_root=Path(spec['checkpoint']['store_root']) if spec['checkpoint'].get('store_root') else None,
        reserve_bytes=spec['checkpoint'].get('reserve_bytes', 100 * 1024**3),
        reclamation_mode=spec['checkpoint'].get('reclamation_mode', 'disabled'),
        retention=plan['retention'] and {k: plan['retention'][k] for k in
            ('max_checkpoint_bytes', 'max_snapshots', 'max_bytes', 'continuous',
             'minimum_rollback_states', 'pinned_snapshots') if k in plan['retention']}
            if spec['checkpoint'].get('retention') is not None else None)
    if reclamation_authorization is not None:
        if storage_backend != 'optimized':
            raise ValueError('Reclamation approval requires optimized storage.')
        store.authorize_reclamation(reclamation_authorization)
    logger = None
    report = dict(run_name=directory.name, configuration_sha256=config_hash,
                  initialized_from=spec['initialization'], exact_resume=resume,
                  state='STARTING', checkpoints=[], budget=budget.document())
    model = optimizer = None
    state = TrainingState()
    counters = dict(context_tokens=0, padded_tokens=0, examples=0, sources={}, validation_bad=0,
                    quality_bad=0, best_quality=spec.get('baseline_quality'))
    stop_reason = None
    pending_final_best = False
    pending_final_validation = None
    try:
        if spec['optimization'].get('allocator', 'default') != os.environ.get('PYTORCH_ALLOC_CONF', 'default'):
            raise ValueError('CUDA allocator differs from the measured run configuration.')
        expected_workspace = spec['optimization'].get('cublas_workspace_config')
        if expected_workspace and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != expected_workspace:
            raise ValueError('CUBLAS workspace differs from the validated deterministic configuration.')
        torch.use_deterministic_algorithms(spec['optimization'].get('deterministic_algorithms', False))
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('This full model workload requires CUDA BF16.')
        set_deterministic_seed(project.training.seed)
        torch.backends.cuda.matmul.allow_tf32 = spec['optimization'].get('tf32', False)
        tokenizer_path = Path(spec['tokenizer']['path'])
        if file_sha256(tokenizer_path) != spec['tokenizer']['sha256']:
            raise ValueError('Tokenizer fingerprint drift.')
        tokenizer = load_tokenizer(tokenizer_path)
        if [tokenizer.token_to_id(t) for t in ['<pad>', '<bos>', '<eos>', '<unk>', '<|system|>', '<|user|>', '<|assistant|>', '<|end|>']] != list(range(8)):
            raise ValueError('Reserved token ID drift.')
        if spec['data'].get('mixture_manifest'):
            from llm_data.chat_mixture import ManifestChatDataset
            data = ManifestChatDataset(Path(spec['data']['train']), tokenizer)
        else:
            data = IndexedChatDataset(Path(spec['data']['train']), tokenizer, cache_bytes=spec['data'].get('cache_bytes', 64 * 1024**2))
        validation = IndexedChatDataset(Path(spec['data']['validation']), tokenizer)
        for name, dataset in [('train', data), ('validation', validation)]:
            if dataset.sha256 != spec['data'][name + '_sha256']:
                raise ValueError('Dataset fingerprint drift.')
        if spec['data'].get('token_probabilities'):
            from llm_data.chat_mixture import TokenQuotaSampler
            sampler = TokenQuotaSampler(data, seed=project.training.seed,
                probabilities=spec['data']['token_probabilities'], bucket_size=spec['data'].get('bucket_size',128))
        else:
            sampler = ConversationSampler(data, seed=project.training.seed,
                balanced=spec['data'].get('balanced', False), bucket_size=spec['data'].get('bucket_size', 128))
        collate = partial(collate_chat, fixed_length=2048 if spec['data'].get('fixed_padding') else None)
        workers = spec['data'].get('workers', 0)
        loader = DataLoader(data, batch_size=project.training.micro_batch_size, sampler=sampler,
            collate_fn=collate, pin_memory=True, num_workers=workers,
            generator=torch.Generator().manual_seed(project.training.seed))
        val_indices = spec['data'].get('validation_indices')
        val_data = Subset(validation, val_indices) if val_indices is not None else validation
        val_loader = DataLoader(val_data, batch_size=1, collate_fn=partial(collate_chat), pin_memory=True)
        model = LanguageModel(project.model).to('cuda')
        model.set_gradient_checkpointing(spec['optimization'].get('gradient_checkpointing', True))
        if count_parameters(model) != 934356480 or model.lm_head.weight is not model.token_embeddings.weight:
            raise ValueError('Architecture/tied weight identity drift.')
        optimizer = create_adamw_optimizer(model, learning_rate=project.training.learning_rate,
            weight_decay=project.training.weight_decay, optimizer_name=spec['optimization']['optimizer'],
            betas=tuple(spec['optimization'].get('betas', [.9, .95])),
            epsilon=spec['optimization'].get('epsilon', 1e-8))
        scheduler = SchedulerState(project.training.warmup_steps, project.training.max_steps,
                                   project.training.learning_rate, project.training.learning_rate *
                                   spec['optimization'].get('minimum_lr_ratio', .1))
        run_id = uuid.uuid4().hex
        if resume:
            restored = load_checkpoint(None if storage_backend == 'optimized' else store.resolve(),
                recovery_store=store if storage_backend == 'optimized' else None,
                allow_previous=storage_backend == 'optimized',
                model=model, optimizer=optimizer, scaler=None,
                expected_config=project, expected_data_manifest_sha256=spec['data']['manifest_sha256'])
            if storage_backend == 'optimized':
                report['recovery_selection'] = store.last_recovery_selection
            if restored.resume_context['configuration_sha256'] != config_hash:
                raise ValueError('Run configuration drift on resume.')
            state, scheduler, run_id = restored.state, restored.scheduler, restored.run_id
            counters = restored.resume_context['chat_counters']
            for name in ('train_metrics.jsonl', 'samples.jsonl'):
                path = directory / name
                if path.exists():
                    archive = path.with_name(path.stem + f'.before-resume-{uuid.uuid4().hex[:8]}.jsonl')
                    shutil.copyfile(path, archive)
                    if name == 'samples.jsonl':
                        retained = [line for line in path.read_text().splitlines()
                                    if json.loads(line)['optimizer_step'] <= state.optimizer_step]
                        path.write_text('\n'.join(retained) + ('\n' if retained else ''))
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
        precision = resolve_precision('bf16', torch.device('cuda'))
        logger = JsonlRunLogger(directory.parent, directory.name, resume=resume,
            overwrite=False, expected_run_id=run_id if resume else None,
            expected_optimizer_step=state.optimizer_step if resume else None)
        logger.write_metadata(dict(run_id=run_id, run_name=directory.name, spec=spec, report=report))

        def save(*, best: bool = False, final: bool = False, validation_loss: float | None = None) -> dict:
            context = dict(configuration_sha256=config_hash, data_manifest_sha256=spec['data']['manifest_sha256'],
                data_cursor=dict(completed_epochs=state.completed_epochs, batches_in_epoch=state.batches_in_epoch),
                microstep_within_gradient_accumulation=0, chat_counters=counters,
                budget=budget.document(), initialization=spec['initialization'], optimizer=spec['optimization']['optimizer'])
            payload = build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None, state=state,
                scheduler=scheduler, config=project, run_name=directory.name, run_id=run_id,
                tokenizer_reference=str(tokenizer_path), tokenizer_sha256=spec['tokenizer']['sha256'], resume_context=context)
            validation_best = validation_loss
            is_validation_best = validation_best is not None and (counters.get('best_saved_validation_loss') is None or validation_best < counters['best_saved_validation_loss'])
            previous_saved_best = counters.get('best_saved_validation_loss')
            if is_validation_best:
                counters['best_saved_validation_loss'] = validation_best
            options = dict(best=best, final=final)
            options['best_validation'] = is_validation_best
            published_error = None
            try:
                record = store.save(payload, **options)
            except BaseException as error:
                if getattr(error, 'checkpoint_published', False):
                    record = error.checkpoint_record
                    published_error = error
                    report['reclamation_error'] = str(error)
                else:
                    if previous_saved_best is None:
                        counters.pop('best_saved_validation_loss', None)
                    else:
                        counters['best_saved_validation_loss'] = previous_saved_best
                    raise
            report['checkpoints'].append(record)
            progress['checkpoint_path'] = str(store.store_root / record['path']) if storage_backend == 'optimized' else str(store.resolve())
            atomic_document(directory / 'budget.json', budget.document())
            if storage_backend == 'optimized' and store.last_retention_simulation is not None:
                progress['checkpoint_retention_simulation'] = store.last_retention_simulation
            if published_error is not None:
                raise published_error
            return record

        if not resume:
            save(best=True)
        baseline_details = {} if spec['evaluation'].get('source_stratified_losses') else None
        baseline_loss = evaluate(model, val_loader, device=torch.device('cuda'), precision=precision,
            max_batches=spec['evaluation']['validation_batches'], deadline_monotonic=budget.training_deadline,
            details=baseline_details)[0]
        if state.best_validation_loss is None:
            state.best_validation_loss = baseline_loss
        report['baseline_validation_loss'] = baseline_loss
        if baseline_details is not None:
            report['baseline_validation_details'] = baseline_details
        if not resume:
            from llm_training.checkpoint_retention import RetentionLimitReached
            try:
                save(validation_loss=baseline_loss)
            except RetentionLimitReached as error:
                stop_reason = 'RECLAMATION_FAILED' if getattr(error, 'checkpoint_published', False) else 'RETENTION_LIMIT_REACHED'
                pending_final_validation = baseline_loss
        previous_quality = counters.get('best_quality')
        if not resume and spec['evaluation'].get('fresh_behavioral_baseline'):
            if quality_evaluator is None or budget.remaining <= budget.reserve_seconds + spec['evaluation']['quality_reserve_seconds']:
                raise RuntimeError('Fresh behavioral baseline cannot fit; no optimizer update permitted.')
            previous_quality = quality_evaluator(model, tokenizer, budget.training_deadline)
            counters['best_quality'] = previous_quality
            report['baseline_quality'] = previous_quality
        last_validation_best = state.best_validation_loss
        last_save_step = state.optimizer_step
        last_save_time = time.monotonic()

        def committed(batches: list[object]) -> None:
            sources = Counter(counters['sources'])
            source_tokens = Counter(counters.get('source_supervised_tokens', {}))
            ids = []
            for batch in batches:
                sources.update(batch['sources'])
                ids.extend(batch['identities'])
                for index in batch['indices']:
                    entry = data.index[index]
                    source_tokens[entry.source] += entry.assistant_tokens
            counters['sources'] = dict(sources)
            counters['source_supervised_tokens'] = dict(source_tokens)
            counters['examples'] += len(ids)
            with (directory / 'samples.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(optimizer_step=state.optimizer_step, identities=ids)) + '\n')

        def update(current: TrainingState, metrics) -> None:
            nonlocal stop_reason, previous_quality, last_validation_best, last_save_step, last_save_time, pending_final_best, pending_final_validation
            counters['context_tokens'] += metrics.context_tokens_processed
            counters['padded_tokens'] += metrics.padded_tokens_processed
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if spec['optimization'].get('release_cached_segments_at_boundary'):
                torch.cuda.empty_cache()
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            if free_bytes < spec['optimization'].get('minimum_headroom_bytes', 1536 * 1024**2):
                stop_reason = 'MEMORY_HEADROOM_GATE'
            quality_improved = False
            validation_value = metrics.validation_loss
            if validation_value is not None:
                counters['validation_bad'] = (counters['validation_bad'] + 1
                    if validation_value > last_validation_best * 1.05 else 0)
                last_validation_best = min(last_validation_best, validation_value)
                if counters['validation_bad'] >= 3:
                    stop_reason = 'VALIDATION_DETERIORATION'
                if (quality_evaluator is not None and current.optimizer_step %
                        spec['evaluation'].get('quality_interval', project.training.evaluation_interval) == 0
                        and budget.remaining > budget.reserve_seconds + spec['evaluation']['quality_reserve_seconds']):
                    q = quality_evaluator(model, tokenizer, budget.training_deadline)
                    if previous_quality:
                        regression = behavioral_regression(q, previous_quality,
                            spec['evaluation'].get('regression_policy'))
                        counters['quality_bad'] = counters['quality_bad'] + 1 if regression else 0
                        quality_improved = (not regression and q['macro_task_accuracy'] >= previous_quality['macro_task_accuracy']
                            and q['END_rate'] >= previous_quality['END_rate']
                            and q['severe_degeneration_rate'] <= previous_quality['severe_degeneration_rate']
                            )
                    else:
                        quality_improved = q['entry_gate_passed']
                    if quality_improved:
                        previous_quality = q
                        counters['best_quality'] = q
                    if counters['quality_bad'] >= spec['evaluation'].get('regression_patience', 2):
                        stop_reason = 'BEHAVIORAL_DETERIORATION'
            progress.update(state='RUNNING', step=current.optimizer_step, training_loss=metrics.training_loss,
                validation_loss=validation_value if validation_value is not None else progress.get('validation_loss'),
                learning_rate=metrics.learning_rate, tokens_per_second=metrics.tokens_per_second,
                total_tokens_seen=current.tokens_seen, context_tokens=counters['context_tokens'],
                effective_epochs=counters['examples'] / len(data), peak_vram_bytes=metrics.peak_reserved_vram_bytes,
                free_gpu_bytes=free_bytes,
                remaining_seconds=budget.remaining, elapsed_seconds=budget.seconds - budget.remaining,
                eta_seconds=min(budget.remaining, metrics.step_time_seconds * (project.training.max_steps - current.optimizer_step)),
                viewer_closed=viewer.poll() is not None, next_checkpoint_step=current.optimizer_step + project.training.checkpoint_interval)
            progress['deadline_warning'] = budget.remaining < budget.reserve_seconds + 300
            progress['loss_history'] = (progress.get('loss_history', []) + [metrics.training_loss])[-30:]
            progress['speed_history'] = (progress.get('speed_history', []) + [metrics.tokens_per_second])[-30:]
            atomic_document(status_path, progress)
            atomic_document(directory / 'budget.json', budget.document())
            if quality_improved or current.optimizer_step % project.training.checkpoint_interval == 0:
                from llm_training.checkpoint_retention import RetentionLimitReached
                try:
                    save(best=quality_improved, validation_loss=validation_value)
                    last_save_step, last_save_time = current.optimizer_step, time.monotonic()
                except RetentionLimitReached as error:
                    pending_final_best = quality_improved
                    pending_final_validation = validation_value
                    stop_reason = 'RECLAMATION_FAILED' if getattr(error, 'checkpoint_published', False) else 'RETENTION_LIMIT_REACHED'

        with StopRequest() as requested:
            def record_validation_details(step, details):
                counters['latest_validation_details'] = details
                with (directory / 'validation-strata.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(optimizer_step=step,**details))+'\n')
            def stop_before_update():
                nonlocal stop_reason
                if requested() or stop_reason is not None:
                    return True
                if spec.get('assistant_token_budget') is not None and state.tokens_seen >= spec['assistant_token_budget']:
                    stop_reason = 'SUPERVISION_BUDGET'
                    return True
                if storage_backend == 'optimized' and store.policy:
                    from llm_training.checkpoint_retention import RetentionLimitReached
                    try:
                        store.prepare_update(run_id)
                    except RetentionLimitReached:
                        stop_reason = 'RETENTION_LIMIT_REACHED'
                        return True
                return False
            try:
                state, metrics = train_model(model, loader, optimizer, device=torch.device('cuda'), precision=precision,
                    max_steps=project.training.max_steps, gradient_accumulation_steps=project.training.gradient_accumulation_steps,
                    gradient_clip=project.training.gradient_clip, warmup_steps=project.training.warmup_steps,
                    peak_learning_rate=project.training.learning_rate, state=state, seed=project.training.seed,
                    minimum_learning_rate=scheduler.minimum_learning_rate,
                    run_name=directory.name, run_id=run_id, logger=logger, validation_loader=val_loader,
                    evaluation_interval=project.training.evaluation_interval, max_validation_batches=spec['evaluation']['validation_batches'],
                    deadline_monotonic=budget.training_deadline,
                    loss_normalization=spec['optimization'].get('loss_normalization', 'supervised_tokens'),
                    target_token_weights={int(key): value for key, value in
                        spec['optimization'].get('target_token_weights', {}).items()},
                    on_committed_batches=committed, on_optimizer_step=update,
                    stop_requested=stop_before_update,
                    validation_allowed=lambda: budget.remaining > budget.reserve_seconds + spec['evaluation']['validation_reserve_seconds'],
                    max_epochs=spec['data'].get('maximum_epochs', 1),
                    on_validation_details=record_validation_details if baseline_details is not None else None)
            except TimeoutError:
                stop_reason = 'TRAINING_DEADLINE'
            if requested():
                stop_reason = f'INTERRUPTED_SIGNAL_{requested.signal_number}'
            elif time.monotonic() >= budget.training_deadline and stop_reason is None:
                stop_reason = 'TRAINING_DEADLINE'
        if stop_reason == 'RECLAMATION_FAILED':
            report['final_reference'] = store.finalize_published_checkpoint(state.optimizer_step)
        elif budget.remaining > spec['checkpoint']['minimum_save_seconds']:
            save(final=True, best=pending_final_best, validation_loss=pending_final_validation)
        else:
            stop_reason = 'FINAL_SAVE_RESERVE_EXHAUSTED'
        report.update(state='INTERRUPTED' if stop_reason and stop_reason.startswith('INTERRUPTED') else
                      'EARLY_STOPPED' if stop_reason and stop_reason != 'TRAINING_DEADLINE' else 'COMPLETED',
            stop_reason=stop_reason or ('STEP_CAP' if state.optimizer_step == project.training.max_steps else 'DATA_COVERAGE_CAP'),
            training_state=state.to_dict(), counters=counters, recovery_index=store.index())
    except BaseException as error:
        # Optimizer/OOM failures can leave uncertain weights. Never save them.
        report.update(state='FAILED', error_type=type(error).__name__, error=str(error),
                      recovery_index=store.index(), recovery_required=True)
        raise
    finally:
        report['budget'] = budget.document()
        progress.update(state=report['state'], remaining_seconds=budget.remaining,
                        elapsed_seconds=budget.seconds - budget.remaining)
        atomic_document(directory / 'report.json', report)
        atomic_document(status_path, progress)
        atomic_document(directory / 'budget.json', budget.document())
        if logger is not None:
            logger.close()
        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report
