"""Fresh-store bounded admission, reader lifetimes and dependency correctness."""

import copy
import gc
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from llm_architecture.component_checkpoint import manifest_document, read_components
from llm_training.checkpoint_retention import (
    RetentionPolicy, RetentionLimitReached, simulate_retention,
)
from llm_training.recovery import RecoveryStore
from llm_training.checkpointing import load_checkpoint
from llm_training.preflight import estimate_bounded_component_storage
from test_component_recovery import fixture, exact, update

BOUND = 1024**2


def bounded(tmp_path, name='run', **options):
    return RecoveryStore(tmp_path / name, backend='optimized',
        store_root=tmp_path / 'group', reserve_bytes=0,
        retention=dict(max_checkpoint_bytes=BOUND, **options))


def advance(payload, step):
    payload['training_state']['optimizer_step'] = step
    payload['training_state']['micro_step'] = step * 2
    payload['resume_context']['budget']['elapsed'] = step


def test_six_snapshots_final_reservation_and_no_seventh(tmp_path):
    _, _, _, payload = fixture()
    store = bounded(tmp_path)
    for step in range(1, 6):
        advance(payload, step)
        store.prepare_update(payload['run_id'])
        store.save(payload, best=step == 1, best_validation=step == 2, milestone=step == 3)
    with pytest.raises(RetentionLimitReached):
        store.prepare_update(payload['run_id'])
    advance(payload, 6)
    store.save(payload, final=True, boundary=True)
    count = len(list((store.store_root / 'snapshots').glob('*/*.ckpt.json')))
    with pytest.raises(RetentionLimitReached):
        store.save(payload)
    assert count == 6 and len(list((store.store_root / 'snapshots').glob('*/*.ckpt.json'))) == count
    assert store.index()['final']['optimizer_step'] == 6
    store.reference('best_validation', 'final')
    assert len(list((store.store_root / 'snapshots').glob('*/*.ckpt.json'))) == 6
    plan = store.plan_retention()
    assert not plan['blocked'] and not plan['executable']
    assert all(n >= 1 for n in plan['object_reference_counts'].values())
    assert plan['physical']['accounted_bytes'] <= store.policy.max_bytes
    assert plan['reservations'][payload['run_id']]['reserved'] is False


def test_group_reference_counts_and_two_final_reservations(tmp_path):
    _, _, _, payload = fixture()
    a = bounded(tmp_path, 'a')
    a.save(payload, best=True)
    b = bounded(tmp_path, 'b')
    other = copy.deepcopy(payload)
    other['run_id'] = 'second'
    b.save(other, best=True)
    am, bm = [manifest_document(s.resolve())[0] for s in (a, b)]
    assert am['components']['model'] == bm['components']['model']
    assert am['components']['optimizer'] == bm['components']['optimizer']
    plan = a.plan_retention()
    shared = am['components']['model']['path']
    assert plan['object_reference_counts'][shared] == 2
    advance(payload, 2)
    a.save(payload)
    advance(other, 2)
    b.save(other)
    with pytest.raises(RetentionLimitReached):
        a.prepare_update(payload['run_id'])
    payload['resume_context']['budget']['elapsed'] += 1
    other['resume_context']['budget']['elapsed'] += 1
    a.save(payload, final=True)
    b.save(other, final=True)
    assert len(list((a.store_root / 'snapshots').glob('*/*.ckpt.json'))) == 6


def test_mapped_tensor_and_detached_view_keep_reader_lease(tmp_path):
    _, _, _, payload = fixture()
    store = bounded(tmp_path)
    store.save(payload)
    data = store.load()
    path = store.index()['latest']['path']
    tensor = next(iter(data['model_state_dict'].values())).detach()
    del data
    gc.collect()
    assert path in store.plan_retention()['active_readers']
    del tensor
    gc.collect()
    assert not store.plan_retention()['active_readers']


def test_previous_fallback_explicit_no_index_rewrite(tmp_path):
    _, _, _, payload = fixture()
    store = bounded(tmp_path)
    store.save(payload)
    first = store.resolve()
    advance(payload, 2)
    store.save(payload)
    latest = store.resolve()
    raw = store.index_path.read_bytes()
    # Only a new synthetic manifest is corrupted.
    latest.write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError):
        store.load()
    with pytest.warns(RuntimeWarning, match='previous'):
        restored = store.load(allow_previous=True)
    assert restored['training_state']['optimizer_step'] == 1
    assert store.last_recovery_selection['fallback']
    assert store.index_path.read_bytes() == raw
    assert first != latest


def test_exact_continuation_and_managed_store_load(tmp_path):
    config, model, optimizer, payload = fixture()
    store = bounded(tmp_path)
    store.save(payload)
    expected = copy.deepcopy(payload)
    loaded = store.load()
    exact(expected, loaded)
    a = update(model, optimizer)
    expected_model, expected_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    info = load_checkpoint(None, recovery_store=store, model=model, optimizer=optimizer,
                           scaler=None, expected_config=config)
    b = update(model, optimizer)
    exact(a, b)
    exact(expected_model, model.state_dict())
    exact(expected_optimizer, optimizer.state_dict())
    assert info.state.optimizer_step == 1


def test_policy_binding_no_adoption_and_orphan_byte_accounting(tmp_path):
    _, _, _, payload = fixture()
    legacy = RecoveryStore(tmp_path / 'old', backend='optimized', reserve_bytes=0)
    legacy.save(payload)
    with pytest.raises(ValueError, match='historical adoption'):
        RecoveryStore(tmp_path / 'new', backend='optimized', store_root=legacy.store_root,
            reserve_bytes=0, retention=dict(max_checkpoint_bytes=BOUND))
    store = bounded(tmp_path)
    store.save(payload)
    (store.store_root / 'orphan.fixture').write_bytes(b'x' * (7 * BOUND))
    with pytest.raises(RetentionLimitReached):
        store.prepare_update(payload['run_id'])
    marker = store.store_root / 'retention-policy.json'
    marker.write_text('{}')
    with pytest.raises(ValueError, match='policy identity'):
        read_components(store.resolve())


def test_corrupt_registry_blocks_plan_and_protects_all(tmp_path):
    _, _, _, payload = fixture()
    store = bounded(tmp_path)
    store.save(payload)
    raw = store.index_path.read_bytes()
    store.index_path.write_text('{}')
    # Missing all role references makes ownership uncertain rather than removable.
    registry = store.store_root / 'retention-runs.json'
    doc = json.loads(registry.read_text())
    doc['unknown'] = dict(index='missing.json', reserved=True)
    registry.write_text(json.dumps(doc))
    plan = store.plan_retention()
    assert plan['blocked'] and plan['uncertainties']
    assert not plan['retirement_candidates']
    assert not plan['executable']
    store.index_path.write_bytes(raw)


def test_pure_simulation_shared_objects_and_pins_over_cap():
    graph = {str(i): dict(order=i, objects=['shared', str(i)]) for i in range(10)}
    result = simulate_retention(graph, {'0', '1'}, max_snapshots=6)
    assert result['retained'] == ['0', '1', '6', '7', '8', '9']
    assert result['object_reference_counts']['shared'] == 6
    assert 'shared' not in result['hypothetical_unreferenced_objects']
    assert simulate_retention(graph, {str(i) for i in range(7)}, max_snapshots=6)['blocked']
    with pytest.raises(ValueError):
        simulate_retention(graph, {'missing'}, max_snapshots=6)


def test_capacity_estimation_and_invalid_limits():
    result = estimate_bounded_component_storage(model_bytes=100, optimizer_bytes=200,
                                                retention={'max_snapshots': 6})
    assert result['required_free_bytes'] >= 180 * 1024**3
    assert result['retained_snapshot_limit'] == 6
    assert not result['garbage_collection_enabled']
    with pytest.raises(ValueError):
        RetentionPolicy(100, max_snapshots=1)
    with pytest.raises(ValueError):
        RetentionPolicy(100, max_bytes=1)


def test_concurrent_writer_admission_preserves_quota(tmp_path):
    _, _, _, payload = fixture()
    a, b = bounded(tmp_path, 'a'), bounded(tmp_path, 'b')
    other = copy.deepcopy(payload)
    other['run_id'] = 'second'
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(a.save, payload), pool.submit(b.save, other)]:
            future.result()
    assert not a.plan_retention()['blocked']


def test_training_stops_before_next_optimizer_update_and_finalizes(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset
    from llm_training.training import train_model, TrainingState, resolve_precision
    from llm_training.checkpointing import build_checkpoint_payload, SchedulerState
    config, model, optimizer, payload = fixture()
    store = bounded(tmp_path, max_snapshots=4)
    payload['scheduler_state'] = SchedulerState(0, 100, .01, .001).to_dict()
    store.save(payload)
    state = TrainingState(**payload['training_state'])
    loader = DataLoader(TensorDataset(torch.ones(8, 4, dtype=torch.long),
                                     torch.ones(8, 4, dtype=torch.long)), batch_size=1)
    stopped = []
    def should_stop():
        try:
            store.prepare_update(payload['run_id'])
            return False
        except RetentionLimitReached:
            stopped.append('RETENTION_LIMIT_REACHED')
            return True
    def save(current, metrics, final=False):
        data = build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None,
            state=current, scheduler=SchedulerState(0, 100, .01, .001), config=config,
            run_name=payload['run_name'], run_id=payload['run_id'],
            tokenizer_reference=None, tokenizer_sha256=payload['tokenizer_sha256'],
            resume_context=payload['resume_context'])
        store.save(data, final=final)
    state, metrics = train_model(model, loader, optimizer, device=torch.device('cpu'),
        precision=resolve_precision('fp32', torch.device('cpu')), max_steps=100,
        gradient_accumulation_steps=2, gradient_clip=1., warmup_steps=0,
        peak_learning_rate=.01, state=state, seed=42, on_optimizer_step=save,
        stop_requested=should_stop, progress=False)
    assert stopped == ['RETENTION_LIMIT_REACHED']
    assert state.optimizer_step == 3 and state.micro_step == 6
    assert len(metrics) == 2
    save(state, None, final=True)
    assert store.load('final')['training_state'] == state.to_dict()
    assert len(list((store.store_root / 'snapshots').glob('*/*.ckpt.json'))) == 4


def test_failed_serialization_and_missing_policy_keep_recovery(tmp_path, monkeypatch):
    import llm_architecture.component_checkpoint as cc
    _, _, _, payload = fixture()
    store = bounded(tmp_path)
    store.save(payload)
    index = store.index_path.read_bytes()
    original = cc.torch.save
    def partial(value, writer):
        writer.write(b'partial fixture')
        raise OSError('injected interrupted bounded save')
    monkeypatch.setattr(cc.torch, 'save', partial)
    with pytest.raises(OSError):
        store.save(payload)
    assert store.index_path.read_bytes() == index
    assert store.load()['training_state'] == payload['training_state']
    assert store.plan_retention()['reservations'][payload['run_id']]['reserved']
    monkeypatch.setattr(cc.torch, 'save', original)
    policy = store.store_root / 'retention-policy.json'
    # Simulate missing metadata without deleting even the synthetic policy file.
    original_exists = Path.exists
    original_bytes = Path.read_bytes
    monkeypatch.setattr(Path, 'exists', lambda path: False if path == policy else original_exists(path))
    def missing_bytes(path):
        if path == policy:
            raise FileNotFoundError('simulated missing policy')
        return original_bytes(path)
    monkeypatch.setattr(Path, 'read_bytes', missing_bytes)
    with pytest.raises((ValueError, FileNotFoundError)):
        store.load()
    with pytest.raises(ValueError, match='policy is missing'):
        RecoveryStore(store.directory, backend='optimized', reserve_bytes=0)


def test_bounded_aliases_and_registered_indexes_count_toward_quota(tmp_path):
    from llm_architecture.component_checkpoint import store_usage
    _, _, _, payload = fixture()
    store = bounded(tmp_path)
    store.save(payload)
    before = store_usage(store.store_root)['accounted_bytes']
    with pytest.raises(ValueError, match='match the snapshot step'):
        store.reference('milestone-999')
    store.reference('milestone-1')
    after = store_usage(store.store_root)['accounted_bytes']
    assert after >= before
    assert store.index_path.stat().st_size > 0
