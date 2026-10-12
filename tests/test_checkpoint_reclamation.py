"""V3 reclamation is exercised only in factory-created disposable namespaces."""

import copy
import gc
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from llm_architecture.component_checkpoint import read_components, manifest_document
from llm_training.checkpoint_retention import RetentionLimitReached, RetentionPolicy
from llm_training.checkpoint_reclamation import (
    create_disposable_fixture_store, plan_reclamation, execute_reclamation,
    load_reclamation_grant,
)
from llm_training.optimized_recovery import OptimizedRecoveryStore
from llm_training.checkpointing import load_checkpoint, capture_random_states
from test_component_recovery import fixture, exact, update
from test_checkpoint_retention import advance

BOUND = 1024**2


def disposable(tmp_path, **options):
    return create_disposable_fixture_store(tmp_path, max_checkpoint_bytes=BOUND, **options)


def history(store, payload, start, stop, **flags):
    for step in range(start, stop):
        advance(payload, step)
        store.prepare_update(payload['run_id'])
        store.save(payload, **flags)


def test_long_history_six_retained_exact_roles_shared_objects(tmp_path):
    _, _, _, payload = fixture()
    store, grant = disposable(tmp_path)
    store.save(payload, best=True, boundary=True)
    initial = store.resolve()
    advance(payload, 2)
    store.save(payload, best_validation=True)
    validation = store.resolve()
    history(store, payload, 3, 66)
    advance(payload, 66)
    store.save(payload, final=True)
    assert len(list((store.store_root / 'snapshots').glob('*/*.ckpt.json'))) == 6
    assert initial.exists() and validation.exists()
    assert store.index()['previous']['optimizer_step'] == 65
    assert store.index()['final']['optimizer_step'] == 66
    plan = plan_reclamation(store)
    assert not plan['inventory']['blocked']
    assert not plan['inventory']['retirement_candidates']
    assert max(plan['inventory']['object_reference_counts'].values()) == 6
    assert plan['inventory']['physical']['accounted_bytes'] <= store.policy.max_bytes
    exact(payload, store.load('final'))
    assert grant.purpose == 'disposable_fixture'


def test_no_boolean_or_historical_store_can_authorize_deletion(tmp_path):
    _, _, _, payload = fixture()
    store = OptimizedRecoveryStore(tmp_path / 'run', store_root=tmp_path / 'store',
        reserve_bytes=0, retention=dict(continuous=True, max_checkpoint_bytes=BOUND))
    history(store, payload, 1, 6)
    with pytest.raises(RetentionLimitReached):
        store.prepare_update(payload['run_id'])
    before = sorted(store.store_root.rglob('*.pt'))
    with pytest.raises(ValueError, match='grant'):
        store.authorize_reclamation(True)
    assert sorted(store.store_root.rglob('*.pt')) == before
    bounded = OptimizedRecoveryStore(tmp_path / 'older', reserve_bytes=0,
                                    retention=dict(max_checkpoint_bytes=BOUND))
    bounded.save(payload)
    with pytest.raises(ValueError, match='historical adoption|drift'):
        OptimizedRecoveryStore(tmp_path / 'new', store_root=bounded.store_root,
            retention=dict(continuous=True, max_checkpoint_bytes=BOUND))


def test_exact_cpu_continuation_after_many_rotations(tmp_path):
    config, model, optimizer, payload = fixture()
    store, _ = disposable(tmp_path)
    for step in range(2, 12):
        update(model, optimizer)
        advance(payload, step)
        payload['random_states'] = capture_random_states()
        store.save(payload)
    loaded = store.load()
    exact(copy.deepcopy(payload), loaded)
    expected_loss = update(model, optimizer)
    weights, opt = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    load_checkpoint(None, model=model, optimizer=optimizer, scaler=None,
                    expected_config=config, recovery_store=store)
    exact(expected_loss, update(model, optimizer))
    exact(weights, model.state_dict())
    exact(opt, optimizer.state_dict())
    assert loaded['scheduler_state'] == payload['scheduler_state']
    assert loaded['resume_context'] == payload['resume_context']


def test_active_storage_protection_and_release(tmp_path):
    _, _, _, payload = fixture()
    store, _ = disposable(tmp_path)
    store.save(payload)
    old = store.resolve()
    data = store.load()
    tensor = next(iter(data['model_state_dict'].values())).detach()
    del data
    gc.collect()
    history(store, payload, 2, 12)
    assert old.exists()
    assert str(old.relative_to(store.store_root)) in plan_reclamation(store)['inventory']['active_readers']
    del tensor
    gc.collect()
    history(store, payload, 12, 15)
    assert not old.exists()


def test_grant_scope_and_external_loading(tmp_path):
    _, _, _, payload = fixture()
    a, grant = disposable(tmp_path)
    b, _ = disposable(tmp_path)
    a.save(payload)
    with pytest.raises(ValueError, match='scope'):
        b.authorize_reclamation(grant)
    document = Path(grant.namespace) / 'reclamation-authorization.json'
    assert load_reclamation_grant(a, document) == grant
    doc = json.loads(document.read_text())
    doc['policy_sha256'] = '0' * 64
    document.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='bound'):
        load_reclamation_grant(a, document)


def pending_plan(tmp_path, monkeypatch):
    """Leave one complete incoming publication pending rotation, only in a fixture."""
    import llm_training.checkpoint_reclamation as reclaim
    _, _, _, payload = fixture()
    store, grant = disposable(tmp_path)
    history(store, payload, 1, 7)
    original = reclaim._execute_locked
    def fail(*args, **kwargs):
        raise OSError('injected reclamation before intent')
    monkeypatch.setattr(reclaim, '_execute_locked', fail)
    advance(payload, 7)
    with pytest.raises(RetentionLimitReached) as error:
        store.save(payload)
    assert error.value.checkpoint_published is True
    assert store.load()['training_state']['optimizer_step'] == 7
    monkeypatch.setattr(reclaim, '_execute_locked', original)
    return store, grant, plan_reclamation(store)


def test_stale_and_mutated_plan_rejected(tmp_path, monkeypatch):
    store, grant, plan = pending_plan(tmp_path, monkeypatch)
    corrupted = copy.deepcopy(plan)
    corrupted['inventory']['retirement_candidates'] = []
    with pytest.raises(ValueError, match='Invalid'):
        execute_reclamation(store, corrupted, grant)
    index = json.loads(store.index_path.read_text())
    index['best_behavioral'] = index['latest']
    store.index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match='Stale'):
        execute_reclamation(store, plan, grant)
    assert store.load()['training_state']['optimizer_step'] == 7


@pytest.mark.parametrize('failure', ['unlink', 'sync', 'status'])
def test_interrupted_reclamation_replay_idempotent(tmp_path, monkeypatch, failure):
    import llm_training.checkpoint_reclamation as reclaim
    store, grant, plan = pending_plan(tmp_path, monkeypatch)
    original_remove, original_sync, original_json = reclaim._remove, reclaim.sync_directory, reclaim.atomic_json
    calls = []
    def interrupted(root, relative, identity):
        original_remove(root, relative, identity)
        if not calls:
            calls.append(relative)
            raise OSError('injected crash after unlink')
    def sync_failure(path):
        if path.name == 'isolated' and not calls:
            calls.append(str(path))
            raise OSError('injected fsync failure')
        return original_sync(path)
    def status_failure(path, value, **kwargs):
        if path.name.endswith('.status.json') and value.get('phase') == 'manifests-retired' and not calls:
            calls.append(str(path))
            raise OSError('injected status write failure')
        return original_json(path, value, **kwargs)
    monkeypatch.setattr(reclaim, '_remove', interrupted if failure == 'unlink' else original_remove)
    monkeypatch.setattr(reclaim, 'sync_directory', sync_failure if failure == 'sync' else original_sync)
    monkeypatch.setattr(reclaim, 'atomic_json', status_failure if failure == 'status' else original_json)
    with pytest.raises(OSError):
        execute_reclamation(store, plan, grant)
    assert store.load()['training_state']['optimizer_step'] == 7
    assert store.load('previous')['training_state']['optimizer_step'] == 6
    monkeypatch.setattr(reclaim, '_remove', original_remove)
    monkeypatch.setattr(reclaim, 'sync_directory', original_sync)
    monkeypatch.setattr(reclaim, 'atomic_json', original_json)
    result = execute_reclamation(store, plan, grant)
    assert result['phase'] == 'complete'
    assert execute_reclamation(store, plan, grant)['already_complete']
    assert len(list((store.store_root / 'snapshots').glob('*/*.ckpt.json'))) == 6


@pytest.mark.parametrize('problem', ['orphan', 'unknown', 'corrupt', 'missing_index'])
def test_uncertain_dependencies_stop_without_cleanup(tmp_path, problem):
    _, _, _, payload = fixture()
    store, grant = disposable(tmp_path)
    store.save(payload)
    if problem == 'orphan':
        target = store.store_root / 'objects' / '00' / ('0' * 64 + '.pt')
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b'unknown fixture object')
    elif problem == 'unknown':
        (store.store_root / 'unregistered.bin').write_bytes(b'unknown')
    elif problem == 'corrupt':
        doc, root = manifest_document(store.resolve())
        (root / doc['components']['optimizer']['path']).write_bytes(b'corrupt')
    else:
        store.index_path.unlink()
    before = {str(p): p.stat().st_size for p in store.store_root.rglob('*') if p.is_file()}
    plan = plan_reclamation(store)
    assert plan['inventory']['blocked']
    with pytest.raises(ValueError):
        execute_reclamation(store, plan, grant)
    after = {str(p): p.stat().st_size for p in store.store_root.rglob('*') if p.is_file()}
    assert before == after


def test_shared_components_multiple_registered_runs(tmp_path):
    _, _, _, payload = fixture()
    a, grant = disposable(tmp_path, max_snapshots=12)
    a.save(payload, best=True)
    b = OptimizedRecoveryStore(Path(grant.namespace) / 'b', store_root=a.store_root,
                               reserve_bytes=0)
    b.authorize_reclamation(grant)
    other = copy.deepcopy(payload)
    other['run_id'] = 'second'
    b.save(other, best=True)
    shared = manifest_document(a.resolve())[0]['components']['model']['path']
    for step in range(2, 18):
        advance(payload, step)
        a.save(payload)
        advance(other, step)
        next(iter(other['optimizer_state_dict']['state'].values()))['exp_avg'].add_(1)
        b.save(other)
    assert (a.store_root / shared).exists()
    assert len(list((a.store_root / 'snapshots').glob('*/*.ckpt.json'))) == 12
    assert not plan_reclamation(a)['inventory']['blocked']
    assert b.load()['training_state']['optimizer_step'] == 17


def test_reader_writer_and_reclaimer_threads(tmp_path):
    _, _, _, payload = fixture()
    store, _ = disposable(tmp_path)
    store.save(payload)
    def reader():
        for _ in range(20):
            loaded = store.load()
            assert loaded['run_id'] == payload['run_id']
            del loaded
    def writer():
        history(store, payload, 2, 22)
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(reader), pool.submit(reader), pool.submit(writer)]:
            future.result(timeout=30)
    assert store.load()['training_state']['optimizer_step'] == 21


def test_rollback_depth_and_protected_quota_conflict(tmp_path):
    _, _, _, payload = fixture()
    store, _ = disposable(tmp_path, minimum_rollback_states=2)
    history(store, payload, 1, 14)
    assert store.index()['rollback-2']['optimizer_step'] == 11
    assert store.load('rollback-2')['training_state']['optimizer_step'] == 11
    index = store.index()
    # Pin every surviving synthetic snapshot, including ones without current roles.
    for path in (store.store_root / 'snapshots').glob('*/*.ckpt.json'):
        doc, _ = manifest_document(path)
        step = doc['training_state']['optimizer_step']
        index[f'milestone-{step}'] = dict(index['latest'],
            path=str(path.relative_to(store.store_root)), sha256=path.name[:-10], optimizer_step=step)
    store.index_path.write_text(json.dumps(index))
    # Six fixed pins plus another state cannot fit; never evict protected roots.
    with pytest.raises(RetentionLimitReached):
        store.prepare_update(payload['run_id'])


def test_v2_policy_document_identity_unchanged():
    policy = RetentionPolicy(BOUND)
    assert set(policy.document()) == {'format', 'max_checkpoint_bytes', 'max_snapshots',
                                      'max_bytes', 'garbage_collection_enabled'}
    assert RetentionPolicy(BOUND, continuous=True).max_bytes == 10 * BOUND + 8 * 1024**2


def _lease_process(snapshot, ready, release):
    data = read_components(Path(snapshot))
    tensor = next(iter(data['model_state_dict'].values())).detach()
    del data
    gc.collect()
    ready.set()
    release.wait(20)
    assert torch.isfinite(tensor).all()


def _crash_executor(directory, root, plan, grant):
    import llm_training.checkpoint_reclamation as reclaim
    store = OptimizedRecoveryStore(Path(directory), store_root=Path(root), reserve_bytes=0)
    original = reclaim._remove
    def crash(*args):
        original(*args)
        os._exit(77)
    reclaim._remove = crash
    execute_reclamation(store, plan, grant)


def test_process_reader_race_and_termination_release(tmp_path, monkeypatch):
    store, grant, plan = pending_plan(tmp_path, monkeypatch)
    candidate = store.store_root / plan['inventory']['retirement_candidates'][0]
    context = multiprocessing.get_context('spawn')
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_lease_process, args=(str(candidate), ready, release))
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(ValueError, match='Stale|protected|reader'):
            execute_reclamation(store, plan, grant)
        assert candidate.exists()
    finally:
        process.terminate()
        process.join(15)
    assert process.exitcode is not None
    result = execute_reclamation(store, plan_reclamation(store), grant)
    assert result['phase'] == 'complete'


def test_actual_process_crash_replays_durable_intent(tmp_path, monkeypatch):
    store, grant, plan = pending_plan(tmp_path, monkeypatch)
    process = multiprocessing.get_context('spawn').Process(target=_crash_executor,
        args=(str(store.directory), str(store.store_root), plan, grant))
    process.start()
    process.join(20)
    if process.is_alive():
        process.terminate()
        process.join(5)
        pytest.fail('Crash-injection process timed out.')
    assert process.exitcode == 77
    assert store.load()['training_state']['optimizer_step'] == 7
    assert execute_reclamation(store, plan, grant)['phase'] == 'complete'


def test_missing_admission_gate_never_falls_back_to_unmanaged_reader(tmp_path):
    _, _, _, payload = fixture()
    store, _ = disposable(tmp_path)
    store.save(payload)
    (store.store_root / '.reader-admission.lock').unlink()
    with pytest.raises(FileNotFoundError):
        store.load()


def test_explicit_cross_run_dependency_and_pin_identity(tmp_path):
    _, _, _, payload = fixture()
    a, grant = disposable(tmp_path,max_snapshots=12)
    a.save(payload)
    original=a.index()['latest']['path']
    b=OptimizedRecoveryStore(Path(grant.namespace)/'b',store_root=a.store_root,reserve_bytes=0)
    b.authorize_reclamation(grant)
    other=copy.deepcopy(payload)
    other['run_id']='second'
    b.save(other)
    b.pin_snapshot(original,label='initializer-for-second')
    history(a,payload,2,19)
    history(b,other,2,19)
    assert (a.store_root/original).exists()
    assert original in plan_reclamation(a)['inventory']['dependency_roots']
    with pytest.raises(ValueError,match='reassigned'):
        b.pin_snapshot(b.index()['latest']['path'],label='initializer-for-second')
    # A foreign path is never adopted or written.
    with pytest.raises(ValueError,match='store-relative'):
        b.pin_snapshot('../outside.ckpt.json',label='bad')


def test_numpy_view_keeps_storage_lease(tmp_path):
    _, _, _, payload=fixture()
    store,_=disposable(tmp_path)
    store.save(payload)
    path=store.index()['latest']['path']
    value=store.load()
    array=next(iter(value['model_state_dict'].values())).numpy()
    del value
    gc.collect()
    assert path in plan_reclamation(store)['inventory']['active_readers']
    del array
    gc.collect()
    assert not plan_reclamation(store)['inventory']['active_readers']


def test_simulation_and_finalize_published_without_reclamation(tmp_path,monkeypatch):
    store,grant,plan=pending_plan(tmp_path,monkeypatch)
    count=len(list((store.store_root/'snapshots').glob('*/*.ckpt.json')))
    record=store.finalize_published_checkpoint(7)
    assert record['optimizer_step']==7 and store.index()['final']==record
    assert len(list((store.store_root/'snapshots').glob('*/*.ckpt.json')))==count
    with pytest.raises(ValueError,match='completed optimizer'):
        store.finalize_published_checkpoint(999)
    with pytest.raises(ValueError,match='separate approval'):
        OptimizedRecoveryStore(tmp_path/'bad',reclamation_mode='execute')


def test_uint8_optimizer_tensors_preserved_without_precision_conversion(tmp_path):
    _,_,_,payload=fixture()
    state=next(iter(payload['optimizer_state_dict']['state'].values()))
    state['__bnb_optimizer_quant_state__']={'state1':torch.arange(256,dtype=torch.uint8),
        'state2':torch.arange(256,dtype=torch.uint8).flip(0),'absmax':torch.randn(17)}
    store,_=disposable(tmp_path)
    store.save(payload)
    exact(copy.deepcopy(payload),store.load())


def test_interrupted_reclamation_can_resume_after_safe_final_alias(tmp_path,monkeypatch):
    import llm_training.checkpoint_reclamation as reclaim
    store,grant,plan=pending_plan(tmp_path,monkeypatch)
    original=reclaim._remove
    called=[]
    def crash(*args):
        original(*args)
        if not called:
            called.append(True)
            raise OSError('fixture crash')
    monkeypatch.setattr(reclaim,'_remove',crash)
    with pytest.raises(OSError):
        execute_reclamation(store,plan,grant)
    monkeypatch.setattr(reclaim,'_remove',original)
    store.finalize_published_checkpoint(7)
    assert execute_reclamation(store,plan,grant)['phase']=='complete'
    assert store.load('final')['training_state']['optimizer_step']==7


def test_owned_partial_status_stage_retained_and_intent_replays(tmp_path,monkeypatch):
    import llm_training.checkpoint_reclamation as reclaim
    store,grant,plan=pending_plan(tmp_path,monkeypatch)
    original=reclaim._remove
    def crash(*args):
        original(*args)
        raise OSError('fixture crash after intent')
    monkeypatch.setattr(reclaim,'_remove',crash)
    with pytest.raises(OSError):execute_reclamation(store,plan,grant)
    monkeypatch.setattr(reclaim,'_remove',original)
    stage=store.store_root/'reclamation'/('.'+plan['plan_id']+'.status-partial')
    stage.write_bytes(b'{partial control write')
    result=execute_reclamation(store,plan,grant)
    assert result['phase']=='complete' and stage.exists()
    assert str(stage.relative_to(store.store_root)) in result['retained_control_stages']


@pytest.mark.parametrize('mutation',['invalid','missing_protocol','missing_policy','unknown_protocol'])
def test_corrupt_store_protocol_never_returns_unleased_payload(tmp_path,mutation):
    _,_,_,payload=fixture()
    store,_=disposable(tmp_path)
    store.save(payload)
    marker=store.store_root/'store.json'
    value=json.loads(marker.read_text())
    if mutation=='invalid':value=[]
    elif mutation=='missing_protocol':value.pop('lifecycle')
    elif mutation=='missing_policy':value.pop('retention_policy_sha256')
    else:value['lifecycle']='unknown-protocol'
    marker.write_text(json.dumps(value))
    with pytest.raises(ValueError):store.load()


def test_completed_journal_cache_revalidates_in_place_corruption(tmp_path):
    from llm_training.checkpoint_reclamation import finish_pending_locked
    _, _, _, payload = fixture()
    store, grant = disposable(tmp_path)
    history(store, payload, 1, 9)
    with store._writer():
        finish_pending_locked(store, grant)
    status = next((store.store_root/'reclamation').glob('*.status.json'))
    status.write_text('{}')
    with store._writer(), pytest.raises(ValueError, match='completion status'):
        finish_pending_locked(store, grant)
    assert store.resolve().exists()


def test_metadata_walk_rejects_symlinks_and_detects_new_unknown_files(tmp_path):
    from llm_architecture.component_checkpoint import store_usage
    _, _, _, payload = fixture()
    store, _ = disposable(tmp_path)
    history(store, payload, 1, 9)
    foreign = tmp_path/'foreign'
    foreign.write_text('protected')
    link = store.store_root/'reclamation'/'foreign'
    link.symlink_to(foreign)
    with pytest.raises(ValueError, match='symlink'):
        store_usage(store.store_root)
    with pytest.raises(ValueError, match='symlink'):
        plan_reclamation(store)
    assert foreign.read_text() == 'protected'


def test_component_roundtrip_signed_zero_and_nonfinite_rejection(tmp_path):
    _, _, _, payload = fixture()
    value = next(iter(payload['model_state_dict'].values()))
    raw = value.reshape(-1).view(torch.int32)
    # Signed zero and two finite subnormal bit patterns exercise byte preservation.
    raw[:3] = torch.tensor([-2147483648, 1, 2], dtype=torch.int32)
    store, _ = disposable(tmp_path)
    store.save(payload)
    loaded = store.load()
    for key, expected in payload['model_state_dict'].items():
        actual = loaded['model_state_dict'][key]
        assert expected.dtype == actual.dtype
        assert torch.equal(expected.reshape(-1).view(torch.uint8),
                           actual.reshape(-1).view(torch.uint8))

    raw[1] = 2143289345
    with pytest.raises(ValueError, match='Nonfinite checkpoint tensor'):
        store.save(payload)
    assert store.load()['model_state_dict'][next(iter(payload['model_state_dict']))].reshape(-1).view(torch.int32)[1].item() == 1
