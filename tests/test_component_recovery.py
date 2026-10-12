"""Exact continuation and fault isolation for the opt-in component backend."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from llm_architecture.component_checkpoint import read_components, manifest_document
from llm_architecture.checkpointing import read_native_checkpoint
from llm_architecture.model import LanguageModel, ModelConfig
from llm_training.checkpointing import (
    build_checkpoint_payload, SchedulerState, load_checkpoint, CheckpointManager,
    capture_random_states, restore_random_states,
)
from llm_training.config import ProjectConfig, TrainingConfig
from llm_training.preflight import estimate_component_storage
from llm_training.recovery import RecoveryStore
from llm_training.training import TrainingState


def fixture():
    config = ProjectConfig(ModelConfig(vocab_size=16, context_length=4, num_layers=1,
        hidden_size=8, num_heads=2, intermediate_size=16, dropout=.1),
        TrainingConfig(1, 2, .01, 0., 0, 1, 1000, 'fp32', 1, 1, 42))
    torch.manual_seed(42)
    model = LanguageModel(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    update(model, optimizer)
    payload = build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None,
        state=TrainingState(optimizer_step=1, micro_step=2, tokens_seen=4),
        scheduler=SchedulerState(0, 10, .01, .001), config=config,
        run_name='isolated', run_id='isolated', tokenizer_reference=None,
        tokenizer_sha256='tokenizer', resume_context=dict(configuration_sha256='same',
            microstep_within_gradient_accumulation=0, data_cursor={'batch': 2},
            sampler={'seed': 42, 'epoch': 0}, budget={'elapsed': 10, 'remaining': 90}))
    return config, model, optimizer, payload


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    # Dropout and global randomness exercise restoration rather than just output equality.
    inputs = torch.randint(0, 16, (1, 4))
    _, loss = model(inputs, inputs)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach().clone()


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and a.stride() == b.stride()
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def store(tmp_path, name='run'):
    return RecoveryStore(tmp_path / name, backend='optimized',
                         store_root=tmp_path / 'group-store', reserve_bytes=0)


def test_roundtrip_aliases_precision_and_bitwise_continuation(tmp_path):
    config, model, optimizer, payload = fixture()
    payload['random_states'] = capture_random_states()
    frozen = copy.deepcopy(payload)
    optimized = store(tmp_path)
    optimized.save(payload, best=True, best_validation=True, final=True)
    loaded = read_native_checkpoint(optimized.resolve())
    exact(frozen, loaded)
    assert loaded['model_state_dict']['lm_head.weight'].untyped_storage()._cdata == loaded['model_state_dict']['token_embeddings.weight'].untyped_storage()._cdata
    draws = (random.random(), np.random.random(), torch.rand(3))
    loss = update(model, optimizer)
    expected_model, expected_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
    restored_model = LanguageModel(config.model)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=.01)
    restored = load_checkpoint(optimized.resolve(), model=restored_model,
        optimizer=restored_optimizer, scaler=None, expected_config=config)
    assert restored.resume_context['budget'] == frozen['resume_context']['budget']
    exact(draws, (random.random(), np.random.random(), torch.rand(3)))
    exact(loss, update(restored_model, restored_optimizer))
    exact(expected_model, restored_model.state_dict())
    exact(expected_optimizer, restored_optimizer.state_dict())


def test_legacy_and_optimized_continuation_agree(tmp_path):
    config, model, optimizer, payload = fixture()
    legacy = RecoveryStore(tmp_path / 'legacy')
    optimized = store(tmp_path)
    legacy.save(payload, best=True)
    optimized.save(payload, best=True)
    results = []
    for path in [legacy.resolve(), optimized.resolve()]:
        m = LanguageModel(config.model)
        o = torch.optim.AdamW(m.parameters(), lr=.01)
        load_checkpoint(path, model=m, optimizer=o, scaler=None)
        losses = [update(m, o) for _ in range(3)]
        results.append((losses, copy.deepcopy(m.state_dict()), copy.deepcopy(o.state_dict()), capture_random_states()))
    exact(*results)


def test_metadata_changes_reuse_objects_and_keep_separate_identity(tmp_path):
    _, _, _, payload = fixture()
    s = store(tmp_path)
    s.save(payload, best=True)
    first = s.resolve()
    before = manifest_document(first)[0]
    payload['resume_context']['budget']['elapsed'] += 1
    s.save(payload, final=True)
    second = s.resolve()
    after = manifest_document(second)[0]
    assert first != second and s.resolve('previous') == first
    for key in ['model', 'optimizer']:
        assert before['components'][key] == after['components'][key]
    assert before['components']['recovery'] != after['components']['recovery']
    assert s.resolve('final') == second and s.resolve('best') == first
    assert s.retention_inventory()['garbage_collection_enabled'] is False


def test_group_sharing_optimizer_changes_and_rotation(tmp_path):
    _, _, _, payload = fixture()
    a, b = store(tmp_path, 'a'), store(tmp_path, 'b')
    a.save(payload, best=True)
    other = copy.deepcopy(payload)
    other['run_id'] = 'another'
    b.save(other, best=True)
    am, bm = [manifest_document(s.resolve())[0] for s in [a, b]]
    assert am['components']['model'] == bm['components']['model']
    initial = a.resolve()
    for step in range(2, 10):
        payload['training_state']['optimizer_step'] = step
        a.save(payload, milestone=step == 5, boundary=step == 9)
    assert a.resolve('best') == initial and a.index()['previous']['optimizer_step'] == 8
    assert a.index()['milestone-5']['optimizer_step'] == 5
    before = manifest_document(a.resolve())[0]
    next(iter(payload['optimizer_state_dict']['state'].values()))['exp_avg'].add_(1)
    a.save(payload)
    after = manifest_document(a.resolve())[0]
    assert before['components']['model'] == after['components']['model']
    assert before['components']['optimizer'] != after['components']['optimizer']


@pytest.mark.parametrize('failure', ['object', 'manifest', 'index'])
def test_failed_publication_preserves_last_verified_state(tmp_path, monkeypatch, failure):
    import llm_architecture.component_checkpoint as cc
    import llm_training.optimized_recovery as recovery
    _, _, _, payload = fixture()
    s = store(tmp_path)
    s.save(payload, best=True)
    previous = s.resolve()
    raw = s.index_path.read_bytes()
    payload['resume_context']['budget']['elapsed'] += 1
    def fail(*args, **kwargs):
        raise OSError('injected interrupted/disk-full write')
    if failure == 'object':
        monkeypatch.setattr(cc.ComponentStore, '_object', fail)
    elif failure == 'manifest':
        original = cc.atomic_json
        def guarded(path, *args, **kwargs):
            if path.name.endswith('.ckpt.json'):
                fail()
            return original(path, *args, **kwargs)
        monkeypatch.setattr(cc, 'atomic_json', guarded)
    else:
        monkeypatch.setattr(recovery, 'atomic_json', fail)
    with pytest.raises(OSError):
        s.save(payload)
    assert s.index_path.read_bytes() == raw and s.resolve() == previous


@pytest.mark.parametrize('part', ['model', 'optimizer', 'recovery', 'manifest'])
def test_corruption_and_missing_components_fail(tmp_path, part):
    _, _, _, payload = fixture()
    s = store(tmp_path)
    s.save(payload)
    path = s.resolve()
    manifest, root = manifest_document(path)
    target = path if part == 'manifest' else root / manifest['components'][part]['path']
    # Only this test's newly created fixture object is modified.
    with target.open('r+b') as stream:
        stream.seek(8)
        stream.write(b'CORRUPT')
    with pytest.raises(ValueError):
        s.resolve()
    target.unlink()
    with pytest.raises(FileNotFoundError):
        s.resolve()


def test_capacity_boundary_alias_and_migration_rejections(tmp_path, monkeypatch):
    import llm_architecture.component_checkpoint as cc
    _, _, _, payload = fixture()
    s = store(tmp_path)
    payload['resume_context']['microstep_within_gradient_accumulation'] = 1
    with pytest.raises(ValueError, match='optimizer boundary'):
        s.save(payload)
    payload['resume_context']['microstep_within_gradient_accumulation'] = 0
    payload['optimizer_state_dict']['alias'] = payload['model_state_dict']['lm_head.weight']
    with pytest.raises(ValueError, match='aliases'):
        s.save(payload)
    del payload['optimizer_state_dict']['alias']
    monkeypatch.setattr(cc.shutil, 'disk_usage', lambda p: type('Disk', (), {'free': 1})())
    with pytest.raises(OSError, match='capacity'):
        s.save(payload)
    assert not s.index_path.exists()
    historical = tmp_path / 'historical'
    historical.mkdir()
    (historical / 'latest.pt').write_bytes(b'historical')
    with pytest.raises(FileExistsError):
        RecoveryStore(historical, backend='optimized', reserve_bytes=0)


def test_concurrent_readers_and_publication(tmp_path):
    _, _, _, payload = fixture()
    s = store(tmp_path)
    s.save(payload)
    def reader():
        for _ in range(8):
            assert read_components(s.resolve())['run_id'] == 'isolated'
    def writer():
        for step in range(2, 6):
            payload['training_state']['optimizer_step'] = step
            s.save(payload)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(reader), pool.submit(reader), pool.submit(writer)]
        for future in futures:
            future.result()


def test_manager_dispatch_and_budget_reserve(tmp_path):
    _, _, _, payload = fixture()
    manager = CheckpointManager(tmp_path, 'manager', backend='optimized', reserve_bytes=0)
    path = manager.save(payload, is_best=True, milestone=True)
    assert path.name.endswith('.ckpt.json')
    assert manager.optimized_store.index()['best_validation']['optimizer_step'] == 1
    estimate = estimate_component_storage(model_bytes=100, optimizer_bytes=50, updated_generations=4)
    assert estimate['retained_payload_bytes'] == 700
    assert estimate['required_free_bytes'] >= 180 * 1024**3
    with pytest.raises(ValueError):
        estimate_component_storage(model_bytes=100, optimizer_bytes=50, updated_generations=-1)


def test_malformed_manifest_and_symlink_are_rejected(tmp_path):
    import hashlib
    _, _, _, data = fixture()
    store = RecoveryStore(tmp_path / 'run', backend='optimized', reserve_bytes=0)
    store.save(data)
    manifest_path = store.resolve()
    doc, root = manifest_document(manifest_path)
    for malformed in ([], dict(doc, components=[]), dict(doc, components={k: [] for k in doc['components']})):
        raw = (json.dumps(malformed, sort_keys=True, separators=(',', ':')) + '\n').encode()
        bad = manifest_path.parent / (hashlib.sha256(raw).hexdigest() + '.ckpt.json')
        bad.write_bytes(raw)
        with pytest.raises(ValueError):
            manifest_document(bad)
    marker = root / 'store.json'
    raw = marker.read_bytes()
    marker.unlink()
    elsewhere = tmp_path / 'marker.json'
    elsewhere.write_bytes(raw)
    marker.symlink_to(elsewhere)
    with pytest.raises(ValueError, match='symlink|escapes'):
        manifest_document(manifest_path)


def test_configuration_drift_nonfinite_and_scaler_state(tmp_path):
    _, _, _, data = fixture()
    data['grad_scaler_state_dict'] = {'scale': 65536., 'growth_tracker': 2}
    store = RecoveryStore(tmp_path / 'run', backend='optimized', reserve_bytes=0)
    store.save(data)
    exact(read_components(store.resolve())['grad_scaler_state_dict'], data['grad_scaler_state_dict'])
    changed = copy.deepcopy(data)
    changed['config']['training']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='identity drift'):
        store.save(changed)
    changed = copy.deepcopy(data)
    next(iter(changed['model_state_dict'].values())).fill_(float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'):
        store.save(changed)
    exact(read_components(store.resolve())['model_state_dict'], data['model_state_dict'])


def test_partial_tensor_serialization_keeps_previous_index(tmp_path, monkeypatch):
    import llm_architecture.component_checkpoint as cc
    _, _, _, payload = fixture()
    s = store(tmp_path)
    s.save(payload)
    previous = s.resolve()
    old_index = s.index_path.read_bytes()
    def partial(value, writer):
        writer.write(b'partial checkpoint bytes')
        raise OSError('interrupted tensor serialization')
    monkeypatch.setattr(cc.torch, 'save', partial)
    with pytest.raises(OSError, match='interrupted'):
        s.save(payload)
    assert s.index_path.read_bytes() == old_index
    assert s.resolve() == previous
    assert not list(s.store_root.glob('.object-*'))


def test_two_writers_share_group_without_partial_publication(tmp_path):
    _, _, _, payload = fixture()
    a, b = store(tmp_path, 'a'), store(tmp_path, 'b')
    other = copy.deepcopy(payload)
    other['run_id'] = 'another'
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(a.save, payload), pool.submit(b.save, other)]
        for f in futures:
            f.result()
    am, bm = [manifest_document(s.resolve())[0] for s in (a, b)]
    assert am['components']['model'] == bm['components']['model']
    assert am['components']['optimizer'] == bm['components']['optimizer']
    assert read_components(a.resolve())['run_id'] != read_components(b.resolve())['run_id']


def test_schedule_bound_covers_evaluation_and_extra_boundaries():
    from llm_training.preflight import checkpoint_publication_bound
    assert checkpoint_publication_bound(max_steps=1000, checkpoint_interval=100,
                                        evaluation_interval=50) == 34
    with pytest.raises(ValueError):
        checkpoint_publication_bound(max_steps=1, checkpoint_interval=0, evaluation_interval=1)
