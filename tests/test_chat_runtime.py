"""Token normalization, transactional publication and cumulative budget contracts."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from llm_architecture.model import LanguageModel, ModelConfig
from llm_training.checkpointing import build_checkpoint_payload, SchedulerState, load_checkpoint
from llm_training.config import ProjectConfig, TrainingConfig
from llm_training.recovery import RecoveryStore
from llm_training.session import SessionBudget, StopRequest
from llm_training.training import TrainingState, resolve_precision, train_model, unpack_training_batch


def project() -> ProjectConfig:
    return ProjectConfig(ModelConfig(vocab_size=16, context_length=4, num_layers=1,
        hidden_size=8, num_heads=2, intermediate_size=16),
        TrainingConfig(1, 2, 0.01, 0.0, 0, 1, 1000, 'fp32', 1, 1, 42))


def test_assistant_token_normalization_matches_summed_reference():
    torch.manual_seed(14)
    a = LanguageModel(project().model)
    b = LanguageModel(project().model)
    b.load_state_dict(a.state_dict())
    inputs = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    labels = torch.tensor([[2, -100, -100, -100], [3, 2, 1, 0]])
    reference = torch.optim.SGD(b.parameters(), lr=0.01)
    _, loss = b(inputs, labels)
    loss.backward()
    reference.step()
    optimizer = torch.optim.SGD(a.parameters(), lr=0.01)
    state, metrics = train_model(a, DataLoader(TensorDataset(inputs, labels), batch_size=1), optimizer,
        device=torch.device('cpu'), precision=resolve_precision('fp32', torch.device('cpu')),
        max_steps=1, gradient_accumulation_steps=2, gradient_clip=1000,
        warmup_steps=0, peak_learning_rate=0.01, loss_normalization='supervised_tokens')
    assert state.tokens_seen == 5 and metrics[0].context_tokens_processed == 8
    for actual, expected in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_budget_resume_counts_downtime_and_rejects_clock_regression(monkeypatch):
    import llm_training.session as session
    wall, monotonic = [1000.0], [10.0]
    monkeypatch.setattr(session.time, 'time', lambda: wall[0])
    monkeypatch.setattr(session.time, 'monotonic', lambda: monotonic[0])
    budget = SessionBudget.start(100, 20)
    wall[0] += 10; monotonic[0] += 10
    saved = budget.document()
    wall[0] += 30; monotonic[0] += 30
    resumed = SessionBudget.resume(saved)
    assert resumed.remaining == 60 and resumed.deadline_utc == 1100
    wall[0] = 1000
    with pytest.raises(ValueError, match='backwards'):
        SessionBudget.resume(saved)
    wall[0] = 1200
    with pytest.raises(TimeoutError):
        SessionBudget.resume(saved)


def test_recovery_failure_keeps_published_checkpoint(tmp_path, monkeypatch):
    import llm_training.recovery as recovery
    model = LanguageModel(project().model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    store = RecoveryStore(tmp_path)
    state = TrainingState()
    def payload():
        return build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None, state=state,
            scheduler=SchedulerState(0, 1, .01, .001), config=project(), run_name='isolated', run_id='fixed',
            tokenizer_reference=None, tokenizer_sha256=None, resume_context={'configuration_sha256': 'same'})
    store.save(payload(), best=True)
    original = store.resolve()
    index_before = store.index_path.read_bytes()
    state.optimizer_step = 1
    monkeypatch.setattr(recovery, '_write_checksum', lambda path: (_ for _ in ()).throw(OSError('injected save failure')))
    with pytest.raises(OSError):
        store.save(payload())
    assert store.index_path.read_bytes() == index_before and store.resolve() == original


def test_recovery_rotation_and_full_restore(tmp_path):
    config = project()
    model = LanguageModel(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    _, loss = model(torch.ones(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long))
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    store = RecoveryStore(tmp_path)
    for step in range(9):
        state = TrainingState(optimizer_step=step)
        store.save(build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None, state=state,
            scheduler=SchedulerState(0, 10, .01, .001), config=config, run_name='isolated', run_id='fixed',
            tokenizer_reference=None, tokenizer_sha256=None, resume_context={}), best=step == 3, final=step == 8)
    assert len(list(tmp_path.glob('slot-*/checkpoint.pt'))) <= 6
    assert store.index()['best']['optimizer_step'] == 3
    loaded_model = LanguageModel(config.model)
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=.01)
    restored = load_checkpoint(store.resolve('final'), model=loaded_model, optimizer=loaded_optimizer, scaler=None)
    assert restored.state.optimizer_step == 8
    assert len(loaded_optimizer.state) == len(optimizer.state)
    for left, right in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(left, right)


def test_right_padding_masks_reject_left_padding_or_targets():
    ids = torch.ones(1, 3, dtype=torch.long)
    labels = torch.tensor([[1, -100, -100]])
    batch = dict(input_ids=ids, labels=labels, attention_mask=torch.tensor([[True, False, False]]))
    assert unpack_training_batch(batch)[2] == 1
    batch['attention_mask'] = torch.tensor([[False, True, True]])
    with pytest.raises(ValueError, match='right-padded'):
        unpack_training_batch(batch)


def test_signal_requests_do_not_raise_inside_accumulation():
    request = StopRequest()
    request._request(15, None)
    assert request() and request.signal_number == 15


@pytest.mark.parametrize('number', [2, 15])
def test_real_sigint_and_sigterm_are_controlled(number):
    import os
    with StopRequest() as request:
        os.kill(os.getpid(), number)
        assert request() and request.signal_number == number


def test_oom_rolls_back_uncommitted_counters_and_gradients():
    model = LanguageModel(project().model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    def failed_step(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError('safe injected OOM before update')
    optimizer.step = failed_step
    loader = DataLoader(TensorDataset(torch.ones(2, 4, dtype=torch.long),
                                     torch.ones(2, 4, dtype=torch.long)), batch_size=1)
    state = TrainingState()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        train_model(model, loader, optimizer, device=torch.device('cpu'),
            precision=resolve_precision('fp32', torch.device('cpu')), max_steps=1,
            gradient_accumulation_steps=2, gradient_clip=1, warmup_steps=0,
            peak_learning_rate=.01, state=state, loss_normalization='supervised_tokens')
    assert state.to_dict() == TrainingState().to_dict()
    assert all(p.grad is None for p in model.parameters())


def test_recovery_rejects_changed_phase_identity(tmp_path):
    config = project()
    model = LanguageModel(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    payload = build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None, state=TrainingState(),
        scheduler=SchedulerState(0, 2, .01, .001), config=config, run_name='isolated', run_id='fixed',
        tokenizer_reference=None, tokenizer_sha256=None, resume_context={'configuration_sha256': 'same'})
    store = RecoveryStore(tmp_path)
    store.save(payload)
    before = store.index_path.read_bytes()
    payload['run_id'] = 'different'
    with pytest.raises(ValueError, match='identity drift'):
        store.save(payload)
    assert store.index_path.read_bytes() == before


def test_nonfinite_save_is_not_promoted(tmp_path):
    model = LanguageModel(project().model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    def payload():
        return build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None, state=TrainingState(),
            scheduler=SchedulerState(0, 2, .01, .001), config=project(), run_name='isolated', run_id='fixed',
            tokenizer_reference=None, tokenizer_sha256=None, resume_context={})
    store = RecoveryStore(tmp_path)
    store.save(payload())
    before = store.index_path.read_bytes()
    with torch.no_grad():
        next(model.parameters()).fill_(float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'):
        store.save(payload())
    assert store.index_path.read_bytes() == before
    assert store.resolve().exists()


def test_right_padding_preserves_real_token_logits():
    model = LanguageModel(project().model).eval()
    with torch.no_grad():
        short, _ = model(torch.tensor([[1, 2]]))
        padded, _ = model(torch.tensor([[1, 2, 0, 0]]))
    torch.testing.assert_close(short, padded[:, :2], atol=1e-6, rtol=1e-5)


def test_validation_deadline_does_not_lose_committed_update(monkeypatch):
    import llm_training.training as training
    model = LanguageModel(project().model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loader = DataLoader(TensorDataset(torch.ones(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.long)), batch_size=1)
    def expired(*args, **kwargs):
        raise TimeoutError('partial validation is discarded')
    monkeypatch.setattr(training, 'evaluate', expired)
    state, metrics = train_model(model, loader, optimizer, device=torch.device('cpu'),
        precision=resolve_precision('fp32', torch.device('cpu')), max_steps=1,
        gradient_accumulation_steps=1, gradient_clip=1, warmup_steps=0, peak_learning_rate=.01,
        validation_loader=loader, evaluation_interval=1)
    assert state.optimizer_step == 1 and len(metrics) == 1
    assert metrics[0].validation_loss is None
    assert metrics[0].validation_status == 'deadline_skipped'


def test_deadline_discards_partial_accumulation_without_weight_update(monkeypatch):
    import llm_training.training as training
    model = LanguageModel(project().model)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    clock = [0.0]
    monkeypatch.setattr(training.time, 'monotonic', lambda: clock[0])
    forward = model.forward
    def late_forward(*args, **kwargs):
        result = forward(*args, **kwargs)
        clock[0] = 10.0
        return result
    monkeypatch.setattr(model, 'forward', late_forward)
    loader = DataLoader(TensorDataset(torch.ones(2, 4, dtype=torch.long),
        torch.ones(2, 4, dtype=torch.long)), batch_size=1)
    state = TrainingState()
    with pytest.raises(TimeoutError):
        train_model(model, loader, optimizer, device=torch.device('cpu'),
            precision=resolve_precision('fp32', torch.device('cpu')), max_steps=1,
            gradient_accumulation_steps=2, gradient_clip=1, warmup_steps=0,
            peak_learning_rate=.01, state=state, deadline_monotonic=9.0)
    assert state.to_dict() == TrainingState().to_dict()
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
