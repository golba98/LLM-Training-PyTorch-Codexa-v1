"""Weighted assistant objectives exclude masks and accumulate exact token counts."""

import pytest
import torch
import torch.nn.functional as F

from llm_training.token_objective import weighted_token_loss, validate_target_weights


def test_weighted_microbatches_match_global_reference_gradient_and_ignore_padding():
    torch.manual_seed(42)
    logits = torch.randn(2, 3, 8, requires_grad=True)
    labels = torch.tensor([[1, 7, -100], [7, 3, 4]])
    accumulated = sum(weighted_token_loss(logits[i:i+1], labels[i:i+1], {7: 3.})
                      * labels[i].ne(-100).sum() for i in range(2)) / labels.ne(-100).sum()
    accumulated.backward()
    actual_gradient = logits.grad.clone()
    reference = logits.detach().clone().requires_grad_(True)
    unweighted = F.cross_entropy(reference.reshape(-1, 8), labels.reshape(-1),
                                ignore_index=-100, reduction='none').reshape_as(labels)
    objective = (unweighted * torch.where(labels.eq(7), 3., 1.)).sum() / labels.ne(-100).sum()
    objective.backward()
    assert torch.allclose(accumulated, objective)
    assert torch.allclose(actual_gradient, reference.grad)
    assert actual_gradient[0, 2].eq(0).all()


@pytest.mark.parametrize('weights', [{7: 0}, {7: float('inf')}, {-1: 3}, {'7': 3}])
def test_invalid_target_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        validate_target_weights(weights)


def test_unit_weights_preserve_cross_entropy_and_empty_targets_reject():
    logits = torch.randn(1, 3, 8)
    labels = torch.tensor([[1, 7, -100]])
    assert torch.allclose(weighted_token_loss(logits, labels, {7: 1.}),
                          F.cross_entropy(logits.reshape(-1, 8), labels.reshape(-1)))
    with pytest.raises(ValueError, match='no target'):
        weighted_token_loss(logits, torch.full_like(labels, -100), {7: 3.})


def test_training_engine_applies_weights_without_counting_extra_tokens():
    from torch.utils.data import DataLoader, TensorDataset
    from llm_training.training import train_model, resolve_precision

    class TableModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.randn(8, 8))

        def forward(self, ids, labels):
            logits = self.table[ids]
            loss = F.cross_entropy(logits.reshape(-1, 8), labels.reshape(-1)) if labels is not None else None
            return logits, loss

    torch.manual_seed(31)
    actual, reference = TableModel(), TableModel()
    reference.load_state_dict(actual.state_dict())
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    labels = torch.tensor([[7, -100, 2], [3, 7, 4]])
    optimizer = torch.optim.SGD(actual.parameters(), lr=.01)
    state, metrics = train_model(actual, DataLoader(TensorDataset(ids, labels), batch_size=1),
        optimizer, device=torch.device('cpu'), precision=resolve_precision('fp32', torch.device('cpu')),
        max_steps=1, gradient_accumulation_steps=2, gradient_clip=1e6,
        warmup_steps=0, peak_learning_rate=.01, loss_normalization='supervised_tokens',
        target_token_weights={7: 3.})
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    logits, _ = reference(ids, labels)
    weighted_token_loss(logits, labels, {7: 3.}).backward()
    reference_optimizer.step()
    assert torch.allclose(actual.table, reference.table)
    assert state.tokens_seen == metrics[0].tokens_processed == 5
    assert metrics[0].context_tokens_processed == 6
