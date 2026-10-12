"""END and per-source loss diagnostics preserve the existing validation objective."""

import pytest
import torch
from torch.utils.data import DataLoader

from llm_architecture.model import LanguageModel
from llm_training.training import evaluate, resolve_precision
from test_chat_runtime import project


def test_source_and_end_losses_match_whole_batch_token_mean():
    model=LanguageModel(project().model)
    ids=torch.tensor([[8,9,10,11],[11,10,9,8]])
    labels=torch.tensor([[9,7,-100,-100],[10,9,7,-100]])
    batch={'input_ids':ids,'labels':labels,'attention_mask':torch.ones_like(ids,dtype=torch.bool),
           'sources':['oasst1','ultrachat']}
    loader=DataLoader([batch],batch_size=None)
    precision=resolve_precision('fp32',torch.device('cpu'))
    expected,tokens=evaluate(model,loader,device=torch.device('cpu'),precision=precision)
    details={}
    actual,n=evaluate(model,loader,device=torch.device('cpu'),precision=precision,details=details)
    assert actual==expected and n==tokens==5
    assert details['end_tokens']==2
    assert details['sources']['oasst1']['tokens']==2 and details['sources']['ultrachat']['tokens']==3
    weighted=sum(row['loss']*row['tokens'] for row in details['sources'].values())/5
    assert weighted==pytest.approx(actual,abs=1e-6)
    assert details['end_loss']>0 and model.training
    previous=dict(details)
    with pytest.raises(TimeoutError):
        evaluate(model,loader,device=torch.device('cpu'),precision=precision,details=details,deadline_monotonic=0)
    assert details==previous  # No partial current score replaces completed evidence.
