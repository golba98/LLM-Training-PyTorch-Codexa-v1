"""Independent best roles and exact scheduled continuation with random batches."""

import copy

import pytest
import torch

from llm_training.checkpointing import (build_checkpoint_payload,load_checkpoint,SchedulerState,
                                       capture_random_states)
from llm_training.recovery import RecoveryStore
from llm_training.training import cosine_learning_rate, TrainingState
from test_chat_runtime import project
from llm_architecture.model import LanguageModel


def test_two_best_roles_are_shared_aliases_and_rotate_without_loss(tmp_path):
    config=project(); model=LanguageModel(config.model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    store=RecoveryStore(tmp_path)
    def payload(step):
        return build_checkpoint_payload(model=model,optimizer=optimizer,scaler=None,
            state=TrainingState(optimizer_step=step),scheduler=SchedulerState(0,20,.01,.001),
            config=config,run_name='isolated',run_id='fixed',tokenizer_reference=None,
            tokenizer_sha256=None,resume_context={'configuration_sha256':'fixed'})
    first=store.save(payload(0),best=True,best_validation=True)
    assert store.index()['best']==store.index()['best_validation']
    validation=store.save(payload(1),best_validation=True)
    for step in range(2,15):
        store.save(payload(step),final=step==14)
    assert store.index()['best']['sha256']==first['sha256']
    assert store.index()['best_validation']['sha256']==validation['sha256']
    for role in store.index():
        assert store.resolve(role).is_file()
    assert len(list(tmp_path.glob('slot-*/checkpoint.pt')))<=6


def test_cpu_schedule_rng_sampler_and_next_updates_are_exact(tmp_path):
    torch.manual_seed(33)
    config=project(); config.training.max_steps=8
    model=LanguageModel(config.model); optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    scheduler=SchedulerState(2,8,.01,.005)
    state=TrainingState(); context={'configuration_sha256':'fixed','sampler':{'epoch':0,'cursor':0}}
    def update():
        nonlocal state
        inputs=torch.randint(0,16,(1,4)); labels=torch.randint(0,16,(1,4))
        labels[:,0]=-100
        optimizer.zero_grad(set_to_none=True)
        _,loss=model(inputs,labels); loss.backward()
        lr=cosine_learning_rate(state.optimizer_step,**scheduler.to_dict())
        for group in optimizer.param_groups:
            group['lr']=lr
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        state.optimizer_step+=1; state.micro_step+=1; state.tokens_seen+=3
        state.batches_in_epoch+=1; context['sampler']['cursor']+=1
    update(); update()
    store=RecoveryStore(tmp_path)
    payload=build_checkpoint_payload(model=model,optimizer=optimizer,scaler=None,state=state,
        scheduler=scheduler,config=config,run_name='cpu',run_id='fixed',tokenizer_reference=None,
        tokenizer_sha256=None,resume_context=context)
    store.save(payload)
    update(); update()
    expected=copy.deepcopy((model.state_dict(),optimizer.state_dict(),state.to_dict(),context,capture_random_states()))
    restored=load_checkpoint(store.resolve(),model=model,optimizer=optimizer,scaler=None,expected_config=config)
    state=restored.state; scheduler=restored.scheduler; context=copy.deepcopy(restored.resume_context)
    update(); update()
    from codexa_fixture_equality import assert_equal
    assert_equal(expected,(model.state_dict(),optimizer.state_dict(),state.to_dict(),context,capture_random_states()))


def test_new_category_quality_guards_apply_patience_in_caller():
    from llm_training.quality_guard import behavioral_regression
    base=dict(macro_task_accuracy=.5,severe_degeneration_rate=.1,END_rate=.8,empty_rate=0,
              role_switch_rate=0,category_pass_rates={'instruction':.8,'factual':.8},correct_END_rate=.7)
    current={**base,'category_pass_rates':{'instruction':.2,'factual':.8}}
    assert behavioral_regression(current,base,{'instruction_drop':.05})
    assert not behavioral_regression(current,base,{})


def test_chat_controller_baseline_and_separate_best_roles_on_cpu(tmp_path,monkeypatch):
    """Exercise orchestration with a tiny CPU model and an injected update boundary."""
    from dataclasses import asdict
    import hashlib,json
    from types import SimpleNamespace
    import llm_training.chat_run as chat
    from llm_data.conversational_audit import content_identity
    from llm_tokenizer.sft import ChatMessage,serialize_conversation
    from llm_training.session import SessionBudget
    from llm_training.training import resolve_precision
    config=project(); config.model.context_length=8
    source=LanguageModel(config.model)
    class Tokenizer:
        def token_to_id(self,value):
            return dict(zip(['<pad>','<bos>','<eos>','<unk>','<|system|>','<|user|>','<|assistant|>','<|end|>'],range(8)))[value]
        def encode(self,text,add_special_tokens=False):
            return SimpleNamespace(ids=[8+int(word)%8 for word in text.split()])
    tokenizer=Tokenizer(); token_path=tmp_path/'tokenizer';token_path.write_text('fixture')
    data_path=tmp_path/'data.jsonl'; rows=[]
    for i in range(2):
        messages=[dict(role='user',content=str(i)),dict(role='assistant',content=f'{i} {i+1}')]
        record=serialize_conversation([ChatMessage(**m) for m in messages],tokenizer)
        rows.append(dict(conversation_id=str(i),root_id=str(i),duplicate_group=str(i),
            source='a' if i else 'b',messages=messages,tokens=len(record.input_ids),
            assistant_tokens=record.supervised_token_count,content_sha256=content_identity(messages)))
    data_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    store=RecoveryStore(tmp_path/'initial')
    store.save(build_checkpoint_payload(model=source,optimizer=torch.optim.AdamW(source.parameters()),
        scaler=None,state=TrainingState(),scheduler=SchedulerState(0,1,.01,.001),config=config,
        run_name='fixture',run_id='fixture',tokenizer_reference=None,tokenizer_sha256=None,resume_context={}))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    spec=dict(purpose='experimental',model=asdict(config.model),training=asdict(config.training),
        initialization=dict(path=str(store.resolve()),sha256=digest(store.resolve())),
        tokenizer=dict(path=str(token_path),sha256=digest(token_path)),
        data=dict(train=str(data_path),validation=str(data_path),train_sha256=digest(data_path),
                  validation_sha256=digest(data_path),manifest_sha256='fixture',maximum_epochs=1),
        optimization=dict(optimizer='adamw',gradient_checkpointing=False),
        checkpoint=dict(storage_backend='legacy',minimum_save_seconds=.01),
        evaluation=dict(validation_batches=1,fresh_behavioral_baseline=True,quality_reserve_seconds=.01,
                        validation_reserve_seconds=.01,quality_interval=1,regression_policy={}))
    model=LanguageModel(config.model)
    optimizer=torch.optim.AdamW(model.parameters())
    monkeypatch.setattr(model,'to',lambda *args,**kwargs:model)
    monkeypatch.setattr(chat,'LanguageModel',lambda config:model)
    monkeypatch.setattr(chat,'count_parameters',lambda model:934356480)
    monkeypatch.setattr(chat,'load_tokenizer',lambda path:tokenizer)
    monkeypatch.setattr(chat,'create_adamw_optimizer',lambda *args,**kwargs:optimizer)
    precision=resolve_precision('fp32',torch.device('cpu'))
    monkeypatch.setattr(chat,'resolve_precision',lambda *args:precision)
    monkeypatch.setattr(chat,'start_chat_viewer',lambda *args,**kwargs:SimpleNamespace(poll=lambda:None))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'is_bf16_supported',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(12*1024**3,16*1024**3))
    monkeypatch.setattr(chat,'evaluate',lambda *args,**kwargs:(.7,3))
    observed=[]
    def quality(*args):
        observed.append('quality')
        return dict(macro_task_accuracy=.5,END_rate=.8,severe_degeneration_rate=.1,empty_rate=0,
            role_switch_rate=0,entry_gate_passed=True,category_pass_rates={'instruction':.5,'factual':.5})
    def update_boundary(model,loader,optimizer,**kwargs):
        assert observed==['quality']
        state=kwargs['state'];state.optimizer_step=1;state.micro_step=2;state.tokens_seen=6
        kwargs['on_committed_batches']([dict(sources=['b','a'],indices=[0,1],identities=['0','1'])])
        metrics=SimpleNamespace(context_tokens_processed=14,padded_tokens_processed=14,
            validation_loss=.7,training_loss=.8,learning_rate=.01,tokens_per_second=6,
            step_time_seconds=1,peak_reserved_vram_bytes=0)
        kwargs['on_optimizer_step'](state,metrics)
        return state,[]
    monkeypatch.setattr(chat,'train_model',update_boundary)
    report=chat.run_chat(spec,tmp_path/'run',SessionBudget.start(30,5),quality_evaluator=quality)
    assert report['state']=='COMPLETED' and observed==['quality','quality']
    assert report['counters']['source_supervised_tokens']=={'a':3,'b':3}
    recovery=RecoveryStore(tmp_path/'run/checkpoints')
    assert recovery.resolve('best_validation').exists() and recovery.resolve('best').exists()
