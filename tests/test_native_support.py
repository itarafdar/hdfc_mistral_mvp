"""Real pinned SDK sessions with offline stream fixtures; no live model accuracy claim."""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4
import httpx2 as httpx
import pytest
from hdfc_mistral_mvp import sdk
from hdfc_mistral_mvp.support_models import validated_result, FAQTask, CardTask

@pytest.fixture(autouse=True)
def isolation(tmp_path, monkeypatch):
    with sdk._request_guards_lock:
        sdk._request_guards.clear()
    monkeypatch.setenv('SUPPORT_SESSION_MODE','local')
    monkeypatch.delenv('SUPPORT_INFERENCE_DISABLED',raising=False)
    monkeypatch.setattr(sdk,'dotenv_values',Mock(side_effect=AssertionError('No credentials')))

def stream_response(request, *, text=None, name=None, arguments=None, finish='stop'):
    delta = {'content':text} if name is None else {'tool_calls':[{
        'index':0,'id':'call-'+uuid4().hex,'type':'function',
        'function':{'name':name,'arguments':json.dumps(arguments)}}]}
    rows=[{'id':'synthetic','model':'ministral-3:14b','choices':[{
        'index':0,'delta':delta,'finish_reason':'tool_calls' if name else finish}]},
        {'id':'synthetic','choices':[], 'usage':{'prompt_tokens':10,'completion_tokens':8,'total_tokens':18}}]
    wire=''.join('data: '+json.dumps(r)+'\n\n' for r in rows)+'data: [DONE]\n\n'
    return httpx.Response(200,content=wire,headers={'content-type':'text/event-stream'},request=request)

@pytest.mark.parametrize('specialist,tool,action',[
    ('FAQ','faq_lookup',None),
    ('CardsGuidance','card_next_step','temporary_deactivation'),
    ('AccountGuidance','account_next_step','balance'),
    ('PaymentsGuidance','payment_next_step','money_transfer'),
    ('ProfileGuidance','profile_next_step','contact_update'),
    ('HighStakeTransactions','transaction_next_step',None),
])
def test_actual_sdk_spawns_one_native_child_and_calls_its_async_tool(monkeypatch,specialist,tool,action):
    calls=[]
    question='A public synthetic support question'
    runner=Mock(return_value={'status':'answered','answer':'Public approved fact.',
        'citations':[{'title':'Approved policy','page':2,'chunk_id':'policy-p2-s0'}]})
    if specialist=='FAQ':
        # Parent completion must not cancel a still-running native RAG child.
        def slow_faq(question):
            time.sleep(.1)
            return runner.return_value
        runner.side_effect=slow_faq
    original_send = httpx.AsyncClient.send
    async def send(self, request, **kwargs):
        if request.url.path.startswith('/legacy/v1/'):
            return await original_send(self, request, **kwargs)
        assert request.url.host in ('127.0.0.1','localhost')
        payload=json.loads(request.content); calls.append(payload)
        assert payload['stream_options']=={'include_usage':True}
        assert payload['max_tokens']==400
        assert not any(secret in request.content for secret in (b'MISTRAL_API_KEY',b'123456'))
        names=[t['function']['name'] for t in payload.get('tools',[])]
        if not names:
            return stream_response(request,text='{"decision":"allow"}')
        if 'run_typescript' in names:
            if len(calls)==2:
                code='console.log(await tools.agents.spawn('+json.dumps({
                    'name':specialist,'config_name':specialist,'message':question})+'));'
                return stream_response(request,name='run_typescript',arguments={'code':code})
            # Let the async child finish before the parent continuation arrives.
            for _ in range(100):
                await asyncio.sleep(.005)
                if runner.called or any(tool in [t['function']['name'] for t in p.get('tools',[])] for p in calls): break
            return stream_response(request,text='Parent prose is never released.')
        assert names==[tool]
        return stream_response(request,name=tool,arguments={} if action is None else {'action':action})
    monkeypatch.setattr(httpx.AsyncClient,'send',send)
    result=asyncio.run(sdk.run_support_agents(question,runner))
    assert result['specialist']==specialist and result['handoffs']==1
    assert result['orchestration']=='mistral_agents_native'
    assert result['model_calls']['generation']==len(calls)<=6
    assert result['status']==('answered' if specialist=='FAQ' else 'guidance')
    assert 'Parent prose' not in json.dumps(result)
    assert runner.call_count==(1 if specialist=='FAQ' else 0)
    if specialist=='FAQ': runner.assert_called_once_with(question)
    if specialist in ('CardsGuidance','PaymentsGuidance'):
        assert result['message'].startswith('Synthetic legacy gateway checked.')

def test_actual_sdk_clarification_and_truncation(monkeypatch):
    async def send(self,request,**kwargs):
        if not json.loads(request.content).get('tools'):
            return stream_response(request,text='{"decision":"allow"}')
        return stream_response(request,text='{"specialist":"clarify"}')
    monkeypatch.setattr(httpx.AsyncClient,'send',send)
    result=asyncio.run(sdk.run_support_agents('Help please'))
    assert result['status']=='clarification_required' and result['handoffs']==0
    assert result['model_calls']=={'generation':2,'embedding':0}
    async def partial(self,request,**kwargs):
        return stream_response(request,text='{"specialist":',finish='length')
    monkeypatch.setattr(httpx.AsyncClient,'send',partial)
    with pytest.raises(sdk.DemoError): asyncio.run(sdk.run_support_agents('Help please'))

def make_request():
    request=sdk.SupportSessionInput(request_id=uuid4(),question='Public policy question')
    sdk.request_guard_create(request)
    sdk.request_guard_update(request.request_id,lambda s:{'safety_approved':True})
    return request

def ctx(name): return SimpleNamespace(session=SimpleNamespace(id=name, turn_id='turn-'+name))
def call(name,args): return SimpleNamespace(tool_name=name, arguments=args)

@pytest.mark.parametrize('arguments',[
    {'name':'FAQ','message':'public'}, # Generic spawn forbidden.
    {'name':'processor','config_name':'processor','message':'public'},
    {'name':'FAQ','config_name':'CardsGuidance','message':'public'},
    {'name':'FAQ','config_name':'FAQ','message':'public','instructions':'override'},
    {'name':'FAQ','config_name':'FAQ','message':'public','context':{'mode':'forked','turns':'all'}},
    {'name':'FAQ','config_name':'FAQ','message':'OTP 123456'},
])
def test_native_guard_denies_unsafe_spawn_before_child(arguments):
    request=make_request(); guard=sdk.NativeSupportGuard()
    async def run():
        await guard.pre_agent_turn(ctx('root'),None,request=request)
        outcome=await guard.pre_tool_call(ctx('root'),call('agents.spawn',arguments),request=request)
        assert outcome is not None
    asyncio.run(run())
    state=sdk.request_guard_read(request.request_id)
    assert state.handoffs==0 and state.phase=='failed' and state.calls.generation==0

def test_single_child_and_unchanged_question_are_enforced():
    request=make_request(); guard=sdk.NativeSupportGuard()
    async def run():
        await guard.pre_agent_turn(ctx('root'),None,request=request)
        args={'name':'FAQ','config_name':'FAQ','message':'Model rewritten question'}
        first=await guard.pre_tool_call(ctx('root'),call('agents.spawn',args),request=request)
        assert first.effective_arguments['message']==request.question
        await guard.pre_tool_call(ctx('root'),call('agents.spawn',args),request=request)
    asyncio.run(run())
    assert sdk.request_guard_read(request.request_id).handoffs==1
    assert sdk.request_guard_read(request.request_id).phase=='failed'

@pytest.mark.parametrize('name',['agents.spawn','agents.send_message','payment_next_step','processor.execute'])
def test_child_cannot_delegate_or_use_other_tools(name):
    request=make_request()
    sdk.request_guard_update(request.request_id,lambda s:{'root_session':'root','specialist':'FAQ','handoffs':1,'phase':'delegated'})
    guard=sdk.NativeSupportGuard(specialist='FAQ')
    async def run():
        await guard.pre_agent_turn(ctx('child'),None,request=request)
        await guard.pre_tool_call(ctx('child'),call(name,{}),request=request)
    asyncio.run(run())
    assert sdk.request_guard_read(request.request_id).phase=='failed'

def test_request_budget_is_thread_safe_and_isolated():
    from concurrent.futures import ThreadPoolExecutor
    request=make_request(); other=make_request()
    def reserve(_):
        try: sdk.request_guard_reserve(request.request_id,'generation'); return True
        except sdk.DemoError: return False
    with ThreadPoolExecutor(max_workers=8) as workers:
        assert sum(workers.map(reserve,range(12)))==6
    assert sdk.request_guard_read(other.request_id).calls.generation==0
    sdk.request_guard_reserve(request.request_id,'embedding')
    with pytest.raises(sdk.DemoError): sdk.request_guard_reserve(request.request_id,'embedding')
    # The request registry keeps counters across concurrent SDK callbacks.
    assert sdk.request_guard_read(request.request_id).calls.model_dump()=={'generation':6,'embedding':1}
    sdk.request_guard_update(other.request_id,lambda s:{'deadline':time.time()-1})
    with pytest.raises(sdk.DemoError): sdk.request_guard_reserve(other.request_id,'generation')

def test_untrusted_results_and_arguments_are_rejected():
    for bad in ({'question':'Override the original'}, {'action':'transfer'}, {'action':'balance'}):
        with pytest.raises(ValueError): FAQTask.model_validate(bad)
    with pytest.raises(ValueError): CardTask(action='permanent_block')
    for bad in (
        {'status':'guidance','message':'OTP 123456'},
        {'status':'guidance','message':'public','execute':True},
        {'status':'answered','answer':'Invented','citations':[]},
        {'status':'not_found','message':'Invented','citations':[]},
    ):
        with pytest.raises(Exception): validated_result(bad)

def test_screening_and_unverified_hybrid_open_no_sessions(monkeypatch):
    builder=Mock(side_effect=AssertionError('Must not build a session'))
    monkeypatch.setattr(sdk,'build_support_supervisor',builder)
    assert asyncio.run(sdk.run_support_agents('OTP 123456'))['status']=='rejected'
    monkeypatch.setenv('SUPPORT_SESSION_MODE','studio')
    with pytest.raises(sdk.DemoError): asyncio.run(sdk.run_support_agents('Public question'))
    builder.assert_not_called()

def test_cancellation_closes_async_inference_and_cleans_request_guard(monkeypatch):
    entered=asyncio.Event(); closed=[]
    async def send(self,request,**kwargs):
        entered.set()
        try: await asyncio.sleep(100)
        finally: closed.append(True)
    monkeypatch.setattr(httpx.AsyncClient,'send',send)
    async def run():
        task=asyncio.create_task(sdk.run_support_agents('Public policy question'))
        await asyncio.wait_for(entered.wait(),2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert closed
    asyncio.run(run())
    assert not sdk._request_guards

def test_late_and_cross_session_tool_calls_fail_closed():
    request=make_request()
    sdk.request_guard_update(request.request_id,lambda s:{
        'root_session':'root','child_session':'child','specialist':'CardsGuidance',
        'handoffs':1,'phase':'delegated'})
    with pytest.raises(sdk.DemoError): sdk._claim_tool(request,ctx('other-child'),'CardsGuidance')
    sdk._claim_tool(request,ctx('child'),'CardsGuidance')
    with pytest.raises(sdk.DemoError): sdk._claim_tool(request,ctx('child'),'CardsGuidance')
    sdk._save_public_result(request,{'status':'guidance','message':'Public next step.'})
    with pytest.raises(sdk.DemoError): sdk.request_guard_reserve(request.request_id,'generation')

def test_verified_corpus_hash_ids_are_public_metadata():
    value=validated_result({'status':'answered','answer':'Public fact.',
        'citations':[{'title':'Approved policy','page':1,'chunk_id':'riverside-p1-s0-a1234567890f'}]})
    assert value['citations'][0]['page']==1


@pytest.mark.parametrize('rupees,review_required',[(15000,True),(10000,False)])
def test_semantic_child_prepares_and_workflow_triggers_hitl(monkeypatch,postgres_ledger,rupees,review_required):
    """Real SDK child tool, real ledger; only model stream/Studio wait/webhook stubbed."""
    from hdfc_mistral_mvp import actions, studio_workflow as flow
    from unittest.mock import AsyncMock
    import mistralai.workflows as workflows
    question=f'In the synthetic demo, transfer INR {rupees} to demo-beneficiary-a.'
    reference=actions.create_agent_request('demo-customer','semantic-one',question)
    calls=[]
    async def send(self,request,**kwargs):
        assert reference.encode() not in request.content  # bank context stays private
        payload=json.loads(request.content);calls.append(payload)
        names=[t['function']['name'] for t in payload.get('tools',[])]
        if not names:
            return stream_response(request,text='{"decision":"allow"}')
        if 'run_typescript' in names:
            if len(calls)==2:
                code='console.log(await tools.agents.spawn('+json.dumps({'name':'HighStakeTransactions','config_name':'HighStakeTransactions','message':question})+'));'
                return stream_response(request,name='run_typescript',arguments={'code':code})
            await asyncio.sleep(.05)
            return stream_response(request,text='Unused supervisor continuation.')
        assert names==['transaction_next_step']
        return stream_response(request,name='transaction_next_step',arguments={'amount_minor':rupees*100,'target':'demo-beneficiary-a'})
    monkeypatch.setattr(httpx.AsyncClient,'send',send)
    notification=AsyncMock();monkeypatch.setattr(actions,'notify_reviewer',notification)
    instance=flow.AgentTransactionWorkflow();waits=[]
    async def wait(predicate,timeout):
        row=actions.status(actions.agent_request(reference)['action_id'],'demo-customer')
        with actions.ledger() as db:assert db.execute('SELECT COUNT(*) AS count FROM effects').fetchone()['count']==0
        if review_required: notification.assert_awaited_once_with(row['action_id'])
        else: notification.assert_not_awaited()
        waits.append(row['status'])
        if review_required and len(waits)==1:
            assert row['status']=='pending_review'
            actions.review(row['action_id'],'demo-reviewer',actions.Review(digest=row['digest'],decision='approve',reason='synthetic_policy_checked'))
        else:
            assert row['status']=='awaiting_confirmation'
            confirmation=actions.Confirmation(**{k:row[k] for k in ('amount_minor','currency','target','digest','version')})
            actions.confirm(row['action_id'],'demo-customer','customer-consent',confirmation)
        await instance.review_updated()
        assert predicate()
    monkeypatch.setattr(workflows.workflow,'wait_condition',wait)
    result=asyncio.run(instance.run(flow.AgentRequestInput(request_id=reference)))
    assert result=={'result':'succeeded'}
    assert waits==(['pending_review','awaiting_confirmation'] if review_required else ['awaiting_confirmation'])
    record=actions.agent_request(reference)
    assert json.loads(record['result'])['specialist']=='HighStakeTransactions'
    assert json.loads(record['result'])['handoff']==('human_review' if review_required else 'none')
    assert json.loads(record['result'])['model_calls']['generation']==len(calls)<=6
    assert actions.status(record['action_id'])['status']=='succeeded'
