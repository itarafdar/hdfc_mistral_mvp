"""Opt-in real PostgreSQL ledger checks; model, Studio and webhook are stubbed."""
import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, Mock
import pytest
from fastapi.testclient import TestClient
from hdfc_mistral_mvp import actions, api, sdk, studio_workflow as flow
import mistralai.workflows as workflows

@pytest.fixture
def sandbox(monkeypatch, postgres_ledger):
    monkeypatch.setenv('DEMO_CUSTOMER_TOKEN', 'customer-synthetic-test-token')
    monkeypatch.setenv('DEMO_REVIEWER_TOKEN', 'reviewer-synthetic-test-token')
    monkeypatch.setenv('DEMO_REVIEW_THRESHOLD_MINOR', '1000000')
    return {'Authorization':'Bearer customer-synthetic-test-token', 'Idempotency-Key':'prepare-one'}


def prepared(amount=1_000_001, owner='demo-customer', key='one'):
    return actions.create(owner, key, actions.Transfer(amount_minor=amount,target='demo-beneficiary-a'))

def confirmation(row):
    return actions.Confirmation(**{k:row[k] for k in ('amount_minor','currency','target','version','digest')})

def confirmed(row):
    return actions.confirm(row['action_id'],'demo-customer','confirm-one',confirmation(row))

def decision(row, choice='approve'):
    return actions.Review(digest=row['digest'],decision=choice,reason='synthetic_policy_checked')

def counts():
    with actions.ledger() as db:
        return tuple(db.execute('SELECT COUNT(*) AS count FROM '+table).fetchone()['count'] for table in ('outbox','effects'))


def test_threshold_confirmation_review_and_exactly_one_effect(sandbox):
    row=prepared()
    assert counts()==(0,0)
    assert actions.dispatch(row['action_id'])=='prepared'
    assert confirmed(row)['status']=='pending_review'
    assert actions.dispatch(row['action_id'])=='pending_review'
    assert counts()==(0,0)
    assert actions.review(row['action_id'],'demo-reviewer',decision(row))['status']=='accepted'
    assert counts()==(1,0)
    assert actions.dispatch(row['action_id'])=='succeeded'
    assert actions.dispatch(row['action_id'])=='succeeded'
    assert counts()==(0,1)
    assert confirmed(row)['status']=='succeeded'
    actions.review(row['action_id'],'demo-reviewer',decision(row))
    assert counts()==(0,1)

@pytest.mark.parametrize('amount',[1,1_000_000])
def test_at_or_below_threshold_still_requires_confirmation(sandbox,amount):
    row=prepared(amount)
    assert actions.dispatch(row['action_id'])=='prepared'
    assert confirmed(row)['status']=='accepted'
    assert actions.dispatch(row['action_id'])=='succeeded'


def test_denied_expired_cross_customer_changed_and_self_approval(sandbox,monkeypatch):
    row=prepared()
    with pytest.raises(actions.ActionError): actions.confirm(row['action_id'],'other','c',confirmation(row))
    with pytest.raises(actions.ActionError): actions.status(row['action_id'],'other')
    with pytest.raises(actions.ActionError): actions.confirm(row['action_id'],'demo-customer','c',confirmation(row).model_copy(update={'amount_minor':2}))
    confirmed(row)
    with pytest.raises(actions.ActionError): actions.review(row['action_id'],'demo-customer',decision(row))
    with pytest.raises(actions.ActionError): actions.review(row['action_id'],'demo-reviewer',decision(row).model_copy(update={'digest':'0'*64}))
    actions.review(row['action_id'],'demo-reviewer',decision(row,'reject'))
    assert actions.dispatch(row['action_id'])=='rejected'
    with pytest.raises(actions.ActionError): actions.review(row['action_id'],'demo-reviewer',decision(row))
    expired=prepared(key='two');confirmed(expired)
    monkeypatch.setattr(actions.time,'time',lambda:10**12)
    with pytest.raises(actions.ActionError): actions.review(expired['action_id'],'demo-reviewer',decision(expired))
    assert actions.expire(expired['action_id'])=='expired'
    assert counts()==(0,0)


def test_confirmation_and_dispatch_deadlines_and_replay(sandbox,monkeypatch):
    row=prepared(10)
    now=actions.time.time()
    monkeypatch.setattr(actions.time,'time',lambda:now+301)
    with pytest.raises(actions.ActionError):confirmed(row)
    monkeypatch.setattr(actions.time,'time',lambda:now)
    confirmed(row)
    monkeypatch.setattr(actions.time,'time',lambda:now+301)
    assert confirmed(row)['status']=='accepted'
    assert actions.dispatch(row['action_id'])=='expired'
    assert counts()[1]==0


def test_creation_idempotency_and_atomic_approval_rollback(sandbox,monkeypatch):
    row=prepared();assert prepared()==row
    with pytest.raises(actions.ActionError): prepared(2)
    confirmed(row)
    def fail(db,row):raise RuntimeError('simulated database failure')
    monkeypatch.setattr(actions,'_accept',fail)
    with pytest.raises(RuntimeError):actions.review(row['action_id'],'demo-reviewer',decision(row))
    with actions.ledger() as db:
        saved=actions._row(db,row['action_id'])
        assert saved['decision'] is None and saved['state']=='pending_review'
    assert counts()==(0,0)


def test_http_auth_confirmation_review_and_signal_retry(sandbox):
    start=AsyncMock();signal=AsyncMock()
    api.app.dependency_overrides[api.get_transaction_starter]=lambda:start
    api.app.dependency_overrides[api.get_transaction_signaller]=lambda:signal
    try:
        http=TestClient(api.app)
        payload={'amount_minor':1_000_001,'target':'demo-beneficiary-a'}
        assert http.post('/v1/synthetic-actions',json=payload,headers={'Idempotency-Key':'p'}).status_code==403
        row=http.post('/v1/synthetic-actions',json=payload,headers=sandbox).json()
        path='/v1/synthetic-actions/'+row['action_id']
        body=confirmation(row).model_dump()
        assert http.post(path+'/confirm',json=body,headers=sandbox).status_code==202
        start.assert_awaited_once_with(row['action_id'])
        assert http.post(path+'/review',json=decision(row).model_dump(),headers=sandbox).status_code==403
        reviewer={'Authorization':'Bearer reviewer-synthetic-test-token'}
        signal.side_effect=RuntimeError('private secret failure')
        out=http.post(path+'/review',json=decision(row).model_dump(),headers=reviewer)
        assert out.status_code==503 and 'private' not in out.text
        assert actions.status(row['action_id'])['status']=='accepted'
        signal.side_effect=None
        assert http.post(path+'/review',json=decision(row).model_dump(),headers=reviewer).status_code==202
        assert counts()==(1,0)  # API cannot execute; only workflow activity dispatches.
        changed={**body,'amount_minor':3}
        assert http.post(path+'/confirm',json=changed,headers=sandbox).status_code==409
        assert http.get(path,headers=sandbox).status_code==200
        assert http.post('/v1/synthetic-actions',json={**payload,'target':'real-account'},headers=sandbox).status_code==422
    finally:api.app.dependency_overrides.clear()


def test_workflow_pause_wake_does_not_approve(sandbox,monkeypatch):
    row=prepared();confirmed(row)
    monkeypatch.setattr(flow,'triage_transaction',AsyncMock())
    monkeypatch.setattr(flow,'notify_transaction',AsyncMock())
    # Real ledger activities, mocked durable wait: this is NOT a live Studio test.
    waits=[]
    async def wait(predicate,timeout):
        assert counts()==(0,0)
        waits.append(True)
        if len(waits)==1:
            await instance.review_updated()  # forged wake alone must not dispatch
        else:
            actions.review(row['action_id'],'demo-reviewer',decision(row))
            await instance.review_updated()
        assert predicate()
    monkeypatch.setattr(workflows.workflow,'wait_condition',wait)
    instance=flow.SyntheticTransactionWorkflow()
    assert asyncio.run(instance.run(flow.TransactionInput(action_id=row['action_id'])))=={'result':'succeeded'}
    assert len(waits)==2 and counts()==(0,1)


def test_webhook_signature_redelivery_and_no_redirects(sandbox,monkeypatch):
    import httpx2
    row=prepared();confirmed(row)
    monkeypatch.setenv('DEMO_REVIEW_WEBHOOK_URL','https://review.example.test/inbox')
    monkeypatch.setenv('DEMO_REVIEW_WEBHOOK_SECRET','synthetic-webhook-secret-value')
    client=AsyncMock();client.__aenter__.return_value=client
    client.post.return_value=Mock(status_code=204)
    factory=Mock(return_value=client);monkeypatch.setattr(httpx2,'AsyncClient',factory)
    asyncio.run(actions.notify_reviewer(row['action_id']))
    kwargs=client.post.call_args.kwargs
    stamp=kwargs['headers']['X-Review-Timestamp']
    expected=hmac.new(b'synthetic-webhook-secret-value',stamp.encode()+b'.'+kwargs['content'],hashlib.sha256).hexdigest()
    assert kwargs['headers']['X-Review-Signature']==expected
    assert json.loads(kwargs['content'])['digest']==row['digest']
    assert factory.call_args.kwargs['follow_redirects'] is False
    client.post.return_value=Mock(status_code=302)
    with pytest.raises(actions.ActionError):asyncio.run(actions.notify_reviewer(row['action_id']))


def test_sdk_submit_signal_minimal_payload_and_registered_workflow(monkeypatch):
    # This checks the SDK payload only; database correlation is covered separately.
    monkeypatch.setattr(actions,'action_workflow_id',lambda _:None)
    monkeypatch.setenv('MISTRAL_STUDIO_API_KEY','synthetic-studio-key')
    client=AsyncMock();client.__aenter__.return_value=client
    monkeypatch.setattr(sdk,'Mistral',Mock(return_value=client))
    action_id='11111111-1111-4111-8111-111111111111'
    from datetime import datetime, timezone
    from mistralai.client.models import WorkflowExecutionResponse, SignalWorkflowResponse
    client.workflows.execute_workflow_async.return_value=WorkflowExecutionResponse(
        workflow_name=flow.TRANSACTION_WORKFLOW,execution_id=action_id,root_execution_id=action_id,
        status='RUNNING',start_time=datetime.now(timezone.utc),end_time=None,result=None)
    client.workflows.executions.signal_workflow_execution_async.return_value=SignalWorkflowResponse()
    asyncio.run(sdk.start_transaction_workflow(action_id))
    kwargs=client.workflows.execute_workflow_async.call_args.kwargs
    assert kwargs['input']=={'action_id':action_id}
    assert kwargs['execution_id']==action_id and kwargs['wait_for_result'] is False
    asyncio.run(sdk.signal_transaction_workflow(action_id))
    client.workflows.executions.signal_workflow_execution_async.assert_awaited_once_with(
        execution_id=action_id,name='review_updated',timeout_ms=5000,retries=None)
    definition=workflows.get_workflow_definition(flow.SyntheticTransactionWorkflow)
    assert definition.name==flow.TRANSACTION_WORKFLOW
    supervisor=sdk.build_support_supervisor()
    assert {child.name for child in supervisor.subagents}=={
        'FAQ','CardsGuidance','AccountGuidance','PaymentsGuidance','ProfileGuidance','HighStakeTransactions'}


def test_no_notification_or_approval_bypass_below_threshold(sandbox,monkeypatch):
    row=prepared(1_000_000);confirmed(row)
    factory=Mock(side_effect=AssertionError('No webhook needed'))
    import httpx2
    monkeypatch.setattr(httpx2,'AsyncClient',factory)
    asyncio.run(actions.notify_reviewer(row['action_id']))
    with pytest.raises(actions.ActionError):actions.review(row['action_id'],'demo-reviewer',decision(row))


def test_concurrent_reviews_and_restart_keep_one_effect(sandbox):
    from concurrent.futures import ThreadPoolExecutor
    row=prepared();confirmed(row)
    def approve(_):return actions.review(row['action_id'],'demo-reviewer',decision(row))['status']
    with ThreadPoolExecutor(4) as pool:assert list(pool.map(approve,range(8)))==['accepted']*8
    with ThreadPoolExecutor(4) as pool:assert list(pool.map(lambda _:actions.dispatch(row['action_id']),range(8)))==['succeeded']*8
    # Every operation closes and reopens PostgreSQL; a new connection retains evidence.
    assert actions.status(row['action_id'],'demo-customer')['status']=='succeeded'
    assert counts()==(0,1)


def test_triage_uses_only_public_template_and_rejects_wrong_child(monkeypatch):
    run=AsyncMock(return_value={'specialist':'HighStakeTransactions','handoffs':1})
    monkeypatch.setattr(sdk,'run_support_agents',run)
    asyncio.run(flow.triage_transaction())
    run.assert_awaited_once_with('Start a synthetic transaction with human approval.')
    run.return_value={'specialist':'PaymentsGuidance','handoffs':1}
    with pytest.raises(sdk.DemoError):asyncio.run(flow.triage_transaction())


def test_timeout_does_not_execute_and_notification_failure_stays_pending(sandbox,monkeypatch):
    row=prepared();confirmed(row)
    monkeypatch.setattr(flow,'triage_transaction',AsyncMock())
    monkeypatch.setattr(flow,'notify_transaction',AsyncMock(side_effect=RuntimeError('delivery failed')))
    instance=flow.SyntheticTransactionWorkflow()
    params=flow.TransactionInput(action_id=row['action_id'])
    with pytest.raises(RuntimeError):asyncio.run(instance.run(params))
    assert actions.status(row['action_id'])['status']=='pending_review' and counts()==(0,0)
    monkeypatch.setattr(flow,'notify_transaction',AsyncMock())
    async def timeout(*args,**kwargs):
        monkeypatch.setattr(actions.time,'time',lambda:10**12)
        raise asyncio.TimeoutError()
    monkeypatch.setattr(workflows.workflow,'wait_condition',timeout)
    assert asyncio.run(instance.run(params))=={'result':'expired'}
    assert counts()==(0,0)


def test_no_webhook_without_review_and_notification_config_fails_closed(sandbox,monkeypatch):
    low=prepared(1);confirmed(low)
    asyncio.run(actions.notify_reviewer(low['action_id']))
    high=prepared(key='high');confirmed(high)
    monkeypatch.delenv('DEMO_REVIEW_WEBHOOK_URL',raising=False)
    with pytest.raises(actions.ActionError):asyncio.run(actions.notify_reviewer(high['action_id']))
    assert actions.dispatch(high['action_id'])=='pending_review'


def test_model_transport_retries_count_toward_budget(monkeypatch):
    from uuid import uuid4
    request=sdk.SupportSessionInput(request_id=uuid4(),question='A public synthetic question')
    sdk.request_guard_create(request)
    token=sdk._native_request.set(request.request_id)
    original=AsyncMock(return_value=Mock())
    monkeypatch.setattr(sdk.MistralChatCompletionProvider,'_open_stream',original)
    provider=sdk.NativeChatCompletionProvider(api_key='synthetic',model='synthetic',retry=sdk.RetryConfig(max_attempts=1))
    try:
        for _ in range(6):asyncio.run(provider._open_stream({}))
        with pytest.raises(sdk.DemoError):asyncio.run(provider._open_stream({}))
        assert original.await_count==6
    finally:
        sdk._native_request.reset(token);sdk.request_guard_delete(request.request_id)
