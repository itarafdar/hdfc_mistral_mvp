import asyncio
import json
from unittest.mock import AsyncMock
import pytest
from fastapi.testclient import TestClient
from hdfc_mistral_mvp import actions,api,sdk

@pytest.fixture
def sandbox(monkeypatch,postgres_ledger):
    monkeypatch.setenv('DEMO_CUSTOMER_TOKEN','synthetic-customer-token-12345')
    monkeypatch.setenv('DEMO_REVIEWER_TOKEN','synthetic-reviewer-token-12345')
    monkeypatch.delenv('SUPPORT_INFERENCE_DISABLED',raising=False)
    return {'Authorization':'Bearer synthetic-customer-token-12345','Idempotency-Key':'semantic-request'}


def test_customer_support_starts_workflow_without_action_api(sandbox,monkeypatch):
    starter=AsyncMock();monkeypatch.setattr(api,'start_agent_support_workflow',starter)
    http=TestClient(api.app)
    question='In the synthetic demo, transfer INR 15000 to demo-beneficiary-a.'
    response=http.post('/v1/customer-support',json={'question':question},headers=sandbox)
    assert response.status_code==202
    reference=response.json()['request_id'];starter.assert_awaited_once_with(reference)
    assert actions.agent_request(reference)['action_id'] is None
    assert http.get(response.json()['status_url'],headers=sandbox).json()=={'result':{'status':'processing'}}
    assert http.get(response.json()['status_url']).status_code==403
    assert http.post('/v1/customer-support',json={'question':question},headers=sandbox).json()['request_id']==reference
    assert http.post('/v1/customer-support',json={'question':question+' changed'},headers=sandbox).status_code==409
    before=starter.await_count
    assert http.post('/v1/customer-support',json={'question':'OTP 123456'},headers=sandbox).status_code==422
    assert starter.await_count==before


@pytest.mark.parametrize('amount,expected',[(999900,'awaiting_confirmation'),(1000000,'awaiting_confirmation'),(1500000,'pending_review')])
def test_threshold_grounding_and_replay(sandbox,amount,expected):
    question=f'In the synthetic demo, transfer INR {amount/100:.2f} to demo-beneficiary-a.'
    reference=actions.create_agent_request('demo-customer','one',question)
    transfer=actions.Transfer(amount_minor=amount,target='demo-beneficiary-a')
    row=actions.prepare_from_agent(reference,transfer)
    assert row['status']==expected
    assert actions.prepare_from_agent(reference,transfer)==row
    assert actions.action_workflow_id(row['action_id'])==reference
    assert actions.dispatch(row['action_id'])==expected
    with pytest.raises(actions.ActionError):actions.prepare_from_agent(reference,transfer.model_copy(update={'amount_minor':123}))
    with pytest.raises(actions.ActionError):actions.agent_request(reference,'other-customer')


def test_confirmation_wakes_existing_workflow_instead_of_starting(sandbox,monkeypatch):
    reference=actions.create_agent_request('demo-customer','one','Synthetic transfer INR 15000 to demo-beneficiary-a.')
    row=actions.prepare_from_agent(reference,actions.Transfer(amount_minor=1500000,target='demo-beneficiary-a'))
    starter=AsyncMock();signal=AsyncMock()
    api.app.dependency_overrides[api.get_transaction_starter]=lambda:starter
    monkeypatch.setattr(api,'signal_transaction_workflow',signal)
    try:
        body={k:row[k] for k in ('amount_minor','currency','target','version','digest')}
        result=TestClient(api.app).post('/v1/synthetic-actions/'+row['action_id']+'/confirm',json=body,headers=sandbox)
        assert result.status_code==202
        starter.assert_not_awaited();signal.assert_awaited_once_with(row['action_id'])
    finally:api.app.dependency_overrides.clear()


def test_reviewer_signal_uses_original_workflow_reference(sandbox,monkeypatch):
    reference=actions.create_agent_request('demo-customer','one','Synthetic transfer INR 15000 to demo-beneficiary-a.')
    row=actions.prepare_from_agent(reference,actions.Transfer(amount_minor=1500000,target='demo-beneficiary-a'))
    monkeypatch.setenv('MISTRAL_STUDIO_API_KEY','synthetic')
    client=AsyncMock();client.__aenter__.return_value=client
    from mistralai.client.models import SignalWorkflowResponse
    client.workflows.executions.signal_workflow_execution_async.return_value=SignalWorkflowResponse()
    monkeypatch.setattr(sdk,'Mistral',lambda **kwargs:client)
    asyncio.run(sdk.signal_transaction_workflow(row['action_id']))
    assert client.workflows.executions.signal_workflow_execution_async.call_args.kwargs['execution_id']==reference


def test_studio_submission_contains_only_opaque_request_reference(sandbox,monkeypatch):
    from datetime import datetime, timezone
    from mistralai.client.models import WorkflowExecutionResponse
    from hdfc_mistral_mvp.studio_workflow import AGENT_TRANSACTION_WORKFLOW
    reference=actions.create_agent_request('demo-customer','one','Synthetic transfer INR 15000 to demo-beneficiary-a.')
    monkeypatch.setenv('MISTRAL_STUDIO_API_KEY','synthetic')
    client=AsyncMock();client.__aenter__.return_value=client
    client.workflows.execute_workflow_async.return_value=WorkflowExecutionResponse(
        workflow_name=AGENT_TRANSACTION_WORKFLOW,execution_id=reference,root_execution_id=reference,
        status='RUNNING',start_time=datetime.now(timezone.utc),end_time=None,result=None)
    monkeypatch.setattr(sdk,'Mistral',lambda **kwargs:client)
    asyncio.run(sdk.start_agent_support_workflow(reference))
    kwargs=client.workflows.execute_workflow_async.call_args.kwargs
    assert kwargs['input']=={'request_id':reference}
    assert kwargs['execution_id']==reference and kwargs['wait_for_result'] is False


def test_review_first_cannot_execute_without_exact_customer_consent(sandbox,monkeypatch):
    reference=actions.create_agent_request('demo-customer','one','Synthetic transfer INR 15000 to demo-beneficiary-a.')
    row=actions.prepare_from_agent(reference,actions.Transfer(amount_minor=1500000,target='demo-beneficiary-a'))
    decision=actions.Review(digest=row['digest'],decision='approve',reason='synthetic_policy_checked')
    actions.review(row['action_id'],'demo-reviewer',decision)
    assert actions.dispatch(row['action_id'])=='awaiting_confirmation'
    with actions.ledger() as db:
        assert actions._row(db,row['action_id'])['decision'] is None  # provisional human vote only
        assert db.execute('SELECT COUNT(*) AS count FROM outbox').fetchone()['count']==0
    body=actions.Confirmation(**{k:row[k] for k in ('amount_minor','currency','target','digest','version')})
    with pytest.raises(actions.ActionError):actions.confirm(row['action_id'],'demo-customer','consent',body.model_copy(update={'amount_minor':100}))
    now=actions.time.time();monkeypatch.setattr(actions.time,'time',lambda:now+301)
    with pytest.raises(actions.ActionError):actions.confirm(row['action_id'],'demo-customer','consent',body)
    assert actions.expire(row['action_id'])=='expired'
    assert actions.dispatch(row['action_id'])=='expired'


def test_rejection_during_agent_review_never_executes(sandbox):
    reference=actions.create_agent_request('demo-customer','one','Synthetic transfer INR 15000 to demo-beneficiary-a.')
    row=actions.prepare_from_agent(reference,actions.Transfer(amount_minor=1500000,target='demo-beneficiary-a'))
    actions.review(row['action_id'],'demo-reviewer',actions.Review(digest=row['digest'],decision='reject',reason='synthetic_policy_failed'))
    body=actions.Confirmation(**{k:row[k] for k in ('amount_minor','currency','target','digest','version')})
    with pytest.raises(actions.ActionError):actions.confirm(row['action_id'],'demo-customer','consent',body)
    assert actions.dispatch(row['action_id'])=='rejected'


def test_webhook_acknowledgement_is_not_approval_and_callback_resumes(sandbox,monkeypatch):
    import httpx2
    from unittest.mock import Mock
    reference=actions.create_agent_request('demo-customer','one','Synthetic transfer INR 15000 to demo-beneficiary-a.')
    row=actions.prepare_from_agent(reference,actions.Transfer(amount_minor=1500000,target='demo-beneficiary-a'))
    monkeypatch.setenv('DEMO_REVIEW_WEBHOOK_URL','https://review.example.test/inbox')
    monkeypatch.setenv('DEMO_REVIEW_WEBHOOK_SECRET','synthetic-notification-secret')
    monkeypatch.setenv('DEMO_REVIEW_CALLBACK_BASE_URL','https://api.example.test')
    client=AsyncMock();client.__aenter__.return_value=client
    client.post.return_value=Mock(status_code=200)  # delivery receipt ONLY
    monkeypatch.setattr(httpx2,'AsyncClient',lambda **kwargs:client)
    asyncio.run(actions.notify_reviewer(row['action_id']))
    payload=json.loads(client.post.call_args.kwargs['content'])
    assert payload['callback_url']=='https://api.example.test/v1/webhooks/human-review/'+row['action_id']
    assert actions.status(row['action_id'])['status']=='pending_review'
    assert actions.dispatch(row['action_id'])=='pending_review'
    signal=AsyncMock();api.app.dependency_overrides[api.get_transaction_signaller]=lambda:signal
    try:
        callback='/v1/webhooks/human-review/'+row['action_id']
        body={'digest':row['digest'],'decision':'approve','reason':'synthetic_policy_checked'}
        http=TestClient(api.app)
        assert http.post(callback,json=body).status_code==403
        assert http.post(callback,json=body,headers=sandbox).status_code==403
        reviewer={'Authorization':'Bearer synthetic-reviewer-token-12345'}
        assert http.post(callback,json={**body,'digest':'0'*64},headers=reviewer).status_code==409
        signal.assert_not_awaited()
        assert http.post(callback,json=body,headers=reviewer).status_code==202
        signal.assert_awaited_once_with(row['action_id'])
        assert actions.status(row['action_id'])['status']=='awaiting_confirmation'
        assert actions.dispatch(row['action_id'])=='awaiting_confirmation'
        assert http.post(callback,json=body,headers=reviewer).status_code==202  # exact replay safe
        assert http.post(callback,json={**body,'decision':'reject'},headers=reviewer).status_code==409
    finally:api.app.dependency_overrides.clear()


def test_no_handoff_for_incomplete_request(sandbox,monkeypatch):
    from hdfc_mistral_mvp import studio_workflow as flow
    run=AsyncMock(return_value=flow.SubagentHandoff())
    notification=AsyncMock()
    monkeypatch.setattr(flow,'run_authenticated_support',run)
    monkeypatch.setattr(flow,'notify_transaction',notification)
    instance=flow.AgentTransactionWorkflow()
    result=asyncio.run(instance.run(flow.AgentRequestInput(request_id='11111111-1111-4111-8111-111111111111')))
    assert result=={'result':'completed_without_action'}
    notification.assert_not_awaited()
