"""Network-free Workflows 3.20.0 contracts, not a live Studio proof."""
import asyncio
from unittest.mock import AsyncMock, Mock
import pytest
import mistralai.workflows as workflows
from mistralai.client.models import WorkflowExecutionSyncResponse
from hdfc_mistral_mvp import sdk
from hdfc_mistral_mvp.studio_workflow import (
    CustomerSupportWorkflow, StudioSupportInput, WORKFLOW_NAME,
)

@pytest.fixture
def studio(monkeypatch):
    monkeypatch.setenv('MISTRAL_STUDIO_API_KEY', 'synthetic-studio-only')
    monkeypatch.setenv('SUPPORT_SESSION_MODE', 'local')
    monkeypatch.delenv('SUPPORT_INFERENCE_DISABLED', raising=False)
    monkeypatch.setattr(sdk, 'dotenv_values', Mock(side_effect=AssertionError('No credential file')))

def test_sdk_definition_and_one_attempt_activity(monkeypatch, studio):
    spec = workflows.get_workflow_definition(CustomerSupportWorkflow)
    assert spec.name == WORKFLOW_NAME
    assert spec.execution_timeout.total_seconds() == 20
    assert spec.enforce_determinism
    run = AsyncMock(return_value={'status':'guidance','message':'Use the secure card page.'})
    monkeypatch.setattr(sdk, 'run_support_agents', run)
    result = asyncio.run(CustomerSupportWorkflow().run(StudioSupportInput(
        question='How do I temporarily deactivate a card?')))
    assert result['response']['status'] == 'guidance'
    run.assert_awaited_once_with('How do I temporarily deactivate a card?')
    run.reset_mock(); run.side_effect = sdk.DemoError('Unavailable')
    with pytest.raises(sdk.DemoError):
        asyncio.run(CustomerSupportWorkflow().run(StudioSupportInput(question='Is Riverside open on Saturdays?')))
    run.assert_awaited_once()  # no retry and no doubled model budget

def test_fastapi_lifespan_starts_worker_once_and_stops_it(monkeypatch):
    import asyncio
    from hdfc_mistral_mvp import api
    started=[]; stopped=[]
    async def start():
        started.append(True)
        return object()
    async def stop(task): stopped.append(task)
    monkeypatch.setattr(api,'start_studio_workflow_worker',start)
    monkeypatch.setattr(api,'stop_studio_workflow_worker',stop)
    async def exercise():
        async with api.lifespan(api.app):
            assert api.app.state.studio_worker_available
            assert api.app.state.studio_worker_task is started_task[0]
    started_task=[]
    async def start_with_task():
        started.append(True)
        task=object(); started_task.append(task); return task
    monkeypatch.setattr(api,'start_studio_workflow_worker',start_with_task)
    asyncio.run(exercise())
    assert started == [True] and stopped == started_task


def test_fastapi_lifespan_survives_missing_worker(monkeypatch):
    import asyncio
    from hdfc_mistral_mvp import api
    async def start(): raise sdk.DemoError('private startup error')
    stopped=[]
    monkeypatch.setattr(api,'start_studio_workflow_worker',start)
    async def stop(task): stopped.append(task)
    monkeypatch.setattr(api,'stop_studio_workflow_worker',stop)
    async def exercise():
        async with api.lifespan(api.app):
            assert not api.app.state.studio_worker_available
            assert api.app.state.studio_worker_task is None
    asyncio.run(exercise())
    assert stopped == [None]


def test_submission_validates_version_matched_result(monkeypatch, studio):
    public = {'status':'answered','answer':'Riverside is open on Saturdays.',
              'citations':[{'title':'Branch hours','page':1,'chunk_id':'branch-p1-s0'}],
              'specialist':'FAQ','handoffs':1,'orchestration':'mistral_agents_native',
              'model_calls':{'generation':4,'embedding':1}}
    execution = WorkflowExecutionSyncResponse(workflow_name=WORKFLOW_NAME,
        execution_id='synthetic', result={'response':public})
    client = AsyncMock(); client.__aenter__.return_value = client
    client.workflows.execute_workflow_async.return_value = execution
    factory = Mock(return_value=client); monkeypatch.setattr(sdk, 'Mistral', factory)
    result = asyncio.run(sdk.execute_studio_support('Is Riverside open on Saturdays?'))
    assert result == public
    factory.assert_called_once_with(api_key='synthetic-studio-only',
                                   server_url='https://api.mistral.ai', timeout_ms=19000)
    kwargs = client.workflows.execute_workflow_async.call_args.kwargs
    assert kwargs['input'] == {'question':'Is Riverside open on Saturdays?'}
    assert kwargs['wait_for_result'] is True
    execution.result = {'response':{**public,'answer':'OTP 123456'}}
    with pytest.raises(Exception):
        asyncio.run(sdk.execute_studio_support('Is Riverside open on Saturdays?'))

@pytest.mark.parametrize('question', ['OTP 123456', '   '])
def test_no_studio_submission_for_sensitive_or_empty_input(monkeypatch, studio, question):
    factory = Mock(side_effect=AssertionError('No network')); monkeypatch.setattr(sdk, 'Mistral', factory)
    with pytest.raises(Exception): asyncio.run(sdk.execute_studio_support(question))
    factory.assert_not_called()

def test_missing_dedicated_key_opens_no_connection(monkeypatch, studio):
    monkeypatch.delenv('MISTRAL_STUDIO_API_KEY')
    factory = Mock(side_effect=AssertionError('No network')); monkeypatch.setattr(sdk, 'Mistral', factory)
    with pytest.raises(sdk.DemoError):
        asyncio.run(sdk.execute_studio_support('Is Riverside open on Saturdays?'))
    factory.assert_not_called()


def test_default_http_route_uses_studio_workflow(monkeypatch, studio):
    from fastapi.testclient import TestClient
    from hdfc_mistral_mvp import api
    public = {'status':'guidance','message':'Use the secure card service.'}
    execution = WorkflowExecutionSyncResponse(workflow_name=WORKFLOW_NAME,
        execution_id='synthetic', result={'response':public})
    client = AsyncMock(); client.__aenter__.return_value = client
    client.workflows.execute_workflow_async.return_value = execution
    factory = Mock(return_value=client); monkeypatch.setattr(sdk, 'Mistral', factory)
    local = Mock(side_effect=AssertionError('API must not run agents directly'))
    monkeypatch.setattr(sdk, 'run_support_agents', local)
    async def start_worker(): return object()
    async def stop_worker(task): pass
    monkeypatch.setattr(api, 'start_studio_workflow_worker', start_worker)
    monkeypatch.setattr(api, 'stop_studio_workflow_worker', stop_worker)
    with TestClient(api.app) as http:
        for suffix in ('', '?stream=true'):
            response = http.post('/v1/customer-support'+suffix,
                json={'question':'How do I update contact details?'})
            assert response.status_code == 200
            if suffix: assert 'event: final' in response.text
            else: assert response.json() == public
        assert client.workflows.execute_workflow_async.await_count == 2
        # Outage: no local fallback, safe HTTP/SSE errors; urgent still available.
        client.workflows.execute_workflow_async.side_effect = RuntimeError('private')
        assert http.post('/v1/customer-support',json={'question':'Help please'}).status_code == 503
        out = http.post('/v1/customer-support?stream=true',json={'question':'Help please'})
        assert 'event: error' in out.text and 'private' not in out.text
        before = factory.call_count
        assert http.post('/v1/customer-support',json={'question':'Lost card'}).json()['status'] == 'urgent_support'
        assert http.post('/v1/customer-support',json={'question':'OTP 123456'}).status_code == 422
        assert factory.call_count == before
    local.assert_not_called()


def test_cancelled_submission_cancels_remote_execution(monkeypatch, studio):
    import asyncio
    client = AsyncMock(); client.__aenter__.return_value = client
    client.workflows.execute_workflow_async.side_effect = asyncio.CancelledError()
    monkeypatch.setattr(sdk, 'Mistral', Mock(return_value=client))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(sdk.execute_studio_support('Help please'))
    submitted = client.workflows.execute_workflow_async.call_args.kwargs['execution_id']
    client.workflows.executions.cancel_workflow_execution_async.assert_awaited_once_with(
        execution_id=submitted, timeout_ms=1000)
