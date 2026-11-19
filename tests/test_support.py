import json
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from hdfc_mistral_mvp import api, support

@pytest.fixture
def client():
    runner=Mock(return_value={'status':'not_found','message':'I couldn’t find that in the approved documents.','citations':[]})
    routes={'Temporarily deactivate a card':'CardsGuidance', 'Help please':'clarify'}
    classifier=Mock(side_effect=lambda question:json.dumps({'specialist':routes.get(question,'FAQ')}))
    async def workflow(question):
        # Explicit orchestration double. Native SDK protocol is tested separately.
        route=support.triage(question,classifier)
        result=support.dispatch(route,question,runner)
        return {**result,'orchestration':'mistral_agents_native','specialist':route.specialist}
    api.app.dependency_overrides[api.get_runner]=lambda:runner
    api.app.dependency_overrides[api.get_support_runner]=lambda:workflow
    with TestClient(api.app) as c:
        c.classifier=classifier
        yield c,runner
    api.app.dependency_overrides.clear()

@pytest.mark.parametrize('query',['','?stream=false'])
def test_json_default(client,query):
    c,r=client
    out=c.post('/v1/customer-support'+query,json={'question':'What is the home loan interest rate?'},headers={'Accept':'text/event-stream'})
    assert out.headers['content-type']=='application/json'
    assert out.json()['status']=='not_found'; r.assert_called_once()

@pytest.mark.parametrize('question,status',[
 ('Temporarily deactivate a card','guidance'),('Help please','clarification_required'),
 ('Lost card','urgent_support'),('Suspected fraud','urgent_support')])
def test_named_routes_and_urgent_without_inference(client,monkeypatch,question,status):
    c,r=client
    if status=='urgent_support': monkeypatch.setenv('SUPPORT_INFERENCE_DISABLED','1')
    out=c.post('/v1/customer-support',json={'question':question})
    assert out.json()['status']==status; r.assert_not_called()
    assert c.classifier.call_count == (0 if status=='urgent_support' else 1)

def test_sensitive_before_runner_and_stream(client):
    c,r=client
    out=c.post('/v1/customer-support?stream=true',json={'question':'My OTP is 123456'})
    assert out.status_code==422 and '123456' not in out.text; r.assert_not_called()
    c.classifier.assert_not_called()

def test_scanner_failure_urgent_reachable(client,monkeypatch):
    c,r=client
    monkeypatch.setattr(support,'screen',Mock(side_effect=RuntimeError('private')))
    assert c.post('/v1/customer-support',json={'question':'Lost card'}).json()['status']=='urgent_support'
    assert c.post('/v1/customer-support',json={'question':'Public question'}).status_code==503
    assert c.get('/v1/support/urgent').status_code==200
    r.assert_not_called()
    c.classifier.assert_not_called()

def test_sse_final_and_error(client):
    c,r=client
    out=c.post('/v1/customer-support?stream=true',json={'question':'How to download statements?'})
    assert out.text.count('event: final')==1 and out.text.count('event: progress')==1
    assert 'event: error' not in out.text
    r.side_effect=RuntimeError('private diagnostic')
    out=c.post('/v1/customer-support?stream=true',json={'question':'How to download statements?'})
    assert out.text.count('event: error')==1 and 'event: final' not in out.text
    assert 'private diagnostic' not in out.text

def test_bad_output_not_streamed(client):
    c,r=client; r.return_value={'status':'answered','answer':'My OTP is 123456','citations':[]}
    out=c.post('/v1/customer-support?stream=true',json={'question':'How to download statements?'})
    assert 'event: error' in out.text and '123456' not in out.text

def test_allowlist():
    with pytest.raises(ValueError): support.Route(specialist='processor')

def test_support_transport_budget_is_enforced():
    import time
    import httpx2
    from hdfc_mistral_mvp.sdk import OllamaHTTPClient, support_budget, DemoError
    sent=[]
    def handler(request):
        sent.append(request)
        return httpx2.Response(200,json={'ok':True})
    with OllamaHTTPClient(transport=httpx2.MockTransport(handler)) as http:
        with support_budget(time.monotonic()+6) as counts:
            http.post('http://127.0.0.1:11434/v1/chat/completions')
            http.post('http://127.0.0.1:11434/v1/chat/completions')
            with pytest.raises(DemoError): http.post('http://127.0.0.1:11434/v1/chat/completions')
            http.post('http://127.0.0.1:11434/v1/embeddings')
            with pytest.raises(DemoError): http.post('http://127.0.0.1:11434/v1/embeddings')
            assert counts=={'generation':2,'embedding':1} and len(sent)==3
        with support_budget(time.monotonic()-1):
            with pytest.raises(DemoError): http.post('http://127.0.0.1:11434/v1/chat/completions')
        assert len(sent)==3

@pytest.mark.parametrize('raw',[
    '{"specialist":"processor"}', '{"specialist":"FAQ","tool":"execute"}',
    '{"specialist":', '["FAQ"]', '{"specialist":null}'
])
@pytest.mark.parametrize('stream',[False,True])
def test_invalid_classification_never_dispatches(client,raw,stream):
    c,r=client; c.classifier.side_effect=None; c.classifier.return_value=raw
    out=c.post('/v1/customer-support'+('?stream=true' if stream else ''),json={'question':'Public question'})
    if stream:
        assert out.text.count('event: error')==1 and 'event: final' not in out.text
    else: assert out.status_code==503 and out.json()['status']=='unavailable'
    assert 'processor' not in out.text and 'execute' not in out.text
    r.assert_not_called()

@pytest.mark.parametrize('stream',[False,True])
def test_classifier_failure_is_sanitized(client,stream):
    c,r=client; c.classifier.side_effect=TimeoutError('private classifier trace')
    out=c.post('/v1/customer-support'+('?stream=true' if stream else ''),json={'question':'Public question'})
    assert ('event: error' in out.text) if stream else out.status_code==503
    assert 'private classifier trace' not in out.text
    r.assert_not_called()

def test_inference_disabled_blocks_classifier(client,monkeypatch):
    c,r=client; monkeypatch.setenv('SUPPORT_INFERENCE_DISABLED','1')
    out=c.post('/v1/customer-support',json={'question':'Help please'})
    assert out.status_code==503
    c.classifier.assert_not_called(); r.assert_not_called()

def test_semantic_paraphrase_is_dispatched(client):
    c,r=client; c.classifier.side_effect=None
    c.classifier.return_value='{"specialist":"CardsGuidance"}'
    question='Stop my plastic from being used for a while'
    out=c.post('/v1/customer-support',json={'question':question})
    assert out.json()['status']=='guidance' and out.json()['orchestration']=='mistral_agents_native'
    c.classifier.assert_called_once_with(question); r.assert_not_called()
