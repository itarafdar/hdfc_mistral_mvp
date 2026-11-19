"""Use the actual pinned SDK with an offline HTTP transport (no credentials)."""
import json
from unittest.mock import Mock
import httpx2
import pytest
from mistralai.client import Mistral
from hdfc_mistral_mvp import sdk


@pytest.mark.parametrize('origin,model',[
    ('http://127.0.0.1:11434', 'ministral-3:14b'),
    ('http://127.0.0.1:8000', 'local-org/mistral-model'),
])
@pytest.mark.parametrize('content,valid',[
    ('{"specialist":"FAQ"}', True),
    ('{"specialist":"processor"}', False),
])
def test_agent_engine_independent_wire_contract(monkeypatch, origin, model, content, valid):
    """Run the actual pinned agent/session/stream parser against an offline server."""
    import httpx2 as httpx
    import time
    from hdfc_mistral_mvp.support import Route
    from unittest.mock import AsyncMock
    streams=[]
    def factory(**kwargs):
        client=Mistral(**kwargs)
        client.chat.stream_async=AsyncMock(wraps=client.chat.stream_async)
        streams.append(client.chat.stream_async)
        return client
    monkeypatch.setattr(sdk,'Mistral',factory)
    calls=[]
    async def send(self, request, **kwargs):
        calls.append(request)
        assert str(request.url).split('#')[0]==origin+'/v1/chat/completions'
        payload=json.loads(request.content)
        assert payload['model']==model
        assert payload['max_tokens']==80 and payload['temperature']==0
        assert payload['stream_options']=={'include_usage':True}
        assert not payload.get('tools')
        chunks=[{'id':'synthetic-agent','model':model,'choices':[{
            'index':0,'delta':{'content':content},'finish_reason':'stop'}]},
            {'id':'synthetic-agent','model':model,'choices':[],
             'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}}]
        wire=''.join('data: '+json.dumps(row)+'\n\n' for row in chunks)+'data: [DONE]\n\n'
        return httpx.Response(200,content=wire,headers={'content-type':'text/event-stream'},request=request)
    monkeypatch.setattr(httpx.AsyncClient,'send',send)
    monkeypatch.setattr(sdk,'dotenv_values',Mock(side_effect=AssertionError('No OCR key read')))
    monkeypatch.setenv('INFERENCE_URL',origin)
    monkeypatch.setenv('SUPPORT_INFERENCE_URL',origin)
    monkeypatch.setenv('SUPPORT_AGENT_MODEL',model)
    with sdk.support_budget(time.monotonic()+6) as counts:
        if valid:
            assert sdk.classify_support_agent('Public policy question',Route).specialist=='FAQ'
        else:
            with pytest.raises(sdk.DemoError): sdk.classify_support_agent('Public policy question',Route)
        assert counts=={'generation':1,'embedding':0}
    assert len(calls)==1
    assert len(streams)==1
    streams[0].assert_awaited_once()
    assert streams[0].call_args.kwargs['retries'] is None


@pytest.mark.parametrize('finish,content,valid',[
    ('stop','{"specialist":"FAQ"}',True),
    ('length','{"specialist":',False),
    ('stop','x'*513,False),
    ('stop',None,False),
])
def test_classification_sdk_contract(monkeypatch,finish,content,valid):
    calls=[]
    def handler(request):
        calls.append(request)
        payload=json.loads(request.content)
        assert request.url.path=='/v1/chat/completions'
        assert request.url.host=='127.0.0.1'
        assert payload['model']=='ministral-3:14b'
        assert payload['response_format']=={'type':'json_object'}
        assert payload['max_tokens']==80 and payload['temperature']==0
        assert 'tools' not in payload
        assert json.loads(payload['messages'][-1]['content'])=={'question':'Pause payments on the plastic'}
        return httpx2.Response(200,json={
            'id':'synthetic-classification','object':'chat.completion','created':0,'model':'ministral-3:14b',
            'choices':[{'index':0,'finish_reason':finish,'message':{'role':'assistant','content':content}}],
            'usage':{'prompt_tokens':10,'completion_tokens':4,'total_tokens':14}})
    monkeypatch.setattr(sdk,'dotenv_values',Mock(side_effect=AssertionError('No hosted key read')))
    ai=sdk.LocalAI()
    ai.http.close()
    ai.http=sdk.OllamaHTTPClient(transport=httpx2.MockTransport(handler))
    ai.client=Mistral(api_key='ollama-local-placeholder',server_url='http://127.0.0.1:11434',client=ai.http,retry_config=None)
    try:
        if valid: assert ai.classify('Pause payments on the plastic')==content
        else:
            with pytest.raises(sdk.DemoError): ai.classify('Pause payments on the plastic')
        assert len(calls)==1
    finally: ai.close()


def test_embedding_wire_contract():
    calls=[]
    def handler(request):
        calls.append(request)
        assert request.url.path=='/v1/embeddings'
        payload=json.loads(request.content)
        assert payload['input']==['synthetic policy']
        assert 'inputs' not in payload
        return httpx2.Response(200,json={'id':'emb-test','object':'list','model':'nomic-embed-text:latest','usage':{'prompt_tokens':2,'total_tokens':2},'data':[{'object':'embedding','index':0,'embedding':[0.1,0.2]}]})
    ai=sdk.LocalAI()
    ai.client=Mistral(api_key='local-placeholder',server_url='http://localhost:11434',client=httpx2.Client(transport=httpx2.MockTransport(handler)),retry_config=None)
    assert ai.embed(['synthetic policy'])==[[0.1,0.2]]
    assert len(calls)==1


def test_generation_contract_and_no_retry():
    calls=[]
    def handler(request):
        calls.append(request)
        payload=json.loads(request.content)
        assert request.url.path=='/v1/chat/completions'
        assert payload['max_tokens']==400 and payload['response_format']=={'type':'json_object'}
        assert 'tools' not in payload
        return httpx2.Response(503,json={'message':'synthetic unavailable'})
    ai=sdk.LocalAI()
    ai.client=Mistral(api_key='local-placeholder',server_url='http://localhost:11434',client=httpx2.Client(transport=httpx2.MockTransport(handler)),retry_config=None)
    with pytest.raises(Exception): ai.generate('public question',[])
    assert len(calls)==1


def test_ocr_sdk_shape_and_credential_isolation(tmp_path,monkeypatch):
    (tmp_path/'mistral_keys.env').write_text('MISTRAL_API_KEY=synthetic-test-placeholder\n')
    pdf=tmp_path/'synthetic.pdf'; pdf.write_bytes(b'%PDF-synthetic-test')
    monkeypatch.setattr(sdk,'ROOT',tmp_path)
    import hashlib
    digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
    (tmp_path/'demo').mkdir()
    (tmp_path/'demo/approved.json').write_text(json.dumps([{'document':'synthetic','title':'Policy','sha256':digest}]))
    real=Mistral; calls=[]
    def handler(request):
        calls.append(request)
        assert request.url.host=='api.mistral.ai' and request.url.path=='/v1/ocr'
        assert request.headers['authorization']=='Bearer synthetic-test-placeholder'
        payload=json.loads(request.content)
        assert payload['model']=='mistral-ocr-4-1'
        assert payload['document']['document_url'].startswith('data:application/pdf;base64,')
        return httpx2.Response(200,json={'model':'mistral-ocr-4-1','pages':[{'index':0,'markdown':'Synthetic policy text','images':[],'dimensions':None}],'usage_info':{'pages_processed':1,'doc_size_bytes':19}})
    monkeypatch.setattr(sdk,'Mistral',lambda **kw:real(**kw,client=httpx2.Client(transport=httpx2.MockTransport(handler))))
    data=sdk.hosted_ocr(pdf,'Policy','synthetic',digest)
    assert data['pages']==[{'page':1,'text':'Synthetic policy text'}]
    assert len(calls)==1


def test_ollama_missing_id_normalization():
    def handler(request):
        return httpx2.Response(200,json={'object':'list','model':'nomic-embed-text:latest','usage':{'prompt_tokens':1,'total_tokens':1},'data':[{'object':'embedding','index':0,'embedding':[0.25,0.75]}]})
    ai=sdk.LocalAI()
    ai.client=Mistral(api_key='local-placeholder',server_url='http://localhost:11434',client=sdk.OllamaHTTPClient(transport=httpx2.MockTransport(handler)),retry_config=None)
    assert ai.embed(['synthetic policy'])==[[0.25,0.75]]
