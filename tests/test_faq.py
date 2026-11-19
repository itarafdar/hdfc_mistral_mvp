import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from hdfc_mistral_mvp import sdk
from hdfc_mistral_mvp.faq import FALLBACK, SUPPORTED, UNSUPPORTED, answer, approved, chunks, fixtures, screen
from hdfc_mistral_mvp.fixture import FixtureAI, MemoryStore

@pytest.fixture
def pipeline():
    ai=FixtureAI()
    return ai,MemoryStore(fixtures(),ai)

def test_supported_and_missing(pipeline):
    ai,store=pipeline
    assert 'Synthetic Statement Policy, page 1' in answer(SUPPORTED,ai,store)
    assert answer(UNSUPPORTED,ai,store)==FALLBACK

@pytest.mark.parametrize('raw',[
    '{"chunk_id":"invented","quote":"Customers can download electronic statements."}',
    '{"chunk_id":"statements-p1-s0-265d5081ee03","quote":"The home loan rate is 2 percent."}',
    '{"chunk_id":1,"quote":null}', 'not JSON', '[]',
])
def test_bad_generation_is_rejected(pipeline,raw):
    ai,store=pipeline; ai.generate=lambda *args:raw
    assert answer(SUPPORTED,ai,store)==FALLBACK

@pytest.mark.parametrize('text',['My OTP is 123456','1234 5678 9012','foo@example.test','ABCDE1234F','my balance is 50','password secret'])
def test_sensitive_input_never_calls_ai(text):
    ai=Mock(); store=Mock()
    with pytest.raises(sdk.DemoError): answer(text,ai,store)
    ai.embed.assert_not_called(); ai.generate.assert_not_called()

def test_empty_weak_and_scanner_failure(pipeline,monkeypatch):
    ai,store=pipeline; ai.generate=Mock()
    store.retrieve=lambda v:[]
    assert answer(SUPPORTED,ai,store)==FALLBACK
    store.retrieve=lambda v:[dict(fixtures()[0],distance=0.9)]
    assert answer(SUPPORTED,ai,store)==FALLBACK
    ai.generate.assert_not_called()
    monkeypatch.setattr('hdfc_mistral_mvp.faq.screen',Mock(side_effect=sdk.DemoError('scanner unavailable')))
    ai.embed=Mock()
    with pytest.raises(sdk.DemoError): answer(SUPPORTED,ai,store)
    ai.embed.assert_not_called()

def test_stable_ids_and_approval():
    a=fixtures(); assert a==fixtures(); assert len(a)>5
    assert len({r['id'] for r in a})==len(a)
    assert all(len([r for r in a if r['document']==doc])>1 for doc in ('statements','accessibility'))
    with pytest.raises(sdk.DemoError): approved('../../private.pdf')
    data=json.loads((sdk.ROOT/'demo/extractions/statements.json').read_text()); data['sha256']='bad'
    with pytest.raises(sdk.DemoError): chunks(data)

def test_missing_key_file_and_key(tmp_path,monkeypatch):
    import hashlib
    pdf=tmp_path/'synthetic.pdf'; pdf.write_bytes(b'%PDF-test')
    digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
    (tmp_path/'demo').mkdir()
    (tmp_path/'demo/approved.json').write_text(json.dumps([{'document':'id','title':'title','sha256':digest}]))
    monkeypatch.setattr(sdk,'ROOT',tmp_path)
    for content in (None,'OTHER=value','MISTRAL_API_KEY='):
        if content is not None: (tmp_path/'mistral_keys.env').write_text(content)
        with pytest.raises(sdk.DemoError,match='mistral_keys.env'):
            sdk.hosted_ocr(pdf,'title','id',digest)

def test_local_client_never_reads_hosted_key(monkeypatch):
    monkeypatch.setattr(sdk,'dotenv_values',Mock(side_effect=AssertionError('must not read key')))
    monkeypatch.setenv('MISTRAL_API_KEY','synthetic-not-a-secret')
    factory=Mock(); monkeypatch.setattr(sdk,'Mistral',factory)
    sdk.LocalAI()
    assert factory.call_args.kwargs['api_key']=='ollama-local-placeholder'
    assert factory.call_args.kwargs['retry_config'] is None

@pytest.mark.parametrize('url',['https://api.mistral.ai','http://evil.test','http://localhost:11434/v1'])
def test_local_endpoint_restriction(url,monkeypatch):
    # Exercise the legacy fallback even when runtime .env sets the canonical URL.
    monkeypatch.delenv('INFERENCE_URL',raising=False)
    monkeypatch.setenv('OLLAMA_URL',url)
    with pytest.raises(sdk.DemoError): sdk.LocalAI()


def test_demo_pdfs_are_image_only():
    for document in ('statements','complaints','accessibility','riverside'):
        _,pdf=approved(document)
        data=pdf.read_bytes()
        assert b'/Subtype /Image' in data
        assert b'/Font' not in data


def test_sensitive_generated_quote_is_not_displayed(pipeline):
    ai,store=pipeline
    ai.generate=lambda *args:json.dumps({'chunk_id':fixtures()[0]['id'],'quote':'My OTP is 123456 and should never be rendered.'})
    assert answer(SUPPORTED,ai,store)==FALLBACK
