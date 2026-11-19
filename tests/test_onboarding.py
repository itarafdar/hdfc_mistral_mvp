"""Network-free checks for data-only approval manifests and safe OCR diagnostics."""
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import httpx2
import pytest
from mistralai.client import Mistral
from hdfc_mistral_mvp import sdk, faq
from hdfc_mistral_mvp.cli import main


@pytest.fixture
def new_document(tmp_path):
    pdf=tmp_path/'policy.pdf'; pdf.write_bytes(b'%PDF-approved-synthetic')
    digest=hashlib.sha256(pdf.read_bytes()).hexdigest()
    manifest=tmp_path/'approved.json'
    manifest.write_text(json.dumps([{'document':'new-policy','title':'Synthetic New Policy','file':str(pdf),'sha256':digest}]))
    extraction={'document':'new-policy','title':'Synthetic New Policy','sha256':digest,
                'pages':[{'page':1,'text':'Policy guides are delivered within five working days.'}]}
    return pdf,manifest,extraction


def test_custom_approval_manifest_and_stable_chunks(new_document):
    pdf,manifest,data=new_document
    assert faq.approved('new-policy',manifest)[1]==pdf
    first=faq.chunks(data,manifest)
    assert first==faq.chunks(data,manifest)
    assert first[0]['title']=='Synthetic New Policy' and first[0]['page']==1
    pdf.write_bytes(b'%PDF-modified')
    with pytest.raises(sdk.DemoError,match='hash mismatch'): faq.chunks(data,manifest)


def test_cli_manifest_requires_extraction(monkeypatch):
    monkeypatch.setattr('sys.argv',['faq','ingest','--manifest','unused.json'])
    with pytest.raises(SystemExit) as exc: main()
    assert exc.value.code==2


@pytest.mark.parametrize('retry_after,expected',[('12','Retry-After: 12'),(None,'No numeric Retry-After'),('private-header-content','No numeric Retry-After')])
def test_ocr_rate_limit_diagnostics_are_sanitized(new_document,tmp_path,monkeypatch,retry_after,expected):
    pdf,manifest,data=new_document
    (tmp_path/'mistral_keys.env').write_text('MISTRAL_API_KEY=synthetic-test-placeholder\n')
    monkeypatch.setattr(sdk,'ROOT',tmp_path)
    calls=[]
    def handler(request):
        calls.append(request)
        headers={} if retry_after is None else {'Retry-After':retry_after}
        return httpx2.Response(429,headers=headers,json={'message':'private body must not be shown'})
    real=Mistral
    monkeypatch.setattr(sdk,'Mistral',lambda **kw:real(**kw,client=httpx2.Client(transport=httpx2.MockTransport(handler))))
    with pytest.raises(sdk.DemoError) as exc:
        sdk.hosted_ocr(pdf,data['title'],data['document'],data['sha256'],manifest)
    message=str(exc.value)
    assert 'HTTP 429' in message and expected in message
    assert 'private' not in message and 'synthetic-test-placeholder' not in message
    assert len(calls)==1
