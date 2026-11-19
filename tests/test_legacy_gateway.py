import asyncio
import json

import httpx2
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock

from hdfc_mistral_mvp import legacy_gateway as gateway
from hdfc_mistral_mvp.mock_legacy import app


@pytest.mark.parametrize('operation',list(gateway.STEPS))
def test_gateway_contract_and_adapter(operation,monkeypatch):
    monkeypatch.setenv('DEMO_LEGACY_GATEWAY_URL','embedded')
    service=gateway.STEPS[operation][0]
    r=TestClient(app).post('/legacy/v1/'+service+'/procedure',
        json={'service':service,'operation':operation})
    assert r.status_code==200
    assert r.json()['confirmation_required'] is True
    result=asyncio.run(gateway.get_legacy_guidance(service,operation))
    assert result['status']=='guidance' and result['message'].startswith('Synthetic legacy gateway checked.')


@pytest.mark.parametrize('payload',[
    {'service':'cards','operation':'money_transfer'},
    {'service':'cards','operation':'permanent_block'},
    {'service':'cards','operation':'temporary_deactivation','account':'synthetic-id'},
    {'service':'payments','operation':'money_transfer'},
])
def test_gateway_rejects_unscoped_requests_without_echo(payload):
    r=TestClient(app).post('/legacy/v1/cards/procedure',json=payload)
    assert r.status_code==422 and r.json()=={'status':'invalid_request'}


@pytest.mark.parametrize('change',[
    {'confirmation_required':False}, {'verification_required':False},
    {'system':'mock_payment_hub'}, {'operation':'money_transfer'},
    {'next_step':'secure_transfer_service'}, {'status':'unavailable'},
    {'environment':'production'}, {'secret':'do-not-log'},
])
def test_untrusted_response_fails_closed(change,monkeypatch,caplog):
    value=TestClient(app).post('/legacy/v1/cards/procedure',json={
        'service':'cards','operation':'temporary_deactivation'}).json()
    async def send(self,request,**kwargs):
        assert json.loads(request.content)=={'service':'cards','operation':'temporary_deactivation'}
        return httpx2.Response(200,json={**value,**change},request=request)
    monkeypatch.setattr(httpx2.AsyncClient,'send',send)
    with pytest.raises(gateway.LegacyUnavailable) as exc:
        asyncio.run(gateway.get_legacy_guidance('cards','temporary_deactivation'))
    assert str(exc.value)=='Synthetic bank gateway is temporarily unavailable.'
    assert 'do-not-log' not in caplog.text


@pytest.mark.parametrize('status',[302,401,503])
def test_http_failure_no_retry(status,monkeypatch):
    calls=[]
    async def send(self,request,**kwargs):
        calls.append(request)
        return httpx2.Response(status,text='private-error',request=request)
    monkeypatch.setattr(httpx2.AsyncClient,'send',send)
    with pytest.raises(gateway.LegacyUnavailable):
        asyncio.run(gateway.get_legacy_guidance('payments','money_transfer'))
    assert len(calls)==1


def test_timeout_cancels_gateway(monkeypatch):
    cancelled=[]
    async def send(self,request,**kwargs):
        try: await asyncio.sleep(60)
        finally: cancelled.append(True)
    monkeypatch.setattr(httpx2.AsyncClient,'send',send)
    with pytest.raises(gateway.LegacyUnavailable):
        asyncio.run(gateway.get_legacy_guidance('cards','temporary_deactivation',timeout=.01))
    assert cancelled


@pytest.mark.parametrize('origin',['https://example.test','http://127.0.0.1:8030/other',
    'http://user:pass@localhost:8030','http://localhost:8030?target=elsewhere'])
def test_model_cannot_pick_destination(origin,monkeypatch):
    monkeypatch.setenv('DEMO_LEGACY_GATEWAY_URL',origin)
    send=Mock(side_effect=AssertionError('No HTTP'))
    monkeypatch.setattr(httpx2.AsyncClient,'send',send)
    with pytest.raises(gateway.LegacyUnavailable):
        asyncio.run(gateway.get_legacy_guidance('cards','temporary_deactivation'))
    send.assert_not_called()
