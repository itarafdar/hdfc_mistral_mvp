"""Network-free HTTP tests calling the real RAG service with fixture dependencies."""
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from hdfc_mistral_mvp import api, sdk
from hdfc_mistral_mvp.faq import FALLBACK, answer_result, fixtures
from hdfc_mistral_mvp.fixture import FixtureAI, MemoryStore

RIVERSIDE = 'Is Riverside open on Saturdays?'


@pytest.fixture
def http_pipeline():
    ai = FixtureAI()
    store = MemoryStore(fixtures(), ai)
    runner = Mock(side_effect=lambda question: answer_result(question, ai, store))
    api.app.dependency_overrides[api.get_runner] = lambda: runner
    with TestClient(api.app) as client:
        yield client, ai, store, runner
    api.app.dependency_overrides.clear()


def test_answer_has_stored_title_and_page(http_pipeline):
    client, ai, store, runner = http_pipeline
    response = client.post('/v1/faq', json={'question': RIVERSIDE})
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'answered'
    assert 'Riverside is open on Saturdays' in data['answer']
    expected = next(r for r in fixtures() if r['document'] == 'riverside')
    assert data['citations'] == [{'title': expected['title'], 'page': 1, 'chunk_id': expected['id']}]
    assert 'Source:' not in data['answer']


@pytest.mark.parametrize('weak', [False, True])
def test_missing_evidence(http_pipeline, weak):
    client, ai, store, runner = http_pipeline
    if weak:
        store.retrieve = lambda vector: [dict(fixtures()[0], distance=0.95)]
    response = client.post('/v1/faq', json={'question': 'What is the home loan interest rate?'})
    assert response.status_code == 200
    assert response.json() == {'status': 'not_found', 'message': FALLBACK, 'citations': []}


@pytest.mark.parametrize('dependency', ['retrieve', 'embed', 'generate'])
def test_dependency_failure_is_safe_503(http_pipeline, dependency):
    client, ai, store, runner = http_pipeline
    target = store if dependency == 'retrieve' else ai
    setattr(target, dependency, Mock(side_effect=RuntimeError('private SDK body and synthetic secret')))
    response = client.post('/v1/faq', json={'question': RIVERSIDE})
    assert response.status_code == 503
    assert response.json() == {'status': 'unavailable', 'message': api.FAQ_RETRY}
    assert response.headers['retry-after'] == '5'
    assert 'private' not in response.text and 'secret' not in response.text


@pytest.mark.parametrize('body', [
    {}, {'question': ''}, {'question': '   '}, {'question': 123},
    {'question': 'x' * 2001}, {'question': 'My OTP is 123456'},
    {'question': 'valid', 'credential': 'synthetic-secret'},
])
def test_input_rejected_before_dependencies_without_echo(http_pipeline, body):
    client, ai, store, runner = http_pipeline
    response = client.post('/v1/faq', json=body)
    assert response.status_code == 422
    assert response.json() == api.INVALID
    runner.assert_not_called()
    assert '123456' not in response.text and 'synthetic-secret' not in response.text


def test_bad_model_output_is_not_found(http_pipeline):
    client, ai, store, runner = http_pipeline
    ai.generate = lambda *args: '{"chunk_id":"invented","quote":"Invented policy answer."}'
    response = client.post('/v1/faq', json={'question': RIVERSIDE})
    assert response.status_code == 200
    assert response.json() == {'status': 'not_found', 'message': FALLBACK, 'citations': []}


def test_route_opens_existing_collection_and_never_loads_ocr_key(monkeypatch):
    ai = Mock(embedding_model='nomic-embed-text:latest')
    ai.embed.return_value = [[0.0, 1.0]]
    store = Mock()
    store.retrieve.return_value = []
    factory = Mock(return_value=store)
    monkeypatch.setattr(api, 'LocalAI', lambda: ai)
    monkeypatch.setattr(api, 'ChromaStore', factory)
    key_reader = Mock(side_effect=AssertionError('API must not load key'))
    monkeypatch.setattr(sdk, 'dotenv_values', key_reader)
    with TestClient(api.app) as client:
        response = client.post('/v1/faq', json={'question': RIVERSIDE})
    assert response.status_code == 200
    factory.assert_called_once_with('nomic-embed-text:latest', create=False)
    store.ingest.assert_not_called()
    ai.close.assert_called_once()
    key_reader.assert_not_called()


def test_unavailable_collection_initialization_is_503(monkeypatch):
    ai = Mock()
    monkeypatch.setattr(api, 'LocalAI', lambda: ai)
    monkeypatch.setattr(api, 'ChromaStore', Mock(side_effect=RuntimeError('private connection string')))
    with TestClient(api.app) as client:
        response = client.post('/v1/faq', json={'question': RIVERSIDE})
    assert response.status_code == 503
    assert response.json()['status'] == 'unavailable'
    assert 'private' not in response.text
    ai.close.assert_called_once()
