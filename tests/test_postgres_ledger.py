"""Offline storage failure contracts and opt-in PostgreSQL concurrency checks."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import psycopg
import pytest
from fastapi.testclient import TestClient

from hdfc_mistral_mvp import actions, api


def test_missing_configuration_does_not_connect(monkeypatch):
    monkeypatch.delenv('DEMO_ACTION_DATABASE_URL', raising=False)
    connect = Mock(side_effect=AssertionError('Must not connect'))
    monkeypatch.setattr(actions.psycopg, 'connect', connect)
    with pytest.raises(actions.ActionError, match='not configured') as error:
        actions.status('missing')
    assert error.value.code == 503
    connect.assert_not_called()


def test_invalid_schema_does_not_connect(monkeypatch):
    monkeypatch.setenv('DEMO_ACTION_DATABASE_URL', 'synthetic-not-used')
    monkeypatch.setenv('DEMO_ACTION_DB_SCHEMA', 'public; DROP SCHEMA public')
    connect = Mock(side_effect=AssertionError('Must not connect'))
    monkeypatch.setattr(actions.psycopg, 'connect', connect)
    with pytest.raises(actions.ActionError, match='not configured'):
        actions.init_ledger()
    connect.assert_not_called()


@pytest.mark.parametrize('operation', ['connect', 'statement', 'commit'])
def test_storage_failure_returns_sanitized_http_503(monkeypatch, operation):
    monkeypatch.setenv('DEMO_CUSTOMER_TOKEN', 'synthetic-customer-test-token')
    monkeypatch.setenv('DEMO_REVIEWER_TOKEN', 'synthetic-reviewer-test-token')
    monkeypatch.setenv('DEMO_ACTION_DATABASE_URL', 'synthetic-not-used')
    monkeypatch.delenv('DEMO_ACTION_DB_SCHEMA', raising=False)
    error = psycopg.OperationalError('private-password-and-SQL-parameters')
    connection = Mock()
    connection.__enter__ = Mock(return_value=connection)
    connection.__exit__ = Mock(return_value=False)
    connect = Mock(return_value=connection)
    if operation == 'connect': connect.side_effect = error
    elif operation == 'statement': connection.execute.side_effect = error
    else:
        connection.__exit__.side_effect = error
        connection.execute.return_value.fetchone.return_value = None
    monkeypatch.setattr(actions.psycopg, 'connect', connect)
    if operation == 'commit':
        # Force an otherwise successful operation through context exit/commit.
        monkeypatch.setattr(actions, '_row', lambda *args: {
            'id':'synthetic', 'state':'prepared', 'digest':'0'*64, 'threshold':10,
            'details':'{"amount_minor":1,"currency":"INR","target":"demo-beneficiary-a"}'})
    response = TestClient(api.app).get(
        '/v1/synthetic-actions/11111111-1111-4111-8111-111111111111',
        headers={'Authorization':'Bearer synthetic-customer-test-token'})
    assert response.status_code == 503
    assert response.json() == {'status':'action_error', 'message':
        'Synthetic action storage unavailable. Please retry.'}
    assert 'private' not in response.text


def test_concurrent_creation_and_request_replays(postgres_ledger):
    transfer = actions.Transfer(amount_minor=1500000, target='demo-beneficiary-a')
    question = 'Synthetic transfer INR 15000 to demo-beneficiary-a.'
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda _: actions.create('owner', 'same-key', transfer), range(16)))
        references = list(pool.map(lambda _: actions.create_agent_request('owner', 'same-key', question), range(16)))
    assert len({row['action_id'] for row in rows}) == 1
    assert len(set(references)) == 1
    with pytest.raises(actions.ActionError):
        actions.create('owner', 'same-key', transfer.model_copy(update={'amount_minor':1}))
    with pytest.raises(actions.ActionError):
        actions.create_agent_request('owner', 'same-key', question+' changed')


def test_concurrent_confirmation_and_review(postgres_ledger):
    reference = actions.create_agent_request('owner', 'one', 'Synthetic transfer INR 15000 to demo-beneficiary-a.')
    row = actions.prepare_from_agent(reference, actions.Transfer(amount_minor=1500000, target='demo-beneficiary-a'))
    confirmation = actions.Confirmation(**{k:row[k] for k in ('amount_minor','currency','target','digest','version')})
    vote = actions.Review(digest=row['digest'], decision='approve', reason='synthetic_policy_checked')
    def apply(i):
        if i % 2: return actions.confirm(row['action_id'], 'owner', 'same-consent', confirmation)
        return actions.review(row['action_id'], 'reviewer', vote)
    with ThreadPoolExecutor(8) as pool: list(pool.map(apply, range(16)))
    assert actions.status(row['action_id'])['status'] == 'accepted'
    with ThreadPoolExecutor(8) as pool:
        assert set(pool.map(lambda _: actions.dispatch(row['action_id']), range(16))) == {'succeeded'}
    with actions.ledger() as db:
        assert db.execute('SELECT COUNT(*) AS count FROM effects').fetchone()['count'] == 1
        assert db.execute('SELECT COUNT(*) AS count FROM outbox').fetchone()['count'] == 0


def test_initialization_is_repeatable_and_rollback_preserves_state(postgres_ledger):
    row = actions.create('owner', 'one', actions.Transfer(amount_minor=1,target='demo-beneficiary-a'))
    actions.init_ledger()
    with pytest.raises(RuntimeError):
        with actions.ledger() as db:
            db.execute('UPDATE actions SET state=%s WHERE id=%s', ('rejected', row['action_id']))
            raise RuntimeError('rollback check')
    assert actions.status(row['action_id'], 'owner') == row
