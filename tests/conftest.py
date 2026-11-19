"""Real database checks are opt-in and use a disposable PostgreSQL schema."""
import os
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest

from hdfc_mistral_mvp import actions


def pytest_addoption(parser):
    parser.addoption('--postgres', action='store_true', help='Run isolated local PostgreSQL ledger checks.')


@pytest.fixture
def postgres_ledger(request, monkeypatch):
    if not request.config.getoption('--postgres'):
        pytest.skip('Requires --postgres and DEMO_ACTION_DATABASE_URL; no database access in offline tests.')
    dsn = os.getenv('DEMO_ACTION_DATABASE_URL')
    if not dsn:
        pytest.fail('--postgres requires DEMO_ACTION_DATABASE_URL')
    schema = 'test_actions_' + uuid4().hex
    monkeypatch.setenv('DEMO_ACTION_DB_SCHEMA', schema)
    try:
        actions.init_ledger()
        yield schema
    finally:
        with psycopg.connect(dsn, connect_timeout=5) as db:
            db.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
