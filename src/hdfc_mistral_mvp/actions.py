"""Synthetic bank-owned action ledger. No real banking integration or model access."""
from contextlib import contextmanager
import hashlib
import hmac
import json
import os
import re
import time
from typing import Literal
from uuid import uuid4

from pydantic import Field, StrictStr
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from .support_models import StrictModel


class ActionError(Exception):
    def __init__(self, code=409, message='Action is not valid in its current state.'):
        self.code, self.message = code, message


class Transfer(StrictModel):
    amount_minor: int = Field(strict=True, gt=0, le=100_000_000)
    currency: Literal['INR'] = 'INR'
    target: Literal['demo-beneficiary-a', 'demo-beneficiary-b']


class Confirmation(Transfer):
    version: Literal[1] = 1
    digest: StrictStr = Field(pattern=r'^[a-f0-9]{64}$')


class Review(StrictModel):
    digest: StrictStr = Field(pattern=r'^[a-f0-9]{64}$')
    decision: Literal['approve', 'reject']
    reason: Literal['synthetic_policy_checked', 'synthetic_policy_failed']


def authorize(authorization, role):
    """Two fixed independent demo identities. Replace with bank IAM before rollout."""
    customer = os.getenv('DEMO_CUSTOMER_TOKEN', '')
    reviewer = os.getenv('DEMO_REVIEWER_TOKEN', '')
    if len(customer) < 24 or len(reviewer) < 24 or hmac.compare_digest(customer.encode(), reviewer.encode()):
        raise ActionError(503, 'Synthetic action authentication is not configured.')
    expected = customer if role == 'customer' else reviewer
    if not authorization or not hmac.compare_digest(authorization.encode(), ('Bearer ' + expected).encode()):
        raise ActionError(403, 'Access denied.')
    return 'demo-customer' if role == 'customer' else 'demo-reviewer'


@contextmanager
def ledger():
    """One short PostgreSQL transaction; no network notification inside it."""
    dsn = os.getenv('DEMO_ACTION_DATABASE_URL', '')
    schema = os.getenv('DEMO_ACTION_DB_SCHEMA', 'bank_actions')
    if not dsn or not re.fullmatch(r'[a-z][a-z0-9_]{0,62}', schema):
        raise ActionError(503, 'Synthetic action storage is not configured.')
    try:
        with psycopg.connect(dsn, connect_timeout=5, row_factory=dict_row,
                             options='-c statement_timeout=5000 -c lock_timeout=5000') as db:
            db.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(schema)))
            yield db
    except psycopg.Error:
        # Driver errors may include credentials, SQL parameters or server details.
        raise ActionError(503, 'Synthetic action storage unavailable. Please retry.') from None


def init_ledger():
    """Explicit, repeatable schema initialization; run before serving actions."""
    schema = os.getenv('DEMO_ACTION_DB_SCHEMA', 'bank_actions')
    with ledger() as db:
        # Serialize concurrent initializers only; ordinary requests lock their rows.
        db.execute('SELECT pg_advisory_xact_lock(73428109)')
        db.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
        db.execute('''
        CREATE TABLE IF NOT EXISTS actions (
          id TEXT PRIMARY KEY, owner TEXT NOT NULL, create_key TEXT NOT NULL,
          details TEXT NOT NULL, digest TEXT NOT NULL, threshold INTEGER NOT NULL,
          state TEXT NOT NULL, confirm_by DOUBLE PRECISION NOT NULL, review_by DOUBLE PRECISION,
          dispatch_by DOUBLE PRECISION, confirmation_key TEXT, decision TEXT, reason TEXT,
          reviewer TEXT, UNIQUE(owner, create_key));
        CREATE TABLE IF NOT EXISTS outbox (action_id TEXT PRIMARY KEY REFERENCES actions(id));
        CREATE TABLE IF NOT EXISTS effects (action_id TEXT PRIMARY KEY REFERENCES actions(id));
        CREATE TABLE IF NOT EXISTS review_votes (
          action_id TEXT PRIMARY KEY REFERENCES actions(id), reviewer TEXT NOT NULL,
          decision TEXT NOT NULL, reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS agent_requests (
          id TEXT PRIMARY KEY, owner TEXT NOT NULL, replay_key TEXT NOT NULL,
          question TEXT NOT NULL, action_id TEXT, result TEXT, UNIQUE(owner,replay_key));
        ''')


def _row(db, action_id, owner=None):
    # Serialize confirmation/review/dispatch for this action, including replays.
    row = db.execute('SELECT * FROM actions WHERE id=%s FOR UPDATE', (action_id,)).fetchone()
    if row is None or (owner is not None and row['owner'] != owner):
        raise ActionError(404, 'Action not found.')
    return dict(row)


def _view(row):
    return {'action_id': row['id'], 'status': row['state'], 'version': 1,
            'digest': row['digest'], **json.loads(row['details']),
            'review_required': json.loads(row['details'])['amount_minor'] > row['threshold']}


def status(action_id, owner=None):
    with ledger() as db:
        row = _row(db, action_id, owner)
        return _view(row)


def create(owner, key, transfer):
    details = json.dumps(transfer.model_dump(), sort_keys=True, separators=(',', ':'))
    with ledger() as db:
        existing = db.execute('SELECT * FROM actions WHERE owner=%s AND create_key=%s', (owner, key)).fetchone()
        if existing:
            if existing['details'] != details: raise ActionError()
            return _view(dict(existing))
        try:
            threshold = int(os.getenv('DEMO_REVIEW_THRESHOLD_MINOR', '1000000'))
        except ValueError:
            raise ActionError(503, 'Invalid synthetic policy configuration.') from None
        if threshold < 0: raise ActionError(503, 'Invalid synthetic policy configuration.')
        action_id = str(uuid4())
        digest = hashlib.sha256((action_id + ':1:' + str(threshold) + ':' + details).encode()).hexdigest()
        db.execute('INSERT INTO actions(id,owner,create_key,details,digest,threshold,state,confirm_by) VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (owner,create_key) DO NOTHING',
                   (action_id, owner, key, details, digest, threshold, 'prepared', time.time()+300))
        saved = db.execute('SELECT * FROM actions WHERE owner=%s AND create_key=%s', (owner,key)).fetchone()
        if saved['details'] != details: raise ActionError()
        return _view(saved)


def _accept(db, row):
    # Approval (if needed), acceptance, deadline and outbox share this transaction.
    db.execute('UPDATE actions SET state=%s,dispatch_by=%s WHERE id=%s', ('accepted', time.time()+60, row['id']))
    db.execute('INSERT INTO outbox VALUES(%s) ON CONFLICT DO NOTHING', (row['id'],))


def confirm(action_id, owner, key, confirmation):
    with ledger() as db:
        row = _row(db, action_id, owner)
        if (confirmation.digest != row['digest'] or
            Transfer.model_validate(confirmation.model_dump(include={'amount_minor','currency','target'})).model_dump() != json.loads(row['details'])):
            raise ActionError()
        if row['confirmation_key']:
            if row['confirmation_key'] != key: raise ActionError()
            return _view(row)  # authorized identical retry, even after expiry
        if row['state'] not in ('prepared','pending_review','awaiting_confirmation') or time.time() >= row['confirm_by']: raise ActionError()
        if row['decision'] == 'reject': raise ActionError()
        db.execute('UPDATE actions SET confirmation_key=%s WHERE id=%s', (key, action_id))
        vote = db.execute('SELECT * FROM review_votes WHERE action_id=%s',(action_id,)).fetchone()
        if vote:
            # Finalize the recorded human vote, effective approval, acceptance and
            # outbox atomically once the customer's exact consent is available.
            if vote['decision'] != 'approve': raise ActionError()
            db.execute('UPDATE actions SET decision=%s,reason=%s,reviewer=%s WHERE id=%s',
                       ('approve',vote['reason'],vote['reviewer'],action_id))
            row['decision'] = 'approve'
        if json.loads(row['details'])['amount_minor'] > row['threshold'] and row['decision'] != 'approve':
            if row['review_by'] is not None and time.time() >= row['review_by']: raise ActionError()
            db.execute('UPDATE actions SET state=%s,review_by=%s WHERE id=%s', ('pending_review', row['review_by'] or time.time()+900, action_id))
        else:
            _accept(db, row)
        return _view(_row(db, action_id))


def review(action_id, reviewer, decision):
    with ledger() as db:
        row = _row(db, action_id)
        if reviewer == row['owner']: raise ActionError(403, 'Independent reviewer required.')
        if decision.digest != row['digest']: raise ActionError()
        if row['decision']:
            if (row['decision'],row['reason'],row['reviewer']) != (decision.decision,decision.reason,reviewer): raise ActionError()
            return _view(row)
        vote = db.execute('SELECT * FROM review_votes WHERE action_id=%s',(action_id,)).fetchone()
        if vote:
            if (vote['decision'],vote['reason'],vote['reviewer']) != (decision.decision,decision.reason,reviewer): raise ActionError()
            return _view(row)
        if row['state'] != 'pending_review' or time.time() >= row['review_by']: raise ActionError()
        if not row['confirmation_key'] and time.time() >= row['confirm_by']: raise ActionError()
        if decision.decision == 'approve' and not row['confirmation_key']:
            db.execute('INSERT INTO review_votes VALUES(%s,%s,%s,%s)',(action_id,reviewer,decision.decision,decision.reason))
            db.execute('UPDATE actions SET state=%s WHERE id=%s',('awaiting_confirmation',action_id))
            return _view(_row(db,action_id))
        db.execute('UPDATE actions SET decision=%s,reason=%s,reviewer=%s WHERE id=%s',
                   (decision.decision,decision.reason,reviewer,action_id))
        if decision.decision == 'approve':
            if row['confirmation_key']: _accept(db, row)
            else: db.execute('UPDATE actions SET state=%s WHERE id=%s',('awaiting_confirmation',action_id))
        else: db.execute('UPDATE actions SET state=%s WHERE id=%s', ('rejected', action_id))
        return _view(_row(db, action_id))


def dispatch(action_id):
    """Local synthetic processor: effect and success commit together, once.

    A remote processor would require send-intent and reconciliation; never swap
    a remote call into this transaction and claim these guarantees still hold.
    """
    with ledger() as db:
        row = _row(db, action_id)
        if row['state'] == 'succeeded': return 'succeeded'
        if row['state'] != 'accepted' or not row['confirmation_key']: return row['state']
        if json.loads(row['details'])['amount_minor'] > row['threshold'] and row['decision'] != 'approve': raise ActionError()
        if time.time() >= row['dispatch_by']:
            db.execute('UPDATE actions SET state=%s WHERE id=%s', ('expired', action_id))
            return 'expired'
        if not db.execute('SELECT 1 FROM outbox WHERE action_id=%s', (action_id,)).fetchone(): raise ActionError()
        db.execute('INSERT INTO effects VALUES(%s) ON CONFLICT DO NOTHING', (action_id,))
        db.execute('UPDATE actions SET state=%s WHERE id=%s', ('succeeded', action_id))
        db.execute('DELETE FROM outbox WHERE action_id=%s', (action_id,))
        return 'succeeded'


def expire(action_id):
    with ledger() as db:
        row = _row(db, action_id)
        if row['state'] == 'pending_review' and (time.time() >= row['review_by'] or
                (not row['confirmation_key'] and time.time() >= row['confirm_by'])):
            db.execute('UPDATE actions SET state=%s WHERE id=%s', ('expired', action_id))
        elif row['state'] == 'awaiting_confirmation' and time.time() >= row['confirm_by']:
            db.execute('UPDATE actions SET state=%s WHERE id=%s', ('expired', action_id))
        return _row(db, action_id)['state']


async def notify_reviewer(action_id):
    """Configured destination only; retries reuse action_id as delivery identifier."""
    import httpx2
    from urllib.parse import urlsplit
    with ledger() as db:
        row = _row(db, action_id)
        if row['state'] != 'pending_review': return
    url = os.getenv('DEMO_REVIEW_WEBHOOK_URL', '')
    secret = os.getenv('DEMO_REVIEW_WEBHOOK_SECRET', '')
    callback_base = os.getenv('DEMO_REVIEW_CALLBACK_BASE_URL','http://127.0.0.1:8021').rstrip('/')
    callback = urlsplit(callback_base)
    if (not callback.hostname or callback.username or callback.password or callback.query or callback.fragment or callback.path or
        not (callback.scheme == 'https' or (callback.scheme == 'http' and callback.hostname in ('localhost','127.0.0.1')))):
        raise ActionError(503,'Reviewer callback is not configured.')
    parts = urlsplit(url)
    if (not parts.hostname or parts.username or parts.password or parts.fragment or parts.query or
        not (parts.scheme == 'https' or (parts.scheme == 'http' and parts.hostname in ('localhost','127.0.0.1'))) or len(secret) < 24):
        raise ActionError(503, 'Reviewer webhook is not configured.')
    payload = json.dumps({'event':'synthetic_action.review_required', 'event_id':action_id,
                          'callback_url':callback_base+'/v1/webhooks/human-review/'+action_id,
                          **_view(row), 'review_by':row['review_by']}, sort_keys=True).encode()
    stamp = str(int(time.time()))
    signature = hmac.new(secret.encode(), stamp.encode()+b'.'+payload, hashlib.sha256).hexdigest()
    async with httpx2.AsyncClient(timeout=5, follow_redirects=False, trust_env=False) as client:
        response = await client.post(url, content=payload, headers={
            'Content-Type':'application/json', 'X-Review-Timestamp':stamp,
            'X-Review-Signature':signature, 'Idempotency-Key':action_id})
        if not 200 <= response.status_code < 300:
            raise ActionError(503, 'Reviewer notification unavailable.')


# Bank-owned request context: never injected into model sessions or Studio traces.
def create_agent_request(owner, key, question):
    from .support import prepare_question
    question = prepare_question(question)
    with ledger() as db:
        existing = db.execute('SELECT * FROM agent_requests WHERE owner=%s AND replay_key=%s', (owner,key)).fetchone()
        if existing:
            if existing['question'] != question: raise ActionError()
            return existing['id']
        request_id = str(uuid4())
        db.execute('INSERT INTO agent_requests(id,owner,replay_key,question) VALUES(%s,%s,%s,%s) ON CONFLICT (owner,replay_key) DO NOTHING',
                   (request_id,owner,key,question))
        saved = db.execute('SELECT * FROM agent_requests WHERE owner=%s AND replay_key=%s', (owner,key)).fetchone()
        if saved['question'] != question: raise ActionError()
        return saved['id']


def agent_request(request_id, owner=None):
    with ledger() as db:
        row = db.execute('SELECT * FROM agent_requests WHERE id=%s',(request_id,)).fetchone()
        if row is None or (owner is not None and row['owner'] != owner): raise ActionError(404,'Request not found.')
        return dict(row)


def agent_result(request_id, result):
    with ledger() as db:
        db.execute('UPDATE agent_requests SET result=%s WHERE id=%s',(json.dumps(result),request_id))


def prepare_from_agent(request_id, transfer):
    request = agent_request(request_id)
    # Ground the untrusted extraction in the explicit synthetic request. Semantic
    # selection belongs to the model; Python rejects invented/ambiguous details.
    from decimal import Decimal
    amounts = re.findall(r'\bINR\s+([0-9]+(?:\.[0-9]{1,2})?)\b',request['question'],re.I)
    targets = set(re.findall(r'\bdemo-beneficiary-[ab]\b',request['question']))
    if len(amounts) != 1 or targets != {transfer.target} or Decimal(amounts[0])*100 != transfer.amount_minor:
        raise ActionError(422,'Specify one INR amount and one synthetic beneficiary.')
    # Stable key prevents duplicate preparations after an uncertain activity result.
    row = create(request['owner'], 'agent:'+request_id, transfer)
    with ledger() as db:
        db.execute('UPDATE agent_requests SET action_id=%s WHERE id=%s',(row['action_id'],request_id))
        saved = _row(db,row['action_id'])
        if saved['state'] == 'prepared':
            high = json.loads(saved['details'])['amount_minor'] > saved['threshold']
            db.execute('UPDATE actions SET state=%s,review_by=%s WHERE id=%s',
                       ('pending_review' if high else 'awaiting_confirmation',time.time()+900 if high else None,row['action_id']))
    return status(row['action_id'],request['owner'])


def action_workflow_id(action_id):
    with ledger() as db:
        row = db.execute('SELECT id FROM agent_requests WHERE action_id=%s',(action_id,)).fetchone()
        return row['id'] if row else None


if __name__ == '__main__':
    try:
        init_ledger()
    except ActionError as exc:
        raise SystemExit(exc.message) from None
    print('PostgreSQL action ledger initialized.')
