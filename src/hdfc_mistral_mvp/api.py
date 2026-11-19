"""Read-only FAQ HTTP interface. Ingestion is an explicit administrator CLI step."""
from typing import Annotated, Callable, Literal
from contextlib import asynccontextmanager
import logging

from fastapi import Depends, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from .faq import ChromaStore, answer_result, screen
from .sdk import (DemoError, LocalAI, execute_studio_support,
                  start_studio_workflow_worker, stop_studio_workflow_worker)


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: StrictStr = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    title: str
    page: int = Field(ge=1)
    chunk_id: str


class Answered(BaseModel):
    status: Literal['answered'] = 'answered'
    answer: str
    citations: list[Citation]


class NotFound(BaseModel):
    status: Literal['not_found'] = 'not_found'
    message: str
    citations: list[Citation]


class Unavailable(BaseModel):
    status: Literal['unavailable'] = 'unavailable'
    message: str


def run_faq(question: str) -> dict:
    ai = LocalAI()
    try:
        # Do not create collections, embed documents or ingest on the query path.
        store = ChromaStore(ai.embedding_model, create=False)
        return answer_result(question, ai, store)
    finally:
        ai.close()


def get_runner() -> Callable[[str], dict]:
    # No connections or inference before the route has screened the question.
    return run_faq


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Exactly one workflow worker per FastAPI process. Missing Studio setup must
    # not prevent urgent support or the independent FAQ API from starting.
    task = None
    app.state.accepting_requests = False
    app.state.studio_worker_task = None
    app.state.studio_worker_available = False
    try:
        task = await start_studio_workflow_worker()
        app.state.studio_worker_task = task
        app.state.studio_worker_available = True
    except Exception:
        logging.getLogger(__name__).warning('Studio workflow worker unavailable at API startup.')
    app.state.accepting_requests = True
    try:
        yield
    finally:
        app.state.accepting_requests = False
        app.state.studio_worker_available = False
        await stop_studio_workflow_worker(task)


app = FastAPI(title='HDFC customer support agent', version='1.0', lifespan=lifespan)
INVALID = {'status': 'invalid_request', 'message': 'Enter a public policy question of 1–2,000 characters without personal data or secrets.'}
RETRY = 'Agent service is temporarily unavailable. Please try again shortly.'
FAQ_RETRY = 'FAQ is temporarily unavailable. Please try again shortly.'


@app.get('/health/live', include_in_schema=False)
async def live():
    return {'status':'alive'}


@app.get('/health/ready', include_in_schema=False)
async def ready():
    # Keep urgent support / direct FAQ reachable during control-plane outages.
    # Drain only on application shutdown; no inference/DB calls in probes.
    if not getattr(app.state, 'accepting_requests', False):
        return JSONResponse(status_code=503, content={'status':'not_ready'})
    task = getattr(app.state, 'studio_worker_task', None)
    worker_running = bool(getattr(app.state, 'studio_worker_available', False) and task and not task.done())
    return {'status':'ready', 'workflow_worker_running':worker_running}


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Default validation errors include submitted input. Do not echo it.
    return JSONResponse(status_code=422, content=INVALID)


@app.post('/v1/faq', response_model=Answered | NotFound,
          responses={503: {'model': Unavailable}})
def faq(request: Question, runner: Annotated[Callable, Depends(get_runner)]):
    try:
        if not request.question.strip():
            raise DemoError('Empty question')
        screen(request.question)
    except DemoError:
        return JSONResponse(status_code=422, content=INVALID)

    try:
        result = runner(request.question)
        # Validate service output before returning it, without parsing CLI text.
        return Answered.model_validate(result) if result['status'] == 'answered' else NotFound.model_validate(result)
    except Exception:
        # SDK/DB exceptions can contain raw request bodies or credentials.
        return JSONResponse(status_code=503, content=Unavailable(message=FAQ_RETRY).model_dump(),
                            headers={'Retry-After': '5'})

# Native Agents SDK supervisor and async named specialists.
import asyncio
import json
import os
import time
from fastapi import Request
from starlette.responses import StreamingResponse
from . import support
def get_support_runner():
    return execute_studio_support


def validate_support_result(result):
    from .support_models import validated_result
    return validated_result(result)


@app.get('/v1/support/urgent')
def urgent_support():
    return support.urgent_support()


@app.post('/v1/customer-support')
async def customer_support(body: Question, request: Request,
                           support_runner: Annotated[Callable, Depends(get_support_runner)], stream: bool = False):
    start = time.monotonic()
    # Urgency is determined locally even if screening subsequently fails.
    urgent = support.is_urgent(body.question)
    try:
        question = support.prepare_question(body.question)
    except SafetyRejected as exc:
        if urgent: return support.urgent_support()
        return JSONResponse(status_code=422, content=exc.result)
    except DemoError:
        if urgent: return support.urgent_support()
        return JSONResponse(status_code=422, content={**INVALID, 'support':support.urgent_support()})
    except Exception:
        if urgent: return support.urgent_support()
        return JSONResponse(status_code=503, content={'status':'unavailable','message':RETRY,'support':support.urgent_support()})

    async def work():
        if urgent: return support.urgent_support()
        if os.getenv('SUPPORT_INFERENCE_DISABLED') == '1': raise DemoError('Inference disabled')
        if request.headers.get('Authorization'):
            owner = actions.authorize(request.headers.get('Authorization'),'customer')
            key = request.headers.get('Idempotency-Key','')
            if not 1 <= len(key) <= 100: raise actions.ActionError(422,'Provide an Idempotency-Key for authenticated support.')
            reference = actions.create_agent_request(owner,key,question)
            await start_agent_support_workflow(reference)
            return {'status':'processing','request_id':reference,
                    'status_url':'/v1/customer-support/requests/'+reference}
        result = await support_runner(question)
        value = validate_support_result({k:v for k,v in result.items()
            if k not in ('orchestration','specialist','handoffs','model_calls')})
        return {**value, **{k:result[k] for k in ('orchestration','specialist','handoffs','model_calls') if k in result}}

    async def bounded():
        async with asyncio.timeout(max(0.001, 20 - (time.monotonic()-start))):
            return await work()

    if not stream:
        try:
            result = await bounded()
            if result.get('status') == 'rejected':
                return JSONResponse(status_code=422, content=result)
            return JSONResponse(status_code=202,content=result) if result.get('status') == 'processing' else result
        except actions.ActionError:
            raise
        except Exception:
            return JSONResponse(status_code=503, content={'status':'unavailable','message':RETRY,'support':support.urgent_support()})

    def event(name, data):
        return 'event: '+name+'\ndata: '+json.dumps(data,ensure_ascii=False)+'\n\n'

    async def events():
        task = None
        try:
            yield event('progress', {'stage':'safety_check'})
            task = asyncio.create_task(bounded())
            # Polling bounds disconnect detection even if model work is blocking.
            while not task.done():
                if await request.is_disconnected(): return
                await asyncio.wait({task}, timeout=0.05)
            result = task.result()
            yield event('final', result)
        except asyncio.CancelledError:
            raise
        except Exception:
            yield event('error', {'status':'unavailable','code':'support_unavailable',
                                  'message':RETRY,'support':support.urgent_support()})
        finally:
            if task and not task.done(): task.cancel()
            if task:
                try: await task
                except (asyncio.CancelledError, Exception): pass
    return StreamingResponse(events(), media_type='text/event-stream',
                             headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

# Secure JSON-only synthetic action service. No action data enters agent context.
from uuid import UUID
from fastapi import Header
from . import actions
from .safety import SafetyRejected
from .sdk import start_transaction_workflow, signal_transaction_workflow, start_agent_support_workflow

@app.exception_handler(actions.ActionError)
async def action_error(request, exc):
    return JSONResponse(status_code=exc.code, content={'status':'action_error','message':exc.message})

Customer = Annotated[str | None, Header(alias='Authorization')]
ReplayKey = Annotated[str, Header(alias='Idempotency-Key', min_length=1, max_length=100)]

def get_transaction_starter():
    return start_transaction_workflow

def get_transaction_signaller():
    return signal_transaction_workflow

@app.post('/v1/synthetic-actions', status_code=201)
def create_action(body: actions.Transfer, idempotency_key: ReplayKey, authorization: Customer = None):
    owner = actions.authorize(authorization, 'customer')
    return actions.create(owner, idempotency_key, body)

@app.get('/v1/synthetic-actions/{action_id}')
def action_status(action_id: UUID, authorization: Customer = None):
    owner = actions.authorize(authorization, 'customer')
    return actions.status(str(action_id), owner)

@app.post('/v1/synthetic-actions/{action_id}/confirm', status_code=202)
async def confirm_action(action_id: UUID, body: actions.Confirmation,
                         idempotency_key: ReplayKey,
                         starter: Annotated[Callable, Depends(get_transaction_starter)], authorization: Customer = None):
    owner = actions.authorize(authorization, 'customer')
    result = actions.confirm(str(action_id), owner, idempotency_key, body)
    try:
        # An identical retry resubmits the same workflow ID after uncertain delivery.
        if actions.action_workflow_id(str(action_id)):
            await signal_transaction_workflow(str(action_id))
        else:
            await starter(str(action_id))
    except Exception:
        return JSONResponse(status_code=503, content={'status':'unavailable',
            'message':'Action is recorded. Retry confirmation with the same details and idempotency key.'})
    return result


@app.get('/v1/customer-support/requests/{request_id}')
def support_request_status(request_id: UUID, authorization: Customer = None):
    owner = actions.authorize(authorization,'customer')
    record = actions.agent_request(str(request_id),owner)
    result = json.loads(record['result']) if record['result'] else {'status':'processing'}
    if record['action_id']:
        return {'result':result,'action':actions.status(record['action_id'],owner)}
    return {'result':result}

@app.post('/v1/synthetic-actions/{action_id}/review', status_code=202)
@app.post('/v1/webhooks/human-review/{action_id}', status_code=202)
async def review_action(action_id: UUID, body: actions.Review,
                        signaller: Annotated[Callable, Depends(get_transaction_signaller)], authorization: Customer = None):
    reviewer = actions.authorize(authorization, 'reviewer')
    result = actions.review(str(action_id), reviewer, body)
    try:
        await signaller(str(action_id))
    except Exception:
        return JSONResponse(status_code=503, content={'status':'unavailable',
            'message':'Decision is recorded. Retry the identical decision to resume the workflow.'})
    return result
