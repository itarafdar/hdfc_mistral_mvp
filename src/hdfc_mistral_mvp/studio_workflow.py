"""Default support orchestration; every native agent runs in this workflow."""
from datetime import timedelta

import mistralai.workflows as workflows
from pydantic import Field, StrictStr, field_validator
from .support_models import StrictModel

WORKFLOW_NAME = 'customer-support-synthetic'

class StudioSupportInput(StrictModel):
    question: StrictStr = Field(min_length=1, max_length=2000)

    @field_validator('question')
    @classmethod
    def screened(cls, value):
        from .support import prepare_question
        return prepare_question(value)

class StudioSupportResult(StrictModel):
    response: dict = Field(description='Validated public support response')

@workflows.activity(name='run-local-support',
                    start_to_close_timeout=timedelta(seconds=20),
                    retry_policy_max_attempts=1)
async def run_local_support(params: StudioSupportInput) -> StudioSupportResult:
    # Imports and all nondeterministic I/O stay inside the activity.
    from .sdk import run_support_agents
    result = await run_support_agents(params.question)
    return StudioSupportResult(response=result)

@workflows.workflow.define(name=WORKFLOW_NAME,
                          workflow_description='Synthetic support with local native agents and Ollama.',
                          execution_timeout=timedelta(seconds=20))
class CustomerSupportWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, params: StudioSupportInput) -> StudioSupportResult:
        return await run_local_support(params)


# Human turnaround is outside the 20-second model budget. Only opaque references
# and safe orchestration outcomes cross the Studio boundary.
TRANSACTION_WORKFLOW = 'synthetic-transaction-review'

class TransactionInput(StrictModel):
    action_id: str = Field(pattern=r'^[a-f0-9-]{36}$')

@workflows.activity(name='triage-synthetic-transaction',
                    start_to_close_timeout=timedelta(seconds=20), retry_policy_max_attempts=1)
async def triage_transaction() -> None:
    from .sdk import run_support_agents, DemoError
    result = await run_support_agents('Start a synthetic transaction with human approval.')
    if result.get('specialist') != 'HighStakeTransactions' or result.get('handoffs') != 1:
        raise DemoError('Synthetic transaction specialist handoff was not verified.')

@workflows.activity(name='notify-synthetic-reviewer',
                    start_to_close_timeout=timedelta(seconds=10), retry_policy_max_attempts=3)
async def notify_transaction(params: TransactionInput) -> None:
    from .actions import notify_reviewer
    await notify_reviewer(params.action_id)

@workflows.activity(name='check-synthetic-action',
                    start_to_close_timeout=timedelta(seconds=10), retry_policy_max_attempts=3)
async def check_transaction(params: TransactionInput) -> str:
    from .actions import expire
    return expire(params.action_id)

@workflows.activity(name='dispatch-synthetic-action',
                    start_to_close_timeout=timedelta(seconds=10), retry_policy_max_attempts=3)
async def dispatch_transaction(params: TransactionInput) -> str:
    from .actions import dispatch
    return dispatch(params.action_id)

@workflows.workflow.define(name=TRANSACTION_WORKFLOW,
                          workflow_description='Synthetic action with bank-owned human approval.',
                          execution_timeout=timedelta(minutes=17))
class SyntheticTransactionWorkflow:
    def __init__(self):
        self.changed = False

    @workflows.workflow.signal(name='review_updated')
    async def review_updated(self) -> None:
        # This is a wake-up, never an approval. The activity re-reads the ledger.
        self.changed = True

    @workflows.workflow.entrypoint
    async def run(self, params: TransactionInput) -> str:
        import asyncio
        await triage_transaction()
        await notify_transaction(params)
        while True:
            self.changed = False
            state = await check_transaction(params)
            if state != 'pending_review':
                if state == 'accepted': return await dispatch_transaction(params)
                return state
            try:
                await workflows.workflow.wait_condition(lambda: self.changed,
                    timeout=timedelta(minutes=15))
            except asyncio.TimeoutError:
                return await check_transaction(params)


AGENT_TRANSACTION_WORKFLOW = 'agent-support-with-review'

class AgentRequestInput(StrictModel):
    request_id: str = Field(pattern=r'^[a-f0-9-]{36}$')

class SubagentHandoff(StrictModel):
    action_id: str | None = None
    human_review_required: bool = Field(default=False,strict=True)

@workflows.activity(name='run-authenticated-support',
                    start_to_close_timeout=timedelta(seconds=20),retry_policy_max_attempts=1)
async def run_authenticated_support(params: AgentRequestInput) -> SubagentHandoff:
    from . import actions
    from .sdk import run_support_agents
    request = actions.agent_request(params.request_id)
    try:
        result = await run_support_agents(request['question'], action_request_id=params.request_id)
    except Exception:
        from .sdk import DemoError
        actions.agent_result(params.request_id,{'status':'unavailable','message':'Support processing failed. No execution was authorized.'})
        raise DemoError('Authenticated support processing failed.') from None
    actions.agent_result(params.request_id,result)
    action_id = actions.agent_request(params.request_id)['action_id']
    if action_id is None: return SubagentHandoff()
    # The native child's tool requested handoff; cross-check the bank policy.
    row = actions.status(action_id,request['owner'])
    requested = result.get('handoff') == 'human_review'
    if result.get('specialist') != 'HighStakeTransactions' or requested != row['review_required']:
        from .sdk import DemoError
        raise DemoError('Subagent handoff does not match bank policy.')
    return SubagentHandoff(action_id=action_id,human_review_required=requested)

@workflows.workflow.define(name=AGENT_TRANSACTION_WORKFLOW,
                          workflow_description='Semantic native subagent prepares and escalates a synthetic action.',
                          execution_timeout=timedelta(minutes=17))
class AgentTransactionWorkflow:
    def __init__(self):
        self.changed = False

    @workflows.workflow.signal(name='review_updated')
    async def review_updated(self) -> None:
        self.changed = True

    @workflows.workflow.entrypoint
    async def run(self, params: AgentRequestInput) -> str:
        import asyncio
        handoff = SubagentHandoff.model_validate(await run_authenticated_support(params))
        if handoff.action_id is None: return 'completed_without_action'
        action = TransactionInput(action_id=handoff.action_id)
        # No second triage, separate start endpoint or model call for escalation.
        if handoff.human_review_required:
            await notify_transaction(action)
        while True:
            self.changed = False
            state = await check_transaction(action)
            if state == 'accepted': return await dispatch_transaction(action)
            if state not in ('pending_review','awaiting_confirmation'): return state
            try:
                await workflows.workflow.wait_condition(lambda:self.changed,timeout=timedelta(minutes=60))
            except asyncio.TimeoutError:
                pass  # Recheck authoritative confirmation/review deadlines in Python.
