"""Public support data only. Banking identity and action state do not belong here."""
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

Specialist = Literal['FAQ', 'CardsGuidance', 'AccountGuidance', 'PaymentsGuidance', 'ProfileGuidance', 'HighStakeTransactions']

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, hide_input_in_errors=True)

class PublicTurn(StrictModel):
    request_id: UUID
    question: StrictStr = Field(min_length=1, max_length=2000)

    @field_validator('question')
    @classmethod
    def public_question(cls, value):
        from .faq import screen
        screen(value)
        if not value.strip(): raise ValueError('Enter a public support question.')
        return value

class Handoff(StrictModel):
    name: Specialist
    config_name: Specialist
    message: StrictStr = Field(min_length=1, max_length=2000)
    context: Literal['fresh'] = 'fresh'

class ClarificationRoute(StrictModel):
    specialist: Literal['clarify']

class CardTask(StrictModel):
    action: Literal['temporary_deactivation']

class FAQTask(StrictModel):
    pass

class TransactionTask(StrictModel):
    amount_minor: int | None = Field(default=None, strict=True, gt=0, le=100_000_000,
                                    description='Synthetic INR amount in paise: rupees multiplied by 100; omit if unspecified.')
    target: Literal['demo-beneficiary-a','demo-beneficiary-b'] | None = None

class ActionPending(StrictModel):
    status: Literal['pending_review','awaiting_confirmation']
    action_id: UUID
    handoff: Literal['human_review','none'] = 'none'
    message: StrictStr

class AccountTask(StrictModel):
    action: Literal['balance', 'statement_download']

class PaymentTask(StrictModel):
    action: Literal['beneficiary_change', 'money_transfer']

class ProfileTask(StrictModel):
    action: Literal['contact_update', 'card_limit_change']

class Citation(StrictModel):
    title: StrictStr
    page: int = Field(ge=1, strict=True)
    # Approved content hashes can contain long digit runs. Validate the metadata
    # format rather than treating a stable corpus ID as customer free text.
    chunk_id: StrictStr = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,160}$')

class FAQAnswer(StrictModel):
    status: Literal['answered']
    answer: StrictStr
    citations: list[Citation] = Field(min_length=1)

class FAQMissing(StrictModel):
    status: Literal['not_found']
    message: Literal['I couldn’t find that in the approved documents.']
    citations: list[Citation] = Field(max_length=0)

class Guidance(StrictModel):
    status: Literal['guidance'] = 'guidance'
    message: StrictStr

class Clarification(StrictModel):
    status: Literal['clarification_required'] = 'clarification_required'
    message: Literal['Please choose one: a public policy question, temporary card deactivation, account access, payments, or profile changes.'] = 'Please choose one: a public policy question, temporary card deactivation, account access, payments, or profile changes.'

class SafetyVerdict(StrictModel):
    decision: Literal['allow','urgent_support','adversarial','inappropriate','irrelevant']

class UrgentSupport(StrictModel):
    status: Literal['urgent_support']
    message: Literal['Contact urgent card support now. This handoff does not block a card or file a fraud report.']
    support_url: StrictStr

class Rejected(StrictModel):
    status: Literal['rejected'] = 'rejected'
    reason: Literal['adversarial','inappropriate','irrelevant','missing_context']
    message: Literal['Please send a banking support question without personal details or instructions to bypass safeguards.'] = 'Please send a banking support question without personal details or instructions to bypass safeguards.'

class CallCounts(StrictModel):
    generation: int = Field(default=0, ge=0, le=6)
    embedding: int = Field(default=0, ge=0, le=1)

class RequestGuardState(StrictModel):
    schema_version: Literal[1] = 1
    request_id: UUID
    question_hash: str
    safety_approved: bool = False
    deadline: float
    phase: Literal['routing', 'delegated', 'completed', 'failed'] = 'routing'
    specialist: Specialist | None = None
    root_session: str | None = None
    child_session: str | None = None
    child_turn: str | None = None
    handoffs: int = Field(default=0, ge=0, le=1)
    tool_used: bool = False
    calls: CallCounts = Field(default_factory=CallCounts)
    result: dict | None = None

def validated_result(result):
    from .faq import screen
    classes = {'answered':FAQAnswer, 'not_found':FAQMissing, 'guidance':Guidance,
               'pending_review':ActionPending,'awaiting_confirmation':ActionPending,
               'clarification_required':Clarification, 'rejected':Rejected,
               'urgent_support':UrgentSupport}
    value = classes[result['status']].model_validate(result).model_dump(mode='json')
    if value['status'] == 'urgent_support':
        from .support import urgent_support
        # Only bank configuration may supply the message/destination, even when
        # a result has crossed the workflow boundary. Never trust a model URL.
        if value != urgent_support(): raise ValueError('Unexpected urgent support destination.')
    for key in ('answer', 'message'):
        if key in value: screen(value[key])
    for citation in value.get('citations', []):
        screen(citation['title'])
    import json
    if len(json.dumps(value).encode()) > 16384: raise ValueError('Support result too large.')
    return value
