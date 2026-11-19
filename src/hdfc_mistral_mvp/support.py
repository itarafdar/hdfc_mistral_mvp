"""Urgent routing and safe public guidance. Native graph lives in sdk.py.

The Phase 1 triage/dispatch helpers remain for historical offline diagnostics.
"""
import os
import re
from typing import Literal
from urllib.parse import urlparse
from pydantic import BaseModel, ConfigDict
from .faq import screen
from .sdk import DemoError

class Route(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    specialist: Literal['FAQ', 'CardsGuidance', 'clarify', 'urgent']

URGENT_DEFAULT = 'https://www.hdfc.bank.in/credit-cards/block-loststolen-card'

def urgent_support():
    url = os.getenv('URGENT_SUPPORT_URL', URGENT_DEFAULT)
    parsed = urlparse(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        url = URGENT_DEFAULT
    return {'status':'urgent_support', 'message':'Contact urgent card support now. This handoff does not block a card or file a fraud report.', 'support_url':url}

def is_urgent(question):
    # Immediate outage-safe detection. Indirect incident descriptions also get
    # semantic urgency assessment by SafetyGuardrail inside the worker activity.
    return bool(re.search(r'\b(lost|stolen|theft|fraud|unauthori[sz]ed|permanent(?:ly)?\s+block)\b|\bcard\s+(?:is\s+)?(?:missing|compromised)\b', question, re.I))

def preflight(question):
    """Screen before opening SSE or making any model request."""
    screen(question)
    if not question.strip(): raise DemoError('Empty question')
    if is_urgent(question): return Route(specialist='urgent')
    return None

def prepare_question(question):
    """Always call the local regex tool before any model or Studio payload."""
    from .safety import scan_banking_input
    value = scan_banking_input(question).question
    preflight(value)  # Existing residual screening stays fail-closed.
    return value

def triage(question, classifier):
    route = preflight(question)
    if route is not None: return route
    if os.getenv('SUPPORT_INFERENCE_DISABLED') == '1':
        raise DemoError('Inference disabled')
    return Route.model_validate_json(classifier(question))

def cards_guidance():
    return {'status':'guidance','message':'Use the authenticated bank card service, select an eligible card, complete bank-native verification, and confirm temporary deactivation there. No card action has been prepared or executed.'}

# Public next steps only: no identity lookup, preparation, balances or writes.
GUIDANCE = {
    'temporary_deactivation': cards_guidance()['message'],
    'balance': 'Sign in to the bank account service and select an account there to view a verified balance. Account data is available only in that secure service.',
    'statement_download': 'Sign in to the bank account service, select an account and statement period, and download the statement there.',
    'beneficiary_change': 'Use the authenticated bank payments service to add or change a beneficiary. Complete its required verification and review the beneficiary there before confirming.',
    'money_transfer': 'Use the authenticated bank payments service to select a beneficiary, enter the amount and review the exact transfer details before bank-native verification and confirmation.',
    'contact_update': 'Use the authenticated bank profile service to change contact details. Complete its verification and confirm the exact changes in that secure service.',
    'card_limit_change': 'Use the authenticated bank card service to review an eligible card and request a limit change. Eligibility, verification and any staff review are handled there.',
}

def dispatch(route, question, faq_runner):
    # Exactly one named handoff; no dynamic names or recursive registry access.
    route=Route.model_validate(route)
    if route.specialist=='FAQ':
        if os.getenv('SUPPORT_INFERENCE_DISABLED')=='1': raise DemoError('Inference disabled')
        return faq_runner(question)
    if route.specialist=='CardsGuidance': return cards_guidance()
    if route.specialist=='urgent': return urgent_support()
    from .support_models import Clarification
    return Clarification().model_dump()
