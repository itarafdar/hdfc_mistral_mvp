"""Bank-owned input scan. Raw input never becomes a model/tool-session payload."""
import re
import unicodedata

from pydantic import Field
from .support_models import StrictModel, Rejected


class SafetyRejected(Exception):
    """Safe reason only; never attach the submitted text or matched values."""
    def __init__(self, reason='adversarial'):
        self.result = Rejected(reason=reason).model_dump()
        super().__init__('Support request rejected.')


class ScanResult(StrictModel):
    question: str = Field(min_length=1, max_length=2000)
    categories: tuple[str, ...] = ()


# Ordered expressions remove whole labelled values before generic digit scans.
# Conservative matching is intentional: IDs are never partially revealed.
PATTERNS = tuple((name, re.compile(pattern, re.I)) for name, pattern in (
    ('SECRET', r'\b(?:otp|one[ -]time\s+password|(?:upi\s+|atm\s+|m)?pin|cvv|cvc|password|passcode|api[ _-]?key|access[ _-]?token|authorization)\b\s*(?:(?:is\b|[:=#])\s*)?(?!(?:is|for|of|policy|reset|change|should|must)\b)(?:[0-9](?:[ \t-]*[0-9]){2,}|"[^"\n]+"|\x27[^\x27\n]+\x27|[^\s,;]+)'),
    ('BANK_ID', r'\b(?:account(?:\s*(?:number|no\.?))|a/c|card(?:\s*(?:number|no\.?))|customer\s*(?:id|number)|cif|crn|utr|rrn|transaction\s*(?:id|reference)|receipt\s*(?:id|number))\s*(?:is\b\s*|[:=#]\s*)?(?:[0-9](?:[ \t-]*[0-9]){2,}|[a-z0-9][a-z0-9_-]{2,})'),
    ('BALANCE', r'\b(?:my\s+)?(?:(?:available|current|closing)\s+)?balance\s*(?:(?:is|of)\b\s*)?(?:[:=]\s*)?(?:(?:INR|Rs\.?)\s*|₹\s*)?[0-9][0-9,.]*(?:\s*(?:rupees|INR))?'),
    ('DOB', r'\b(?:dob|date\s+of\s+birth|born\s+on)\s*(?:is\b\s*|[:=]\s*)?\d{1,4}[ /.-]\d{1,2}[ /.-]\d{1,4}'),
    ('ADDRESS', r'\b(?:(?:my|mailing|residential|home)\s+)?address\s*(?:is\b|[:=])\s*[^\n.;!?]{1,200}'),
    ('NAME', r'\b(?:my\s+name\s+is|customer\s+name\s*[:=]|name\s*[:=])\s*[^\n,;.!?]{1,80}'),
    ('PAN', r'(?<!\w)(?:[a-z][ \t-]*){5}(?:[0-9][ \t-]*){4}[a-z](?!\w)'),
    ('IFSC', r'\b[a-z]{4}0[a-z0-9]{6}\b'),
    ('PASSPORT', r'\b[a-z][0-9]{7}\b'),
    ('DRIVING_LICENCE', r'\b[a-z]{2}[ -]?[0-9]{2}[ -]?[0-9]{4}[ -]?[0-9]{7}\b'),
    ('GROUPED_ID', r'(?<!\w)[0-9]{4}(?:[./][0-9]{4}){2,4}(?!\w)'),
    ('PHONE', r'(?<!\w)(?:\+?91[ -]*)?\(?[6-9][0-9]{4}\)?[ -]*[0-9]{5}(?!\w)'),
    ('CONTACT', r'(?<![\w@])[\w.+-]+\s*@\s*[\w.-]+'),
    ('NUMBER', r'(?<!\w)(?:\+\s*)?[0-9](?:[ \t-]*[0-9]){5,}(?!\w)'),
))

INJECTION = re.compile(
    r'\b(?:ignore|override|disregard)\b.{0,60}\b(?:instructions?|rules?|safety|guardrails?|policy)\b'
    r'|\b(?:reveal|print|show|leak)\b.{0,60}\b(?:system\s+prompt|hidden\s+instructions?|credentials?)\b'
    r'|\b(?:bypass|disable)\b.{0,40}\b(?:approval|confirmation|authorization|authentication|safety|guardrails?)\b'
    r'|\b(?:jailbreak|developer\s+mode|drop\s+table)\b|<\|(?:system|im_start)\|>', re.I | re.S)


def scan_banking_input(question: str) -> ScanResult:
    """Mandatory Python scan tool, called before any AI SDK tool/session exists.

    Returns only redacted text and category names. No original values or hashes
    are returned, logged or retained. Regex cannot identify every name/address.
    """
    if not isinstance(question, str) or not 1 <= len(question) <= 2000:
        raise SafetyRejected('missing_context')
    text = ''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c
        for c in unicodedata.normalize('NFKC', question) if unicodedata.category(c) != 'Cf')
    if any(ord(c) < 32 and c not in '\n\r\t' for c in text):
        raise SafetyRejected('adversarial')
    # Inspect before redaction so secret labels cannot hide an override request.
    if INJECTION.search(text):
        raise SafetyRejected('adversarial')
    categories = []
    for category, pattern in PATTERNS:
        text, count = pattern.subn('[REDACTED_'+category+']', text)
        if count: categories.append(category)
    # Preserve the public banking intent without the old scanner's possessive labels.
    text = re.sub(r'\bmy\s+card\b', 'a card', text, flags=re.I)
    text = re.sub(r'\bmy\s+account\b', 'an account', text, flags=re.I)
    remainder = re.sub(r'\[REDACTED_[A-Z_]+\]', '', text)
    words = set(re.findall(r'[a-z]+', remainder.casefold()))
    if not words - {'my','is','the','a','an','and','here','this','it','please'}:
        raise SafetyRejected('missing_context')
    if len(text) > 2000: raise SafetyRejected('missing_context')
    return ScanResult(question=text.strip(), categories=tuple(categories))


def require_redacted(question):
    """Validate already screened data crossing an internal session boundary."""
    if scan_banking_input(question).question != question:
        raise SafetyRejected('adversarial')
    return question
