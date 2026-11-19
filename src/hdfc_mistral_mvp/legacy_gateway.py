"""Synthetic service contracts and a bounded, read-only legacy gateway adapter."""
import asyncio
import os
from typing import Literal
from urllib.parse import urlparse

import httpx2
from pydantic import model_validator

from .support_models import StrictModel


class LegacyRequest(StrictModel):
    service: Literal['cards', 'payments']
    operation: Literal['temporary_deactivation', 'beneficiary_change', 'money_transfer']

    @model_validator(mode='after')
    def allowed_pair(self):
        if (self.service == 'cards') != (self.operation == 'temporary_deactivation'):
            raise ValueError('Operation is outside this service.')
        return self


class LegacyProcedure(LegacyRequest):
    environment: Literal['synthetic']
    system: Literal['mock_card_processor', 'mock_payment_hub']
    status: Literal['ready', 'unavailable']
    verification_required: Literal[True]
    confirmation_required: Literal[True]
    next_step: Literal['secure_card_service', 'secure_beneficiary_service', 'secure_transfer_service']


STEPS = {
    'temporary_deactivation': ('cards', 'mock_card_processor', 'secure_card_service'),
    'beneficiary_change': ('payments', 'mock_payment_hub', 'secure_beneficiary_service'),
    'money_transfer': ('payments', 'mock_payment_hub', 'secure_transfer_service'),
}


class LegacyUnavailable(Exception):
    def __init__(self):
        super().__init__('Synthetic bank gateway is temporarily unavailable.')


async def get_legacy_guidance(service: str, operation: str, *, timeout: float = 2) -> dict:
    """Only procedure discovery; never accepts customer identifiers or writes."""
    try:
        request = LegacyRequest(service=service, operation=operation)
        origin = os.getenv('DEMO_LEGACY_GATEWAY_URL', 'embedded')
        transport = None
        if origin == 'embedded':
            from .mock_legacy import app
            transport = httpx2.ASGITransport(app=app)
            origin = 'http://127.0.0.1:8030'  # ASGI transport: no socket/network.
        else:
            parsed = urlparse(origin)
            if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
                    or parsed.path not in ('', '/') or parsed.username is not None
                    or parsed.password is not None or parsed.query or parsed.fragment):
                raise ValueError('Only the loopback synthetic gateway is supported.')
        # No retry/fallback: an unavailable configured gateway must be visible.
        async with asyncio.timeout(max(.001, min(2, timeout))):
            async with httpx2.AsyncClient(transport=transport, timeout=2,
                                         follow_redirects=False, trust_env=False) as client:
                async with client.stream('POST', origin.rstrip('/') + '/legacy/v1/' + request.service + '/procedure',
                                         json=request.model_dump()) as response:
                    if response.status_code != 200:
                        raise ValueError('Gateway failed.')
                    body = bytearray()
                    async for block in response.aiter_bytes():
                        body.extend(block)
                        if len(body) > 8192: raise ValueError('Oversized gateway response.')
        value = LegacyProcedure.model_validate_json(bytes(body))
        expected = STEPS[request.operation]
        if (value.service != request.service or value.operation != request.operation
                or (value.service, value.system, value.next_step) != expected or value.status != 'ready'):
            raise ValueError('Gateway contract mismatch.')
        from .support import GUIDANCE
        # Render Python-owned text, not arbitrary legacy content or model prose.
        return {'status': 'guidance', 'message': 'Synthetic legacy gateway checked. ' + GUIDANCE[request.operation]}
    except Exception:
        raise LegacyUnavailable() from None
