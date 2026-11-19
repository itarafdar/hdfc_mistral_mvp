"""Mock card/payment legacy APIs. No real HDFC connection, customer data or writes."""
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .legacy_gateway import LegacyRequest, LegacyProcedure, STEPS

app = FastAPI(title='Synthetic legacy bank gateway')


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    return JSONResponse(status_code=422, content={'status': 'invalid_request'})


def procedure(params: LegacyRequest, service: str):
    if params.service != service:
        return JSONResponse(status_code=422, content={'status': 'invalid_request'})
    _, system, next_step = STEPS[params.operation]
    return LegacyProcedure(**params.model_dump(), environment='synthetic', system=system,
        status='ready', verification_required=True, confirmation_required=True, next_step=next_step)


@app.post('/legacy/v1/cards/procedure', response_model=LegacyProcedure)
async def card_procedure(params: LegacyRequest):
    return procedure(params, 'cards')


@app.post('/legacy/v1/payments/procedure', response_model=LegacyProcedure)
async def payment_procedure(params: LegacyRequest):
    return procedure(params, 'payments')
