"""CreditVision AI FastAPI entry point."""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.config import ALLOWED_ORIGINS, DISCLAIMER
from api.schemas import ApplicantData, HealthResponse, PredictionResponse
from api.service import ModelUnavailableError, load_artifact, model_version, predict

logger = logging.getLogger("creditvision")
app = FastAPI(title="CreditVision AI API", version="3.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; script-src 'self'; img-src 'self'; connect-src 'self'"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": "Los datos de demostración son incompletos o están fuera de los rangos permitidos."})


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Demasiadas solicitudes. Espera un momento e inténtalo de nuevo."})


@app.get("/api/health", response_model=HealthResponse)
@limiter.limit("60/minute")
async def health_check(request: Request):
    try:
        load_artifact()
    except Exception:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    return HealthResponse(status="ok", model_loaded=True, model_version=model_version())


@app.get("/api/ready", response_model=HealthResponse)
@limiter.limit("60/minute")
async def readiness_check(request: Request):
    return await health_check(request)


@app.post("/api/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
async def predict_risk(data: ApplicantData, request: Request):
    started = time.perf_counter()
    prediction_id = str(uuid.uuid4())
    try:
        probability, band, factors = predict(data)
    except ModelUnavailableError as exc:
        logger.error("prediction_id=%s technical_error=%s", prediction_id, str(exc))
        raise HTTPException(status_code=503, detail="El servicio de modelo no está disponible.")
    except Exception:
        logger.exception("prediction_id=%s technical_error=inference_failure", prediction_id)
        raise HTTPException(status_code=503, detail="No fue posible calcular el resultado informativo.")
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    logger.info("prediction_id=%s model_version=%s latency_ms=%s status=200", prediction_id, model_version(), latency_ms)
    return PredictionResponse(prediction_id=prediction_id, model_version=model_version(), probability_of_default=probability, risk_band=band, recommendation="requiere revisión", disclaimer=DISCLAIMER, explanatory_factors=factors)


# One local strategy: FastAPI serves the repository root after API routes.
from api.config import ROOT_DIR
app.mount("/", StaticFiles(directory=ROOT_DIR, html=True), name="frontend")
