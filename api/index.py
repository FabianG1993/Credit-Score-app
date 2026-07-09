"""
CreditVision AI — FastAPI Backend (Vercel Serverless Function)

Serves ML credit risk predictions with SHAP interpretability.
Deployed as a Vercel Python serverless function at /api/*.
"""

import logging
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("creditvision")

# ---------------------------------------------------------------------------
# App & Middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CreditVision AI API",
    description="Credit risk prediction with SHAP interpretability",
    version="2.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# GZip compression for all responses > 500 bytes
app.add_middleware(GZipMiddleware, minimum_size=500)

# CORS — restrict to allowed origins via env var; defaults to permissive for local dev
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Rate limiting — 30 requests/minute per IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Return a friendly 429 when rate limit is exceeded."""
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please wait a moment and try again."},
    )


# ---------------------------------------------------------------------------
# Model Loading (lazy, cached)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(MODEL_DIR, "credit_score_model.pkl")


@lru_cache(maxsize=1)
def load_model():
    """Load the trained model from disk. Cached after first call."""
    import joblib

    logger.info("Loading model from %s", MODEL_PATH)
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return None


@lru_cache(maxsize=1)
def load_explainer():
    """Create SHAP TreeExplainer. Cached after first call."""
    import shap

    model = load_model()
    if model is None:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP explainer initialized.")
        return explainer
    except Exception as e:
        logger.error("Failed to create SHAP explainer: %s", e)
        return None


# ---------------------------------------------------------------------------
# Feature Configuration (single source of truth)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Maps API field names (underscores) → model feature names (hyphens)
FIELD_MAP = {
    "RevolvingUtilizationOfUnsecuredLines": "RevolvingUtilizationOfUnsecuredLines",
    "age": "age",
    "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio": "DebtRatio",
    "MonthlyIncome": "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans": "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate": "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines": "NumberRealEstateLoansOrLines",
    "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents": "NumberOfDependents",
}

# Friendly display names for SHAP breakdown (single source of truth)
FRIENDLY_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit Utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 Days Late",
    "DebtRatio": "Debt Ratio",
    "MonthlyIncome": "Monthly Income",
    "NumberOfOpenCreditLinesAndLoans": "Open Credit Lines",
    "NumberOfTimes90DaysLate": "Times 90+ Days Late",
    "NumberRealEstateLoansOrLines": "Real Estate Loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 Days Late",
    "NumberOfDependents": "Dependents",
}


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------
class ApplicantData(BaseModel):
    """Input schema for credit risk prediction.

    All fields correspond to features in the Give Me Some Credit dataset.
    """

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0, le=1, description="Ratio of unsecured credit balance to credit limit"
    )
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., ge=0, description="Number of times 30-59 days past due"
    )
    DebtRatio: float = Field(
        ..., ge=0, le=5.0, description="Ratio of monthly debt to income (can exceed 1.0)"
    )
    MonthlyIncome: float = Field(..., ge=0, description="Monthly income in dollars")
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ..., ge=0, description="Number of open credit lines and loans"
    )
    NumberOfTimes90DaysLate: int = Field(
        ..., ge=0, description="Number of times 90+ days past due"
    )
    NumberRealEstateLoansOrLines: int = Field(
        ..., ge=0, description="Number of real estate loans or lines"
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., ge=0, description="Number of times 60-89 days past due"
    )
    NumberOfDependents: float = Field(
        ..., ge=0, description="Number of dependents in the household"
    )

    model_config = {"json_schema_extra": {"example": {
        "RevolvingUtilizationOfUnsecuredLines": 0.2,
        "age": 45,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 5000,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1,
    }}}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Check model and SHAP explainer loading status."""
    model = load_model()
    explainer = load_explainer()
    status = {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
    }
    if not status["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model is not available.")
    return status


@app.post("/api/predict")
@limiter.limit("30/minute")
async def predict_risk(data: ApplicantData, request: Request):
    """Predict credit default risk and return SHAP feature breakdown.

    Returns:
        risk_score: Probability of default (0-1)
        risk_percentage: Probability as percentage
        risk_category: Low / Medium / High
        base_value: Baseline default probability from the model
        shap_breakdown: Top feature impacts with friendly names
        shap_error: Error message if SHAP computation failed (optional)
    """
    import numpy as np

    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.")

    # Build feature array from the request (NumPy instead of DataFrame for performance)
    payload = data.model_dump()
    feature_values = [payload[field] for field in FIELD_MAP]
    # Map to model column order
    feature_dict = {
        feature_name: payload[field_name]
        for field_name, feature_name in FIELD_MAP.items()
    }
    ordered_values = [feature_dict[col] for col in FEATURE_COLUMNS]
    features_array = np.array([ordered_values], dtype=np.float64)

    # Predict probability of default (class 1)
    try:
        prob = model.predict_proba(features_array)[0, 1]
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=422, detail="Prediction failed. Please check input values.")

    risk_score = float(prob)
    risk_category = (
        "Low Risk" if risk_score <= 0.2 else
        "Medium Risk" if risk_score <= 0.5 else
        "High Risk"
    )

    # SHAP interpretability
    shap_error = None
    feature_impacts = []
    base_value = 0.0

    explainer = load_explainer()
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(features_array)

            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])

            feature_impacts = [
                {
                    "feature": col,
                    "friendly_name": FRIENDLY_NAMES.get(col, col),
                    "value": float(ordered_values[i]),
                    "impact": float(sv[i]),
                }
                for i, col in enumerate(FEATURE_COLUMNS)
            ]
            feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
        except Exception as e:
            logger.warning("SHAP computation failed: %s", e)
            shap_error = "SHAP interpretability is temporarily unavailable."
    else:
        shap_error = "SHAP explainer could not be initialized."

    response = {
        "risk_score": risk_score,
        "risk_percentage": round(risk_score * 100, 2),
        "risk_category": risk_category,
        "base_value": float(base_value),
        "shap_breakdown": feature_impacts,
    }
    if shap_error:
        response["shap_error"] = shap_error

    logger.info(
        "Prediction: score=%.4f category=%s",
        risk_score,
        risk_category,
    )

    return response
