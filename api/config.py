"""Configuration shared by the API and model service."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "credit_score_model.pkl"))
MODEL_METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", f"{MODEL_PATH}.metadata.json"))
MODEL_VERSION_FALLBACK = os.getenv("MODEL_VERSION", "legacy-demo-model")
DEFAULT_ALLOWED_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000"
ALLOWED_ORIGINS = [value.strip() for value in os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",") if value.strip()]

# Communication bands for this demonstrator, not credit policy.
RISK_BANDS = ((0.10, "Riesgo bajo"), (0.30, "Riesgo medio"), (1.00, "Riesgo alto"))
DISCLAIMER = "Resultado informativo para demostración. No constituye aprobación, rechazo ni oferta de crédito. Requiere revisión humana y controles adicionales."

FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents",
]
FIELD_MAP = {
    "RevolvingUtilizationOfUnsecuredLines": "RevolvingUtilizationOfUnsecuredLines", "age": "age",
    "NumberOfTime30_59DaysPastDueNotWorse": "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio": "DebtRatio",
    "MonthlyIncome": "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans": "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate": "NumberOfTimes90DaysLate", "NumberRealEstateLoansOrLines": "NumberRealEstateLoansOrLines",
    "NumberOfTime60_89DaysPastDueNotWorse": "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents": "NumberOfDependents",
}
FRIENDLY_NAMES = {
    "RevolvingUtilizationOfUnsecuredLines": "Utilización de crédito", "age": "Edad", "NumberOfTime30-59DaysPastDueNotWorse": "Atrasos de 30 a 59 días",
    "DebtRatio": "Relación deuda/ingreso", "MonthlyIncome": "Ingreso mensual", "NumberOfOpenCreditLinesAndLoans": "Líneas de crédito abiertas",
    "NumberOfTimes90DaysLate": "Atrasos de 90 días o más", "NumberRealEstateLoansOrLines": "Préstamos inmobiliarios",
    "NumberOfTime60-89DaysPastDueNotWorse": "Atrasos de 60 a 89 días", "NumberOfDependents": "Dependientes",
}
