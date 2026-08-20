"""Pydantic schemas. Limits reject implausible demo values."""
from pydantic import BaseModel, Field


class ApplicantData(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, le=1)
    age: int = Field(..., ge=18, le=100)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0, le=24)
    DebtRatio: float = Field(..., ge=0, le=5)
    MonthlyIncome: float = Field(..., ge=0, le=1_000_000)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, le=100)
    NumberOfTimes90DaysLate: int = Field(..., ge=0, le=24)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0, le=20)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0, le=24)
    NumberOfDependents: int = Field(..., ge=0, le=20)


class ExplanatoryFactor(BaseModel):
    feature: str
    label: str
    direction: str


class PredictionResponse(BaseModel):
    prediction_id: str
    model_version: str
    probability_of_default: float = Field(ge=0, le=1)
    risk_band: str
    recommendation: str
    disclaimer: str
    explanatory_factors: list[ExplanatoryFactor]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
