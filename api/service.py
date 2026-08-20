"""Safe model loading and inference. It never logs applicant data."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any
import numpy as np

from api.config import FEATURE_COLUMNS, FIELD_MAP, FRIENDLY_NAMES, MODEL_METADATA_PATH, MODEL_PATH, MODEL_VERSION_FALLBACK, RISK_BANDS
from api.schemas import ApplicantData, ExplanatoryFactor


class ModelUnavailableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_artifact() -> tuple[Any, dict[str, Any]]:
    if not MODEL_PATH.is_file():
        raise ModelUnavailableError("El artefacto del modelo no está disponible.")
    import joblib
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8")) if MODEL_METADATA_PATH.is_file() else {}
    return model, metadata


def model_version() -> str:
    try:
        _, metadata = load_artifact()
        return str(metadata.get("model_version", MODEL_VERSION_FALLBACK))
    except Exception:
        return MODEL_VERSION_FALLBACK


def risk_band(probability: float) -> str:
    return next(label for upper, label in RISK_BANDS if probability <= upper)


def ordered_values(data: ApplicantData) -> list[float]:
    values = data.model_dump()
    mapped = {model_name: values[api_name] for api_name, model_name in FIELD_MAP.items()}
    return [float(mapped[column]) for column in FEATURE_COLUMNS]


def validate_feature_order(model: Any) -> None:
    names = getattr(model, "feature_names_in_", None)
    if names is not None and list(names) != FEATURE_COLUMNS:
        raise ModelUnavailableError("El orden de variables del modelo no coincide con la configuración.")


def predict(data: ApplicantData) -> tuple[float, str, list[ExplanatoryFactor]]:
    model, _ = load_artifact()
    validate_feature_order(model)
    values = ordered_values(data)
    probability = float(model.predict_proba(np.array([values], dtype=np.float64))[0, 1])
    if not 0 <= probability <= 1:
        raise ModelUnavailableError("El modelo devolvió una probabilidad inválida.")
    return probability, risk_band(probability), explain(model, values)


def explain(model: Any, values: list[float]) -> list[ExplanatoryFactor]:
    """Return qualitative SHAP factors; do not mislabel log-odds as probabilities."""
    try:
        import shap
        result = shap.TreeExplainer(model).shap_values(np.array([values], dtype=np.float64))
        impacts = result[1][0] if isinstance(result, list) else np.asarray(result)[0]
        top = sorted(enumerate(impacts), key=lambda item: abs(float(item[1])), reverse=True)[:5]
        return [ExplanatoryFactor(feature=FEATURE_COLUMNS[i], label=FRIENDLY_NAMES[FEATURE_COLUMNS[i]], direction="aumenta" if float(impact) > 0 else "reduce") for i, impact in top]
    except Exception:
        return []
