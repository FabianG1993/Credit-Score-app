# CreditVision AI

Prototipo educativo de evaluación asistida de riesgo. El modelo estima una **probabilidad de incumplimiento** usando datos de ejemplo/no locales; no aprueba, rechaza ni ofrece créditos.

## Ejecución local

Requiere Python 3.10+. Cree un entorno, instale y ejecute un único proceso que sirve API y web:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn api.index:app --reload
```

Abra `http://localhost:8000`. La documentación API está en `/api/docs`; salud en `/api/health` y disponibilidad en `/api/ready`.

## Pruebas y calidad

```powershell
pytest
ruff check api model.py tests
```

## Entrenamiento reproducible

Los CSV no se versionan. Con un dataset autorizado que contenga las columnas esperadas:

```powershell
python model.py --train D:\ruta\train.csv --model credit_score_model.pkl --seed 42 --threshold 0.30
```

Genera el modelo y un archivo de metadatos (hash del dataset, variables, versiones y métricas ROC-AUC, PR-AUC, KS/Gini, Brier, calibración y matriz de confusión). Los límites de bandas están centralizados en `api/config.py`; son categorías informativas, no política crediticia.

## Arquitectura

`index.html` y `app.js` consumen FastAPI. `api/schemas.py` valida rangos, `api/service.py` carga el artefacto y genera factores cualitativos, y `api/config.py` concentra configuración. La API no registra payloads: registra solo identificador, versión, latencia y errores técnicos sanitizados.

## Despliegue en Vercel

El archivo `vercel.json` sirve los archivos estáticos y enruta `/api/*`. Configure `ALLOWED_ORIGINS` con los orígenes HTTPS exactos y, si corresponde, `MODEL_PATH`. Verifique que el artefacto de demostración esté disponible en el build. El limitador en memoria no protege un entorno serverless distribuido: use un gateway o Redis. Los cold starts y SHAP pueden incrementar la latencia.

## Limitaciones y controles pendientes

No usar con datos reales sin autorización/finalidad, minimización, retención y borrado, atención de derechos, auditoría, autenticación y revisión humana. También faltan validación local, evaluación de sesgo, monitoreo de drift y aprobación legal/compliance. Consulte [model-card.md](docs/model-card.md) y [data-governance.md](docs/data-governance.md).
