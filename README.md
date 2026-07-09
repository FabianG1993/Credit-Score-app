# CreditVision AI - Credit Scoring API

API en FastAPI (Serverless Vercel) para predecir el riesgo de crédito utilizando un modelo de Machine Learning (`HistGradientBoosting`). Incluye explicabilidad en tiempo real de las predicciones a través de la librería SHAP.

## 🚀 Características principales

- **Machine Learning**: Modelo `HistGradientBoostingClassifier` entrenado para predecir la probabilidad de incumplimiento de pago (Credit Scoring).
- **Interpretabilidad**: Integración con **SHAP** para entender cómo cada variable financiera del solicitante impacta en la decisión del modelo.
- **FastAPI**: Backend rápido e interactivo con documentación automática.
- **Serverless**: Configurado para desplegarse fácilmente en Vercel.

## 🛠️ Tecnologías

- Python 3.9+
- FastAPI
- Scikit-Learn
- SHAP
- Polars / Pandas
- Vercel

## 📦 Instalación y uso Local

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/FabianG1993/Credit-Score-app.git
   cd Credit-Score-app
   ```

2. **Crear y activar un entorno virtual** (opcional pero recomendado):
   ```bash
   python -m venv env
   # En Windows:
   env\Scripts\activate
   # En Mac/Linux:
   source env/bin/activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la API de forma local**:
   ```bash
   uvicorn api.index:app --reload
   ```

5. **Probar la API**:
   La API estará disponible en `http://localhost:8000`. Puedes ir a `http://localhost:8000/api/docs` para ver la documentación interactiva generada por FastAPI (Swagger UI) y probar el endpoint `/api/predict`.

## 📡 Ejemplo de Petición (Endpoint `/api/predict`)

**POST** `/api/predict`

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.2,
  "age": 45,
  "NumberOfTime30_59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.35,
  "MonthlyIncome": 5000,
  "NumberOfOpenCreditLinesAndLoans": 8,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60_89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 1
}
```

**Respuesta**:
La API devolverá la puntuación de riesgo (`risk_score`), la categoría de riesgo y un desglose (`shap_breakdown`) detallando cómo contribuyó cada variable al resultado final.
