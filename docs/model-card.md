# Model card — CreditVision AI

## Uso previsto

Demostración asistida para explorar probabilidad de incumplimiento. No es un sistema de decisión, aprobación, rechazo ni oferta de crédito.

## Datos y salida

El artefacto actual proviene de datos públicos/de ejemplo, no de una cartera local validada. La salida `probability_of_default` está entre 0 y 1; `risk_band` comunica una categoría configurable y `recommendation` solo indica “requiere revisión”.

## Evaluación

El script de entrenamiento produce ROC-AUC, PR-AUC, KS, Gini, Brier, matriz de confusión y curva de calibración. No hay métricas locales validadas incluidas en el repositorio.

## Limitaciones

No se ha evaluado sesgo, estabilidad temporal, drift ni desempeño por segmentos. Las explicaciones SHAP son factores técnicos cualitativos y no razones legales definitivas.
