import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
import shap
import joblib

# 1. Load Data
print("Loading data...")
train_df = pl.read_csv("train.csv")
test_df = pl.read_csv("test.csv")

# The first column is often an index or ID without a name in some versions, 
# but polars usually gives it a name if it's in the CSV.
# In your case, 'Id' was shown in the JSON output.

# 2. Preprocessing
# Target variable: SeriousDlqin2yrs
target = "SeriousDlqin2yrs"
features = [col for col in train_df.columns if col not in [target, "Id", ""]]

X = train_df.select(features).to_pandas()
y = train_df.select(target).to_series().to_pandas()

X_test_final = test_df.select(features).to_pandas()

# 3. Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Model
# HistGradientBoostingClassifier is great because it handles missing values (NaNs) automatically
print("Training HistGradientBoostingClassifier...")
model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    class_weight="balanced"  # Important for imbalanced credit scoring data
)

model.fit(X_train, y_train)

# 5. Evaluation
y_pred_proba = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_pred_proba)

print(f"\nValidation ROC AUC: {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, model.predict(X_val)))

# 6. SHAP Interpretability
print("\nCalculating SHAP values for model interpretation...")
explainer = shap.TreeExplainer(model)
# Sample 100 observations for faster shap calculation
X_sample = X_val.sample(min(100, len(X_val)), random_state=42)
shap_values = explainer.shap_values(X_sample)

# 7. Make Predictions on Test Set
print("Generating predictions for test.csv...")
test_probs = model.predict_proba(X_test_final)[:, 1]

# Create submission file
submission = pd.DataFrame({
    "Id": test_df["Id"].to_list(),
    "Probability": test_probs
})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to 'submission.csv'")

# Save the model
joblib.dump(model, "credit_score_model.pkl")
print("Model saved as 'credit_score_model.pkl'")
