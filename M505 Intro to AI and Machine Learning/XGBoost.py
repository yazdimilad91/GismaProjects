import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier, plot_importance

# --- Load expression (samples = rows, genes = columns) and metadata ---
X = pd.read_csv("X_tcga.csv", index_col=0)
meta = pd.read_csv("meta_tcga.csv")

# Align samples
common_ids = X.index.intersection(meta["sample"])
X = X.loc[common_ids]
meta = meta.set_index("sample").loc[common_ids]

# Map labels
label_map = {
    "Normal Tissue": 0,
    "Solid Tissue Normal": 0,
    "Primary Tumor": 1,
    "Metastatic": 1
}
y = meta["_sample_type"].map(label_map)

print("X shape:", X.shape)
print("y distribution:\n", y.value_counts())

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train XGBoost model ---
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)
xgb.fit(X_train, y_train)

# --- Evaluation ---
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_proba))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Tumor"],
            yticklabels=["Normal", "Tumor"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix (XGBoost)")
plt.tight_layout()
plt.show()

# --- Feature Importance ---
plt.figure(figsize=(8, 6))
plot_importance(xgb, max_num_features=10, importance_type="weight")
plt.title("Top 10 Important Features (XGBoost)")
plt.show()
