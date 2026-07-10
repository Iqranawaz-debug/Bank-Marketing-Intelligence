"""
Trains the final model used by the Streamlit app, using the exact same
cleaning / feature engineering / preprocessing logic as the analysis notebook.
Run once to produce model.pkl, columns.pkl, and bank_clean.csv.
"""
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

df = pd.read_csv('bank.csv')

# --- Missing value treatment (unknown -> NaN -> meaningful category) ---
df.replace('unknown', np.nan, inplace=True)
df['poutcome'] = df['poutcome'].fillna('not_contacted')
df['contact'] = df['contact'].fillna('other_method')
df['job'] = df['job'].fillna('unknown')
df['education'] = df['education'].fillna('unknown')

# --- Feature engineering ---
df['was_contacted_previously'] = df['pdays'].apply(lambda x: 0 if x == -1 else 1)

# --- Leakage fix ---
df.drop(columns=['duration'], inplace=True, errors='ignore')

# --- Outlier treatment ---
df['balance'] = winsorize(df['balance'], limits=[0.01, 0.01])
df['campaign'] = winsorize(df['campaign'], limits=[0.0, 0.01])

# Save a clean, human-readable version for the EDA dashboard
df.to_csv('bank_clean.csv', index=False)

# --- Preprocessing for modeling ---
model_df = df.copy()
model_df['deposit'] = model_df['deposit'].map({'yes': 1, 'no': 0})
categorical_cols = model_df.select_dtypes(include='object').columns.tolist()
model_df = pd.get_dummies(model_df, columns=categorical_cols, drop_first=True)

X = model_df.drop(columns=['deposit'])
y = model_df['deposit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                       scale_pos_weight=scale_pos, eval_metric='logloss',
                       random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

metrics = {
    'Accuracy': accuracy_score(y_test, preds),
    'Precision': precision_score(y_test, preds),
    'Recall': recall_score(y_test, preds),
    'F1': f1_score(y_test, preds),
    'ROC-AUC': roc_auc_score(y_test, probs)
}
print("Test metrics:", metrics)

joblib.dump(model, 'model.pkl')
joblib.dump(list(X.columns), 'columns.pkl')
joblib.dump(metrics, 'metrics.pkl')

# Feature importance for the app
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.to_csv('feature_importances.csv', header=['importance'])

print("Saved: model.pkl, columns.pkl, metrics.pkl, bank_clean.csv, feature_importances.csv")
