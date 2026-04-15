import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

warnings.filterwarnings('ignore')


SEED = 42
SPLIT_RATIO = 0.30
BENIGN_LIMIT = 50000

raw_df = pd.read_csv('data/combined_cleaned.csv', low_memory=False)

# Strip identifiers and PII
meta_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Timestamp']
features = raw_df.drop(columns=[c for c in meta_cols if c in raw_df.columns])
labels = features.pop('Label')

# Cast to numeric and scrub infinities/NaNs
for col in features.columns:
    features[col] = pd.to_numeric(features[col], errors='coerce')

features = features.replace([np.inf, -np.inf], np.nan).dropna()
labels = labels[features.index]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=SPLIT_RATIO, random_state=SEED, stratify=labels
)

# Downsample majority class
sampler = RandomUnderSampler(sampling_strategy={'BENIGN': BENIGN_LIMIT}, random_state=SEED)
X_train, y_train = sampler.fit_resample(X_train, y_train)

# Clean up types and residual nulls
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

for col in X_train.columns:
    X_train[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test[col].replace([np.inf, -np.inf], np.nan, inplace=True)

    col_median = X_train[col].median()
    X_train[col].fillna(col_median, inplace=True)
    X_test[col].fillna(col_median, inplace=True)

# Normalization
norm_scaler = StandardScaler()
X_train_scaled = norm_scaler.fit_transform(X_train)
X_test_scaled = norm_scaler.transform(X_test)

X_train_scaled = np.clip(X_train_scaled, -10, 10)
X_test_scaled = np.clip(X_test_scaled, -10, 10)

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# --- Model Training ---

# Gradient Boosting
xgb_mod = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=SEED, n_jobs=-1)
xgb_mod.fit(X_train_scaled, y_train_encoded, verbose=0)
xgb_preds = xgb_mod.predict(X_test_scaled)
xgb_f1 = f1_score(y_test_encoded, xgb_preds, average='weighted')

# Random Forest
rf_mod = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=SEED, n_jobs=-1)
rf_mod.fit(X_train_scaled, y_train_encoded)
rf_preds = rf_mod.predict(X_test_scaled)
rf_f1 = f1_score(y_test_encoded, rf_preds, average='weighted')

# Weigh models based on their relative F1 performance
model_weights = np.array([xgb_f1, rf_f1]) / (xgb_f1 + rf_f1)

xgb_probs = xgb_mod.predict_proba(X_test_scaled)
rf_probs = rf_mod.predict_proba(X_test_scaled)

blended_probs = (model_weights[0] * xgb_probs) + (model_weights[1] * rf_probs)
final_preds = np.argmax(blended_probs, axis=1)

print(f"Ensemble Weighted F1: {f1_score(y_test_encoded, final_preds, average='weighted'):.4f}")
print(f"Overall Accuracy: {accuracy_score(y_test_encoded, final_preds):.4f}")
print("\nDetailed Report:")
print(classification_report(y_test_encoded, final_preds, target_names=encoder.classes_, digits=4))

bundle = {
    'xgb': xgb_mod,
    'rf': rf_mod,
    'scaler': norm_scaler,
    'le': encoder,
    'weights': model_weights
}
joblib.dump(bundle, 'temporal_dna_model.joblib')
