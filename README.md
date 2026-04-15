# Temporal DNA Model Training Pipeline

This repository contains the training pipeline for the "Temporal DNA" network traffic classification model. It processes raw network flow data, handles class imbalances, and trains a weighted ensemble model (XGBoost + Random Forest).

## Overview

The pipeline performs the following steps:
1. **Data Preprocessing:** Drops metadata (IPs, Flow IDs, Timestamps), coerces data types, and handles `NaN` and infinite values via median imputation.
2. **Resampling:** Applies undersampling to the majority class (`BENIGN`) to prevent model bias.
3. **Feature Scaling:** Uses standard scaling and clips extreme outliers.
4. **Model Training:** Trains an XGBoost classifier and a Random Forest classifier.
5. **Ensembling:** Blends predictions using a weighted average based on the individual models' validation F1 scores.
6. **Export:** Saves the trained models, scaler, label encoder, and ensemble weights to a single `.joblib` bundle.

## Prerequisites

Requires Python 3.8+ and the following packages:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn joblib
```
## Setup
The script expects a CSV dataset located at data/combined_cleaned.csv.
Ensure your dataset includes a Label column for the target classes. The script is configured to specifically downsample the BENIGN class to 50,000 samples to balance the training set.

## Usage
``` bash
python temporal_dna.py
```
Outputs

During execution, the script will output classification metrics to the console:

    Ensemble Weighted F1 Score

    Overall Accuracy

    Detailed Classification Report (Precision, Recall, F1 per class)

Upon successful completion, the script exports the artifact bundle to the current directory:

    temporal_dna_model.joblib

Using the Saved Model (Inference)

To load and use the saved model bundle in your detection application:
Python

import joblib
import numpy as np

# Load the bundle
bundle = joblib.load('temporal_dna_model.joblib')
xgb_mod = bundle['xgb']
rf_mod = bundle['rf']
scaler = bundle['scaler']
le = bundle['le']
weights = bundle['weights']

# Assume `X_new` is a raw pandas DataFrame of new network flows matching the training features
# ... [Apply the same NaN/Inf handling and median imputation here] ...

# Scale features
X_scaled = scaler.transform(X_new)
X_scaled = np.clip(X_scaled, -10, 10)

# Generate ensemble predictions
xgb_probs = xgb_mod.predict_proba(X_scaled)
rf_probs = rf_mod.predict_proba(X_scaled)

blended_probs = (weights[0] * xgb_probs) + (weights[1] * rf_probs)
final_preds = np.argmax(blended_probs, axis=1)

# Map back to string labels
predicted_labels = le.inverse_transform(final_preds)
