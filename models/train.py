import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from features.feature_engineering import engineer_features
from labeling.label_generator import generate_labels

FEATURE_COLS = [
    'SMA_20', 'SMA_50', 'SMA_200', 'EMA_20',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'ROC_10', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
    'ATR', 'Rolling_Std', 'Rolling_Return', 'Z_Score', 'Drawdown',
    'Volume_Pct_Change', 'SMA_20_50_Cross', 'SMA_50_200_Cross'
]

MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'RandomForest':       RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'XGBoost':            XGBClassifier(n_estimators=200, eval_metric='mlogloss', random_state=42),
    'SVM':                SVC(kernel='rbf', class_weight='balanced', probability=True),
}


def load_and_prepare(csv_path: str = "data/raw_data.csv"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = engineer_features(df)
    df = generate_labels(df)

    df = df.replace([float('inf'), float('-inf')], float('nan'))
    df = df.dropna(subset=FEATURE_COLS + ['label'])

    X = df[FEATURE_COLS]
    y = df['label']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    return X, y_enc, le


def walk_forward_train(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_mask = np.isfinite(X_train).all(axis=1)
        val_mask   = np.isfinite(X_val).all(axis=1)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_val,   y_val   = X_val[val_mask],     y_val[val_mask]

        if len(X_train) == 0 or len(X_val) == 0:
            print(f"  Fold {fold+1} — Skipped (empty after cleaning)")
            continue

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc   = scaler.transform(X_val)

        model.fit(X_train_sc, y_train)
        preds = model.predict(X_val_sc)
        report = classification_report(
            y_val, preds,
            output_dict=True,
            zero_division=0
        )
        f1 = report['macro avg']['f1-score']
        results.append(f1)
        print(f"  Fold {fold+1} — Macro F1: {f1:.4f}")

    if results:
        print(f"  Mean Macro F1: {np.mean(results):.4f}\n")
    return results


def run():
    print("Loading and preparing data...")
    X, y, le = load_and_prepare()
    print(f"Dataset shape: {X.shape}, Classes: {le.classes_}\n")

    best_model_name = None
    best_f1 = -1

    for name, model in MODELS.items():
        print(f"Training {name}...")
        fold_f1s = walk_forward_train(X, y, model)
        if fold_f1s:
            mean_f1 = np.mean(fold_f1s)
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_model_name = name

    best_model_name = 'XGBoost'
    print(f"\nFinal model: {best_model_name} (supports feature importance + SHAP)")
    print("Training final model on full dataset...")

    clean_mask = np.isfinite(X).all(axis=1)
    X_clean = X[clean_mask]
    y_clean = y[clean_mask]

    final_model = MODELS[best_model_name]
    scaler_final = StandardScaler()
    X_sc = scaler_final.fit_transform(X_clean)
    final_model.fit(X_sc, y_clean)

    os.makedirs("models", exist_ok=True)
    with open("models/saved_model.pkl", "wb") as f:
        pickle.dump({
            'model':         final_model,
            'scaler':        scaler_final,
            'label_encoder': le,
            'features':      FEATURE_COLS
        }, f)

    print("Model saved to models/saved_model.pkl")


if __name__ == "__main__":
    run()