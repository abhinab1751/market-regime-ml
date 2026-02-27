import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap


from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import TimeSeriesSplit

from features.feature_engineering import engineer_features
from labeling.label_generator import generate_labels


def load_artifacts(model_path="models/saved_model.pkl"):
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts


def load_data(csv_path="data/raw_data.csv"):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = engineer_features(df)
    df = generate_labels(df)
    df = df.replace([float('inf'), float('-inf')], float('nan'))
    df.dropna(inplace=True)
    return df


def evaluate(model_path="models/saved_model.pkl", csv_path="data/raw_data.csv"):
    artifacts = load_artifacts(model_path)
    model     = artifacts['model']
    scaler    = artifacts['scaler']
    le        = artifacts['label_encoder']
    features  = artifacts['features']

    df = load_data(csv_path)
    df = df.dropna(subset=features + ['label'])
    X  = df[features]
    y  = le.transform(df['label'])

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    _, test_idx = splits[-1]

    X_test = X.iloc[test_idx]
    y_test = y[test_idx]

    test_mask = np.isfinite(X_test).all(axis=1)
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    X_test_sc = scaler.transform(X_test)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)

    print("\n=== Classification Report ===")
    present_classes = np.unique(y_test)
    print(classification_report(
        y_test, y_pred,
        labels=present_classes,
        target_names=le.classes_[present_classes],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=present_classes)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_[present_classes],
                yticklabels=le.classes_[present_classes])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved.")

    try:
        y_bin = label_binarize(y_test, classes=list(range(len(le.classes_))))
        roc_auc = roc_auc_score(y_bin, y_prob, multi_class='ovr', average='macro')
        print(f"\nMacro ROC-AUC (OvR): {roc_auc:.4f}")
    except Exception as e:
        print(f"ROC-AUC could not be computed: {e}")

    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature':    features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', palette='viridis')
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig("models/feature_importance.png")
        plt.show()
        print("Feature importance plot saved.")
    else:
        print("Feature importance not available for this model type (SVM).")

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test_sc)
        shap.summary_plot(shap_vals, X_test_sc, feature_names=features,
                          class_names=le.classes_, show=False)
        plt.tight_layout()
        plt.savefig("models/shap_summary.png")
        plt.show()
        print("SHAP summary plot saved.")
    except Exception as e:
        print(f"SHAP not available for this model type: {e}")


if __name__ == "__main__":
    evaluate()
