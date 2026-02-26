"""
Cognitive Firewall v2 - Model Trainer
Trains an ensemble of ML models with:
  - Manual oversampling (no imblearn required)
  - Class-weighted learning
  - Stratified K-fold cross-validation
  - Feature importance analysis
  - Full metrics report

Designed to reach 92%+ weighted F1 on the combined India phishing dataset.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feature_extraction import extract_features_batch, FEATURE_NAMES, NUM_FEATURES


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    required = {"url", "binary_label"}
    if not required.issubset(df.columns):
        # Try to infer
        url_col = next((c for c in df.columns if "url" in c.lower()), df.columns[0])
        label_col = next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), df.columns[-1])
        df = df[[url_col, label_col]].copy()
        df.columns = ["url", "label"]

        def _to_binary(lbl):
            lbl = str(lbl).strip().lower()
            return 0 if "safe" in lbl or lbl == "0" else 1

        df["binary_label"] = df["label"].apply(_to_binary)

    df = df[["url", "binary_label"]].dropna()
    df["url"] = df["url"].astype(str).str.strip()
    return df


# ── Manual oversampling (no imblearn) ─────────────────────────────────────────

def manual_oversample(X: np.ndarray, y: np.ndarray, target_ratio: float = 0.4) -> tuple:
    """
    Oversample the minority class by duplicating with small Gaussian noise.
    target_ratio: desired fraction of minority class (0.4 = 40% safe)
    """
    classes, counts = np.unique(y, return_counts=True)
    minority_cls = classes[np.argmin(counts)]
    majority_cls = classes[np.argmax(counts)]
    n_majority = counts[np.argmax(counts)]

    target_minority = int(n_majority * target_ratio / (1 - target_ratio))
    current_minority = counts[np.argmin(counts)]

    if target_minority <= current_minority:
        return X, y

    minority_idx = np.where(y == minority_cls)[0]
    n_needed = target_minority - current_minority
    chosen = np.random.choice(minority_idx, size=n_needed, replace=True)

    X_minority = X[chosen].copy()
    # Add tiny Gaussian noise (0.5% of std) to avoid pure duplicates
    noise = np.random.normal(0, 0.005 * X.std(axis=0).clip(0.001), X_minority.shape)
    X_minority += noise

    X_aug = np.vstack([X, X_minority])
    y_aug = np.concatenate([y, np.full(n_needed, minority_cls)])

    # Shuffle
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


# ── Feature extraction ────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, verbose: bool = True) -> tuple:
    if verbose:
        print(f"  Extracting {NUM_FEATURES} features for {len(df)} URLs...")
    t0 = time.time()
    feat_df = extract_features_batch(df["url"].tolist())

    # Ensure column alignment
    for col in FEATURE_NAMES:
        if col not in feat_df.columns:
            feat_df[col] = 0
    feat_df = feat_df[FEATURE_NAMES].fillna(0).astype(float)

    X = feat_df.values
    y = df["binary_label"].values.astype(int)

    if verbose:
        print(f"  Feature matrix: {X.shape}  ({time.time()-t0:.1f}s)")
        print(f"  Class distribution: safe={np.sum(y==0)}, malicious={np.sum(y==1)}")
    return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def train(csv_path: str, models_dir: str, seed: int = 42) -> dict:
    np.random.seed(seed)
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  COGNITIVE FIREWALL v2 — MODEL TRAINER")
    print("=" * 65)

    # Load
    print("\n[1/5] Loading dataset...")
    df = load_dataset(csv_path)
    print(f"  Loaded {len(df)} URLs")

    # Feature extraction
    print("\n[2/5] Feature extraction...")
    X, y = build_feature_matrix(df)

    # Train/test split (stratified)
    print("\n[3/5] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Train safe/malicious: {np.sum(y_train==0)}/{np.sum(y_train==1)}")

    # Oversample minority class in training set only
    X_train_aug, y_train_aug = manual_oversample(X_train, y_train, target_ratio=0.35)
    print(f"  After oversampling — Train: {len(X_train_aug)} "
          f"(safe: {np.sum(y_train_aug==0)}, malicious: {np.sum(y_train_aug==1)})")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)

    # ── Model definitions ─────────────────────────────────────────────────────
    print("\n[4/5] Training models...")

    class_weight = {0: 3.0, 1: 1.0}  # penalise safe misclassification more

    model_configs = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight=class_weight,
                random_state=seed,
                n_jobs=-1,
            ),
            False,  # needs_scaling
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                class_weight=class_weight,
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                min_samples_split=5,
                random_state=seed,
            ),
            False,
        ),
        (
            "LogisticRegression",
            LogisticRegression(
                C=0.5,
                max_iter=2000,
                class_weight=class_weight,
                random_state=seed,
                solver="lbfgs",
            ),
            True,  # needs_scaling
        ),
    ]

    trained_models = {}
    results = {}

    for name, model, needs_scaling in model_configs:
        t0 = time.time()
        print(f"\n  → {name}...")

        Xtr = X_train_scaled if needs_scaling else X_train_aug
        Xte = X_test_scaled if needs_scaling else X_test

        model.fit(Xtr, y_train_aug)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        metrics = _compute_metrics(y_test, y_pred, y_prob)
        results[name] = metrics
        trained_models[name] = (model, needs_scaling)

        elapsed = time.time() - t0
        print(f"    Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1_weighted']:.4f} | "
              f"AUC: {metrics['auc_roc']:.4f} | "
              f"Prec(safe): {metrics['precision_safe']:.4f} | "
              f"Rec(safe): {metrics['recall_safe']:.4f} | "
              f"{elapsed:.1f}s")

        joblib.dump(model, os.path.join(models_dir, f"{name.lower()}.pkl"))

    # ── Voting Ensemble ───────────────────────────────────────────────────────
    print("\n  → VotingEnsemble (RF + ET + GB)...")
    voting = VotingClassifier(
        estimators=[
            ("rf", trained_models["RandomForest"][0]),
            ("et", trained_models["ExtraTrees"][0]),
            ("gb", trained_models["GradientBoosting"][0]),
        ],
        voting="soft",
        n_jobs=-1,
    )
    voting.fit(X_train_aug, y_train_aug)
    y_pred_v = voting.predict(X_test)
    y_prob_v = voting.predict_proba(X_test)[:, 1]

    metrics_v = _compute_metrics(y_test, y_pred_v, y_prob_v)
    results["VotingEnsemble"] = metrics_v
    trained_models["VotingEnsemble"] = (voting, False)

    print(f"    Acc: {metrics_v['accuracy']:.4f} | "
          f"F1: {metrics_v['f1_weighted']:.4f} | "
          f"AUC: {metrics_v['auc_roc']:.4f} | "
          f"Prec(safe): {metrics_v['precision_safe']:.4f} | "
          f"Rec(safe): {metrics_v['recall_safe']:.4f}")
    joblib.dump(voting, os.path.join(models_dir, "ensemble.pkl"))

    # ── 5-fold Cross-Validation on best model ─────────────────────────────────
    print("\n  → 5-fold CV on VotingEnsemble...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(
        voting, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1
    )
    print(f"    CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    results["VotingEnsemble"]["cv_f1_mean"] = round(float(cv_scores.mean()), 4)
    results["VotingEnsemble"]["cv_f1_std"] = round(float(cv_scores.std()), 4)

    # ── Feature importance ────────────────────────────────────────────────────
    rf_model = trained_models["RandomForest"][0]
    importances = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": rf_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)

    # ── Save scaler + metadata ────────────────────────────────────────────────
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n[5/5] RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Model':<22} {'Accuracy':>9} {'F1(weighted)':>13} {'AUC-ROC':>9} {'Prec(safe)':>11} {'Rec(safe)':>10}")
    print("-" * 65)
    for name, m in results.items():
        print(
            f"  {name:<20} {m['accuracy']:>9.4f} {m['f1_weighted']:>13.4f} "
            f"{m['auc_roc']:>9.4f} {m['precision_safe']:>11.4f} {m['recall_safe']:>10.4f}"
        )
    print("=" * 65)

    best_name = max(results, key=lambda n: results[n]["f1_weighted"])
    best = results[best_name]
    print(f"\n  ✅ Best model: {best_name}")
    print(f"     F1 (weighted): {best['f1_weighted']:.4f}")
    print(f"     Accuracy:      {best['accuracy']:.4f}")
    print(f"     AUC-ROC:       {best['auc_roc']:.4f}")

    # Confusion matrix for best model
    print(f"\n  Confusion Matrix ({best_name}):")
    best_model, best_scaling = trained_models[best_name]
    Xte_final = X_test_scaled if best_scaling else X_test
    cm = confusion_matrix(y_test, best_model.predict(Xte_final))
    print(f"                 Predicted Safe  Predicted Malicious")
    print(f"  Actual Safe          {cm[0][0]:>5}              {cm[0][1]:>5}")
    print(f"  Actual Malicious     {cm[1][0]:>5}              {cm[1][1]:>5}")

    print(f"\n  Top 10 most predictive features:")
    for i, row in importances.head(10).iterrows():
        bar = "█" * int(row["importance"] * 300)
        print(f"    {row['feature']:<35} {row['importance']:.4f}  {bar}")

    print(f"\n  Models saved to: {models_dir}")
    return results


def _compute_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1_malicious": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "f1_safe": round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "precision_malicious": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_malicious": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "precision_safe": round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "recall_safe": round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, "data", "combined.csv")
    models_dir = os.path.join(base, "models")
    train(csv_path, models_dir)