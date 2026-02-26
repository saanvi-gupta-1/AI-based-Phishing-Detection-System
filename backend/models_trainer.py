"""
Cognitive Firewall v6 — Model Trainer
======================================
Designed to work with the combined dataset from prepare_data.py v6
(India CSVs + PhiUSIIL Kaggle dataset).

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  Input URL                                                  │
  │      │                                                      │
  │      ▼                                                      │
  │  [1] Trusted Domain Whitelist  → SAFE instantly             │
  │      │ (not trusted)                                        │
  │      ▼                                                      │
  │  [2] Feature Extraction (90 URL features)                   │
  │      │                                                      │
  │      ▼                                                      │
  │  [3] Feature Audit (drop zero-variance, cap leaky)          │
  │      │                                                      │
  │      ▼                                                      │
  │  [4] Ensemble (RF + ET + GB + LR) via VotingClassifier      │
  │      │                                                      │
  │      ▼                                                      │
  │  [5] Calibrated Probability → Tuned Threshold               │
  │      │                                                      │
  │      ▼                                                      │
  │  SAFE / MALICIOUS + confidence score                        │
  └─────────────────────────────────────────────────────────────┘

Key design decisions:
  • OOF cross-validation  — real performance, no data leakage
  • SMOTE inside CV folds — oversampling never contaminates eval
  • Threshold tuned on OOF probs optimising F1-safe (not default 0.5)
  • Asymmetric class weights {safe:5, malicious:1} — FP costs more
  • Feature leakage audit — caps features correlated >0.85 with label
  • Overfit detector — flags train/test F1 gap > 0.12
  • XGBoost + LightGBM support — auto-enabled if installed
"""

import os, sys, json, re, copy, time, warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats as scipy_stats

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                              classification_report, confusion_matrix,
                              precision_score, recall_score)

warnings.filterwarnings("ignore")

# Optional boosting libraries
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feature_extraction import extract_features_batch, FEATURE_NAMES, NUM_FEATURES


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — TRUSTED DOMAIN WHITELIST
#  Hard override: these domains bypass ML entirely and return SAFE
# ═══════════════════════════════════════════════════════════════════════════════

TRUSTED_DOMAINS = {
    # Indian banks
    "sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com",
    "axisbank.com", "kotakbank.com", "kotak.com", "bankofbaroda.in",
    "canarabank.in", "pnbindia.in", "unionbankofindia.co.in",
    "indusind.com", "yesbank.in", "idfcfirstbank.com",
    "federalbank.co.in", "southindianbank.com", "rbl.co.in",
    # Indian fintech / payments
    "zerodha.com", "groww.in", "paytm.com", "phonepe.com", "gpay.app",
    "razorpay.com", "policybazaar.com", "bankbazaar.com", "cleartax.in",
    "upstox.com", "angelone.in", "motilaloswal.com", "smallcase.com",
    "kite.zerodha.com", "console.zerodha.com", "hdfcsec.com",
    "icicisec.com", "bajajfinserv.in",
    # Indian telecom
    "airtel.in", "jio.com", "vi.in", "bsnl.co.in",
    # Indian govt & regulatory
    "gov.in", "nic.in", "irctc.co.in", "incometax.gov.in",
    "uidai.gov.in", "digilocker.gov.in", "mca.gov.in",
    "epfindia.gov.in", "sebi.gov.in", "rbi.org.in",
    "npci.org.in", "india.gov.in", "gst.gov.in",
    "bseindia.com", "nseindia.com", "irdai.gov.in",
    # Indian ecommerce / services
    "flipkart.com", "amazon.in", "myntra.com", "bigbasket.com",
    "zomato.com", "swiggy.com", "makemytrip.com", "irctc.co.in",
    "naukri.com", "housing.com", "99acres.com",
    # Indian IT / media
    "tcs.com", "infosys.com", "wipro.com", "zoho.com",
    "ndtv.com", "thehindu.com", "moneycontrol.com",
    # Global trusted
    "google.com", "gmail.com", "youtube.com", "google.co.in",
    "microsoft.com", "linkedin.com", "github.com", "stackoverflow.com",
    "amazon.com", "apple.com", "wikipedia.org", "cloudflare.com",
    "twitter.com", "x.com", "instagram.com", "facebook.com",
    "whatsapp.com", "medium.com", "reddit.com",
}

TWO_PART_TLDS = {
    "co.in","gov.in","org.in","net.in","ac.in","nic.in","edu.in",
    "co.uk","org.uk","co.nz","com.au","co.za","co.jp",
}

def extract_apex(url: str) -> str:
    url = str(url).lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = url.split("/")[0].split("?")[0].split(":")[0].split("@")[-1]
    parts = url.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_PART_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else url

def is_trusted(url: str) -> bool:
    apex = extract_apex(url)
    if apex in TRUSTED_DOMAINS:
        return True
    return any(apex == td or apex.endswith("." + td) for td in TRUSTED_DOMAINS)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Accept both 'binary_label' (from prepare_data.py) and raw label column
    if "binary_label" in df.columns and "url" in df.columns:
        df = df[["url", "binary_label"]].copy()
    else:
        url_col   = next((c for c in df.columns if "url" in c.lower()), df.columns[0])
        label_col = next((c for c in df.columns
                          if "label" in c.lower() or "class" in c.lower()), df.columns[-1])
        df = df[[url_col, label_col]].copy()
        df.columns = ["url", "label"]

        def _to_binary(v):
            s = str(v).strip().lower()
            try:
                return int(float(s))    # handles "0", "1", "0.0"
            except ValueError:
                pass
            return 0 if s in {"safe","legitimate","benign","good","clean","0"} else 1

        df["binary_label"] = df["label"].apply(_to_binary)
        df = df[["url", "binary_label"]]

    df = df.dropna().copy()
    df["url"]          = df["url"].astype(str).str.strip()
    df["binary_label"] = df["binary_label"].astype(int)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    n0    = (df["binary_label"] == 0).sum()
    n1    = (df["binary_label"] == 1).sum()
    total = len(df)
    ratio = n0 / total if total else 0

    print(f"  Safe: {n0:,} ({ratio*100:.1f}%)  |  Malicious: {n1:,} ({(1-ratio)*100:.1f}%)")
    print(f"  Total: {total:,}")

    if ratio < 0.15:
        print(f"\n  ⚠️  DANGER: Safe class is only {ratio*100:.1f}%.")
        print("     Model will learn 'predict everything malicious'.")
        print("     Re-run prepare_data.py with higher --phiusiil-safe value.")
    elif ratio < 0.30:
        print(f"  ⚠️  Low safe ratio ({ratio*100:.1f}%). Results may be biased.")
    else:
        print(f"  ✅ Class balance acceptable.")

    if total < 2000:
        print(f"  ⚠️  Small dataset ({total}). Consider more data for better generalisation.")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df: pd.DataFrame, verbose: bool = True) -> tuple:
    if verbose:
        print(f"  Extracting {NUM_FEATURES} features for {len(df):,} URLs …")
    t0      = time.time()
    feat_df = extract_features_batch(df["url"].tolist())
    for col in FEATURE_NAMES:
        if col not in feat_df.columns:
            feat_df[col] = 0
    feat_df = feat_df[FEATURE_NAMES].fillna(0).replace([np.inf, -np.inf], 0).astype(float)
    X = feat_df.values
    y = df["binary_label"].values.astype(int)
    if verbose:
        print(f"  Matrix: {X.shape}  ({time.time()-t0:.1f}s)")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — FEATURE AUDIT
#  Detects leaky features (high correlation with label = shortcut learning)
#  Detects zero-variance features (useless columns)
# ═══════════════════════════════════════════════════════════════════════════════

def audit_features(X: np.ndarray, y: np.ndarray,
                   feature_names: list,
                   leakage_threshold: float = 0.85) -> tuple:
    """
    Returns:
      X_clean      — matrix with zero-variance dropped, leaky features capped
      active_names — list of remaining feature names
      keep_mask    — boolean mask of kept columns (for inference)
    """
    print(f"\n  Feature audit …")

    # Step 1: drop zero-variance columns
    stds     = X.std(axis=0)
    keep     = stds > 1e-6
    n_zv     = (~keep).sum()
    if n_zv:
        print(f"  Dropped {n_zv} zero-variance features")

    X_kept   = X[:, keep]
    names_kept = [f for f, k in zip(feature_names, keep) if k]

    # Step 2: detect leaky features (high point-biserial correlation with label)
    leaky = []
    for i, name in enumerate(names_kept):
        col = X_kept[:, i]
        if col.std() < 1e-6:
            continue
        r, _ = scipy_stats.pointbiserialr(y, col)
        if abs(r) > leakage_threshold:
            leaky.append((i, name, round(r, 4)))

    if leaky:
        print(f"  Leaky features (|r|>{leakage_threshold}) → winsorised to ±3σ:")
        for idx, name, r in sorted(leaky, key=lambda x: abs(x[2]), reverse=True):
            print(f"    {name:<40} r={r:+.4f}")
        # Winsorise instead of dropping — keeps signal, breaks shortcut
        X_kept = X_kept.copy()
        for idx, name, r in leaky:
            mu, sigma = X_kept[:, idx].mean(), X_kept[:, idx].std()
            if sigma > 0:
                X_kept[:, idx] = np.clip(X_kept[:, idx],
                                          mu - 3*sigma, mu + 3*sigma)
    else:
        print(f"  ✅ No severe feature leakage detected.")

    print(f"  Active features: {len(names_kept)}/{len(feature_names)}")

    # Build final keep mask in original feature space
    final_keep = keep.copy()
    return X_kept, names_kept, final_keep


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — SMOTE (inside CV folds only)
# ═══════════════════════════════════════════════════════════════════════════════

def smote_fold(X: np.ndarray, y: np.ndarray,
               target_ratio: float = 0.40, k: int = 5) -> tuple:
    """
    Lightweight SMOTE — only called on training portion of each CV fold.
    Never called on validation or test data.
    """
    from sklearn.neighbors import NearestNeighbors
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y

    minority_cls = classes[np.argmin(counts)]
    n_maj        = counts[np.argmax(counts)]
    n_min        = counts[np.argmin(counts)]
    n_need       = int(n_maj * target_ratio / (1.0 - target_ratio)) - n_min

    if n_need <= 0:
        return X, y

    X_min    = X[y == minority_cls]
    k_actual = min(k, len(X_min) - 1)
    if k_actual < 1:
        return X, y

    nn = NearestNeighbors(n_neighbors=k_actual + 1, n_jobs=1)
    nn.fit(X_min)
    _, idxs = nn.kneighbors(X_min)

    rng   = np.random.default_rng(42)
    synth = []
    for _ in range(n_need):
        i   = rng.integers(0, len(X_min))
        j   = idxs[i, rng.integers(1, k_actual + 1)]
        lam = rng.random()
        synth.append(X_min[i] + lam * (X_min[j] - X_min[i]))

    X_aug = np.vstack([X, np.array(synth)])
    y_aug = np.concatenate([y, np.full(n_need, minority_cls)])
    perm  = rng.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — THRESHOLD TUNING
#  Sweeps 0.10 → 0.90 and picks threshold maximising f1_safe
#  (minimises false positives on legitimate domains)
# ═══════════════════════════════════════════════════════════════════════════════

def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                   optimize_for: str = "f1_safe") -> tuple:
    best_t, best_score = 0.5, 0.0
    rows = []
    for t in np.arange(0.10, 0.91, 0.01):
        y_pred = (y_prob >= t).astype(int)
        scores = {
            "threshold":   round(t, 2),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
            "f1_safe":     f1_score(y_true, y_pred, pos_label=0,        zero_division=0),
            "f1_mal":      f1_score(y_true, y_pred, pos_label=1,        zero_division=0),
            "recall_safe": recall_score(y_true, y_pred, pos_label=0,    zero_division=0),
        }
        rows.append(scores)
        if scores[optimize_for] > best_score:
            best_score = scores[optimize_for]
            best_t     = t
    return round(float(best_t), 2), round(float(best_score), 4), pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MODEL DEFINITIONS
#  Regularised to prevent overfitting on ~10K dataset
#  Asymmetric class weights: safe=5x, malicious=1x → penalise false alarms
# ═══════════════════════════════════════════════════════════════════════════════

def build_models(seed: int) -> dict:
    cw = {0: 5.0, 1: 1.0}   # safe misclassification costs 5×

    models = {
        "RF": RandomForestClassifier(
            n_estimators   = 400,
            max_depth      = 12,      # hard cap — prevents memorisation
            min_samples_split = 8,    # needs ≥8 samples to split a node
            min_samples_leaf  = 4,    # each leaf needs ≥4 samples
            max_features   = "sqrt",
            class_weight   = cw,
            random_state   = seed,
            n_jobs         = -1,
        ),
        "ET": ExtraTreesClassifier(
            n_estimators   = 400,
            max_depth      = 12,
            min_samples_split = 8,
            min_samples_leaf  = 4,
            max_features   = "sqrt",
            class_weight   = cw,
            random_state   = seed,
            n_jobs         = -1,
        ),
        "GB": GradientBoostingClassifier(
            n_estimators   = 200,
            max_depth      = 4,       # very shallow — best generalisation
            learning_rate  = 0.05,    # slow learning = less overfit
            subsample      = 0.7,
            max_features   = 0.7,
            min_samples_split = 10,
            min_samples_leaf  = 5,
            random_state   = seed,
        ),
        "LR": LogisticRegression(
            C            = 0.05,      # strong L2 regularisation
            max_iter     = 3000,
            class_weight = cw,
            solver       = "saga",
            penalty      = "l2",
            random_state = seed,
        ),
    }

    if HAS_XGB:
        models["XGB"] = XGBClassifier(
            n_estimators      = 300,
            max_depth         = 5,
            learning_rate     = 0.05,
            subsample         = 0.7,
            colsample_bytree  = 0.7,
            scale_pos_weight  = 5.0,  # handles imbalance natively
            eval_metric       = "logloss",
            use_label_encoder = False,
            random_state      = seed,
            n_jobs            = -1,
            verbosity         = 0,
        )
        print("  ✅ XGBoost available — added to ensemble")

    if HAS_LGBM:
        models["LGBM"] = LGBMClassifier(
            n_estimators     = 300,
            max_depth        = 5,
            learning_rate    = 0.05,
            num_leaves       = 31,
            subsample        = 0.7,
            colsample_bytree = 0.7,
            class_weight     = cw,
            random_state     = seed,
            n_jobs           = -1,
            verbosity        = -1,
        )
        print("  ✅ LightGBM available — added to ensemble")

    return models


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    cm             = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":            round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted":         round(f1_score(y_true, y_pred, average="weighted",  zero_division=0), 4),
        "f1_macro":            round(f1_score(y_true, y_pred, average="macro",     zero_division=0), 4),
        "f1_safe":             round(f1_score(y_true, y_pred, pos_label=0,         zero_division=0), 4),
        "f1_malicious":        round(f1_score(y_true, y_pred, pos_label=1,         zero_division=0), 4),
        "precision_safe":      round(precision_score(y_true, y_pred, pos_label=0,  zero_division=0), 4),
        "recall_safe":         round(recall_score(y_true, y_pred,    pos_label=0,  zero_division=0), 4),
        "precision_malicious": round(precision_score(y_true, y_pred, pos_label=1,  zero_division=0), 4),
        "recall_malicious":    round(recall_score(y_true, y_pred,    pos_label=1,  zero_division=0), 4),
        "auc_roc":             round(roc_auc_score(y_true, y_prob), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — OOF CROSS-VALIDATION
#  The ONLY honest performance estimate — model never sees val data during fit
# ═══════════════════════════════════════════════════════════════════════════════

def oof_cross_validate(model_factory, X: np.ndarray, y: np.ndarray,
                        n_splits: int = 5, seed: int = 42,
                        needs_scale: bool = False) -> tuple:
    """
    Returns (oof_probs, per_fold_f1s).
    SMOTE applied inside each fold training portion only.
    Scaler fitted inside each fold training portion only.
    """
    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y))
    fold_f1s  = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Oversample inside this fold only
        X_tr_aug, y_tr_aug = smote_fold(X_tr, y_tr, target_ratio=0.40)

        # Scale inside this fold only (no leakage to val)
        if needs_scale:
            sc     = StandardScaler()
            X_tr_aug = sc.fit_transform(X_tr_aug)
            X_va     = sc.transform(X_va)

        m = model_factory()
        m.fit(X_tr_aug, y_tr_aug)

        probs             = m.predict_proba(X_va)[:, 1]
        oof_probs[va_idx] = probs

        fold_f1 = f1_score(y_va, (probs >= 0.5).astype(int),
                           average="weighted", zero_division=0)
        fold_f1s.append(round(fold_f1, 4))

    return oof_probs, fold_f1s


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def train(csv_path: str, models_dir: str, seed: int = 42) -> dict:
    np.random.seed(seed)
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "═" * 70)
    print("  COGNITIVE FIREWALL v6 — MODEL TRAINER")
    print("═" * 70)

    # ── Step 1: Load & validate ────────────────────────────────────────────────
    print("\n[1/8] Loading dataset …")
    df = load_dataset(csv_path)
    validate_dataset(df)

    # ── Step 2: Feature extraction ─────────────────────────────────────────────
    print("\n[2/8] Feature extraction …")
    X_raw, y = build_feature_matrix(df)

    # ── Step 3: Feature audit ──────────────────────────────────────────────────
    print("\n[3/8] Feature audit (leakage + zero-variance) …")
    X_clean, active_features, keep_mask = audit_features(
        X_raw, y, FEATURE_NAMES, leakage_threshold=0.85
    )

    # ── Step 4: Train / Val / Test split (stratified, 70/15/15) ───────────────
    print("\n[4/8] Splitting 70 / 15 / 15 (train / val / test) …")
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_clean, y, test_size=0.15, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=0.15 / 0.85,
        random_state=seed, stratify=y_tv
    )
    print(f"  Train : {len(X_train):,}  safe={( y_train==0).sum():,}  mal={(y_train==1).sum():,}")
    print(f"  Val   : {len(X_val):,}   safe={(y_val==0).sum():,}  mal={(y_val==1).sum():,}")
    print(f"  Test  : {len(X_test):,}  safe={(y_test==0).sum():,}  mal={(y_test==1).sum():,}")

    # ── Step 5: OOF cross-validation ───────────────────────────────────────────
    print("\n[5/8] OOF 5-fold cross-validation (honest performance estimate) …")

    models_dict = build_models(seed)
    oof_results = {}

    for name, model in models_dict.items():
        needs_scale = (name == "LR")
        t0 = time.time()

        oof_probs, fold_f1s = oof_cross_validate(
            lambda m=model: copy.deepcopy(m),
            X_train, y_train,
            n_splits=5, seed=seed,
            needs_scale=needs_scale,
        )

        # Tune threshold on OOF probs (val data never seen during training)
        best_t, best_score, _ = tune_threshold(y_train, oof_probs, "f1_safe")

        oof_pred = (oof_probs >= best_t).astype(int)
        m_oof    = compute_metrics(y_train, oof_pred, oof_probs)

        oof_results[name] = {
            "oof_f1_weighted": m_oof["f1_weighted"],
            "oof_f1_safe":     m_oof["f1_safe"],
            "oof_auc":         m_oof["auc_roc"],
            "oof_threshold":   best_t,
            "fold_f1s":        fold_f1s,
        }

        elapsed = time.time() - t0
        print(f"  {name:<6}  "
              f"OOF F1w={m_oof['f1_weighted']:.4f}  "
              f"F1-safe={m_oof['f1_safe']:.4f}  "
              f"AUC={m_oof['auc_roc']:.4f}  "
              f"Thresh={best_t}  "
              f"Folds={fold_f1s}  "
              f"({elapsed:.1f}s)")

    # ── Step 6: Full training on complete train set ────────────────────────────
    print("\n[6/8] Final model training on full train set …")

    # Global SMOTE for final model fitting (not used in eval)
    X_tr_aug, y_tr_aug = smote_fold(X_train, y_train, target_ratio=0.40)
    print(f"  After SMOTE: {len(X_tr_aug):,}  "
          f"safe={(y_tr_aug==0).sum():,}  mal={(y_tr_aug==1).sum():,}")

    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_tr_aug)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    final_models = {}
    test_results = {}
    thresholds   = {}

    for name, model in models_dict.items():
        t0          = time.time()
        needs_scale = (name == "LR")
        Xtr         = X_tr_sc    if needs_scale else X_tr_aug
        Xva         = X_val_sc   if needs_scale else X_val
        Xte         = X_test_sc  if needs_scale else X_test

        print(f"\n  ── {name} {'─'*(50-len(name))}")

        # Fit
        if name == "XGB" and HAS_XGB:
            # XGBoost supports native early stopping
            Xtr2, Xes, ytr2, yes = train_test_split(
                Xtr, y_tr_aug, test_size=0.1, random_state=seed, stratify=y_tr_aug
            )
            model.set_params(early_stopping_rounds=20)
            model.fit(Xtr2, ytr2, eval_set=[(Xes, yes)], verbose=False)
            print(f"     Best iteration: {model.best_iteration}")
        else:
            model.fit(Xtr, y_tr_aug)

        # Calibrate probabilities on val set (never on test)
        cal = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        cal.fit(Xva, y_val)

        # Use OOF-tuned threshold (not re-tuned on test)
        best_t            = oof_results[name]["oof_threshold"]
        thresholds[name]  = best_t

        y_prob_te = cal.predict_proba(Xte)[:, 1]
        y_pred_te = (y_prob_te >= best_t).astype(int)
        m         = compute_metrics(y_test, y_pred_te, y_prob_te)
        test_results[name] = m
        final_models[name] = (cal, needs_scale)

        # Overfit gap check
        y_prob_tr = cal.predict_proba(Xtr)[:, 1]
        train_f1  = f1_score(y_tr_aug, (y_prob_tr >= best_t).astype(int),
                             average="weighted", zero_division=0)
        gap = train_f1 - m["f1_weighted"]
        overfit_flag = "  ⚠️  OVERFIT" if gap > 0.12 else "  ✅"

        print(f"     Threshold    : {best_t}")
        print(f"     Train F1     : {train_f1:.4f}  Test F1: {m['f1_weighted']:.4f}  "
              f"Gap: {gap:+.4f}{overfit_flag}")
        print(f"     Accuracy     : {m['accuracy']:.4f}  AUC: {m['auc_roc']:.4f}")
        print(f"     Recall(safe) : {m['recall_safe']:.4f}  "
              f"Prec(safe): {m['precision_safe']:.4f}  "
              f"({time.time()-t0:.1f}s)")

        joblib.dump(cal, os.path.join(models_dir, f"{name.lower()}.pkl"))

    # ── Step 7: Voting Ensemble ────────────────────────────────────────────────
    print("\n  ── VotingEnsemble ──────────────────────────────────────────────")
    t0 = time.time()

    # Build ensemble from all non-LR models (trees work better in soft voting)
    ensemble_members = []

    for n, m in models_dict.items():
       if n == "LR":
          continue

    m_copy = copy.deepcopy(m)

    # Remove early stopping if present (critical fix)
    if n == "XGB" and HAS_XGB:
        m_copy.set_params(early_stopping_rounds=None)

    ensemble_members.append((n, m_copy))

    voting_raw = VotingClassifier(
        estimators=ensemble_members,
        voting="soft",
        n_jobs=-1,
    )
    voting_raw.fit(X_tr_aug, y_tr_aug)

    # Calibrate on val set
    cal_v = CalibratedClassifierCV(voting_raw, cv="prefit", method="isotonic")
    cal_v.fit(X_val, y_val)

    # Tune threshold on val (not test)
    y_prob_val_v = cal_v.predict_proba(X_val)[:, 1]
    best_t_v, _, sweep_df = tune_threshold(y_val, y_prob_val_v, "f1_safe")
    thresholds["Ensemble"] = best_t_v

    # Evaluate on test
    y_prob_te_v = cal_v.predict_proba(X_test)[:, 1]
    y_pred_te_v = (y_prob_te_v >= best_t_v).astype(int)
    m_v         = compute_metrics(y_test, y_pred_te_v, y_prob_te_v)
    test_results["Ensemble"] = m_v
    final_models["Ensemble"] = (cal_v, False)

    # Overfit gap for ensemble
    y_prob_tr_v  = cal_v.predict_proba(X_tr_aug)[:, 1]
    train_f1_v   = f1_score(y_tr_aug, (y_prob_tr_v >= best_t_v).astype(int),
                            average="weighted", zero_division=0)
    gap_v = train_f1_v - m_v["f1_weighted"]

    print(f"  Threshold    : {best_t_v}")
    print(f"  Train F1     : {train_f1_v:.4f}  Test F1: {m_v['f1_weighted']:.4f}  "
          f"Gap: {gap_v:+.4f}{'  ⚠️  OVERFIT' if gap_v > 0.12 else '  ✅'}")
    print(f"  Accuracy     : {m_v['accuracy']:.4f}  AUC: {m_v['auc_roc']:.4f}")
    print(f"  Recall(safe) : {m_v['recall_safe']:.4f}  "
          f"Prec(safe): {m_v['precision_safe']:.4f}  ({time.time()-t0:.1f}s)")

    joblib.dump(cal_v, os.path.join(models_dir, "ensemble.pkl"))

    # Save threshold sweep for inspection
    sweep_df.to_csv(os.path.join(models_dir, "threshold_sweep.csv"), index=False)

    # ── Step 8: Save all artefacts ─────────────────────────────────────────────
    print("\n[8/8] Saving artefacts …")
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    np.save(os.path.join(models_dir, "keep_mask.npy"), keep_mask)

    with open(os.path.join(models_dir, "active_features.json"), "w") as f:
        json.dump(active_features, f)
    with open(os.path.join(models_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(models_dir, "trusted_domains.json"), "w") as f:
        json.dump(sorted(TRUSTED_DOMAINS), f, indent=2)
    with open(os.path.join(models_dir, "oof_results.json"), "w") as f:
        json.dump(oof_results, f, indent=2)
    with open(os.path.join(models_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    # Feature importance from RF
    try:
        rf_est   = voting_raw.named_estimators_["RF"]
        imp_df   = pd.DataFrame({
            "feature":    active_features,
            "importance": rf_est.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp_df.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)
    except Exception:
        imp_df = pd.DataFrame({"feature": active_features,
                               "importance": [0]*len(active_features)})

    # ── Final report ───────────────────────────────────────────────────────────
    print("\n" + "═" * 82)
    print(f"  {'Model':<10} {'OOF-F1w':>8} {'OOF-AUC':>8} "
          f"{'Test-F1w':>9} {'Test-AUC':>9} "
          f"{'Rec(safe)':>10} {'Prec(safe)':>11} {'Thresh':>7}")
    print("─" * 82)
    for name in list(models_dict.keys()) + ["Ensemble"]:
        oof = oof_results.get(name, {})
        tst = test_results.get(name, {})
        t   = thresholds.get(name, 0.5)
        print(f"  {name:<10} "
              f"{oof.get('oof_f1_weighted', 0):>8.4f} "
              f"{oof.get('oof_auc', 0):>8.4f} "
              f"{tst.get('f1_weighted', 0):>9.4f} "
              f"{tst.get('auc_roc', 0):>9.4f} "
              f"{tst.get('recall_safe', 0):>10.4f} "
              f"{tst.get('precision_safe', 0):>11.4f} "
              f"{t:>7.2f}")
    print("═" * 82)

    best_name = max(test_results, key=lambda n: test_results[n]["f1_weighted"])
    best      = test_results[best_name]
    print(f"\n  ✅  Best model  : {best_name}")
    print(f"     Test F1w     : {best['f1_weighted']:.4f}")
    print(f"     Test AUC     : {best['auc_roc']:.4f}")
    print(f"     Recall(safe) : {best['recall_safe']:.4f}")
    print(f"     Prec(safe)   : {best['precision_safe']:.4f}")
    print(f"     Threshold    : {thresholds[best_name]}")

    # Confusion matrix for best model
    bm, bs    = final_models[best_name]
    Xte_b     = X_test_sc if bs else X_test
    y_pb      = bm.predict_proba(Xte_b)[:, 1]
    y_pred_b  = (y_pb >= thresholds[best_name]).astype(int)
    cm        = confusion_matrix(y_test, y_pred_b)

    print(f"\n  Confusion Matrix ({best_name}):")
    print(f"  {'':20}  Pred Safe   Pred Malicious")
    print(f"  {'Actual Safe':20}  {cm[0][0]:>9}   {cm[0][1]:>14}")
    print(f"  {'Actual Malicious':20}  {cm[1][0]:>9}   {cm[1][1]:>14}")
    print(f"\n{classification_report(y_test, y_pred_b, target_names=['safe','malicious'])}")

    print("  Top 15 most predictive features:")
    for _, row in imp_df.head(15).iterrows():
        bar = "█" * int(row["importance"] * 500)
        print(f"    {row['feature']:<42} {row['importance']:.4f}  {bar}")

    print(f"\n  Artefacts saved → {models_dir}")
    print(f"  Files: ensemble.pkl, scaler.pkl, keep_mask.npy, thresholds.json,")
    print(f"         feature_importance.csv, threshold_sweep.csv, oof_results.json")
    return test_results


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — INFERENCE ENGINE (drop-in for your IPS)
# ═══════════════════════════════════════════════════════════════════════════════

class PhishingDetector:
    """
    Production-ready detector with two-layer defence:
    Layer 1 — Trusted domain whitelist (instant, zero latency)
    Layer 2 — ML ensemble with calibrated probabilities + tuned threshold

    Usage:
        det = PhishingDetector("models/")
        results = det.predict([
            "https://sbi.co.in/login",          # → safe (whitelist)
            "https://sbi-kyc-update.xyz/login",  # → malicious (ML)
            "https://google.com",               # → safe (whitelist)
        ])
    """

    def __init__(self, models_dir: str):
        self.ensemble  = joblib.load(os.path.join(models_dir, "ensemble.pkl"))
        self.keep_mask = np.load(os.path.join(models_dir, "keep_mask.npy"))

        with open(os.path.join(models_dir, "thresholds.json")) as f:
            thr = json.load(f)
        self.threshold = thr.get("Ensemble", 0.5)

        td_path = os.path.join(models_dir, "trusted_domains.json")
        if os.path.exists(td_path):
            with open(td_path) as f:
                self._trusted = set(json.load(f))
        else:
            self._trusted = TRUSTED_DOMAINS.copy()

    def predict(self, urls: list) -> list:
        results  = [None] * len(urls)
        ml_urls  = []
        ml_idxs  = []

        # Layer 1: whitelist check
        for i, url in enumerate(urls):
            if is_trusted(url):
                results[i] = {
                    "url":        url,
                    "label":      "safe",
                    "confidence": 1.0,
                    "reason":     "trusted_whitelist",
                }
            else:
                ml_urls.append(url)
                ml_idxs.append(i)

        # Layer 2: ML for everything else
        if ml_urls:
            feat_df = extract_features_batch(ml_urls)
            for col in FEATURE_NAMES:
                if col not in feat_df.columns:
                    feat_df[col] = 0
            X = feat_df[FEATURE_NAMES].fillna(0).replace([np.inf,-np.inf], 0).astype(float).values
            X = X[:, self.keep_mask]   # same feature filter applied at training

            probs = self.ensemble.predict_proba(X)[:, 1]
            for j, (url, prob) in enumerate(zip(ml_urls, probs)):
                label = "malicious" if prob >= self.threshold else "safe"
                results[ml_idxs[j]] = {
                    "url":        url,
                    "label":      label,
                    "confidence": round(float(prob), 4),
                    "reason":     "ml_model",
                }

        return results

    def predict_one(self, url: str) -> dict:
        return self.predict([url])[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Cognitive Firewall v6 — Model Trainer")
    p.add_argument("csv",      nargs="?",
                   default=os.path.join(os.path.dirname(HERE), "data", "combined.csv"),
                   help="Path to combined.csv from prepare_data.py")
    p.add_argument("--models", default=os.path.join(os.path.dirname(HERE), "models"),
                   help="Directory to save trained models")
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"❌ Dataset not found: {args.csv}")
        print("   Run prepare_data.py first to generate combined.csv")
        sys.exit(1)

    train(args.csv, args.models, args.seed)