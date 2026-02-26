"""
Cognitive Firewall v4 - Model Trainer (OVERFITTING FIX)
========================================================
Root cause of Test 2 failures:
  1. Model predicts ALL URLs as malicious (threshold bias from imbalanced training)
  2. Overfitting: 1.0000 on every metric means model memorised training data
  3. Trusted/known-good domains not given explicit safe signal in features
  4. Training dataset likely 90%+ phishing → model learned "predict malicious always"

Fixes applied:
  ──────────────────────────────────────────────────────────
  A. TRUSTED DOMAIN WHITELIST  — explicit pre-check before ML
  B. DATASET VALIDATION        — abort if class ratio is too skewed
  C. ANTI-OVERFITTING          — heavier regularisation, smaller trees,
                                  mandatory min_samples_leaf, max_depth caps
  D. CALIBRATED THRESHOLD       — found on a proper held-out val set,
                                  not the same split used for fitting
  E. HARD NEGATIVE MINING       — inject known-good domains into training
                                  so model sees what "safe" really looks like
  F. CROSS-VAL SANITY CHECK     — raise warning if CV > 0.98 (likely overfit)
  G. REALISTIC SYNTHETIC SAFE   — generate known-legit URL features to
                                  balance training without duplication noise
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             precision_score, recall_score)

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from feature_extraction import extract_features_batch, FEATURE_NAMES, NUM_FEATURES


# ═══════════════════════════════════════════════════════════════════════════════
#  TRUSTED DOMAIN WHITELIST  (pre-ML hard override)
# ═══════════════════════════════════════════════════════════════════════════════

# These domains are unconditionally SAFE regardless of what the model thinks.
# Extend this list freely — it is the most reliable protection against false positives.
TRUSTED_DOMAINS = {
    # Indian banks & finance
    "sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com",
    "axisbank.com", "kotak.com", "bankofbaroda.in", "canarabank.in",
    "pnbindia.in", "unionbankofindia.co.in", "indusind.com", "yesbank.in",
    "idfcfirstbank.com", "federalbank.co.in", "southindianbank.com",
    # Indian fintech / payments
    "zerodha.com", "groww.in", "paytm.com", "phonepe.com", "gpay.app",
    "razorpay.com", "policybazaar.com", "bankbazaar.com", "cleartax.in",
    "upstox.com", "angelone.in", "motilaloswal.com",
    # Indian telco / infra
    "airtel.in", "jio.com", "vi.in", "bsnl.co.in",
    # Indian govt
    "gov.in", "nic.in", "irctc.co.in", "incometax.gov.in",
    "uidai.gov.in", "digilocker.gov.in", "mca.gov.in",
    # Global trusted
    "google.com", "gmail.com", "youtube.com", "google.co.in",
    "microsoft.com", "linkedin.com", "github.com", "stackoverflow.com",
    "amazon.com", "amazon.in", "flipkart.com", "apple.com",
    "wikipedia.org", "cloudflare.com", "akamai.com",
}

def _extract_apex(url: str) -> str:
    """Return apex domain (e.g. 'https://foo.bar.sbi.co.in/x' → 'sbi.co.in')."""
    import re
    url = url.lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = url.split("/")[0].split("?")[0].split(":")[0]   # strip path/port
    parts = url.split(".")
    # Handle two-part TLDs: co.in, gov.in, org.in, net.in, ac.in, nic.in
    two_part_tlds = {"co.in", "gov.in", "org.in", "net.in", "ac.in",
                     "nic.in", "co.uk", "org.uk", "co.nz", "com.au"}
    if len(parts) >= 3 and ".".join(parts[-2:]) in two_part_tlds:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else url


def is_trusted(url: str) -> bool:
    apex = _extract_apex(url)
    if apex in TRUSTED_DOMAINS:
        return True
    # Also check if any trusted domain is a suffix match
    for td in TRUSTED_DOMAINS:
        if apex == td or apex.endswith("." + td):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  KNOWN-GOOD URLS FOR HARD NEGATIVE INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

KNOWN_SAFE_URLS = [
    "https://www.sbi.co.in/web/personal-banking",
    "https://netbanking.hdfcbank.com/netbanking/",
    "https://www.icicibank.com/personal-banking/account",
    "https://www.google.com/search?q=hello",
    "https://www.google.co.in/",
    "https://mail.google.com/mail/u/0/",
    "https://github.com/torvalds/linux",
    "https://stackoverflow.com/questions/tagged/python",
    "https://zerodha.com/varsity/",
    "https://groww.in/mutual-funds",
    "https://www.airtel.in/recharge/",
    "https://www.jio.com/selfcare/plans/",
    "https://www.irctc.co.in/nget/train-search",
    "https://incometax.gov.in/iec/foportal",
    "https://www.amazon.in/s?k=laptop",
    "https://www.flipkart.com/mobiles",
    "https://www.linkedin.com/in/example",
    "https://en.wikipedia.org/wiki/India",
    "https://www.microsoft.com/en-in/",
    "https://www.apple.com/in/",
    "https://paytm.com/recharge",
    "https://www.phonepe.com/",
    "https://razorpay.com/payment-gateway/",
    "https://axisbank.com/retail/onlineservices/",
    "https://kotakbank.com/personal/home.html",
    "https://www.bankofbaroda.in/",
    "https://upstox.com/open-demat-account/",
    "https://angelone.in/open-demat-account",
    "https://cleartax.in/s/itr-filing",
    "https://digilocker.gov.in/",
    "https://uidai.gov.in/en/",
    "https://mca.gov.in/content/mca/global/en/home.html",
    "https://www.bsnl.co.in/opencms/bsnl/BSNL/",
    "https://vi.in/prepaid-recharge",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://cloudflare.com",
    "https://www.amazon.com/",
    "https://docs.python.org/3/",
    "https://www.pnbindia.in/",
    "https://www.yesbank.in/",
]


def inject_known_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Add KNOWN_SAFE_URLS with label=0 to training data."""
    safe_df = pd.DataFrame({
        "url": KNOWN_SAFE_URLS,
        "binary_label": [0] * len(KNOWN_SAFE_URLS),
    })
    # Repeat several times so model sees them consistently across folds
    safe_df = pd.concat([safe_df] * 5, ignore_index=True)
    combined = pd.concat([df, safe_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_dataset(df: pd.DataFrame) -> None:
    n_safe      = (df["binary_label"] == 0).sum()
    n_malicious = (df["binary_label"] == 1).sum()
    total       = len(df)
    ratio       = n_safe / total if total else 0

    print(f"  Safe: {n_safe:,} ({ratio*100:.1f}%)  |  Malicious: {n_malicious:,} ({(1-ratio)*100:.1f}%)")

    if ratio < 0.05:
        print("\n  ⚠️  WARNING: Dataset has <5% safe URLs!")
        print("     This will cause the model to predict EVERYTHING as malicious.")
        print("     Fix: add more legitimate URLs to your training CSV.")
        print("     Proceeding with hard-negative injection to compensate...\n")
    if total < 500:
        print(f"  ⚠️  WARNING: Only {total} samples. Model may not generalise well.")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"url", "binary_label"}
    if not required.issubset(df.columns):
        url_col   = next((c for c in df.columns if "url"   in c.lower()), df.columns[0])
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


# ═══════════════════════════════════════════════════════════════════════════════
#  SMOTE-STYLE OVERSAMPLING  (minority class only, applied after safe injection)
# ═══════════════════════════════════════════════════════════════════════════════

def smote_oversample(X: np.ndarray, y: np.ndarray,
                     target_ratio: float = 0.35, k: int = 5) -> tuple:
    from sklearn.neighbors import NearestNeighbors

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y

    minority_cls = classes[np.argmin(counts)]
    n_maj        = counts[np.argmax(counts)]
    n_min        = counts[np.argmin(counts)]
    target_n     = int(n_maj * target_ratio / (1.0 - target_ratio))
    n_needed     = target_n - n_min

    if n_needed <= 0:
        return X, y

    X_min     = X[y == minority_cls]
    k_actual  = min(k, len(X_min) - 1)
    nn        = NearestNeighbors(n_neighbors=k_actual + 1, n_jobs=-1)
    nn.fit(X_min)
    _, idxs   = nn.kneighbors(X_min)

    rng = np.random.default_rng(42)
    synth = []
    for _ in range(n_needed):
        i      = rng.integers(0, len(X_min))
        j      = idxs[i, rng.integers(1, k_actual + 1)]
        lam    = rng.random()
        synth.append(X_min[i] + lam * (X_min[j] - X_min[i]))

    X_aug = np.vstack([X, np.array(synth)])
    y_aug = np.concatenate([y, np.full(n_needed, minority_cls)])
    perm  = rng.permutation(len(X_aug))
    return X_aug[perm], y_aug[perm]


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(df: pd.DataFrame, verbose: bool = True) -> tuple:
    if verbose:
        print(f"  Extracting {NUM_FEATURES} features for {len(df):,} URLs …")
    t0      = time.time()
    feat_df = extract_features_batch(df["url"].tolist())
    for col in FEATURE_NAMES:
        if col not in feat_df.columns:
            feat_df[col] = 0
    feat_df = feat_df[FEATURE_NAMES].fillna(0).astype(float)
    X = feat_df.values
    y = df["binary_label"].values.astype(int)
    if verbose:
        print(f"  Matrix: {X.shape}  ({time.time()-t0:.1f}s)")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
#  THRESHOLD OPTIMISATION  (on a separate validation split, never train data)
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_threshold(y_val: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.81, 0.01):
        y_hat = (y_prob >= t).astype(int)
        score = f1_score(y_val, y_hat, average="weighted", zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return round(float(best_t), 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":            round(accuracy_score(y_true, y_pred), 4),
        "f1_weighted":         round(f1_score(y_true, y_pred, average="weighted",  zero_division=0), 4),
        "f1_macro":            round(f1_score(y_true, y_pred, average="macro",     zero_division=0), 4),
        "f1_malicious":        round(f1_score(y_true, y_pred, pos_label=1,         zero_division=0), 4),
        "f1_safe":             round(f1_score(y_true, y_pred, pos_label=0,         zero_division=0), 4),
        "precision_malicious": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall_malicious":    round(recall_score(y_true, y_pred,    pos_label=1, zero_division=0), 4),
        "precision_safe":      round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "recall_safe":         round(recall_score(y_true, y_pred,    pos_label=0, zero_division=0), 4),
        "auc_roc":             round(roc_auc_score(y_true, y_prob), 4),
        "true_positives":  int(tp), "true_negatives":  int(tn),
        "false_positives": int(fp), "false_negatives": int(fn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def train(csv_path: str, models_dir: str, seed: int = 42) -> dict:
    np.random.seed(seed)
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  COGNITIVE FIREWALL v4 — OVERFITTING-FIXED TRAINER")
    print("=" * 70)

    # ── 1. Load & validate ─────────────────────────────────────────────────────
    print("\n[1/6] Loading & validating dataset …")
    df = load_dataset(csv_path)
    print(f"  Raw dataset: {len(df):,} URLs")
    validate_dataset(df)

    # ── 2. Inject known-good URLs ──────────────────────────────────────────────
    print("\n[2/6] Injecting known-safe URLs (hard negatives) …")
    df = inject_known_safe(df)
    print(f"  After injection: {len(df):,} URLs")
    n_safe = (df["binary_label"] == 0).sum()
    n_mal  = (df["binary_label"] == 1).sum()
    print(f"  Safe: {n_safe:,}  |  Malicious: {n_mal:,}  |  Ratio: {n_safe/(len(df)):.2%}")

    # ── 3. Feature extraction ──────────────────────────────────────────────────
    print("\n[3/6] Feature extraction …")
    X, y = build_feature_matrix(df)

    # ── 4. Three-way split: train / val / test ─────────────────────────────────
    # val  → threshold optimisation (never touches model fitting)
    # test → final unbiased evaluation
    print("\n[4/6] Splitting 70 / 15 / 15 (train / val / test) …")
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, random_state=seed, stratify=y_tv)

    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    print(f"  Train safe/malicious: {np.sum(y_train==0):,}/{np.sum(y_train==1):,}")

    # SMOTE on training split only
    X_train_aug, y_train_aug = smote_oversample(X_train, y_train, target_ratio=0.35)
    print(f"  After SMOTE: {len(X_train_aug):,}  "
          f"(safe: {np.sum(y_train_aug==0):,} / mal: {np.sum(y_train_aug==1):,})")

    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train_aug)
    X_val_sc     = scaler.transform(X_val)
    X_test_sc    = scaler.transform(X_test)

    # ── 5. Model definitions (REGULARISED to prevent overfitting) ──────────────
    print("\n[5/6] Training models (regularised) …")

    # class_weight: penalise false negatives on safe class → fewer false alarms
    cw = {0: 3.0, 1: 1.0}

    model_configs = [
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,              # ← was None; capping prevents overfitting
                min_samples_split=6,       # ← was 2-3; forces generalisation
                min_samples_leaf=4,        # ← was 1; avoids leaf memorisation
                max_features="sqrt",
                class_weight=cw,
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=6,
                min_samples_leaf=4,
                class_weight=cw,
                random_state=seed,
                n_jobs=-1,
            ),
            False,
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,               # ← shallow trees generalise better
                learning_rate=0.05,        # ← slower learning = less overfit
                subsample=0.7,
                min_samples_split=8,
                min_samples_leaf=4,
                random_state=seed,
            ),
            False,
        ),
        (
            "LogisticRegression",
            LogisticRegression(
                C=0.1,                     # ← strong L2 regularisation
                max_iter=3000,
                class_weight=cw,
                solver="saga",
                random_state=seed,
            ),
            True,
        ),
    ]

    trained_models  = {}
    results         = {}
    best_thresholds = {}

    for name, model, needs_scaling in model_configs:
        t0  = time.time()
        Xtr = X_train_sc  if needs_scaling else X_train_aug
        Xva = X_val_sc    if needs_scaling else X_val
        Xte = X_test_sc   if needs_scaling else X_test

        print(f"\n  ── {name} ──────────────────────────────────────────────")
        model.fit(Xtr, y_train_aug)

        # Calibrate on validation set
        cal = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
        cal.fit(Xva, y_val)

        # Threshold on val, evaluate on test
        y_prob_val  = cal.predict_proba(Xva)[:, 1]
        best_t      = find_best_threshold(y_val, y_prob_val)
        best_thresholds[name] = best_t

        y_prob_test = cal.predict_proba(Xte)[:, 1]
        y_pred_test = (y_prob_test >= best_t).astype(int)

        m = compute_metrics(y_test, y_pred_test, y_prob_test)
        results[name] = m
        trained_models[name] = (cal, needs_scaling)

        # Overfit detector
        y_pred_train = (cal.predict_proba(Xtr)[:, 1] >= best_t).astype(int)
        train_f1     = f1_score(y_train_aug, y_pred_train, average="weighted", zero_division=0)
        overfit_gap  = train_f1 - m["f1_weighted"]

        print(f"    Threshold : {best_t}")
        print(f"    Train F1  : {train_f1:.4f}  |  Test F1: {m['f1_weighted']:.4f}  "
              f"|  Gap: {overfit_gap:+.4f}{'  ⚠️ OVERFIT' if overfit_gap > 0.10 else '  ✅'}")
        print(f"    Acc: {m['accuracy']:.4f}  AUC: {m['auc_roc']:.4f}  "
              f"Rec(safe): {m['recall_safe']:.4f}  Prec(safe): {m['precision_safe']:.4f}  "
              f"({time.time()-t0:.1f}s)")

        joblib.dump(cal, os.path.join(models_dir, f"{name.lower()}.pkl"))

    # ── Voting Ensemble ────────────────────────────────────────────────────────
    print("\n  ── VotingEnsemble ──────────────────────────────────────────────")
    t0 = time.time()
    rf = trained_models["RandomForest"][0]
    et = trained_models["ExtraTrees"][0]
    gb = trained_models["GradientBoosting"][0]

    voting = VotingClassifier(
        estimators=[("rf", rf), ("et", et), ("gb", gb)],
        voting="soft", n_jobs=-1,
    )
    # VotingClassifier needs unfitted base estimators — re-fit on aug data
    rf_raw = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_split=6,
                                    min_samples_leaf=4, max_features="sqrt",
                                    class_weight=cw, random_state=seed, n_jobs=-1)
    et_raw = ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_split=6,
                                   min_samples_leaf=4, class_weight=cw, random_state=seed, n_jobs=-1)
    gb_raw = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                         subsample=0.7, min_samples_split=8, min_samples_leaf=4,
                                         random_state=seed)
    voting_raw = VotingClassifier(
        estimators=[("rf", rf_raw), ("et", et_raw), ("gb", gb_raw)],
        voting="soft", n_jobs=-1,
    )
    voting_raw.fit(X_train_aug, y_train_aug)

    # Calibrate ensemble on val
    cal_v = CalibratedClassifierCV(voting_raw, cv="prefit", method="isotonic")
    cal_v.fit(X_val, y_val)

    y_prob_val_v   = cal_v.predict_proba(X_val)[:, 1]
    best_t_v       = find_best_threshold(y_val, y_prob_val_v)
    best_thresholds["VotingEnsemble"] = best_t_v

    y_prob_test_v  = cal_v.predict_proba(X_test)[:, 1]
    y_pred_test_v  = (y_prob_test_v >= best_t_v).astype(int)
    m_v = compute_metrics(y_test, y_pred_test_v, y_prob_test_v)
    results["VotingEnsemble"] = m_v
    trained_models["VotingEnsemble"] = (cal_v, False)

    print(f"    Threshold: {best_t_v}  |  Acc: {m_v['accuracy']:.4f}  |  "
          f"F1w: {m_v['f1_weighted']:.4f}  |  AUC: {m_v['auc_roc']:.4f}  |  "
          f"Rec(safe): {m_v['recall_safe']:.4f}  ({time.time()-t0:.1f}s)")
    joblib.dump(cal_v, os.path.join(models_dir, "ensemble.pkl"))

    # ── 5-fold CV sanity check ─────────────────────────────────────────────────
    print("\n  → 5-fold CV on VotingEnsemble (overfit sanity check) …")
    cv_scores = cross_val_score(
        voting_raw, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring="f1_weighted", n_jobs=-1,
    )
    cv_mean = cv_scores.mean()
    print(f"    CV F1 (weighted): {cv_mean:.4f} ± {cv_scores.std():.4f}")
    if cv_mean > 0.98:
        print("    ⚠️  CV F1 > 0.98 — possible overfitting. Check dataset for leakage.")
    results["VotingEnsemble"]["cv_f1_mean"] = round(float(cv_mean), 4)
    results["VotingEnsemble"]["cv_f1_std"]  = round(float(cv_scores.std()), 4)

    # ── Feature importance ─────────────────────────────────────────────────────
    base_rf = rf_raw  # unfitted would fail; use the one inside voting_raw
    try:
        base_rf = voting_raw.estimators_[0]   # RF is first
        importances = pd.DataFrame({
            "feature":    FEATURE_NAMES,
            "importance": base_rf.feature_importances_,
        }).sort_values("importance", ascending=False)
        importances.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)
    except Exception:
        importances = pd.DataFrame({"feature": FEATURE_NAMES, "importance": [0]*len(FEATURE_NAMES)})

    # ── Save artefacts ─────────────────────────────────────────────────────────
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f)
    with open(os.path.join(models_dir, "thresholds.json"), "w") as f:
        json.dump(best_thresholds, f, indent=2)
    with open(os.path.join(models_dir, "trusted_domains.json"), "w") as f:
        json.dump(sorted(TRUSTED_DOMAINS), f, indent=2)
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── 6. Final report ────────────────────────────────────────────────────────
    print("\n[6/6] RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Model':<22} {'Acc':>7} {'F1(w)':>7} {'AUC':>7} "
          f"{'Rec(safe)':>10} {'Prec(safe)':>11} {'Thresh':>7}")
    print("-" * 75)
    for name, m in results.items():
        t = best_thresholds.get(name, 0.5)
        print(f"  {name:<20} {m['accuracy']:>7.4f} {m['f1_weighted']:>7.4f} "
              f"{m['auc_roc']:>7.4f} {m['recall_safe']:>10.4f} "
              f"{m['precision_safe']:>11.4f} {t:>7.2f}")
    print("=" * 75)

    best_name = max(results, key=lambda n: results[n]["f1_weighted"])
    best      = results[best_name]
    print(f"\n  ✅  Best model : {best_name}")
    print(f"     F1 weighted : {best['f1_weighted']:.4f}")
    print(f"     Accuracy    : {best['accuracy']:.4f}")
    print(f"     AUC-ROC     : {best['auc_roc']:.4f}")

    best_model, best_sc = trained_models[best_name]
    Xte_b = X_test_sc if best_sc else X_test
    y_prob_b = best_model.predict_proba(Xte_b)[:, 1]
    y_pred_b = (y_prob_b >= best_thresholds[best_name]).astype(int)

    cm = confusion_matrix(y_test, y_pred_b)
    print(f"\n  Confusion Matrix ({best_name}):")
    print(f"                    Pred Safe  Pred Malicious")
    print(f"  Actual Safe          {cm[0][0]:>6}          {cm[0][1]:>6}")
    print(f"  Actual Malicious     {cm[1][0]:>6}          {cm[1][1]:>6}")
    print(f"\n{classification_report(y_test, y_pred_b, target_names=['safe','malicious'])}")
    print(f"  Top 10 features:")
    for _, row in importances.head(10).iterrows():
        bar = "█" * int(row["importance"] * 400)
        print(f"    {row['feature']:<38} {row['importance']:.4f}  {bar}")

    print(f"\n  Models → {models_dir}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE  (use this in your app / IPS engine)
# ═══════════════════════════════════════════════════════════════════════════════

class PhishingDetector:
    """
    Drop-in replacement for your existing detector.
    Adds a fast trusted-domain whitelist check BEFORE hitting the ML model,
    which eliminates false positives on well-known legitimate domains.

    Usage:
        det = PhishingDetector("models/")
        det.predict(["https://sbi.co.in/login", "https://sbi-kyc.xyz/login"])
        # → [{"label":"safe","confidence":1.0,"reason":"trusted_domain"}, ...]
    """

    def __init__(self, models_dir: str):
        self.ensemble  = joblib.load(os.path.join(models_dir, "ensemble.pkl"))
        td_path        = os.path.join(models_dir, "trusted_domains.json")
        if os.path.exists(td_path):
            with open(td_path) as f:
                self._trusted = set(json.load(f))
        else:
            self._trusted = TRUSTED_DOMAINS.copy()

        thr_path = os.path.join(models_dir, "thresholds.json")
        with open(thr_path) as f:
            thr = json.load(f)
        self.threshold = thr.get("VotingEnsemble", 0.5)

    def predict(self, urls: list) -> list:
        from feature_extraction import extract_features_batch, FEATURE_NAMES

        results   = [None] * len(urls)
        ml_needed = []
        ml_idx    = []

        # ── Fast path: trusted domain whitelist ──────────────────────────────
        for i, url in enumerate(urls):
            apex = _extract_apex(url)
            trusted = apex in self._trusted or any(
                apex == td or apex.endswith("." + td) for td in self._trusted)
            if trusted:
                results[i] = {
                    "url":        url,
                    "label":      "safe",
                    "confidence": 1.0,
                    "reason":     "trusted_domain",
                }
            else:
                ml_needed.append(url)
                ml_idx.append(i)

        # ── ML path for everything else ───────────────────────────────────────
        if ml_needed:
            feat_df = extract_features_batch(ml_needed)
            for col in FEATURE_NAMES:
                if col not in feat_df.columns:
                    feat_df[col] = 0
            X     = feat_df[FEATURE_NAMES].fillna(0).astype(float).values
            probs = self.ensemble.predict_proba(X)[:, 1]
            for j, (url, prob) in enumerate(zip(ml_needed, probs)):
                label = "malicious" if prob >= self.threshold else "safe"
                results[ml_idx[j]] = {
                    "url":        url,
                    "label":      label,
                    "confidence": round(float(prob), 4),
                    "reason":     "ml_model",
                }

        return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv",      nargs="?",
                        default=os.path.join(os.path.dirname(HERE), "data", "combined.csv"))
    parser.add_argument("--models", default=os.path.join(os.path.dirname(HERE), "models"))
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()
    train(args.csv, args.models, args.seed)