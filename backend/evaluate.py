"""
evaluate.py — Test Cognitive Firewall on unseen labeled data
Usage:
    python evaluate.py ../data/test.csv
    python evaluate.py ../data/test.csv --col-url url --col-label label --sample 2000
"""

import os, sys, json, argparse, time
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from feature_extraction import extract_features, FEATURE_NAMES

# ── Try to load trained model ─────────────────────────────────────────────────
import joblib
MODELS_DIR = os.path.join(HERE, "..", "models")

_model = _scaler = _feature_names = _keep_mask = None
_threshold = 0.5

for fname in ["ensemble.pkl", "randomforest.pkl", "random_forest.pkl"]:
    p = os.path.join(MODELS_DIR, fname)
    if os.path.exists(p):
        _model = joblib.load(p)
        print(f"[Model] Loaded: {fname}")
        break

sp = os.path.join(MODELS_DIR, "scaler.pkl")
if os.path.exists(sp):
    _scaler = joblib.load(sp)

fp = os.path.join(MODELS_DIR, "feature_names.json")
if os.path.exists(fp):
    with open(fp) as f:
        _feature_names = json.load(f)

mp = os.path.join(MODELS_DIR, "keep_mask.npy")
if os.path.exists(mp):
    _keep_mask = np.load(mp)

tp = os.path.join(MODELS_DIR, "thresholds.json")
if os.path.exists(tp):
    with open(tp) as f:
        thr = json.load(f)
    _threshold = thr.get("Ensemble", thr.get("RandomForest", 0.5))

if not _model:
    print("[Model] No trained model — using heuristic fallback\n")
else:
    print(f"[Model] Threshold: {_threshold}\n")


# ── Extended trusted whitelist (same as main.py) ──────────────────────────────
import re

TRUSTED_DOMAINS = {
    "sbi.co.in","onlinesbi.sbi","hdfcbank.com","icicibank.com","axisbank.com",
    "kotakbank.com","kotak.com","bankofbaroda.in","canarabank.in","pnbindia.in",
    "paytm.com","phonepe.com","razorpay.com","flipkart.com","amazon.in",
    "google.com","gmail.com","youtube.com","microsoft.com","linkedin.com",
    "github.com","apple.com","amazon.com","wikipedia.org","twitter.com","x.com",
    "instagram.com","facebook.com","irctc.co.in","gov.in","nic.in",
    "incometax.gov.in","uidai.gov.in","npci.org.in","zerodha.com","groww.in",
    "airtel.in","jio.com","zomato.com","swiggy.com",
}
TWO_PART_TLDS = {"co.in","gov.in","org.in","net.in","ac.in","co.uk","com.au"}

def _apex(url):
    url = str(url).lower().strip()
    url = re.sub(r"^https?://","",url)
    url = url.split("/")[0].split("?")[0].split(":")[0].split("@")[-1]
    parts = url.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_PART_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else url

def _is_trusted(url):
    apex = _apex(url)
    return apex in TRUSTED_DOMAINS or any(
        apex == td or apex.endswith("."+td) for td in TRUSTED_DOMAINS
    )


# ── Predict single URL ────────────────────────────────────────────────────────

def predict(url: str) -> dict:
    if _is_trusted(url):
        return {"confidence": 0.01, "label": 0, "reason": "whitelist"}

    norm = re.sub(r"^https?://(www\.)?","", str(url).strip(), flags=re.IGNORECASE)
    feats = extract_features(norm)
    fn = _feature_names if _feature_names else FEATURE_NAMES
    x = np.array([feats.get(k, 0) for k in fn], dtype=float).reshape(1, -1)
    if _keep_mask is not None and len(_keep_mask) != x.shape[1]:
       from feature_extraction import FEATURE_NAMES as ALL_FEATS
       x = np.array([feats.get(k, 0) for k in ALL_FEATS], dtype=float).reshape(1, -1)
    

    if _model is None:
        risk = feats.get("heuristic_risk_score", 0)
        risk += feats.get("brand_abuse", 0) * 0.25
        risk += feats.get("indian_bank_phishing", 0) * 0.35
        risk += feats.get("brand_as_subdomain_fqdn", 0) * 0.40
        risk += feats.get("gov_brand_wrong_tld", 0) * 0.35
        conf = min(0.95, risk)
        if feats.get("is_trusted_domain", 0):
            conf = max(0.0, conf - 0.4)
        return {"confidence": round(conf, 4), "label": int(conf > 0.55), "reason": "heuristic"}

    xm = x[:, _keep_mask] if _keep_mask is not None else x
    if _scaler is not None:
        xm = _scaler.transform(xm)
    prob = float(_model.predict_proba(xm)[0][1])
    return {"confidence": round(prob, 4), "label": int(prob >= _threshold), "reason": "ml"}


# ── Label normaliser ──────────────────────────────────────────────────────────

def to_binary(v):
    s = str(v).strip().lower()
    try:
        return int(float(s))
    except ValueError:
        pass
    return 0 if s in {"safe","legitimate","benign","good","clean","0"} else 1


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(csv_path, url_col, label_col, sample, threshold_override):
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    # Auto-detect columns if not specified
    if url_col not in df.columns:
        url_col = next((c for c in df.columns if "url" in c.lower()), df.columns[0])
    if label_col not in df.columns:
        label_col = next((c for c in df.columns
                         if "label" in c.lower() or "class" in c.lower()), df.columns[-1])

    print(f"Using columns: url='{url_col}' label='{label_col}'")
    df = df[[url_col, label_col]].dropna().copy()
    df.columns = ["url", "label"]
    df["label"] = df["label"].apply(to_binary)

    # Sample if requested
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample} rows (balanced if possible)")

    n_safe = (df["label"] == 0).sum()
    n_mal  = (df["label"] == 1).sum()
    print(f"Dataset: {len(df):,} URLs — Safe: {n_safe:,} | Malicious: {n_mal:,}\n")

    # ── Run predictions ───────────────────────────────────────────────────────
    preds, confs, reasons = [], [], []
    t0 = time.time()

    for i, row in df.iterrows():
        r = predict(row["url"])
        preds.append(r["label"])
        confs.append(r["confidence"])
        reasons.append(r["reason"])

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(df) - i - 1) / rate
            print(f"  Progress: {i+1}/{len(df)} ({rate:.0f} URLs/s, ~{remaining:.0f}s remaining)")

    elapsed = time.time() - t0
    df["pred"]       = preds
    df["confidence"] = confs
    df["reason"]     = reasons

    thr = threshold_override or _threshold

    # ── Metrics ───────────────────────────────────────────────────────────────
    y_true = df["label"].values
    y_pred = df["pred"].values
    y_conf = df["confidence"].values

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0  # false positive rate
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0  # false negative rate (missed threats)

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_conf)
    except Exception:
        auc = None

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  EVALUATION RESULTS")
    print("═" * 55)
    print(f"  URLs evaluated   : {len(df):,}")
    print(f"  Time elapsed     : {elapsed:.1f}s ({len(df)/elapsed:.0f} URLs/sec)")
    print(f"  Engine           : {'ML model' if _model else 'Heuristic fallback'}")
    print("─" * 55)
    print(f"  Accuracy         : {accuracy*100:.2f}%")
    print(f"  Precision        : {precision*100:.2f}%  (of flagged, how many real?)")
    print(f"  Recall           : {recall*100:.2f}%   (of real threats, how many caught?)")
    print(f"  F1 Score         : {f1*100:.2f}%")
    if auc:
        print(f"  AUC-ROC          : {auc:.4f}")
    print("─" * 55)
    print(f"  True Positives   : {tp:,}   (phishing caught ✅)")
    print(f"  True Negatives   : {tn:,}   (clean correctly allowed ✅)")
    print(f"  False Positives  : {fp:,}   (clean wrongly blocked ⚠️ )")
    print(f"  False Negatives  : {fn:,}   (phishing missed 🚨)")
    print("─" * 55)
    print(f"  False Positive Rate : {fpr*100:.2f}%  (legit URLs wrongly flagged)")
    print(f"  False Negative Rate : {fnr*100:.2f}%  (threats that slipped through)")
    print("═" * 55)

    # ── Worst false negatives (missed threats) ────────────────────────────────
    missed = df[(df["label"] == 1) & (df["pred"] == 0)].sort_values("confidence")
    if not missed.empty:
        print(f"\n  🚨 Top 10 MISSED THREATS (false negatives):")
        for _, r in missed.head(10).iterrows():
            print(f"     [{r['confidence']:.2f}] {r['url']}")

    # ── Worst false positives (wrongly blocked) ───────────────────────────────
    wrong = df[(df["label"] == 0) & (df["pred"] == 1)].sort_values("confidence", ascending=False)
    if not wrong.empty:
        print(f"\n  ⚠️  Top 10 WRONG BLOCKS (false positives):")
        for _, r in wrong.head(10).iterrows():
            print(f"     [{r['confidence']:.2f}] {r['url']}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = csv_path.replace(".csv", "_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Full results saved → {out_path}")

    return accuracy


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv",                   help="Path to labeled CSV")
    p.add_argument("--col-url",   default="url",   help="URL column name")
    p.add_argument("--col-label", default="label", help="Label column name")
    p.add_argument("--sample",    type=int, default=None, help="Sample N rows")
    p.add_argument("--threshold", type=float, default=None, help="Override decision threshold")
    args = p.parse_args()

    evaluate(args.csv, args.col_url, args.col_label, args.sample, args.threshold)