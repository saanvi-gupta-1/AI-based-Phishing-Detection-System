"""
evaluate.py — Cognitive Firewall v3 Evaluator
==============================================
Usage:
    python evaluate.py ../data/combined.csv
    python evaluate.py mydata.csv --col-url url --col-label label --sample 1000

FIXED bugs vs original:
  1. Uses PhishingDetector (not raw model.predict) — whitelist + correct features
  2. No feature dimension mismatch (keep_mask now matches NUM_FEATURES=81)
  3. No phantom feature references (brand_as_subdomain_fqdn etc.)
  4. Whitelist covers claude.ai, anthropic.com and 200+ domains
  5. Handles both raw and normalised URLs correctly
"""

import os, sys, json, argparse, time
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MDL = os.path.join(HERE, "..", "models")


# ── Load detector ─────────────────────────────────────────────────────────────
_detector = None
_heuristic_only = False

try:
    from models_trainer import PhishingDetector
    _detector = PhishingDetector(MDL)
    print(f"[✓] ML model loaded  (threshold={_detector.thr:.3f})")
    print(f"[✓] Active features: {len(_detector.active)}")
except Exception as e:
    print(f"[!] No trained model — heuristic fallback ({e})")
    _heuristic_only = True

    # Minimal heuristic fallback using just feature_extraction
    import re
    from feature_extraction import extract_features, TRUSTED_DOMAINS

    _TW = {"co.in","co.uk","com.au","org.in","net.in","gov.in","ac.in","nic.in"}

    def _apex(url):
        url=re.sub(r"^https?://","",str(url).lower().strip())
        url=url.split("/")[0].split("?")[0].split(":")[0].split("@")[-1]
        pts=url.split(".")
        if len(pts)>=3 and ".".join(pts[-2:]) in _TW: return ".".join(pts[-3:])
        return ".".join(pts[-2:]) if len(pts)>=2 else url

    def _is_trusted(url):
        a=_apex(url)
        return a in TRUSTED_DOMAINS or any(a==t or a.endswith("."+t) for t in TRUSTED_DOMAINS)


# ── Label normaliser ──────────────────────────────────────────────────────────

def to_binary(v):
    s = str(v).strip().lower()
    try:   return int(float(s))
    except: pass
    return 0 if s in {"safe","legitimate","benign","good","clean","0"} else 1


# ── Predict single URL ────────────────────────────────────────────────────────

def predict(url: str) -> dict:
    if _detector is not None:
        r = _detector.predict_one(url)
        return {"confidence": r["confidence"], "label": r["label"],
                "reason": r["reason"]}

    # Heuristic fallback
    if _is_trusted(url):
        return {"confidence": 0.01, "label": 0, "reason": "whitelist"}

    norm = re.sub(r"^https?://(www\.)?","",str(url).strip(),flags=re.I)
    feats = extract_features(norm)
    risk = feats.get("heuristic_risk_score", 0.0)
    conf = min(0.95, risk)
    return {"confidence": round(conf,4), "label": int(conf > 0.50), "reason": "heuristic"}


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(csv_path, url_col, label_col, sample, threshold_override):
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)

    if url_col not in df.columns:
        url_col = next((c for c in df.columns if "url" in c.lower()), df.columns[0])
    if label_col not in df.columns:
        label_col = next((c for c in df.columns
                          if "label" in c.lower() or "class" in c.lower()), df.columns[-1])

    print(f"Columns: url='{url_col}'  label='{label_col}'")
    df = df[[url_col, label_col]].dropna().copy()
    df.columns = ["url","label"]
    df["label"] = df["label"].apply(to_binary)

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample} rows")

    n0=(df["label"]==0).sum(); n1=(df["label"]==1).sum()
    print(f"Dataset: {len(df):,}  Safe: {n0:,}  Malicious: {n1:,}\n")

    # Override threshold if requested
    if threshold_override and _detector:
        _detector.thr = threshold_override

    preds, confs, reasons = [], [], []
    t0 = time.time()
    for i, row in df.iterrows():
        r = predict(row["url"])
        preds.append(r["label"]); confs.append(r["confidence"]); reasons.append(r["reason"])
        if (i+1) % 200 == 0:
            elapsed=time.time()-t0; rate=(i+1)/elapsed
            print(f"  {i+1}/{len(df)} ({rate:.0f} URLs/s)")

    elapsed = time.time()-t0
    df["pred"]       = preds
    df["confidence"] = confs
    df["reason"]     = reasons

    y_true = df["label"].values
    y_pred = df["pred"].values
    y_conf = df["confidence"].values

    tp=((y_true==1)&(y_pred==1)).sum()
    tn=((y_true==0)&(y_pred==0)).sum()
    fp=((y_true==0)&(y_pred==1)).sum()
    fn=((y_true==1)&(y_pred==0)).sum()

    acc   = (tp+tn)/len(y_true)
    prec  = tp/(tp+fp) if (tp+fp)>0 else 0
    rec   = tp/(tp+fn) if (tp+fn)>0 else 0
    f1    = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    fpr   = fp/(fp+tn) if (fp+tn)>0 else 0
    fnr   = fn/(fn+tp) if (fn+tp)>0 else 0

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_conf)
    except: auc=None

    print("\n"+"═"*60)
    print("  EVALUATION RESULTS")
    print("═"*60)
    print(f"  Engine           : {'ML Ensemble' if not _heuristic_only else 'Heuristic'}")
    print(f"  URLs evaluated   : {len(df):,}")
    print(f"  Time             : {elapsed:.1f}s  ({len(df)/elapsed:.0f} URLs/s)")
    print("─"*60)
    print(f"  Accuracy         : {acc*100:.2f}%")
    print(f"  Precision        : {prec*100:.2f}%")
    print(f"  Recall           : {rec*100:.2f}%")
    print(f"  F1 Score         : {f1*100:.2f}%")
    if auc: print(f"  AUC-ROC          : {auc:.4f}")
    print("─"*60)
    print(f"  True Positives   : {tp:>6,}  ✅ phishing caught")
    print(f"  True Negatives   : {tn:>6,}  ✅ clean allowed")
    print(f"  False Positives  : {fp:>6,}  ⚠️  clean wrongly blocked")
    print(f"  False Negatives  : {fn:>6,}  🚨 threats missed")
    print("─"*60)
    print(f"  False Positive Rate : {fpr*100:.2f}%")
    print(f"  False Negative Rate : {fnr*100:.2f}%")
    print("═"*60)

    missed = df[(df["label"]==1)&(df["pred"]==0)].sort_values("confidence")
    if not missed.empty:
        print(f"\n  🚨 Top 10 MISSED THREATS:")
        for _,r in missed.head(10).iterrows():
            print(f"     [{r['confidence']:.3f}] {r['url']}")

    wrong = df[(df["label"]==0)&(df["pred"]==1)].sort_values("confidence",ascending=False)
    if not wrong.empty:
        print(f"\n  ⚠️  Top 10 FALSE POSITIVES:")
        for _,r in wrong.head(10).iterrows():
            print(f"     [{r['confidence']:.3f}] {r['url']}")

    out = csv_path.replace(".csv","_eval_results.csv")
    df.to_csv(out,index=False)
    print(f"\n  Results saved → {out}")
    return acc


if __name__=="__main__":
    p=argparse.ArgumentParser(description="Cognitive Firewall — Evaluator")
    p.add_argument("csv",                   help="Labeled CSV path")
    p.add_argument("--col-url",   default="url",   help="URL column name")
    p.add_argument("--col-label", default="label", help="Label column name")
    p.add_argument("--sample",    type=int,   default=None)
    p.add_argument("--threshold", type=float, default=None)
    args=p.parse_args()
    evaluate(args.csv, args.col_url, args.col_label, args.sample, args.threshold)