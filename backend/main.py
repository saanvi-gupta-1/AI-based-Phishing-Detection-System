"""
Cognitive Firewall v2.2 — FastAPI Backend + SQLite Database
Fixed: Extended trusted domain whitelist (includes claude.ai, anthropic.com, etc.)
       Better heuristic fallback to reduce false positives
       Claude AI-powered analysis integration support
"""

import os, sys, re, json, time, io, warnings
import joblib, numpy as np, pandas as pd
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from feature_extraction import extract_features, extract_features_batch, FEATURE_NAMES
from ips_ids import IPSIDSEngine
from database import (
    init_db,
    upsert_malicious_url, list_malicious_urls, get_malicious_url_by_norm,
    mark_resolved, delete_malicious_url, get_malicious_stats, export_malicious_csv,
    log_scan, get_scan_history,
    upsert_ip, get_blocked_ips_db,
    log_event, get_events,
    get_global_stats,
)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cognitive Firewall API",
    description="AI Phishing Detection + IPS/IDS + SQLite persistence",
    version="2.2.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Extended Trusted Domain Whitelist (prevents false positives) ──────────────

EXTENDED_TRUSTED_DOMAINS = {
    # Anthropic / Claude
    "claude.ai", "anthropic.com", "api.anthropic.com",
    "console.anthropic.com", "docs.anthropic.com",
    # Major AI companies
    "openai.com", "chat.openai.com", "platform.openai.com",
    "gemini.google.com", "bard.google.com", "copilot.microsoft.com",
    "huggingface.co", "cohere.com", "mistral.ai",
    # Major tech
    "google.com", "google.co.in", "gmail.com", "youtube.com",
    "microsoft.com", "outlook.com", "office.com", "azure.com",
    "apple.com", "icloud.com",
    "amazon.com", "amazon.in", "aws.amazon.com",
    "github.com", "gitlab.com", "bitbucket.org",
    "stackoverflow.com", "stackexchange.com",
    "wikipedia.org", "wikimedia.org",
    "linkedin.com", "twitter.com", "x.com",
    "instagram.com", "facebook.com", "meta.com",
    "whatsapp.com", "telegram.org",
    "reddit.com", "medium.com", "substack.com",
    "cloudflare.com", "1.1.1.1",
    "vercel.com", "netlify.com", "heroku.com",
    "docker.com", "kubernetes.io",
    "mozilla.org", "firefox.com",
    # Indian banks
    "sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com",
    "axisbank.com", "kotakbank.com", "kotak.com", "bankofbaroda.in",
    "canarabank.in", "pnbindia.in", "unionbankofindia.co.in",
    "yesbank.in", "indusind.com", "idfcfirstbank.com",
    # Indian fintech
    "zerodha.com", "groww.in", "paytm.com", "phonepe.com",
    "razorpay.com", "policybazaar.com", "cleartax.in",
    "upstox.com", "angelone.in",
    # Indian telecom
    "airtel.in", "jio.com", "vi.in", "bsnl.co.in",
    # Indian govt
    "gov.in", "nic.in", "irctc.co.in", "incometax.gov.in",
    "uidai.gov.in", "digilocker.gov.in", "mca.gov.in",
    "epfindia.gov.in", "sebi.gov.in", "rbi.org.in",
    "npci.org.in", "india.gov.in", "gst.gov.in",
    "bseindia.com", "nseindia.com",
    # Indian ecommerce
    "flipkart.com", "myntra.com", "bigbasket.com",
    "zomato.com", "swiggy.com", "makemytrip.com",
    "naukri.com",
    # Indian IT
    "tcs.com", "infosys.com", "wipro.com", "zoho.com",
    # News
    "ndtv.com", "thehindu.com", "moneycontrol.com",
    "bbc.com", "reuters.com", "apnews.com", "cnn.com",
    # Dev tools
    "npmjs.com", "pypi.org", "docs.python.org",
    "developer.mozilla.org", "w3schools.com",
}

TWO_PART_TLDS = {
    "co.in", "gov.in", "org.in", "net.in", "ac.in", "nic.in", "edu.in",
    "co.uk", "org.uk", "co.nz", "com.au", "co.za", "co.jp",
}


def _extract_apex(url: str) -> str:
    url = str(url).lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = url.split("/")[0].split("?")[0].split(":")[0].split("@")[-1]
    parts = url.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_PART_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else url


def _is_trusted(url: str) -> bool:
    apex = _extract_apex(url)
    if apex in EXTENDED_TRUSTED_DOMAINS:
        return True
    # Check if apex is a subdomain of a trusted domain
    return any(apex == td or apex.endswith("." + td) for td in EXTENDED_TRUSTED_DOMAINS)


# ── Model state ───────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(HERE, "..", "models")
_model = _scaler = _feature_names = None
_keep_mask = None
_threshold = 0.5

ids_engine = IPSIDSEngine(
    block_threshold_score=75.0,
    rate_limit_rpm=200,
    max_events=10000,
    log_path=os.path.join(HERE, "..", "logs", "threat_events.jsonl"),
)


def _load_model():
    global _model, _scaler, _feature_names, _keep_mask, _threshold
    os.makedirs(MODELS_DIR, exist_ok=True)

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
        print("[Model] No trained model found — using improved heuristic fallback")
    else:
        print(f"[Model] Threshold: {_threshold}")


@app.on_event("startup")
def startup():
    os.makedirs(os.path.join(HERE, "..", "data"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "..", "logs"), exist_ok=True)
    init_db()
    _load_model()
    print("[Startup] Cognitive Firewall v2.2 ready")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", str(url).strip(), flags=re.IGNORECASE)


def _predict(url: str) -> dict:
    # Layer 0: Trusted domain whitelist → always clean
    if _is_trusted(url):
        feats = extract_features(_normalize(url))
        return {
            "label": 0,
            "confidence": 0.01,
            "features": feats,
            "reason": "trusted_whitelist",
        }

    norm = _normalize(url)
    feats = extract_features(norm)
    fn = _feature_names or FEATURE_NAMES
    x = np.array([feats.get(k, 0) for k in fn], dtype=float).reshape(1, -1)

    if _model is None:
        # Improved heuristic fallback — less aggressive, more calibrated
        risk = feats.get("heuristic_risk_score", 0)
        brand_boost = feats.get("brand_abuse", 0) * 0.25
        bank_boost = feats.get("indian_bank_phishing", 0) * 0.35
        conf = min(0.95, risk + brand_boost + bank_boost)
        # Trusted domains already handled above; apply a small safe discount
        if feats.get("is_trusted_domain", 0):
            conf = max(0.0, conf - 0.4)
        return {
            "label": int(conf > 0.55),
            "confidence": round(conf, 4),
            "features": feats,
            "reason": "heuristic",
        }

    # Apply keep_mask if available
    if _keep_mask is not None:
        x_masked = x[:, _keep_mask]
    else:
        x_masked = x

    if _scaler is not None:
        x_masked = _scaler.transform(x_masked)

    prob = float(_model.predict_proba(x_masked)[0][1])
    lbl = int(prob >= _threshold)
    return {
        "label": lbl,
        "confidence": round(prob, 4),
        "features": feats,
        "reason": "ml_model",
    }


def _verdict(label: int, conf: float) -> str:
    if label == 1 and conf >= 0.80:
        return "phishing"
    if label == 1 and conf >= 0.45:
        return "suspicious"
    return "clean"


def _severity(verdict: str, conf: float) -> str:
    if verdict == "phishing" and conf >= 0.90:
        return "critical"
    if verdict == "phishing":
        return "high"
    if verdict == "suspicious":
        return "medium"
    return "low"


def _persist(url: str, pred: dict, ids_result: dict, client_ip: str, t0: float):
    v = _verdict(pred["label"], pred["confidence"])
    sev = _severity(v, pred["confidence"])
    norm = _normalize(url)
    duration_ms = int((time.time() - t0) * 1000)

    log_scan(
        url=url, url_normalized=norm, verdict=v,
        confidence=pred["confidence"], action=ids_result["action"],
        severity=sev, source_ip=client_ip, scan_duration_ms=duration_ms,
    )

    if v in ("phishing", "suspicious"):
        upsert_malicious_url(
            url=url, url_normalized=norm, verdict=v,
            confidence=pred["confidence"], severity=sev,
            action_taken=ids_result["action"],
            matched_rules=ids_result.get("matched_rules", []),
            source_ip=client_ip,
            features=pred["features"],
        )

    if ids_result.get("ip_reputation"):
        upsert_ip(ids_result["ip_reputation"])

    if ids_result.get("event"):
        log_event(ids_result["event"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    url: str
    source_ip: Optional[str] = None

class BulkScanRequest(BaseModel):
    urls: List[str]
    source_ip: Optional[str] = None

class IPActionRequest(BaseModel):
    ip: str
    reason: Optional[str] = "Manual action"

class ResolveRequest(BaseModel):
    url_normalized: str


# ── Scan endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Cognitive Firewall API",
        "version": "2.2.0",
        "model_loaded": _model is not None,
        "model_threshold": _threshold,
        "database": "SQLite — data/firewall.db",
        "docs": "/docs",
        "trusted_domains_count": len(EXTENDED_TRUSTED_DOMAINS),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded" if _model else "heuristic",
        "timestamp": time.time(),
        "version": "2.2.0",
    }


@app.post("/scan")
async def scan_url(req: ScanRequest, request: Request):
    """Scan one URL. Saves to DB automatically if malicious."""
    t0 = time.time()
    ip = req.source_ip or request.client.host
    pred = _predict(req.url)
    ids_result = ids_engine.analyze(
        url=req.url, ml_confidence=pred["confidence"],
        ml_label=pred["label"], features=pred["features"], source_ip=ip,
    )
    v = _verdict(pred["label"], pred["confidence"])
    sev = _severity(v, pred["confidence"])
    _persist(req.url, pred, ids_result, ip, t0)

    return {
        "url": req.url,
        "url_normalized": _normalize(req.url),
        "verdict": v,
        "confidence": pred["confidence"],
        "confidence_pct": f"{pred['confidence']*100:.1f}%",
        "action": ids_result["action"],
        "severity": sev,
        "threat_type": ids_result["threat_type"],
        "matched_rules": ids_result["matched_rules"],
        "features": pred["features"],
        "ip_reputation": ids_result["ip_reputation"],
        "scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scan_duration_ms": int((time.time() - t0) * 1000),
        "saved_to_db": v in ("phishing", "suspicious"),
        "prediction_reason": pred.get("reason", "unknown"),
    }


@app.post("/scan/bulk")
async def scan_bulk(req: BulkScanRequest, request: Request):
    if len(req.urls) > 500:
        raise HTTPException(400, "Max 500 URLs per request.")
    ip = req.source_ip or request.client.host
    results, t0 = [], time.time()

    for url in req.urls:
        ts = time.time()
        pred = _predict(url)
        ids_r = ids_engine.analyze(
            url=url, ml_confidence=pred["confidence"],
            ml_label=pred["label"], features=pred["features"], source_ip=ip,
        )
        v = _verdict(pred["label"], pred["confidence"])
        _persist(url, pred, ids_r, ip, ts)
        results.append({
            "url": url,
            "verdict": v,
            "confidence": pred["confidence"],
            "action": ids_r["action"],
            "severity": _severity(v, pred["confidence"]),
        })

    summary = {
        "total":       len(results),
        "phishing":    sum(1 for r in results if r["verdict"] == "phishing"),
        "suspicious":  sum(1 for r in results if r["verdict"] == "suspicious"),
        "clean":       sum(1 for r in results if r["verdict"] == "clean"),
        "blocked":     sum(1 for r in results if r["action"] == "blocked"),
        "saved_to_db": sum(1 for r in results if r["verdict"] in ("phishing", "suspicious")),
        "elapsed_ms":  int((time.time() - t0) * 1000),
    }
    return {"summary": summary, "results": results}


@app.post("/scan/csv")
async def scan_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file.")
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(400, f"CSV parse error: {e}")

    url_col = next((c for c in df.columns if "url" in c.lower()), df.columns[0])
    urls = df[url_col].dropna().astype(str).tolist()
    if len(urls) > 2000:
        raise HTTPException(400, "Max 2000 URLs per CSV.")

    results = []
    for url in urls:
        pred = _predict(url)
        v = _verdict(pred["label"], pred["confidence"])
        sev = _severity(v, pred["confidence"])
        log_scan(
            url=url, url_normalized=_normalize(url), verdict=v,
            confidence=pred["confidence"], action="alerted",
            severity=sev, source_ip="csv_upload",
        )
        if v in ("phishing", "suspicious"):
            upsert_malicious_url(
                url=url, url_normalized=_normalize(url), verdict=v,
                confidence=pred["confidence"], severity=sev,
                action_taken="alerted", matched_rules=[],
                source_ip="csv_upload", features=pred["features"],
            )
        results.append({"url": url, "verdict": v, "confidence": pred["confidence"]})

    summary = {
        "total": len(results),
        "phishing": sum(1 for r in results if r["verdict"] == "phishing"),
        "suspicious": sum(1 for r in results if r["verdict"] == "suspicious"),
        "clean": sum(1 for r in results if r["verdict"] == "clean"),
        "saved_to_db": sum(1 for r in results if r["verdict"] in ("phishing", "suspicious")),
    }
    return {"summary": summary, "results": results}


# ── Malicious URL DB endpoints ────────────────────────────────────────────────

@app.get("/db/malicious")
def db_list(
    verdict: Optional[str] = None,
    severity: Optional[str] = None,
    search: Optional[str] = None,
    active_only: bool = True,
    limit: int = 100,
    offset: int = 0,
):
    return list_malicious_urls(
        verdict=verdict, severity=severity, search=search,
        active_only=active_only, limit=limit, offset=offset,
    )


@app.get("/db/malicious/stats")
def db_mal_stats():
    return get_malicious_stats()


@app.get("/db/malicious/export")
def db_export():
    return PlainTextResponse(
        content=export_malicious_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=malicious_urls.csv"},
    )


@app.post("/db/malicious/resolve")
def db_resolve(req: ResolveRequest):
    mark_resolved(req.url_normalized)
    return {"status": "resolved", "url_normalized": req.url_normalized}


@app.delete("/db/malicious/{url_normalized:path}")
def db_delete(url_normalized: str):
    delete_malicious_url(url_normalized)
    return {"status": "deleted", "url_normalized": url_normalized}


@app.get("/db/stats")
def db_global_stats():
    return get_global_stats()


@app.get("/db/history")
def db_history(
    limit: int = 100,
    offset: int = 0,
    verdict: Optional[str] = None,
):
    return get_scan_history(limit=limit, offset=offset, verdict=verdict)


# ── IPS/IDS endpoints ─────────────────────────────────────────────────────────

@app.get("/ids/stats")
def ids_stats():
    return {**ids_engine.get_stats(), **get_global_stats()}


@app.get("/ids/events")
def ids_events(limit: int = 50, severity: Optional[str] = None):
    db = get_events(limit=limit, severity=severity)
    return db if db else ids_engine.get_recent_events(limit=limit, severity_filter=severity)


@app.get("/ids/blocked-ips")
def ids_blocked_ips():
    db = get_blocked_ips_db()
    return db if db else ids_engine.get_blocked_ips()


@app.get("/ids/top-threats")
def ids_top_threats(limit: int = 10):
    return ids_engine.get_top_threat_ips(limit=limit)


@app.get("/ids/ip/{ip}")
def ids_ip_rep(ip: str):
    return ids_engine.get_ip_reputation(ip)


@app.post("/ids/block-ip")
def ids_block(req: IPActionRequest):
    result = ids_engine.block_ip(req.ip, req.reason)
    upsert_ip(ids_engine.get_ip_reputation(req.ip))
    return result


@app.post("/ids/unblock-ip")
def ids_unblock(req: IPActionRequest):
    result = ids_engine.unblock_ip(req.ip)
    upsert_ip(ids_engine.get_ip_reputation(req.ip))
    return result


@app.get("/ids/rules")
def ids_rules():
    return ids_engine.get_rules()


# ── Model endpoints ───────────────────────────────────────────────────────────

@app.get("/model/metrics")
def model_metrics():
    p = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(p):
        return {
            "error": "No trained model found.",
            "help": "Run: python models_trainer.py <path_to_combined.csv>",
            "model_loaded": False,
            "using_heuristic": True,
        }
    return json.load(open(p))


@app.get("/model/feature-importance")
def feature_importance(top: int = 20):
    p = os.path.join(MODELS_DIR, "feature_importance.csv")
    if not os.path.exists(p):
        return {"error": "No feature_importance.csv found."}
    return pd.read_csv(p).head(top).to_dict(orient="records")


@app.post("/model/reload")
def model_reload():
    _load_model()
    return {
        "status": "reloaded",
        "model_loaded": _model is not None,
        "threshold": _threshold,
    }


@app.get("/model/status")
def model_status():
    return {
        "model_loaded": _model is not None,
        "has_scaler": _scaler is not None,
        "has_keep_mask": _keep_mask is not None,
        "threshold": _threshold,
        "feature_count": len(_feature_names) if _feature_names else len(FEATURE_NAMES),
        "using_heuristic": _model is None,
        "trusted_domains_count": len(EXTENDED_TRUSTED_DOMAINS),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")