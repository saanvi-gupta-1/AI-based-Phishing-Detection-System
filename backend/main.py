"""
Cognitive Firewall v2.1 — FastAPI Backend + SQLite Database
Every scan is logged. Every malicious URL is stored and deduplicated.
IP reputation and threat events persist across server restarts.
Database file: data/firewall.db
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

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cognitive Firewall API",
    description="AI Phishing Detection + IPS/IDS + SQLite persistence",
    version="2.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Model state ────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(HERE, "..", "models")
_model = _scaler = _feature_names = None

ids_engine = IPSIDSEngine(
    block_threshold_score=75.0,
    rate_limit_rpm=100,
    max_events=5000,
    log_path=os.path.join(HERE, "..", "logs", "threat_events.jsonl"),
)


def _load_model():
    global _model, _scaler, _feature_names
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
    if not _model:
        print("[Model] No model found — using heuristic fallback")


@app.on_event("startup")
def startup():
    init_db()
    _load_model()
    print("[Startup] Cognitive Firewall v2.1 ready")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", str(url).strip(), flags=re.IGNORECASE)


def _predict(url: str) -> dict:
    norm = _normalize(url)
    feats = extract_features(norm)
    fn = _feature_names or FEATURE_NAMES
    x = np.array([feats.get(k, 0) for k in fn], dtype=float).reshape(1, -1)
    if _model is None:
        risk = feats.get("heuristic_risk_score", 0)
        conf = min(0.99, risk + feats.get("brand_abuse", 0) * 0.3)
        return {"label": int(conf > 0.5), "confidence": round(conf, 4), "features": feats}
    lbl = int(_model.predict(x)[0])
    prob = float(_model.predict_proba(x)[0][1])
    return {"label": lbl, "confidence": round(prob, 4), "features": feats}


def _verdict(label: int, conf: float) -> str:
    if label == 1 and conf >= 0.85:
        return "phishing"
    if label == 1 and conf >= 0.40:
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
    """Write scan result to all relevant DB tables."""
    v = _verdict(pred["label"], pred["confidence"])
    sev = _severity(v, pred["confidence"])
    norm = _normalize(url)
    duration_ms = int((time.time() - t0) * 1000)

    # 1. Always log to scan_history
    log_scan(
        url=url, url_normalized=norm, verdict=v,
        confidence=pred["confidence"], action=ids_result["action"],
        severity=sev, source_ip=client_ip, scan_duration_ms=duration_ms,
    )

    # 2. Store malicious URLs in dedicated table (deduplicated)
    if v in ("phishing", "suspicious"):
        upsert_malicious_url(
            url=url, url_normalized=norm, verdict=v,
            confidence=pred["confidence"], severity=sev,
            action_taken=ids_result["action"],
            matched_rules=ids_result.get("matched_rules", []),
            source_ip=client_ip,
            features=pred["features"],
        )

    # 3. Persist IP reputation
    if ids_result.get("ip_reputation"):
        upsert_ip(ids_result["ip_reputation"])

    # 4. Persist threat event
    if ids_result.get("event"):
        log_event(ids_result["event"])


# ── Schemas ────────────────────────────────────────────────────────────────────

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


# ── Scan endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "Cognitive Firewall API",
        "version": "2.1.0",
        "model_loaded": _model is not None,
        "database": "SQLite — data/firewall.db",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "loaded" if _model else "heuristic",
        "timestamp": time.time(),
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
    }


@app.post("/scan/bulk")
async def scan_bulk(req: BulkScanRequest, request: Request):
    """Scan up to 500 URLs. All malicious ones saved to DB."""
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
    """Upload a CSV with a 'url' column. Malicious URLs saved to DB."""
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
        "total":       len(results),
        "phishing":    sum(1 for r in results if r["verdict"] == "phishing"),
        "suspicious":  sum(1 for r in results if r["verdict"] == "suspicious"),
        "clean":       sum(1 for r in results if r["verdict"] == "clean"),
        "saved_to_db": sum(1 for r in results if r["verdict"] in ("phishing", "suspicious")),
    }
    return {"summary": summary, "results": results}


# ── Malicious URL database endpoints ──────────────────────────────────────────

@app.get("/db/malicious")
def db_list(
    verdict: Optional[str] = None,
    severity: Optional[str] = None,
    search: Optional[str] = None,
    active_only: bool = True,
    limit: int = 100,
    offset: int = 0,
):
    """
    Browse the malicious URL database.
    Filter by: verdict=phishing|suspicious, severity=critical|high|medium,
               search=keyword, active_only=true|false
    Paginate with: limit, offset
    """
    return list_malicious_urls(
        verdict=verdict, severity=severity, search=search,
        active_only=active_only, limit=limit, offset=offset,
    )


@app.get("/db/malicious/stats")
def db_mal_stats():
    """Count breakdown: total, phishing, suspicious, critical, active."""
    return get_malicious_stats()


@app.get("/db/malicious/export")
def db_export():
    """Download all active malicious URLs as a CSV file."""
    return PlainTextResponse(
        content=export_malicious_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=malicious_urls.csv"},
    )


@app.post("/db/malicious/resolve")
def db_resolve(req: ResolveRequest):
    """Mark a URL as resolved — it stays in DB but removed from active list."""
    mark_resolved(req.url_normalized)
    return {"status": "resolved", "url_normalized": req.url_normalized}


@app.delete("/db/malicious/{url_normalized:path}")
def db_delete(url_normalized: str):
    """Permanently delete a URL record from the database."""
    delete_malicious_url(url_normalized)
    return {"status": "deleted", "url_normalized": url_normalized}


@app.get("/db/stats")
def db_global_stats():
    """Persistent global stats — survives server restarts."""
    return get_global_stats()


@app.get("/db/history")
def db_history(
    limit: int = 100,
    offset: int = 0,
    verdict: Optional[str] = None,
):
    """Full scan history including clean URLs."""
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


# ── Model endpoints ────────────────────────────────────────────────────────────

@app.get("/model/metrics")
def model_metrics():
    p = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(p):
        return {"error": "Run model_trainer.py first."}
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
    return {"status": "reloaded", "model_loaded": _model is not None}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")