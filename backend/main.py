"""
Cognitive Firewall v3 — FastAPI Backend
Uses PhishingDetector (2-layer whitelist + ML) from model_trainer.py
"""
import os,sys,re,json,time,io,warnings
import pandas as pd
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)

from ips_ids import IPSIDSEngine
from database import (
    init_db, upsert_malicious_url, list_malicious_urls,
    mark_resolved, delete_malicious_url, get_malicious_stats, export_malicious_csv,
    log_scan, get_scan_history, upsert_ip, get_blocked_ips_db,
    log_event, get_events, get_global_stats,
)
from threat_intel import analyze_threats

app=FastAPI(title="Cognitive Firewall API",version="3.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

MDL=os.path.join(HERE,"..","models")
_det=None

ids_engine=IPSIDSEngine(
    block_threshold_score=75.0, rate_limit_rpm=200, max_events=5000,
    log_path=os.path.join(HERE,"..","logs","events.jsonl"),
)

def _load():
    global _det
    try:
        from models_trainer import PhishingDetector
        _det=PhishingDetector(MDL)
        print(f"[Model] Loaded  thr={_det.thr:.3f}  feats={len(_det.active)}")
    except Exception as e:
        print(f"[Model] Fallback heuristic ({e})")
        _det=None

@app.on_event("startup")
def startup():
    os.makedirs(os.path.join(HERE,"..","logs"),exist_ok=True)
    init_db(); _load()
    print("[Startup] Cognitive Firewall v3 ready")

def _norm(url):
    return re.sub(r"^https?://(www\.)?","",str(url).strip(),flags=re.I).rstrip("/")

def _predict(url):
    # 1. External Threat Intelligence & Dynamic Page Scanning (Zero-Day catch-all)
    threat = analyze_threats(url)
    if threat["flagged"]:
        return {
            "label": 1,
            "confidence": threat["confidence"],
            "features": {},
            "reason": threat["reason"]
        }

    # 2. Local Machine Learning Model
    if _det:
        r=_det.predict_one(url)
        return {"label":r["label"],"confidence":r["confidence"],
                "features":r.get("features",{}),"reason":r.get("reason","ml")}
    from feature_extraction import extract_features
    f=extract_features(_norm(url))
    risk=f.get("heuristic_risk_score",0)
    conf=min(0.99,risk+f.get("brand_abuse",0)*0.3)
    return {"label":int(conf>0.5),"confidence":round(conf,4),"features":f,"reason":"heuristic"}

def _verdict(lbl,conf):
    if lbl==1 and conf>=0.85: return "phishing"
    if lbl==1 and conf>=0.40: return "suspicious"
    return "clean"

def _severity(v,conf):
    if v=="phishing" and conf>=0.90: return "critical"
    if v=="phishing": return "high"
    if v=="suspicious": return "medium"
    return "low"

def _persist(url,pred,ids_r,ip,t0):
    v=_verdict(pred["label"],pred["confidence"])
    sev=_severity(v,pred["confidence"])
    norm=_norm(url)
    log_scan(url=url,url_normalized=norm,verdict=v,confidence=pred["confidence"],
             action=ids_r["action"],severity=sev,source_ip=ip,
             scan_duration_ms=int((time.time()-t0)*1000))
    if v in ("phishing","suspicious"):
        upsert_malicious_url(url=url,url_normalized=norm,verdict=v,
                             confidence=pred["confidence"],severity=sev,
                             action_taken=ids_r["action"],
                             matched_rules=ids_r.get("matched_rules",[]),
                             source_ip=ip,features=pred.get("features",{}))
    if ids_r.get("ip_reputation"): upsert_ip(ids_r["ip_reputation"])
    if ids_r.get("event"):         log_event(ids_r["event"])

class ScanReq(BaseModel):
    url: str
    source_ip: Optional[str]=None

class BulkReq(BaseModel):
    urls: List[str]
    source_ip: Optional[str]=None

class IPReq(BaseModel):
    ip: str
    reason: Optional[str]="Manual"

class ResolveReq(BaseModel):
    url_normalized: str

@app.get("/")
def root(): return {"name":"Cognitive Firewall API","version":"3.0.0",
                    "model_loaded":_det is not None,"docs":"/docs"}

@app.get("/health")
def health(): return {"status":"healthy","model":"loaded" if _det else "heuristic",
                      "timestamp":time.time()}

@app.post("/scan")
async def scan(req:ScanReq, request:Request):
    t0=time.time(); ip=req.source_ip or request.client.host
    pred=_predict(req.url)
    ids_r=ids_engine.analyze(url=req.url,ml_confidence=pred["confidence"],
                              ml_label=pred["label"],features=pred.get("features",{}),source_ip=ip)
    v=_verdict(pred["label"],pred["confidence"])
    sev=_severity(v,pred["confidence"])
    _persist(req.url,pred,ids_r,ip,t0)
    return {"url":req.url,"url_normalized":_norm(req.url),"verdict":v,
            "confidence":pred["confidence"],"confidence_pct":f"{pred['confidence']*100:.1f}%",
            "action":ids_r["action"],"severity":sev,
            "threat_type":ids_r["threat_type"],"matched_rules":ids_r["matched_rules"],
            "detection_reason":pred.get("reason","ml"),
            "features":pred.get("features",{}),"ip_reputation":ids_r["ip_reputation"],
            "scan_time":time.strftime("%Y-%m-%d %H:%M:%S"),
            "scan_duration_ms":int((time.time()-t0)*1000),
            "saved_to_db":v in ("phishing","suspicious")}

@app.post("/scan/bulk")
async def scan_bulk(req:BulkReq, request:Request):
    if len(req.urls)>500: raise HTTPException(400,"Max 500 URLs")
    ip=req.source_ip or request.client.host
    results=[]; t0=time.time()
    for url in req.urls:
        ts=time.time()
        pred=_predict(url)
        ids_r=ids_engine.analyze(url=url,ml_confidence=pred["confidence"],
                                  ml_label=pred["label"],features=pred.get("features",{}),source_ip=ip)
        v=_verdict(pred["label"],pred["confidence"])
        _persist(url,pred,ids_r,ip,ts)
        results.append({"url":url,"verdict":v,"confidence":pred["confidence"],
                        "action":ids_r["action"],"severity":_severity(v,pred["confidence"])})
    s={"total":len(results),
       "phishing":sum(1 for r in results if r["verdict"]=="phishing"),
       "suspicious":sum(1 for r in results if r["verdict"]=="suspicious"),
       "clean":sum(1 for r in results if r["verdict"]=="clean"),
       "blocked":sum(1 for r in results if r["action"]=="blocked"),
       "saved_to_db":sum(1 for r in results if r["verdict"] in ("phishing","suspicious")),
       "elapsed_ms":int((time.time()-t0)*1000)}
    return {"summary":s,"results":results}

@app.post("/scan/csv")
async def scan_csv(file:UploadFile=File(...)):
    if not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Upload .csv")
    content=await file.read()
    try: df=pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e: raise HTTPException(400,f"CSV error: {e}")
    uc=next((c for c in df.columns if "url" in c.lower()),df.columns[0])
    urls=df[uc].dropna().astype(str).tolist()
    if len(urls)>2000: raise HTTPException(400,"Max 2000 URLs")
    results=[]
    for url in urls:
        pred=_predict(url); v=_verdict(pred["label"],pred["confidence"])
        sev=_severity(v,pred["confidence"])
        log_scan(url=url,url_normalized=_norm(url),verdict=v,
                 confidence=pred["confidence"],action="alerted",severity=sev,source_ip="csv")
        if v in ("phishing","suspicious"):
            upsert_malicious_url(url=url,url_normalized=_norm(url),verdict=v,
                                 confidence=pred["confidence"],severity=sev,
                                 action_taken="alerted",matched_rules=[],
                                 source_ip="csv",features=pred.get("features",{}))
        results.append({"url":url,"verdict":v,"confidence":pred["confidence"]})
    return {"summary":{"total":len(results),
                        "phishing":sum(1 for r in results if r["verdict"]=="phishing"),
                        "suspicious":sum(1 for r in results if r["verdict"]=="suspicious"),
                        "clean":sum(1 for r in results if r["verdict"]=="clean"),
                        "saved_to_db":sum(1 for r in results if r["verdict"] in ("phishing","suspicious"))},
            "results":results}

@app.get("/db/malicious")
def db_list(verdict:Optional[str]=None,severity:Optional[str]=None,
            search:Optional[str]=None,active_only:bool=True,limit:int=100,offset:int=0):
    return list_malicious_urls(verdict=verdict,severity=severity,search=search,
                               active_only=active_only,limit=limit,offset=offset)

@app.get("/db/malicious/stats")
def db_stats(): return get_malicious_stats()

@app.get("/db/malicious/export")
def db_export():
    return PlainTextResponse(content=export_malicious_csv(),media_type="text/csv",
        headers={"Content-Disposition":"attachment; filename=malicious_urls.csv"})

@app.post("/db/malicious/resolve")
def db_resolve(req:ResolveReq):
    mark_resolved(req.url_normalized)
    return {"status":"resolved"}

@app.delete("/db/malicious/{url_normalized:path}")
def db_delete(url_normalized:str):
    delete_malicious_url(url_normalized); return {"status":"deleted"}

@app.get("/db/stats")
def db_global(): return get_global_stats()

@app.get("/db/history")
def db_hist(limit:int=100,offset:int=0,verdict:Optional[str]=None):
    return get_scan_history(limit=limit,offset=offset,verdict=verdict)

@app.get("/ids/stats")
def ids_stats(): return {**ids_engine.get_stats(),**get_global_stats()}

@app.get("/ids/events")
def ids_events(limit:int=50,severity:Optional[str]=None):
    db=get_events(limit=limit,severity=severity)
    return db if db else ids_engine.get_recent_events(limit=limit,severity_filter=severity)

@app.get("/ids/blocked-ips")
def ids_blocked():
    db=get_blocked_ips_db()
    return db if db else ids_engine.get_blocked_ips()

@app.get("/ids/top-threats")
def ids_top(limit:int=10): return ids_engine.get_top_threat_ips(limit=limit)

@app.get("/ids/ip/{ip}")
def ids_ip(ip:str): return ids_engine.get_ip_reputation(ip)

@app.post("/ids/block-ip")
def ids_block(req:IPReq):
    r=ids_engine.block_ip(req.ip,req.reason); upsert_ip(ids_engine.get_ip_reputation(req.ip))
    return r

@app.post("/ids/unblock-ip")
def ids_unblock(req:IPReq):
    r=ids_engine.unblock_ip(req.ip); upsert_ip(ids_engine.get_ip_reputation(req.ip))
    return r

@app.get("/ids/rules")
def ids_rules(): return ids_engine.get_rules()

@app.get("/model/metrics")
def metrics():
    p=os.path.join(MDL,"metrics.json")
    return json.load(open(p)) if os.path.exists(p) else {"error":"Run model_trainer.py first"}

@app.get("/model/feature-importance")
def feat_imp(top:int=20):
    p=os.path.join(MDL,"feature_importance.csv")
    if not os.path.exists(p): return {"error":"No feature_importance.csv"}
    return pd.read_csv(p).head(top).to_dict(orient="records")

@app.post("/model/reload")
def reload():
    _load(); return {"status":"reloaded","loaded":_det is not None}

if __name__=="__main__":
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=False,log_level="info")