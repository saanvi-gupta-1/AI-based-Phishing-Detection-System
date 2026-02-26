"""
Cognitive Firewall v2 — SQLite Database Layer
Stores: malicious URLs, scan history, IP reputation, threat events.
Uses Python built-in sqlite3 — zero extra packages required.

Tables:
  malicious_urls   — every phishing/suspicious URL detected (deduplicated)
  scan_history     — every single scan ever run
  ip_reputation    — per-IP threat scores and block status
  threat_events    — IPS/IDS engine event log
"""

import sqlite3
import json
import time
import os
import threading
from typing import Optional, Dict, List
from contextlib import contextmanager

# DB lives in the data/ folder next to the backend
_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "firewall.db"
)
DB_PATH = os.environ.get("FIREWALL_DB", _DEFAULT_DB)

# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Deduplicated table of every malicious URL the system has seen
CREATE TABLE IF NOT EXISTS malicious_urls (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    url            TEXT    NOT NULL,
    url_normalized TEXT    NOT NULL UNIQUE,
    verdict        TEXT    NOT NULL CHECK(verdict IN ('phishing','suspicious')),
    confidence     REAL    NOT NULL,
    severity       TEXT    NOT NULL,
    action_taken   TEXT    NOT NULL,
    matched_rules  TEXT    DEFAULT '[]',   -- JSON array of rule IDs
    source_ip      TEXT    DEFAULT 'unknown',
    first_seen     REAL    NOT NULL,
    last_seen      REAL    NOT NULL,
    seen_count     INTEGER DEFAULT 1,
    features_json  TEXT    DEFAULT '{}',   -- snapshot of key ML features
    is_active      INTEGER DEFAULT 1,      -- 1=active threat, 0=resolved
    notes          TEXT    DEFAULT ''
);

-- Full scan log — every URL scanned including clean ones
CREATE TABLE IF NOT EXISTS scan_history (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    url              TEXT NOT NULL,
    url_normalized   TEXT NOT NULL,
    verdict          TEXT NOT NULL,
    confidence       REAL NOT NULL,
    action           TEXT NOT NULL,
    severity         TEXT NOT NULL,
    source_ip        TEXT DEFAULT 'unknown',
    scanned_at       REAL NOT NULL,
    scan_duration_ms INTEGER DEFAULT 0
);

-- Per-IP reputation tracking
CREATE TABLE IF NOT EXISTS ip_reputation (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ip              TEXT NOT NULL UNIQUE,
    threat_score    REAL    DEFAULT 0.0,
    request_count   INTEGER DEFAULT 0,
    phishing_hits   INTEGER DEFAULT 0,
    suspicious_hits INTEGER DEFAULT 0,
    blocked_count   INTEGER DEFAULT 0,
    is_blocked      INTEGER DEFAULT 0,
    block_reason    TEXT    DEFAULT '',
    first_seen      REAL NOT NULL,
    last_seen       REAL NOT NULL
);

-- IPS/IDS event log (persisted copy of the in-memory deque)
CREATE TABLE IF NOT EXISTS threat_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    url           TEXT NOT NULL,
    source_ip     TEXT DEFAULT 'unknown',
    threat_type   TEXT NOT NULL,
    severity      TEXT NOT NULL,
    confidence    REAL NOT NULL,
    action_taken  TEXT NOT NULL,
    matched_rules TEXT DEFAULT '[]',
    event_time    REAL NOT NULL
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_mal_url      ON malicious_urls(url_normalized);
CREATE INDEX IF NOT EXISTS idx_mal_verdict  ON malicious_urls(verdict);
CREATE INDEX IF NOT EXISTS idx_mal_active   ON malicious_urls(is_active);
CREATE INDEX IF NOT EXISTS idx_mal_seen     ON malicious_urls(last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_scan_time    ON scan_history(scanned_at DESC);
CREATE INDEX IF NOT EXISTS idx_scan_url     ON scan_history(url_normalized);
CREATE INDEX IF NOT EXISTS idx_ip           ON ip_reputation(ip);
CREATE INDEX IF NOT EXISTS idx_event_time   ON threat_events(event_time DESC);
"""

# ── Thread-local connections ───────────────────────────────────────────────────

_local = threading.local()


def _conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
        c = sqlite3.connect(DB_PATH, check_same_thread=False)
        c.row_factory = sqlite3.Row
        _local.conn = c
    return _local.conn


@contextmanager
def _tx():
    """Yield a connection, commit on success, rollback on error."""
    c = _conn()
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise


# ── Init ───────────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables. Safe to call multiple times."""
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.executescript(_SCHEMA)
    c.commit()
    c.close()
    print(f"[DB] Ready → {DB_PATH}")


# ── Malicious URL helpers ──────────────────────────────────────────────────────

def _top_features(features: dict) -> str:
    keys = [
        "heuristic_risk_score", "brand_abuse", "indian_bank_phishing",
        "is_suspicious_tld", "has_ip_in_url", "domain_entropy",
        "suspicious_keyword_count", "uses_shortener",
        "brand_similarity_score", "is_trusted_domain",
    ]
    return json.dumps({k: features.get(k, 0) for k in keys})


def upsert_malicious_url(
    url: str,
    url_normalized: str,
    verdict: str,
    confidence: float,
    severity: str,
    action_taken: str,
    matched_rules: list,
    source_ip: str,
    features: dict,
) -> dict:
    """
    Insert a new malicious URL or bump seen_count if it already exists.
    Returns the stored record as a dict.
    """
    now = time.time()
    with _tx() as c:
        existing = c.execute(
            "SELECT id FROM malicious_urls WHERE url_normalized = ?",
            (url_normalized,),
        ).fetchone()

        if existing:
            c.execute(
                """UPDATE malicious_urls SET
                       last_seen    = ?,
                       seen_count   = seen_count + 1,
                       confidence   = MAX(confidence, ?),
                       is_active    = 1
                   WHERE url_normalized = ?""",
                (now, confidence, url_normalized),
            )
        else:
            c.execute(
                """INSERT INTO malicious_urls
                   (url, url_normalized, verdict, confidence, severity,
                    action_taken, matched_rules, source_ip,
                    first_seen, last_seen, features_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    url, url_normalized, verdict, confidence, severity,
                    action_taken, json.dumps(matched_rules), source_ip,
                    now, now, _top_features(features),
                ),
            )
    return get_malicious_url_by_norm(url_normalized) or {}


def get_malicious_url_by_norm(url_normalized: str) -> Optional[dict]:
    with _tx() as c:
        row = c.execute(
            "SELECT * FROM malicious_urls WHERE url_normalized = ?",
            (url_normalized,),
        ).fetchone()
    return dict(row) if row else None


def list_malicious_urls(
    verdict: Optional[str] = None,
    severity: Optional[str] = None,
    search: Optional[str] = None,
    active_only: bool = True,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """
    Paginated list of malicious URLs.
    Returns {total, limit, offset, items}.
    """
    clauses, params = [], []
    if active_only:
        clauses.append("is_active = 1")
    if verdict:
        clauses.append("verdict = ?")
        params.append(verdict)
    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    if search:
        clauses.append("url LIKE ?")
        params.append(f"%{search}%")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    with _tx() as c:
        total = c.execute(
            f"SELECT COUNT(*) FROM malicious_urls {where}", params
        ).fetchone()[0]
        rows = c.execute(
            f"SELECT * FROM malicious_urls {where} "
            f"ORDER BY last_seen DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

    # Pretty-print timestamps
    items = []
    for r in rows:
        d = dict(r)
        d["first_seen_human"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d["first_seen"]))
        d["last_seen_human"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d["last_seen"]))
        try:
            d["matched_rules"] = json.loads(d.get("matched_rules", "[]"))
        except Exception:
            d["matched_rules"] = []
        try:
            d["features_json"] = json.loads(d.get("features_json", "{}"))
        except Exception:
            d["features_json"] = {}
        items.append(d)

    return {"total": total, "limit": limit, "offset": offset, "items": items}


def get_malicious_stats() -> dict:
    with _tx() as c:
        r = c.execute(
            """SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN verdict='phishing'   THEN 1 END) AS phishing,
                SUM(CASE WHEN verdict='suspicious' THEN 1 END) AS suspicious,
                SUM(CASE WHEN severity='critical'  THEN 1 END) AS critical_count,
                SUM(CASE WHEN severity='high'      THEN 1 END) AS high_count,
                SUM(CASE WHEN is_active=1          THEN 1 END) AS active,
                SUM(seen_count)                                 AS total_detections
            FROM malicious_urls"""
        ).fetchone()
    return dict(r) if r else {}


def mark_resolved(url_normalized: str):
    with _tx() as c:
        c.execute(
            "UPDATE malicious_urls SET is_active = 0 WHERE url_normalized = ?",
            (url_normalized,),
        )


def delete_malicious_url(url_normalized: str):
    with _tx() as c:
        c.execute(
            "DELETE FROM malicious_urls WHERE url_normalized = ?",
            (url_normalized,),
        )


def export_malicious_csv() -> str:
    """Return CSV string of all active malicious URLs."""
    result = list_malicious_urls(active_only=True, limit=50000)
    rows = result["items"]
    lines = ["url,verdict,confidence,severity,seen_count,first_seen,last_seen,source_ip"]
    for r in rows:
        lines.append(
            f'"{r["url"]}",{r["verdict"]},{r["confidence"]:.4f},'
            f'{r["severity"]},{r["seen_count"]},'
            f'"{r["first_seen_human"]}","{r["last_seen_human"]}",{r["source_ip"]}'
        )
    return "\n".join(lines)


# ── Scan history ───────────────────────────────────────────────────────────────

def log_scan(
    url: str,
    url_normalized: str,
    verdict: str,
    confidence: float,
    action: str,
    severity: str,
    source_ip: str,
    scan_duration_ms: int = 0,
):
    with _tx() as c:
        c.execute(
            """INSERT INTO scan_history
               (url, url_normalized, verdict, confidence, action,
                severity, source_ip, scanned_at, scan_duration_ms)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (url, url_normalized, verdict, confidence, action,
             severity, source_ip, time.time(), scan_duration_ms),
        )


def get_scan_history(
    limit: int = 100,
    offset: int = 0,
    verdict: Optional[str] = None,
) -> dict:
    where = f"WHERE verdict = '{verdict}'" if verdict else ""
    with _tx() as c:
        total = c.execute(f"SELECT COUNT(*) FROM scan_history {where}").fetchone()[0]
        rows = c.execute(
            f"SELECT * FROM scan_history {where} ORDER BY scanned_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
    items = []
    for r in rows:
        d = dict(r)
        d["scanned_at_human"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d["scanned_at"]))
        items.append(d)
    return {"total": total, "limit": limit, "offset": offset, "items": items}


# ── IP reputation ──────────────────────────────────────────────────────────────

def upsert_ip(rep: dict):
    now = time.time()
    ip = rep.get("ip", "unknown")
    with _tx() as c:
        exists = c.execute(
            "SELECT id FROM ip_reputation WHERE ip = ?", (ip,)
        ).fetchone()
        if exists:
            c.execute(
                """UPDATE ip_reputation SET
                       threat_score    = ?,
                       request_count   = ?,
                       phishing_hits   = ?,
                       suspicious_hits = ?,
                       blocked_count   = ?,
                       is_blocked      = ?,
                       block_reason    = ?,
                       last_seen       = ?
                   WHERE ip = ?""",
                (
                    rep.get("threat_score", 0),
                    rep.get("request_count", 0),
                    rep.get("phishing_hits", 0),
                    rep.get("suspicious_hits", 0),
                    rep.get("blocked_requests", 0),
                    1 if rep.get("is_blocked") else 0,
                    rep.get("block_reason", ""),
                    now, ip,
                ),
            )
        else:
            c.execute(
                """INSERT INTO ip_reputation
                   (ip, threat_score, request_count, phishing_hits,
                    suspicious_hits, blocked_count, is_blocked,
                    block_reason, first_seen, last_seen)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    ip,
                    rep.get("threat_score", 0),
                    rep.get("request_count", 0),
                    rep.get("phishing_hits", 0),
                    rep.get("suspicious_hits", 0),
                    rep.get("blocked_requests", 0),
                    1 if rep.get("is_blocked") else 0,
                    rep.get("block_reason", ""),
                    now, now,
                ),
            )


def get_blocked_ips_db() -> list:
    with _tx() as c:
        rows = c.execute(
            "SELECT * FROM ip_reputation WHERE is_blocked = 1 ORDER BY threat_score DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ── Threat events ──────────────────────────────────────────────────────────────

def log_event(event: dict):
    with _tx() as c:
        c.execute(
            """INSERT INTO threat_events
               (url, source_ip, threat_type, severity, confidence,
                action_taken, matched_rules, event_time)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                event.get("url", ""),
                event.get("ip", "unknown"),
                event.get("threat_type", "unknown"),
                event.get("severity", "low"),
                event.get("confidence", 0),
                event.get("action_taken", "allowed"),
                json.dumps(event.get("matched_rules", [])),
                event.get("timestamp", time.time()),
            ),
        )


def get_events(limit: int = 50, severity: Optional[str] = None) -> list:
    where = f"WHERE severity = '{severity}'" if severity else ""
    with _tx() as c:
        rows = c.execute(
            f"SELECT * FROM threat_events {where} ORDER BY event_time DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["timestamp_human"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(d["event_time"])
        )
        try:
            d["matched_rules"] = json.loads(d["matched_rules"])
        except Exception:
            d["matched_rules"] = []
        out.append(d)
    return out


# ── Global stats (from DB — survives restarts) ─────────────────────────────────

def get_global_stats() -> dict:
    with _tx() as c:
        sh = c.execute(
            """SELECT
                COUNT(*)                                       AS total_scanned,
                SUM(CASE WHEN action='blocked'    THEN 1 END) AS total_blocked,
                SUM(CASE WHEN verdict='phishing'  THEN 1 END) AS total_phishing,
                SUM(CASE WHEN verdict='suspicious'THEN 1 END) AS total_suspicious,
                SUM(CASE WHEN verdict='clean'     THEN 1 END) AS total_clean
            FROM scan_history"""
        ).fetchone()
        ips = c.execute(
            "SELECT COUNT(*) FROM ip_reputation WHERE is_blocked = 1"
        ).fetchone()[0]
        unique_mal = c.execute(
            "SELECT COUNT(*) FROM malicious_urls WHERE is_active = 1"
        ).fetchone()[0]
    d = dict(sh) if sh else {}
    d["ips_blocked"] = ips
    d["unique_malicious_urls"] = unique_mal
    # Replace None with 0
    return {k: (v or 0) for k, v in d.items()}