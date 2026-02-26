"""
Cognitive Firewall v2 — IPS/IDS Engine
Intrusion Prevention & Detection System.
Handles: rate limiting, IP reputation, rule matching,
automatic blocking, event log, and alert callbacks.
Thread-safe. No external dependencies.
"""

import time
import threading
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class ThreatEvent:
    timestamp: float
    url: str
    ip: str
    threat_type: str      # "phishing" | "suspicious" | "scan_abuse" | "clean"
    severity: str         # "critical" | "high" | "medium" | "low" | "info"
    confidence: float
    action_taken: str     # "blocked" | "alerted" | "allowed"
    matched_rules: list
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp_human"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)
        )
        return d


@dataclass
class IPRecord:
    ip: str
    threat_score: float = 0.0
    request_count: int = 0
    phishing_hits: int = 0
    suspicious_hits: int = 0
    blocked_requests: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    is_blocked: bool = False
    block_reason: str = ""
    block_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ip": self.ip,
            "threat_score": round(self.threat_score, 1),
            "request_count": self.request_count,
            "phishing_hits": self.phishing_hits,
            "suspicious_hits": self.suspicious_hits,
            "blocked_requests": self.blocked_requests,
            "is_blocked": self.is_blocked,
            "block_reason": self.block_reason,
            "first_seen": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.first_seen)),
            "last_seen": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_seen)),
            "block_time": (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.block_time))
                if self.block_time else None
            ),
        }


# ── IDS Rules ────────────────────────────────────────────────────────────────

IDS_RULES = [
    {
        "id": "R001",
        "name": "Confirmed Phishing URL",
        "type": "ml_confidence",
        "threshold": 0.85,
        "label": 1,
        "severity": "critical",
        "action": "blocked",
    },
    {
        "id": "R002",
        "name": "Suspected Phishing URL",
        "type": "ml_confidence",
        "threshold": 0.60,
        "label": 1,
        "severity": "high",
        "action": "alerted",
    },
    {
        "id": "R003",
        "name": "Low-Confidence Suspicious URL",
        "type": "ml_confidence",
        "threshold": 0.40,
        "label": 1,
        "severity": "medium",
        "action": "alerted",
    },
    {
        "id": "R004",
        "name": "IP Address Used as URL",
        "type": "feature_flag",
        "feature": "has_ip_in_url",
        "value": 1,
        "severity": "high",
        "action": "alerted",
    },
    {
        "id": "R005",
        "name": "Indian Bank Brand Abuse",
        "type": "feature_flag",
        "feature": "indian_bank_phishing",
        "value": 1,
        "severity": "critical",
        "action": "blocked",
    },
    {
        "id": "R006",
        "name": "URL Shortener Detected",
        "type": "feature_flag",
        "feature": "uses_shortener",
        "value": 1,
        "severity": "medium",
        "action": "alerted",
    },
    {
        "id": "R007",
        "name": "Suspicious TLD",
        "type": "feature_flag",
        "feature": "is_suspicious_tld",
        "value": 1,
        "severity": "low",
        "action": "alerted",
    },
    {
        "id": "R008",
        "name": "High Domain Entropy",
        "type": "feature_threshold",
        "feature": "domain_entropy",
        "threshold": 3.8,
        "severity": "medium",
        "action": "alerted",
    },
    {
        "id": "R009",
        "name": "Brand Abuse Detected",
        "type": "feature_flag",
        "feature": "brand_abuse",
        "value": 1,
        "severity": "high",
        "action": "alerted",
    },
    {
        "id": "R010",
        "name": "Repeat Offender IP",
        "type": "ip_reputation",
        "threshold": 75.0,
        "severity": "critical",
        "action": "blocked",
    },
    {
        "id": "R011",
        "name": "High Heuristic Risk Score",
        "type": "feature_threshold",
        "feature": "heuristic_risk_score",
        "threshold": 0.7,
        "severity": "high",
        "action": "alerted",
    },
    {
        "id": "R012",
        "name": "Suspicious Path (Login/Redirect)",
        "type": "feature_flag",
        "feature": "path_has_login",
        "value": 1,
        "severity": "medium",
        "action": "alerted",
    },
]

SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


# ── Rate Limiter ──────────────────────────────────────────────────────────────

class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._windows: dict = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple:
        """Returns (is_allowed: bool, remaining: int, reset_in: float)."""
        now = time.time()
        cutoff = now - self.window
        with self._lock:
            q = self._windows[key]
            while q and q[0] < cutoff:
                q.popleft()
            count = len(q)
            if count >= self.max_requests:
                reset_in = q[0] + self.window - now if q else 0
                return False, 0, round(reset_in, 1)
            q.append(now)
            return True, self.max_requests - count - 1, 0.0


# ── Core Engine ───────────────────────────────────────────────────────────────

class IPSIDSEngine:
    """
    Production IPS/IDS Engine for Cognitive Firewall.

    Usage:
        engine = IPSIDSEngine()
        result = engine.analyze(url, ml_confidence, ml_label, features, source_ip)
        print(result["action"])  # "blocked" | "alerted" | "allowed"
    """

    def __init__(
        self,
        block_threshold_score: float = 75.0,
        rate_limit_rpm: int = 60,
        max_events: int = 10000,
        log_path: Optional[str] = None,
        auto_expire_blocks_hours: float = 0,  # 0 = permanent
    ):
        self.block_threshold_score = block_threshold_score
        self.auto_expire_hours = auto_expire_blocks_hours
        self.log_path = log_path

        self._ip_table: dict[str, IPRecord] = {}
        self._event_log: deque = deque(maxlen=max_events)
        self._rate_limiter = SlidingWindowRateLimiter(rate_limit_rpm, 60)
        self._alert_callbacks: list[Callable] = []
        self._lock = threading.RLock()

        self._stats = {
            "total_analyzed": 0,
            "total_blocked": 0,
            "total_alerted": 0,
            "total_allowed": 0,
            "phishing_detected": 0,
            "suspicious_detected": 0,
            "ips_blocked": 0,
        }

        if log_path:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        url: str,
        ml_confidence: float,
        ml_label: int,
        features: dict,
        source_ip: Optional[str] = None,
    ) -> dict:
        """
        Run URL through all IDS rules.
        Returns decision dict with: action, severity, threat_type,
        matched_rules, ip_reputation, and event.
        """
        with self._lock:
            ip = source_ip or "unknown"
            ip_rec = self._get_ip(ip)

            # Update IP record
            ip_rec.request_count += 1
            ip_rec.last_seen = time.time()

            self._stats["total_analyzed"] += 1

            # ── Pre-check: is this IP already hard-blocked? ────────────────
            if ip_rec.is_blocked:
                if self.auto_expire_hours > 0:
                    age_h = (time.time() - ip_rec.block_time) / 3600
                    if age_h > self.auto_expire_hours:
                        ip_rec.is_blocked = False
                        ip_rec.block_reason = ""
                    else:
                        return self._finalize(
                            url, ip, ml_confidence, features, ip_rec,
                            "blocked", "critical", "repeat_offender", ["R010"]
                        )
                else:
                    return self._finalize(
                        url, ip, ml_confidence, features, ip_rec,
                        "blocked", "critical", "repeat_offender", ["R010"]
                    )

            # ── Rate limiting ─────────────────────────────────────────────
            allowed, remaining, reset_in = self._rate_limiter.check(ip)
            if not allowed:
                ip_rec.threat_score = min(100, ip_rec.threat_score + 5)
                return self._finalize(
                    url, ip, ml_confidence, features, ip_rec,
                    "alerted", "medium", "rate_abuse", ["R_RATE"]
                )

            # ── Apply IDS rules ───────────────────────────────────────────
            matched = []
            action = "allowed"
            severity = "info"
            threat_type = "clean"

            for rule in IDS_RULES:
                hit = False
                rtype = rule["type"]

                if rtype == "ml_confidence":
                    if ml_label == rule.get("label", 1) and ml_confidence >= rule["threshold"]:
                        hit = True

                elif rtype == "feature_flag":
                    if features.get(rule["feature"], 0) == rule.get("value", 1):
                        hit = True

                elif rtype == "feature_threshold":
                    if features.get(rule["feature"], 0) >= rule["threshold"]:
                        hit = True

                elif rtype == "ip_reputation":
                    if ip_rec.threat_score >= rule["threshold"]:
                        hit = True

                if hit:
                    matched.append(rule["id"])
                    # Escalate severity and action
                    if SEVERITY_RANK.get(rule["severity"], 0) > SEVERITY_RANK.get(severity, 0):
                        severity = rule["severity"]
                    if rule["action"] == "blocked":
                        action = "blocked"
                    elif rule["action"] == "alerted" and action == "allowed":
                        action = "alerted"

            # ── Determine threat type ─────────────────────────────────────
            if ml_label == 1 and ml_confidence >= 0.85:
                threat_type = "phishing"
            elif ml_label == 1 and ml_confidence >= 0.40:
                threat_type = "suspicious"
            elif features.get("indian_bank_phishing", 0):
                threat_type = "phishing"

            # ── Update IP reputation ──────────────────────────────────────
            if threat_type == "phishing":
                ip_rec.phishing_hits += 1
                ip_rec.threat_score = min(100, ip_rec.threat_score + 25)
                self._stats["phishing_detected"] += 1
            elif threat_type == "suspicious":
                ip_rec.suspicious_hits += 1
                ip_rec.threat_score = min(100, ip_rec.threat_score + 10)
                self._stats["suspicious_detected"] += 1

            if action == "blocked":
                ip_rec.blocked_requests += 1

            # Auto-block if threshold reached
            if ip_rec.threat_score >= self.block_threshold_score and not ip_rec.is_blocked:
                ip_rec.is_blocked = True
                ip_rec.block_reason = (
                    f"Auto-blocked: threat score {ip_rec.threat_score:.0f}/100"
                )
                ip_rec.block_time = time.time()
                self._stats["ips_blocked"] += 1
                if "R010" not in matched:
                    matched.append("R010")
                action = "blocked"
                severity = "critical"

            return self._finalize(
                url, ip, ml_confidence, features, ip_rec,
                action, severity, threat_type, matched
            )

    def block_ip(self, ip: str, reason: str = "Manual block") -> dict:
        with self._lock:
            rec = self._get_ip(ip)
            rec.is_blocked = True
            rec.block_reason = reason
            rec.block_time = time.time()
            self._stats["ips_blocked"] += 1
            return {"status": "blocked", "ip": ip, "reason": reason}

    def unblock_ip(self, ip: str) -> dict:
        with self._lock:
            rec = self._get_ip(ip)
            rec.is_blocked = False
            rec.block_reason = ""
            rec.threat_score = max(0, rec.threat_score - 30)
            self._stats["ips_blocked"] = max(0, self._stats["ips_blocked"] - 1)
            return {"status": "unblocked", "ip": ip}

    def get_ip_reputation(self, ip: str) -> dict:
        with self._lock:
            return self._get_ip(ip).to_dict()

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def get_recent_events(
        self, limit: int = 50, severity_filter: Optional[str] = None
    ) -> list:
        with self._lock:
            events = list(self._event_log)
            if severity_filter:
                events = [e for e in events if e.get("severity") == severity_filter]
            return list(reversed(events))[:limit]

    def get_blocked_ips(self) -> list:
        with self._lock:
            return [
                rec.to_dict()
                for rec in self._ip_table.values()
                if rec.is_blocked
            ]

    def get_top_threat_ips(self, limit: int = 10) -> list:
        with self._lock:
            sorted_recs = sorted(
                self._ip_table.values(),
                key=lambda r: r.threat_score,
                reverse=True,
            )[:limit]
            return [r.to_dict() for r in sorted_recs]

    def get_rules(self) -> list:
        return IDS_RULES

    def register_alert_callback(self, fn: Callable):
        """Register a function called for critical/high severity events."""
        self._alert_callbacks.append(fn)

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_ip(self, ip: str) -> IPRecord:
        if ip not in self._ip_table:
            self._ip_table[ip] = IPRecord(ip=ip)
        return self._ip_table[ip]

    def _finalize(
        self, url, ip, confidence, features, ip_rec,
        action, severity, threat_type, matched_rules
    ) -> dict:
        event = ThreatEvent(
            timestamp=time.time(),
            url=url,
            ip=ip,
            threat_type=threat_type,
            severity=severity,
            confidence=round(confidence, 4),
            action_taken=action,
            matched_rules=matched_rules,
            details={"feature_highlights": {
                k: features.get(k)
                for k in [
                    "heuristic_risk_score", "brand_abuse", "indian_bank_phishing",
                    "is_trusted_domain", "is_suspicious_tld", "has_ip_in_url"
                ]
            }},
        ).to_dict()

        self._event_log.append(event)

        if action == "blocked":
            self._stats["total_blocked"] += 1
        elif action == "alerted":
            self._stats["total_alerted"] += 1
        else:
            self._stats["total_allowed"] += 1

        # Fire callbacks
        if severity in ("critical", "high"):
            for cb in self._alert_callbacks:
                try:
                    cb(event)
                except Exception:
                    pass

        # Persist to log file
        if self.log_path:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception:
                pass

        return {
            "action": action,
            "severity": severity,
            "threat_type": threat_type,
            "matched_rules": matched_rules,
            "ip_reputation": ip_rec.to_dict(),
            "event": event,
        }