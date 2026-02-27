import { useState, useEffect, useCallback, useRef } from "react";
import {
  AlertTriangle, Shield, ShieldOff, Activity, Search, Upload,
  Database, Terminal, XCircle, CheckCircle, Clock, RefreshCw,
  Download, Trash2, CheckSquare, Filter, ChevronDown, ChevronUp,
  Zap, Globe, Eye, BarChart2, Lock, Unlock, Info, Copy, ExternalLink,
  AlertOctagon, TrendingUp, Wifi, Server, FileText, Settings
} from "lucide-react";

const API = "http://localhost:8000";

// ── Demo / Fallback Data ──────────────────────────────────────────────────────
const DEMO_STATS = {
  total_scanned: 14832, total_blocked: 2341, total_phishing: 1890,
  total_suspicious: 1203, total_clean: 11739, ips_blocked: 47,
  unique_malicious_urls: 312,
};
const DEMO_EVENTS = [
  { timestamp_human:"2025-08-01 14:32:11", url:"sbi-netbanking-verify.xyz/login",   severity:"critical", action_taken:"blocked",  threat_type:"phishing",   confidence:0.97, source_ip:"103.21.244.82" },
  { timestamp_human:"2025-08-01 14:31:58", url:"amazon-secure-alert.top/account",   severity:"high",     action_taken:"alerted",  threat_type:"suspicious", confidence:0.73, source_ip:"45.77.33.91" },
  { timestamp_human:"2025-08-01 14:31:44", url:"hdfc-update-kyc.shop/verify",       severity:"critical", action_taken:"blocked",  threat_type:"phishing",   confidence:0.94, source_ip:"185.220.101.12" },
  { timestamp_human:"2025-08-01 14:31:29", url:"paypa1-secure-login.ru/auth",       severity:"critical", action_taken:"blocked",  threat_type:"phishing",   confidence:0.98, source_ip:"91.108.4.22" },
  { timestamp_human:"2025-08-01 14:30:50", url:"icicibank-secure.net/login",        severity:"high",     action_taken:"blocked",  threat_type:"phishing",   confidence:0.88, source_ip:"45.142.212.100" },
];
const DEMO_MALICIOUS = [
  { id:1, url:"hdfc-update-kyc.shop/verify/account",  verdict:"phishing",   confidence:0.97, severity:"critical", seen_count:14, first_seen_human:"2025-07-28 09:11", last_seen_human:"2025-08-01 14:31", source_ip:"185.220.101.12", is_active:1 },
  { id:2, url:"sbi-netbanking-verify.xyz/login",       verdict:"phishing",   confidence:0.95, severity:"critical", seen_count:9,  first_seen_human:"2025-07-30 11:43", last_seen_human:"2025-08-01 14:32", source_ip:"103.21.244.82",  is_active:1 },
  { id:3, url:"icicibank-support.net/reset",           verdict:"phishing",   confidence:0.88, severity:"high",     seen_count:4,  first_seen_human:"2025-07-29 16:05", last_seen_human:"2025-08-01 12:10", source_ip:"45.142.212.100", is_active:1 },
  { id:4, url:"amazon-secure-alert.top/account",       verdict:"suspicious", confidence:0.73, severity:"medium",   seen_count:3,  first_seen_human:"2025-08-01 14:31", last_seen_human:"2025-08-01 14:31", source_ip:"45.77.33.91",    is_active:1 },
];

// ── Palettes ──────────────────────────────────────────────────────────────────
const C = {
  bg:       "#080b14",
  panel:    "#0d1220",
  panelAlt: "#111827",
  border:   "rgba(255,255,255,0.07)",
  border2:  "rgba(255,255,255,0.12)",
  accent:   "#6366f1",
  accent2:  "#818cf8",
  text:     "#e2e8f0",
  muted:    "#64748b",
  muted2:   "#94a3b8",
  red:      "#ef4444",
  orange:   "#f97316",
  yellow:   "#eab308",
  green:    "#22c55e",
  blue:     "#3b82f6",
  purple:   "#a855f7",
};

const SEV = {
  critical: { color: C.red,    bg: "rgba(239,68,68,0.1)",    border: "rgba(239,68,68,0.25)",  label: "CRITICAL" },
  high:     { color: C.orange, bg: "rgba(249,115,22,0.1)",   border: "rgba(249,115,22,0.25)", label: "HIGH"     },
  medium:   { color: C.yellow, bg: "rgba(234,179,8,0.1)",    border: "rgba(234,179,8,0.25)",  label: "MEDIUM"   },
  low:      { color: C.green,  bg: "rgba(34,197,94,0.1)",    border: "rgba(34,197,94,0.25)",  label: "LOW"      },
  info:     { color: C.blue,   bg: "rgba(59,130,246,0.1)",   border: "rgba(59,130,246,0.25)", label: "INFO"     },
};
const VERDICT_C = {
  phishing:   { color: C.red,    bg: "rgba(239,68,68,0.12)",   label: "PHISHING",   icon: "🎣" },
  suspicious: { color: C.orange, bg: "rgba(249,115,22,0.12)",  label: "SUSPICIOUS", icon: "⚠️" },
  clean:      { color: C.green,  bg: "rgba(34,197,94,0.12)",   label: "CLEAN",      icon: "✅" },
};

// ── Shared style atoms ────────────────────────────────────────────────────────
const card = (extra = {}) => ({
  background: C.panel,
  border: `1px solid ${C.border}`,
  borderRadius: 12,
  padding: "1.25rem 1.5rem",
  ...extra,
});
const input = {
  background: "rgba(0,0,0,0.35)",
  border: `1px solid ${C.border2}`,
  borderRadius: 8,
  padding: "0.65rem 1rem",
  color: C.text,
  fontSize: 14,
  fontFamily: "'Fira Code', 'JetBrains Mono', monospace",
  outline: "none",
  width: "100%",
  boxSizing: "border-box",
};
const btn = (bg = C.accent, small = false) => ({
  background: `linear-gradient(135deg, ${bg}, ${bg}cc)`,
  border: "none",
  borderRadius: 8,
  padding: small ? "0.4rem 0.9rem" : "0.65rem 1.4rem",
  color: "#fff",
  fontSize: small ? 12 : 13,
  fontWeight: 700,
  cursor: "pointer",
  letterSpacing: "0.04em",
  whiteSpace: "nowrap",
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  transition: "opacity 0.15s",
});
const btnGhost = (col = C.accent, small = false) => ({
  background: `${col}18`,
  border: `1px solid ${col}40`,
  borderRadius: 8,
  padding: small ? "0.35rem 0.75rem" : "0.55rem 1.1rem",
  color: col,
  fontSize: small ? 11 : 12,
  fontWeight: 600,
  cursor: "pointer",
  whiteSpace: "nowrap",
  display: "inline-flex",
  alignItems: "center",
  gap: 5,
  transition: "background 0.15s",
});

// ── Fetch helper ──────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const r = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}

// ── Micro-components ──────────────────────────────────────────────────────────
function Badge({ text, color, bg }) {
  return (
    <span style={{
      background: bg || `${color}18`,
      color,
      border: `1px solid ${color}40`,
      borderRadius: 5,
      padding: "2px 8px",
      fontSize: 10,
      fontWeight: 700,
      letterSpacing: "0.08em",
    }}>{text}</span>
  );
}

function VerdictBadge({ verdict }) {
  const c = VERDICT_C[verdict] || VERDICT_C.clean;
  return <Badge text={c.label} color={c.color} bg={c.bg} />;
}

function SeverityBadge({ severity }) {
  const c = SEV[severity] || SEV.low;
  return <Badge text={c.label} color={c.color} bg={c.bg} />;
}

function ConfidenceBar({ val, width = "100%" }) {
  const pct = Math.round((val || 0) * 100);
  const col = pct >= 80 ? C.red : pct >= 55 ? C.orange : C.green;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, width }}>
      <div style={{ flex: 1, height: 4, background: "rgba(255,255,255,0.08)", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: col, borderRadius: 4, transition: "width 0.5s ease" }} />
      </div>
      <span style={{ fontSize: 11, color: col, fontFamily: "monospace", minWidth: 36 }}>{pct}%</span>
    </div>
  );
}

function StatCard({ label, value, color, icon, sub }) {
  return (
    <div style={{ ...card(), display: "flex", flexDirection: "column", gap: 8, minWidth: 0 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <span style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.1em", lineHeight: 1.3 }}>{label}</span>
        <span style={{ color, opacity: 0.7, flexShrink: 0 }}>{icon}</span>
      </div>
      <span style={{ fontSize: 28, fontWeight: 800, color: color || C.text, fontFamily: "monospace", lineHeight: 1 }}>
        {typeof value === "number" ? value.toLocaleString() : (value ?? "—")}
      </span>
      {sub && <span style={{ fontSize: 11, color: C.muted }}>{sub}</span>}
    </div>
  );
}

function Spinner() {
  const [angle, setAngle] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setAngle(a => (a + 12) % 360), 40);
    return () => clearInterval(t);
  }, []);
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" style={{ transform: `rotate(${angle}deg)` }}>
      <circle cx="9" cy="9" r="7" fill="none" stroke={C.muted} strokeWidth="2" strokeDasharray="30 14" />
    </svg>
  );
}

function Dot({ color, pulse }) {
  return (
    <span style={{
      display: "inline-block", width: 8, height: 8, borderRadius: "50%",
      background: color, flexShrink: 0,
      boxShadow: pulse ? `0 0 8px ${color}` : "none",
    }} />
  );
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

// ═══════════════════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════════════════
function TabDashboard({ stats, events, loading, onRefresh }) {
  const s = stats || DEMO_STATS;
  const evts = events?.length ? events : DEMO_EVENTS;
  const isDemo = !stats;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {isDemo && (
        <div style={{ background: "rgba(99,102,241,0.08)", border: `1px solid ${C.accent}30`, borderRadius: 10, padding: "0.75rem 1.1rem", fontSize: 12, color: C.accent2, display: "flex", gap: 8, alignItems: "center" }}>
          <Info size={14} /> Backend offline — showing demo data. Start FastAPI server on port 8000.
        </div>
      )}

      {/* Stats Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(190px, 1fr))", gap: 12 }}>
        <StatCard label="Total Scanned"    value={s.total_scanned}         color={C.accent2}  icon={<Activity size={16}/>} />
        <StatCard label="Phishing Caught"  value={s.total_phishing}        color={C.red}      icon={<AlertOctagon size={16}/>} />
        <StatCard label="Suspicious"       value={s.total_suspicious}      color={C.orange}   icon={<AlertTriangle size={16}/>} />
        <StatCard label="Clean"            value={s.total_clean}           color={C.green}    icon={<CheckCircle size={16}/>} />
        <StatCard label="Blocked Requests" value={s.total_blocked}         color="#ec4899"    icon={<ShieldOff size={16}/>} />
        <StatCard label="IPs Blocked"      value={s.ips_blocked}           color={C.purple}   icon={<Lock size={16}/>} />
        <StatCard label="Unique Malicious" value={s.unique_malicious_urls} color={C.yellow}   icon={<Database size={16}/>} />
      </div>

      {/* Detection Rate */}
      {s.total_scanned > 0 && (
        <div style={{ ...card(), display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.08em" }}>Detection Breakdown</div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              {[
                { label: "Phishing Rate", val: s.total_phishing / s.total_scanned, color: C.red },
                { label: "Suspicious Rate", val: s.total_suspicious / s.total_scanned, color: C.orange },
                { label: "Clean Rate", val: s.total_clean / s.total_scanned, color: C.green },
              ].map(({ label, val, color }) => (
                <div key={label} style={{ minWidth: 160 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 12, color: C.muted2 }}>{label}</span>
                    <span style={{ fontSize: 12, color, fontFamily: "monospace" }}>{(val * 100).toFixed(1)}%</span>
                  </div>
                  <div style={{ height: 5, background: "rgba(255,255,255,0.06)", borderRadius: 4 }}>
                    <div style={{ width: `${Math.min(val * 100, 100)}%`, height: "100%", background: color, borderRadius: 4 }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div style={{ marginLeft: "auto" }}>
            <button style={btnGhost(C.accent)} onClick={onRefresh}>
              <RefreshCw size={13} /> Refresh
            </button>
          </div>
        </div>
      )}

      {/* Live Threat Feed */}
      <div style={card()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Activity size={14} color={C.accent} />
            <span style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 600 }}>Live Threat Feed</span>
            <Dot color={C.green} pulse />
          </div>
          {loading && <Spinner />}
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
          {evts.slice(0, 10).map((e, i) => {
            const sc = SEV[e.severity] || SEV.low;
            const tt = e.threat_type || "clean";
            return (
              <div key={i} style={{
                display: "grid",
                gridTemplateColumns: "8px 1fr auto auto auto",
                gap: 12,
                alignItems: "center",
                background: sc.bg,
                border: `1px solid ${sc.border}`,
                borderRadius: 9,
                padding: "0.6rem 1rem",
              }}>
                <Dot color={sc.color} pulse={sc.color === C.red} />
                <span style={{ fontFamily: "monospace", fontSize: 12, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {e.url}
                </span>
                <VerdictBadge verdict={tt} />
                <SeverityBadge severity={e.severity} />
                <span style={{ fontSize: 11, color: C.muted, whiteSpace: "nowrap" }}>{e.timestamp_human}</span>
              </div>
            );
          })}
          {evts.length === 0 && (
            <div style={{ textAlign: "center", padding: "2rem", color: C.muted, fontSize: 13 }}>
              No threat events yet. Scan some URLs to get started.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// URL SCANNER
// ═══════════════════════════════════════════════════════════
function TabScanner() {
  const [url, setUrl] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const [showFeatures, setShowFeatures] = useState(false);

  const scan = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await api("/scan", {
        method: "POST",
        body: JSON.stringify({ url: url.trim() }),
      });
      setResult(r);
      setHistory(h => [r, ...h].slice(0, 20));
    } catch (e) {
      // Heuristic fallback when backend is offline
      const isLikelyPhishing = /\.xyz|\.top|\.shop|login.*verify|verify.*login|bank.*secure|secure.*bank/i.test(url);
      const fakeResult = {
        url: url.trim(),
        verdict: isLikelyPhishing ? "suspicious" : "clean",
        confidence: isLikelyPhishing ? 0.72 : 0.05,
        action: isLikelyPhishing ? "alerted" : "allowed",
        severity: isLikelyPhishing ? "medium" : "low",
        prediction_reason: "local_heuristic",
        scan_time: new Date().toLocaleString(),
        _offline: true,
      };
      setResult(fakeResult);
      setError("Backend offline — local heuristic used. Start FastAPI for full ML analysis.");
    }
    setLoading(false);
  };

  const vc = result ? (VERDICT_C[result.verdict] || VERDICT_C.clean) : null;
  const sc = result ? (SEV[result.severity] || SEV.low) : null;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: 20, alignItems: "start" }}>
      {/* Main scanner */}
      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {/* Input */}
        <div style={card()}>
          <div style={{ fontSize: 13, color: C.muted2, marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
            <Search size={14} color={C.accent} /> URL Scanner — Enter any URL to analyze
          </div>
          <div style={{ display: "flex", gap: 10 }}>
            <input
              style={{ ...input, flex: 1 }}
              value={url}
              onChange={e => setUrl(e.target.value)}
              onKeyDown={e => e.key === "Enter" && scan()}
              placeholder="https://example.com/login or paste a suspicious link..."
            />
            <button style={btn(loading ? C.muted : C.accent)} onClick={scan} disabled={loading}>
              {loading ? <Spinner /> : <Search size={14} />}
              {loading ? "Scanning..." : "Scan URL"}
            </button>
          </div>
          <div style={{ marginTop: 10, display: "flex", gap: 8, flexWrap: "wrap" }}>
            {["https://claude.ai", "http://sbi-verify.xyz/login", "https://hdfcbank.com", "http://paypa1-secure.ru/auth"].map(u => (
              <button key={u} style={btnGhost(C.muted2, true)} onClick={() => setUrl(u)}>
                {u.length > 30 ? u.slice(0, 30) + "…" : u}
              </button>
            ))}
          </div>
        </div>

        {/* Result */}
        {result && (
          <div style={{ ...card(), borderColor: vc ? `${vc.color}40` : C.border }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
                  <span style={{ fontSize: 22 }}>{vc?.icon || "🔍"}</span>
                  <span style={{ fontSize: 24, fontWeight: 800, color: vc?.color || C.text }}>
                    {result.verdict?.toUpperCase() || "UNKNOWN"}
                  </span>
                  {result._offline && <Badge text="OFFLINE MODE" color={C.yellow} />}
                </div>
                <div style={{ fontFamily: "monospace", fontSize: 12, color: C.muted2, wordBreak: "break-all" }}>
                  {result.url}
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
                <SeverityBadge severity={result.severity} />
                {result.action && <Badge text={result.action?.toUpperCase()} color={result.action === "blocked" ? C.red : result.action === "alerted" ? C.orange : C.green} />}
              </div>
            </div>

            {/* Confidence */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 12, color: C.muted }}>Confidence Score</span>
                <span style={{ fontSize: 12, color: vc?.color, fontFamily: "monospace" }}>{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <ConfidenceBar val={result.confidence} />
            </div>

            {/* Key facts */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 16 }}>
              {[
                { label: "Threat Type", val: result.threat_type || result.verdict },
                { label: "Scan Method", val: result.prediction_reason || "ml_model" },
                { label: "Scan Time", val: result.scan_duration_ms ? `${result.scan_duration_ms}ms` : "—" },
              ].map(({ label, val }) => (
                <div key={label} style={{ background: "rgba(0,0,0,0.25)", borderRadius: 7, padding: "0.6rem 0.9rem" }}>
                  <div style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>{label}</div>
                  <div style={{ fontSize: 13, color: C.text, fontFamily: "monospace" }}>{val}</div>
                </div>
              ))}
            </div>

            {/* Matched rules */}
            {result.matched_rules?.length > 0 && (
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, color: C.muted, marginBottom: 6 }}>TRIGGERED RULES</div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {result.matched_rules.map(r => <Badge key={r} text={r} color={C.orange} />)}
                </div>
              </div>
            )}

            {/* Features toggle */}
            {result.features && (
              <div>
                <button style={btnGhost(C.muted2, true)} onClick={() => setShowFeatures(v => !v)}>
                  {showFeatures ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                  {showFeatures ? "Hide" : "Show"} Feature Details ({Object.keys(result.features).length} features)
                </button>
                {showFeatures && (
                  <div style={{ marginTop: 10, display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 6 }}>
                    {Object.entries(result.features)
                      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                      .slice(0, 20)
                      .map(([k, v]) => (
                        <div key={k} style={{ background: "rgba(0,0,0,0.2)", borderRadius: 6, padding: "0.4rem 0.7rem", display: "flex", justifyContent: "space-between" }}>
                          <span style={{ fontSize: 11, color: C.muted2 }}>{k}</span>
                          <span style={{ fontSize: 11, color: v > 0 ? C.text : C.muted, fontFamily: "monospace" }}>
                            {typeof v === "number" ? v.toFixed(3) : String(v)}
                          </span>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {error && (
          <div style={{ background: "rgba(234,179,8,0.08)", border: `1px solid ${C.yellow}40`, borderRadius: 9, padding: "0.75rem 1rem", fontSize: 12, color: C.yellow, display: "flex", gap: 8 }}>
            <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 1 }} /> {error}
          </div>
        )}
      </div>

      {/* Scan History sidebar */}
      <div style={card()}>
        <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
          <Clock size={13} /> Recent Scans
        </div>
        {history.length === 0 ? (
          <div style={{ color: C.muted, fontSize: 12, textAlign: "center", padding: "2rem 0" }}>
            No scans yet this session
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
            {history.map((h, i) => {
              const hc = VERDICT_C[h.verdict] || VERDICT_C.clean;
              return (
                <div key={i} style={{ background: "rgba(0,0,0,0.25)", borderRadius: 8, padding: "0.6rem 0.8rem", cursor: "pointer", border: `1px solid ${C.border}` }}
                     onClick={() => { setUrl(h.url); setResult(h); }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <VerdictBadge verdict={h.verdict} />
                    <span style={{ fontSize: 11, color: hc.color, fontFamily: "monospace" }}>
                      {(h.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div style={{ fontFamily: "monospace", fontSize: 11, color: C.muted2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {h.url}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// BULK SCAN
// ═══════════════════════════════════════════════════════════
function TabBulk() {
  const [text, setText] = useState("");
  const [results, setResults] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState("all");
  const [progress, setProgress] = useState(0);

  const runBulk = async () => {
    const urls = text.split("\n").map(u => u.trim()).filter(Boolean);
    if (!urls.length) return;
    setLoading(true);
    setResults(null);
    setSummary(null);
    setProgress(0);

    // Try API first; fall back to local heuristic
    try {
      setProgress(30);
      const r = await api("/scan/bulk", {
        method: "POST",
        body: JSON.stringify({ urls }),
      });
      setProgress(100);
      setResults(r.results);
      setSummary(r.summary);
    } catch {
      // Local heuristic batch
      const res = urls.map((u, idx) => {
        setProgress(Math.round((idx / urls.length) * 100));
        const isPhish = /\.xyz|\.top|\.shop|login.*verify|bank.*kyc|paypa1|\.ru\/.*login/i.test(u);
        const isSusp = /login|verify|secure|alert|update/i.test(u) && !isPhish;
        return {
          url: u,
          verdict: isPhish ? "phishing" : isSusp ? "suspicious" : "clean",
          confidence: isPhish ? 0.82 : isSusp ? 0.61 : 0.04,
          action: isPhish ? "blocked" : isSusp ? "alerted" : "allowed",
          severity: isPhish ? "high" : isSusp ? "medium" : "low",
        };
      });
      setResults(res);
      setSummary({
        total: res.length,
        phishing: res.filter(r => r.verdict === "phishing").length,
        suspicious: res.filter(r => r.verdict === "suspicious").length,
        clean: res.filter(r => r.verdict === "clean").length,
        blocked: res.filter(r => r.action === "blocked").length,
      });
      setProgress(100);
    }
    setLoading(false);
  };

  const filtered = results ? (filter === "all" ? results : results.filter(r => r.verdict === filter)) : [];

  const exportCSV = () => {
    if (!results) return;
    const lines = ["url,verdict,confidence,action,severity",
      ...results.map(r => `"${r.url}",${r.verdict},${r.confidence.toFixed(4)},${r.action},${r.severity}`)];
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = "bulk_scan_results.csv"; a.click();
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={card()}>
        <div style={{ fontSize: 13, color: C.muted2, marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
          <Upload size={14} color={C.accent} /> Bulk URL Scanner — one URL per line
        </div>
        <textarea
          style={{ ...input, height: 160, resize: "vertical", fontFamily: "monospace", lineHeight: 1.6 }}
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder={"https://suspicious-site.xyz/login\nhttp://bank-verify.top/account\nhttps://google.com\n..."}
        />
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 12, flexWrap: "wrap", gap: 8 }}>
          <span style={{ fontSize: 12, color: C.muted }}>
            {text.split("\n").filter(Boolean).length} URLs entered (max 500)
          </span>
          <div style={{ display: "flex", gap: 8 }}>
            {results && <button style={btnGhost(C.green)} onClick={exportCSV}><Download size={13}/> Export CSV</button>}
            <button style={btn(loading ? C.muted : C.accent)} onClick={runBulk} disabled={loading}>
              {loading ? <Spinner /> : <Zap size={14}/>}
              {loading ? `Scanning ${progress}%…` : "Scan All URLs"}
            </button>
          </div>
        </div>

        {/* Progress bar */}
        {loading && (
          <div style={{ marginTop: 10, height: 3, background: "rgba(255,255,255,0.06)", borderRadius: 4 }}>
            <div style={{ width: `${progress}%`, height: "100%", background: C.accent, borderRadius: 4, transition: "width 0.3s" }} />
          </div>
        )}
      </div>

      {summary && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))", gap: 10 }}>
          {[
            { label: "Total", val: summary.total, color: C.accent2 },
            { label: "Phishing", val: summary.phishing, color: C.red },
            { label: "Suspicious", val: summary.suspicious, color: C.orange },
            { label: "Clean", val: summary.clean, color: C.green },
            { label: "Blocked", val: summary.blocked, color: "#ec4899" },
          ].map(({ label, val, color }) => (
            <div key={label} style={{ ...card({ padding: "0.9rem 1.1rem" }), textAlign: "center" }}>
              <div style={{ fontSize: 22, fontWeight: 800, color, fontFamily: "monospace" }}>{val || 0}</div>
              <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</div>
            </div>
          ))}
        </div>
      )}

      {results && (
        <div style={card()}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14, flexWrap: "wrap", gap: 8 }}>
            <span style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em" }}>
              Results ({filtered.length} shown)
            </span>
            <div style={{ display: "flex", gap: 6 }}>
              {["all", "phishing", "suspicious", "clean"].map(f => (
                <button key={f} style={filter === f ? btn(C.accent, true) : btnGhost(C.muted2, true)} onClick={() => setFilter(f)}>
                  {f.charAt(0).toUpperCase() + f.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 480, overflowY: "auto" }}>
            {filtered.map((r, i) => {
              const rc = VERDICT_C[r.verdict] || VERDICT_C.clean;
              return (
                <div key={i} style={{
                  display: "grid", gridTemplateColumns: "1fr auto auto auto",
                  gap: 12, alignItems: "center",
                  background: `${rc.color}0a`, border: `1px solid ${rc.color}20`,
                  borderRadius: 8, padding: "0.55rem 0.9rem",
                }}>
                  <span style={{ fontFamily: "monospace", fontSize: 12, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {r.url}
                  </span>
                  <VerdictBadge verdict={r.verdict} />
                  <span style={{ fontSize: 11, color: rc.color, fontFamily: "monospace", minWidth: 40, textAlign: "right" }}>
                    {(r.confidence * 100).toFixed(0)}%
                  </span>
                  <Badge text={r.action} color={r.action === "blocked" ? C.red : r.action === "alerted" ? C.orange : C.green} />
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// THREAT DATABASE
// ═══════════════════════════════════════════════════════════
function TabDatabase() {
  const [data, setData] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState({ verdict: "", severity: "", search: "" });
  const [page, setPage] = useState(0);
  const PER = 30;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: PER, offset: page * PER, active_only: true });
      if (filter.verdict)  params.set("verdict",  filter.verdict);
      if (filter.severity) params.set("severity", filter.severity);
      if (filter.search)   params.set("search",   filter.search);
      const [d, s] = await Promise.all([
        api(`/db/malicious?${params}`),
        api("/db/malicious/stats"),
      ]);
      setData(d);
      setStats(s);
    } catch {
      setData({ total: DEMO_MALICIOUS.length, items: DEMO_MALICIOUS });
      setStats({ total: DEMO_MALICIOUS.length, phishing: 3, suspicious: 1, critical_count: 2, active: 4 });
    }
    setLoading(false);
  }, [filter, page]);

  useEffect(() => { load(); }, [load]);

  const doResolve = async (url_normalized) => {
    try { await api("/db/malicious/resolve", { method: "POST", body: JSON.stringify({ url_normalized }) }); load(); }
    catch { alert("Backend offline"); }
  };

  const exportAll = () => window.open(`${API}/db/malicious/export`);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Stats row */}
      {stats && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))", gap: 10 }}>
          {[
            { label: "Total Malicious", val: stats.total || 0, color: C.red },
            { label: "Phishing", val: stats.phishing || 0, color: C.red },
            { label: "Suspicious", val: stats.suspicious || 0, color: C.orange },
            { label: "Critical", val: stats.critical_count || 0, color: "#dc2626" },
            { label: "Active", val: stats.active || 0, color: C.yellow },
          ].map(({ label, val, color }) => (
            <div key={label} style={{ ...card({ padding: "0.85rem 1rem" }) }}>
              <div style={{ fontSize: 18, fontWeight: 700, color, fontFamily: "monospace" }}>{val}</div>
              <div style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginTop: 2 }}>{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div style={{ ...card(), display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
        <Filter size={14} color={C.muted} />
        <input
          style={{ ...input, width: 220 }}
          placeholder="Search URLs..."
          value={filter.search}
          onChange={e => setFilter(f => ({ ...f, search: e.target.value }))}
        />
        <select style={{ ...input, width: 140 }} value={filter.verdict} onChange={e => setFilter(f => ({ ...f, verdict: e.target.value }))}>
          <option value="">All Verdicts</option>
          <option value="phishing">Phishing</option>
          <option value="suspicious">Suspicious</option>
        </select>
        <select style={{ ...input, width: 140 }} value={filter.severity} onChange={e => setFilter(f => ({ ...f, severity: e.target.value }))}>
          <option value="">All Severity</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
        </select>
        <button style={btn(C.accent)} onClick={() => { setPage(0); load(); }}>
          {loading ? <Spinner /> : <Search size={13}/>} Search
        </button>
        <button style={btnGhost(C.green)} onClick={exportAll}>
          <Download size={13}/> Export CSV
        </button>
      </div>

      {/* Table */}
      <div style={card()}>
        <div style={{ fontSize: 12, color: C.muted, marginBottom: 14 }}>
          {data ? `${data.total || 0} records` : "Loading…"}
        </div>

        {/* Header */}
        <div style={{ display: "grid", gridTemplateColumns: "2fr 100px 100px 80px 120px 120px 100px", gap: 8, padding: "0.5rem 0.75rem", borderBottom: `1px solid ${C.border}`, marginBottom: 6 }}>
          {["URL", "Verdict", "Severity", "Conf.", "First Seen", "Last Seen", "Actions"].map(h => (
            <span key={h} style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em" }}>{h}</span>
          ))}
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 5, maxHeight: 520, overflowY: "auto" }}>
          {(data?.items || []).map((row, i) => (
            <div key={i} style={{
              display: "grid", gridTemplateColumns: "2fr 100px 100px 80px 120px 120px 100px",
              gap: 8, alignItems: "center",
              padding: "0.6rem 0.75rem",
              background: i % 2 === 0 ? "rgba(0,0,0,0.15)" : "transparent",
              borderRadius: 7,
            }}>
              <span style={{ fontFamily: "monospace", fontSize: 12, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={row.url}>
                {row.url}
              </span>
              <VerdictBadge verdict={row.verdict} />
              <SeverityBadge severity={row.severity} />
              <span style={{ fontSize: 12, fontFamily: "monospace", color: C.text }}>{(row.confidence * 100).toFixed(0)}%</span>
              <span style={{ fontSize: 11, color: C.muted }}>{row.first_seen_human?.split(" ")[0]}</span>
              <span style={{ fontSize: 11, color: C.muted }}>{row.last_seen_human?.split(" ")[0]}</span>
              <button style={btnGhost(C.green, true)} onClick={() => doResolve(row.url_normalized)}>
                <CheckSquare size={11}/> Resolve
              </button>
            </div>
          ))}
          {(data?.items || []).length === 0 && (
            <div style={{ textAlign: "center", padding: "2.5rem", color: C.muted, fontSize: 13 }}>
              No malicious URLs found. Scan some URLs to populate this database.
            </div>
          )}
        </div>

        {/* Pagination */}
        {data && data.total > PER && (
          <div style={{ display: "flex", justifyContent: "center", gap: 8, marginTop: 14 }}>
            <button style={btnGhost(C.muted2, true)} disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Prev</button>
            <span style={{ fontSize: 12, color: C.muted, display: "flex", alignItems: "center" }}>
              Page {page + 1} of {Math.ceil(data.total / PER)}
            </span>
            <button style={btnGhost(C.muted2, true)} disabled={(page + 1) * PER >= data.total} onClick={() => setPage(p => p + 1)}>Next →</button>
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// IPS / IDS
// ═══════════════════════════════════════════════════════════
function TabIDS({ events }) {
  const [blocked, setBlocked] = useState([]);
  const [rules, setRules] = useState([]);
  const [topThreats, setTopThreats] = useState([]);
  const [blockIP, setBlockIP] = useState("");
  const [blockReason, setBlockReason] = useState("");
  const [loading, setLoading] = useState(false);

  const loadData = useCallback(async () => {
    try {
      const [b, r, t] = await Promise.all([
        api("/ids/blocked-ips"),
        api("/ids/rules"),
        api("/ids/top-threats"),
      ]);
      setBlocked(Array.isArray(b) ? b : []);
      setRules(Array.isArray(r) ? r : []);
      setTopThreats(Array.isArray(t) ? t : []);
    } catch { /* offline */ }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const doBlock = async () => {
    if (!blockIP.trim()) return;
    try { await api("/ids/block-ip", { method: "POST", body: JSON.stringify({ ip: blockIP, reason: blockReason || "Manual block" }) }); loadData(); setBlockIP(""); setBlockReason(""); }
    catch { alert("Backend offline"); }
  };

  const doUnblock = async (ip) => {
    try { await api("/ids/unblock-ip", { method: "POST", body: JSON.stringify({ ip }) }); loadData(); }
    catch { alert("Backend offline"); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Block IP form */}
        <div style={card()}>
          <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
            <Lock size={13} color={C.red} /> Manual IP Block
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <input style={input} value={blockIP} onChange={e => setBlockIP(e.target.value)} placeholder="IP address e.g. 103.21.244.82" />
            <input style={input} value={blockReason} onChange={e => setBlockReason(e.target.value)} placeholder="Block reason (optional)" />
            <button style={btn(C.red)} onClick={doBlock}><Lock size={13}/> Block IP</button>
          </div>
        </div>

        {/* Top Threats */}
        <div style={card()}>
          <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
            <TrendingUp size={13} color={C.orange} /> Top Threat IPs
          </div>
          {topThreats.length === 0
            ? <div style={{ fontSize: 12, color: C.muted }}>No threat data yet</div>
            : topThreats.slice(0, 5).map((t, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 7, background: "rgba(0,0,0,0.2)", borderRadius: 7, padding: "0.5rem 0.75rem" }}>
                <Terminal size={13} color={C.orange} />
                <span style={{ fontFamily: "monospace", color: C.orange, fontSize: 12, flex: 1 }}>{t.ip}</span>
                <Badge text={`Score: ${t.threat_score?.toFixed(0)}`} color={C.red} />
                {t.is_blocked && <Badge text="BLOCKED" color={C.red} />}
              </div>
            ))
          }
        </div>
      </div>

      {/* Blocked IPs */}
      <div style={card()}>
        <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
          <ShieldOff size={13} color={C.red} /> Blocked IPs ({blocked.length})
        </div>
        {blocked.length === 0
          ? <div style={{ fontSize: 12, color: C.muted }}>No IPs currently blocked</div>
          : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 8 }}>
              {blocked.map((b, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, background: "rgba(239,68,68,0.07)", border: "1px solid rgba(239,68,68,0.2)", borderRadius: 8, padding: "0.65rem 1rem" }}>
                  <Dot color={C.red} pulse />
                  <span style={{ fontFamily: "monospace", color: C.red, fontSize: 12, flex: 1 }}>{b.ip}</span>
                  <span style={{ fontSize: 11, color: C.muted, flex: 1 }}>{b.block_reason}</span>
                  <button style={btnGhost(C.green, true)} onClick={() => doUnblock(b.ip)}>
                    <Unlock size={11}/> Unblock
                  </button>
                </div>
              ))}
            </div>
          )
        }
      </div>

      {/* IDS Rules */}
      <div style={card()}>
        <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 14, display: "flex", alignItems: "center", gap: 6 }}>
          <FileText size={13} color={C.accent} /> Active IDS Rules ({rules.length || 12})
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 8 }}>
          {(rules.length ? rules : [
            { id: "R001", name: "Confirmed Phishing URL", severity: "critical", action: "blocked" },
            { id: "R002", name: "Suspected Phishing URL", severity: "high", action: "alerted" },
            { id: "R003", name: "Low-Confidence Suspicious", severity: "medium", action: "alerted" },
            { id: "R004", name: "IP Address Used as URL", severity: "high", action: "alerted" },
            { id: "R005", name: "Indian Bank Brand Abuse", severity: "critical", action: "blocked" },
            { id: "R006", name: "URL Shortener Detected", severity: "medium", action: "alerted" },
            { id: "R007", name: "Suspicious TLD", severity: "low", action: "alerted" },
            { id: "R009", name: "Brand Abuse Detected", severity: "high", action: "alerted" },
            { id: "R010", name: "Repeat Offender IP", severity: "critical", action: "blocked" },
            { id: "R011", name: "High Heuristic Risk Score", severity: "high", action: "alerted" },
            { id: "R012", name: "Suspicious Login Path", severity: "medium", action: "alerted" },
          ]).map(r => {
            const sc = SEV[r.severity] || SEV.low;
            return (
              <div key={r.id} style={{ display: "flex", alignItems: "center", gap: 10, background: "rgba(0,0,0,0.2)", border: `1px solid ${C.border}`, borderRadius: 8, padding: "0.6rem 0.9rem" }}>
                <span style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", minWidth: 36 }}>{r.id}</span>
                <span style={{ fontSize: 12, color: C.text, flex: 1 }}>{r.name}</span>
                <SeverityBadge severity={r.severity} />
                <Badge text={r.action} color={r.action === "blocked" ? C.red : C.orange} />
              </div>
            );
          })}
        </div>
      </div>

      {/* Recent Events */}
      <div style={card()}>
        <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 12, display: "flex", alignItems: "center", gap: 6 }}>
          <Activity size={13} color={C.accent} /> Recent Threat Events
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 380, overflowY: "auto" }}>
          {(events?.length ? events : DEMO_EVENTS).slice(0, 15).map((e, i) => {
            const sc = SEV[e.severity] || SEV.low;
            return (
              <div key={i} style={{ display: "grid", gridTemplateColumns: "8px 1fr auto auto 130px", gap: 10, alignItems: "center", background: sc.bg, border: `1px solid ${sc.border}`, borderRadius: 8, padding: "0.55rem 0.9rem" }}>
                <Dot color={sc.color} />
                <span style={{ fontFamily: "monospace", fontSize: 12, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{e.url}</span>
                <VerdictBadge verdict={e.threat_type} />
                <SeverityBadge severity={e.severity} />
                <span style={{ fontSize: 11, color: C.muted, textAlign: "right" }}>{e.timestamp_human}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// MODEL STATUS
// ═══════════════════════════════════════════════════════════
function TabModel() {
  const [status, setStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [importance, setImportance] = useState([]);

  useEffect(() => {
    Promise.all([
      api("/model/status").catch(() => null),
      api("/model/metrics").catch(() => null),
      api("/model/feature-importance?top=15").catch(() => []),
    ]).then(([s, m, imp]) => {
      setStatus(s);
      setMetrics(m);
      setImportance(Array.isArray(imp) ? imp : []);
    });
  }, []);

  const reload = async () => {
    try { await api("/model/reload", { method: "POST" }); const s = await api("/model/status"); setStatus(s); }
    catch { alert("Backend offline"); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Status card */}
      <div style={{ ...card(), display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>Model Status</div>
          {status ? (
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                <Dot color={status.model_loaded ? C.green : C.yellow} pulse={status.model_loaded} />
                <span style={{ fontSize: 14, fontWeight: 700, color: status.model_loaded ? C.green : C.yellow }}>
                  {status.model_loaded ? "ML Model Active" : "Heuristic Mode"}
                </span>
              </div>
              <div style={{ fontSize: 12, color: C.muted }}>Threshold: {status.threshold?.toFixed(2)}</div>
              <div style={{ fontSize: 12, color: C.muted }}>Features: {status.feature_count}</div>
              <div style={{ fontSize: 12, color: C.muted }}>Trusted Domains: {status.trusted_domains_count}</div>
            </div>
          ) : (
            <div style={{ fontSize: 12, color: C.muted }}>Connect backend to view status</div>
          )}
        </div>
        <div>
          <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>Architecture</div>
          <div style={{ fontSize: 12, color: C.text, lineHeight: 1.8 }}>
            Ensemble (RF + ET + GB + LR)<br/>
            SMOTE oversampling<br/>
            Calibrated probabilities<br/>
            Tuned decision threshold<br/>
            90+ URL features
          </div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>Train Model</div>
          <div style={{ fontSize: 12, color: C.muted2, marginBottom: 10, lineHeight: 1.7 }}>
            Download a phishing dataset (e.g. PhiUSIIL from Kaggle) then run:
          </div>
          <code style={{ display: "block", background: "rgba(0,0,0,0.4)", borderRadius: 7, padding: "0.6rem 0.9rem", fontSize: 11, color: C.accent2, marginBottom: 10, fontFamily: "monospace" }}>
            python models_trainer.py data/combined.csv
          </code>
          <button style={btnGhost(C.accent)} onClick={reload}><RefreshCw size={13}/> Reload Model</button>
        </div>
      </div>

      {/* Metrics */}
      {metrics && !metrics.error && (
        <div style={{ ...card() }}>
          <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 14 }}>Model Performance Metrics</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 10 }}>
            {Object.entries(metrics).filter(([k]) => typeof metrics[k] === "number").map(([k, v]) => (
              <div key={k} style={{ background: "rgba(0,0,0,0.2)", borderRadius: 8, padding: "0.75rem 1rem" }}>
                <div style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 4 }}>{k.replace(/_/g, " ")}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: C.accent2, fontFamily: "monospace" }}>
                  {typeof v === "number" && v <= 1 ? (v * 100).toFixed(1) + "%" : v}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Feature importance */}
      {importance.length > 0 && (
        <div style={card()}>
          <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 14 }}>Feature Importance (Top 15)</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {importance.map((f, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 12, color: C.muted2, fontFamily: "monospace", minWidth: 28 }}>#{i + 1}</span>
                <span style={{ fontSize: 12, color: C.text, width: 240 }}>{f.feature}</span>
                <div style={{ flex: 1, height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 4 }}>
                  <div style={{ width: `${f.importance * 100 / (importance[0]?.importance || 1)}%`, height: "100%", background: C.accent, borderRadius: 4 }} />
                </div>
                <span style={{ fontSize: 11, color: C.accent2, fontFamily: "monospace", minWidth: 50 }}>{f.importance.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Setup guide */}
      <div style={card()}>
        <div style={{ fontSize: 12, color: C.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 14, display: "flex", alignItems: "center", gap: 6 }}>
          <Settings size={13} /> Setup Guide
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, fontSize: 13, color: C.muted2, lineHeight: 1.8 }}>
          <div>
            <div style={{ color: C.text, fontWeight: 600, marginBottom: 6 }}>1. Install Dependencies</div>
            <code style={{ display: "block", background: "rgba(0,0,0,0.35)", borderRadius: 7, padding: "0.75rem", fontSize: 12, color: C.accent2, fontFamily: "monospace" }}>
              pip install fastapi uvicorn scikit-learn joblib pandas numpy
            </code>
          </div>
          <div>
            <div style={{ color: C.text, fontWeight: 600, marginBottom: 6 }}>2. Start Backend</div>
            <code style={{ display: "block", background: "rgba(0,0,0,0.35)", borderRadius: 7, padding: "0.75rem", fontSize: 12, color: C.accent2, fontFamily: "monospace" }}>
              cd backend && python main.py
            </code>
          </div>
          <div>
            <div style={{ color: C.text, fontWeight: 600, marginBottom: 6 }}>3. Get Training Data</div>
            <div>Download PhiUSIIL dataset from Kaggle or any phishing URL dataset with url + label columns.</div>
          </div>
          <div>
            <div style={{ color: C.text, fontWeight: 600, marginBottom: 6 }}>4. Train Model</div>
            <code style={{ display: "block", background: "rgba(0,0,0,0.35)", borderRadius: 7, padding: "0.75rem", fontSize: 12, color: C.accent2, fontFamily: "monospace" }}>
              python models_trainer.py data/combined.csv --models ../models
            </code>
          </div>
        </div>
        <div style={{ marginTop: 16, padding: "0.75rem 1rem", background: "rgba(99,102,241,0.08)", border: `1px solid ${C.accent}30`, borderRadius: 8, fontSize: 12, color: C.accent2 }}>
          💡 <strong>Without a trained model</strong> — the system uses an improved heuristic engine that checks: domain entropy, brand abuse, suspicious TLDs, trusted domain whitelist (includes claude.ai, anthropic.com, google.com, etc.), IP-in-URL, and 80+ other signals.
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// ROOT APP
// ═══════════════════════════════════════════════════════════
export default function CognitiveFirewall() {
  const [tab, setTab]       = useState("dashboard");
  const [stats, setStats]   = useState(null);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [backendOk, setBackendOk] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [s, e] = await Promise.all([
        api("/db/stats"),
        api("/ids/events?limit=20"),
      ]);
      setStats(s);
      setEvents(Array.isArray(e) ? e : []);
      setBackendOk(true);
    } catch {
      setBackendOk(false);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 20000);
    return () => clearInterval(t);
  }, [refresh]);

  const TABS = [
    { id: "dashboard", label: "Dashboard",    icon: <Activity size={14}/> },
    { id: "scanner",   label: "URL Scanner",  icon: <Search size={14}/> },
    { id: "database",  label: "Threat DB",    icon: <Database size={14}/> },
    { id: "bulk",      label: "Bulk Scan",    icon: <Upload size={14}/> },
    { id: "ids",       label: "IPS / IDS",    icon: <Shield size={14}/> },
    { id: "model",     label: "Model",        icon: <BarChart2 size={14}/> },
  ];

  return (
    <div style={{
      minHeight: "100vh",
      background: C.bg,
      fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
      color: C.text,
    }}>
      {/* Header */}
      <div style={{
        background: "rgba(8,11,20,0.95)",
        borderBottom: `1px solid ${C.border}`,
        padding: "0 2.5rem",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        height: 58,
        position: "sticky",
        top: 0,
        zIndex: 100,
        backdropFilter: "blur(16px)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 32, height: 32, background: `linear-gradient(135deg, ${C.accent}, ${C.purple})`, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Shield size={18} color="#fff" />
          </div>
          <div>
            <span style={{ fontSize: 16, fontWeight: 800, letterSpacing: "0.02em" }}>
              Cognitive<span style={{ color: C.accent2 }}>Firewall</span>
            </span>
            <span style={{ marginLeft: 8, fontSize: 10, color: C.muted, background: "rgba(99,102,241,0.12)", border: `1px solid ${C.accent}30`, borderRadius: 5, padding: "2px 7px", fontWeight: 600 }}>v2.2</span>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          {backendOk !== null && (
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Dot color={backendOk ? C.green : C.red} pulse={backendOk} />
              <span style={{ fontSize: 11, color: C.muted }}>
                {backendOk ? "Backend Connected" : "Backend Offline"}
              </span>
            </div>
          )}
          <button style={btnGhost(C.accent, true)} onClick={refresh}>
            {loading ? <Spinner /> : <RefreshCw size={12}/>} Refresh
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{
        background: "rgba(13,18,32,0.8)",
        borderBottom: `1px solid ${C.border}`,
        padding: "0 2.5rem",
        display: "flex",
        gap: 2,
        backdropFilter: "blur(8px)",
      }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: "0.85rem 1.2rem",
              color: tab === t.id ? C.accent2 : C.muted,
              borderBottom: tab === t.id ? `2px solid ${C.accent}` : "2px solid transparent",
              fontSize: 13,
              fontWeight: tab === t.id ? 700 : 500,
              display: "flex",
              alignItems: "center",
              gap: 6,
              transition: "color 0.15s",
              whiteSpace: "nowrap",
            }}
          >
            {t.icon}{t.label}
          </button>
        ))}
      </div>

      {/* Page content */}
      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "1.75rem 2.5rem", boxSizing: "border-box" }}>
        {tab === "dashboard" && <TabDashboard stats={stats}   events={events} loading={loading} onRefresh={refresh} />}
        {tab === "scanner"   && <TabScanner />}
        {tab === "database"  && <TabDatabase />}
        {tab === "bulk"      && <TabBulk />}
        {tab === "ids"       && <TabIDS events={events} />}
        {tab === "model"     && <TabModel />}
      </div>
    </div>
  );
}