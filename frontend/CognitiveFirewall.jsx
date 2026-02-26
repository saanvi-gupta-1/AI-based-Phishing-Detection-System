import { useState, useEffect, useCallback } from "react";
import { AlertTriangle, Shield, ShieldOff, Activity, Search, Upload,
         Database, Eye, Terminal, XCircle, CheckCircle, Clock,
         RefreshCw, Download, Trash2, CheckSquare, Filter } from "lucide-react";

const API = "http://localhost:8000";

// ── Demo data (shown when backend is offline) ──────────────────────────────
const DEMO_STATS = {
  total_scanned: 14832, total_blocked: 2341, total_phishing: 1890,
  total_suspicious: 1203, total_clean: 11739, ips_blocked: 47,
  unique_malicious_urls: 312,
};
const DEMO_EVENTS = [
  { timestamp_human:"2025-08-01 14:32:11", url:"sbi-netbanking-verify.xyz/login",   severity:"critical", action_taken:"blocked",  threat_type:"phishing",   confidence:0.97, ip:"103.21.244.82" },
  { timestamp_human:"2025-08-01 14:31:58", url:"amazon-secure-alert.top/account",   severity:"high",     action_taken:"alerted",  threat_type:"suspicious", confidence:0.73, ip:"45.77.33.91" },
  { timestamp_human:"2025-08-01 14:31:44", url:"hdfc-update-kyc.shop/verify",       severity:"critical", action_taken:"blocked",  threat_type:"phishing",   confidence:0.94, ip:"185.220.101.12" },
  { timestamp_human:"2025-08-01 14:31:29", url:"paypa1-secure-login.ru/auth",       severity:"critical", action_taken:"blocked",  threat_type:"phishing",   confidence:0.98, ip:"91.108.4.22" },
  { timestamp_human:"2025-08-01 14:30:50", url:"icicibank-secure.net/login",        severity:"high",     action_taken:"blocked",  threat_type:"phishing",   confidence:0.88, ip:"45.142.212.100" },
];
const DEMO_MALICIOUS = [
  { id:1, url:"hdfc-update-kyc.shop/verify/account",  verdict:"phishing",   confidence:0.97, severity:"critical", seen_count:14, first_seen_human:"2025-07-28 09:11:02", last_seen_human:"2025-08-01 14:31:44", source_ip:"185.220.101.12", is_active:1 },
  { id:2, url:"sbi-netbanking-verify.xyz/login",       verdict:"phishing",   confidence:0.95, severity:"critical", seen_count:9,  first_seen_human:"2025-07-30 11:43:21", last_seen_human:"2025-08-01 14:32:11", source_ip:"103.21.244.82",  is_active:1 },
  { id:3, url:"airtelrecharge.co.in/offer",            verdict:"phishing",   confidence:0.91, severity:"high",     seen_count:6,  first_seen_human:"2025-07-31 08:22:10", last_seen_human:"2025-08-01 13:55:00", source_ip:"unknown",        is_active:1 },
  { id:4, url:"icicibank-support.net/reset",           verdict:"phishing",   confidence:0.88, severity:"high",     seen_count:4,  first_seen_human:"2025-07-29 16:05:44", last_seen_human:"2025-08-01 12:10:33", source_ip:"45.142.212.100", is_active:1 },
  { id:5, url:"amazon-secure-alert.top/account",       verdict:"suspicious", confidence:0.73, severity:"medium",   seen_count:3,  first_seen_human:"2025-08-01 14:31:58", last_seen_human:"2025-08-01 14:31:58", source_ip:"45.77.33.91",    is_active:1 },
  { id:6, url:"bit.ly/claim-hdfc-reward",              verdict:"suspicious", confidence:0.68, severity:"medium",   seen_count:2,  first_seen_human:"2025-07-30 20:14:00", last_seen_human:"2025-07-31 08:00:00", source_ip:"unknown",        is_active:1 },
];

// ── Colour helpers ─────────────────────────────────────────────────────────
const SEV = {
  critical: { color:"#ff2244", bg:"rgba(255,34,68,0.12)",  border:"rgba(255,34,68,0.3)",  label:"CRITICAL" },
  high:     { color:"#ff8c00", bg:"rgba(255,140,0,0.12)",  border:"rgba(255,140,0,0.3)",  label:"HIGH"     },
  medium:   { color:"#f5c542", bg:"rgba(245,197,66,0.12)", border:"rgba(245,197,66,0.3)", label:"MEDIUM"   },
  low:      { color:"#4caf50", bg:"rgba(76,175,80,0.12)",  border:"rgba(76,175,80,0.3)",  label:"LOW"      },
};
const VERDICT = {
  phishing:   { color:"#ff2244", bg:"rgba(255,34,68,0.12)",  label:"PHISHING"   },
  suspicious: { color:"#ff8c00", bg:"rgba(255,140,0,0.12)",  label:"SUSPICIOUS" },
  clean:      { color:"#4caf50", bg:"rgba(76,175,80,0.12)",  label:"CLEAN"      },
};
const S = {
  card:    { background:"rgba(255,255,255,0.03)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:14, padding:"1.4rem" },
  input:   { flex:1, background:"rgba(0,0,0,0.5)", border:"1px solid rgba(255,255,255,0.12)", borderRadius:10, padding:"0.8rem 1.1rem", color:"#e0e0ff", fontSize:14, fontFamily:"monospace", outline:"none" },
  btn:     (col="#7c6aff") => ({ background:`linear-gradient(135deg,${col},${col}cc)`, border:"none", borderRadius:10, padding:"0.8rem 1.6rem", color:"#fff", fontSize:13, fontWeight:700, cursor:"pointer", letterSpacing:"0.06em", whiteSpace:"nowrap" }),
  btnSm:   (col="#7c6aff") => ({ background:`${col}22`, border:`1px solid ${col}44`, borderRadius:8, padding:"0.35rem 0.75rem", color:col, fontSize:12, fontWeight:600, cursor:"pointer" }),
  mono:    { fontFamily:"monospace", fontSize:12, color:"#8b8fa8" },
};

// ── Fetch wrapper ──────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const r = await fetch(`${API}${path}`, { headers:{"Content-Type":"application/json"}, ...opts });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

// ── Stat card ──────────────────────────────────────────────────────────────
function StatCard({ label, value, color = "#7c6aff", icon }) {
  return (
    <div style={{ ...S.card, display:"flex", flexDirection:"column", gap:6 }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <span style={{ fontSize:11, color:"#8b8fa8", textTransform:"uppercase", letterSpacing:"0.1em" }}>{label}</span>
        <span style={{ color, opacity:0.6 }}>{icon}</span>
      </div>
      <span style={{ fontSize:26, fontWeight:700, color, fontFamily:"monospace" }}>
        {typeof value === "number" ? value.toLocaleString() : value}
      </span>
    </div>
  );
}

// ── Verdict badge ──────────────────────────────────────────────────────────
function VBadge({ verdict }) {
  const c = VERDICT[verdict] || VERDICT.clean;
  return (
    <span style={{ background:c.bg, color:c.color, border:`1px solid ${c.color}44`,
      borderRadius:6, padding:"2px 8px", fontSize:11, fontWeight:700, letterSpacing:"0.08em" }}>
      {c.label}
    </span>
  );
}

function SBadge({ severity }) {
  const c = SEV[severity] || SEV.low;
  return (
    <span style={{ background:c.bg, color:c.color, border:`1px solid ${c.border}`,
      borderRadius:6, padding:"2px 8px", fontSize:11, fontWeight:700 }}>
      {c.label}
    </span>
  );
}

function ConfBar({ val }) {
  const pct = Math.round((val || 0) * 100);
  const col = pct >= 85 ? "#ff2244" : pct >= 60 ? "#ff8c00" : "#4caf50";
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8 }}>
      <div style={{ flex:1, height:5, background:"rgba(255,255,255,0.06)", borderRadius:3, overflow:"hidden" }}>
        <div style={{ width:`${pct}%`, height:"100%", background:col, borderRadius:3, transition:"width 0.4s" }} />
      </div>
      <span style={{ fontSize:11, color:col, fontFamily:"monospace", minWidth:34 }}>{pct}%</span>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: Dashboard
// ═══════════════════════════════════════════════════════════════════════════
function TabDashboard({ stats, events }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:20 }}>
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(170px,1fr))", gap:12 }}>
        <StatCard label="Total Scanned"    value={stats.total_scanned}          color="#7c6aff" icon={<Activity size={16}/>} />
        <StatCard label="Phishing"         value={stats.total_phishing}         color="#ff2244" icon={<XCircle   size={16}/>} />
        <StatCard label="Suspicious"       value={stats.total_suspicious}       color="#ff8c00" icon={<AlertTriangle size={16}/>} />
        <StatCard label="Blocked"          value={stats.total_blocked}          color="#ff5566" icon={<ShieldOff size={16}/>} />
        <StatCard label="IPs Blocked"      value={stats.ips_blocked}            color="#e040fb" icon={<Terminal  size={16}/>} />
        <StatCard label="Unique Malicious" value={stats.unique_malicious_urls}  color="#ff6b35" icon={<Database  size={16}/>} />
      </div>

      <div style={S.card}>
        <div style={{ fontSize:11, color:"#8b8fa8", textTransform:"uppercase", letterSpacing:"0.1em", marginBottom:14, display:"flex", alignItems:"center", gap:6 }}>
          <Activity size={13}/> Live Threat Feed
        </div>
        <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
          {events.slice(0,8).map((e,i) => {
            const c = SEV[e.severity] || SEV.low;
            return (
              <div key={i} style={{ display:"flex", alignItems:"center", gap:12,
                background:c.bg, border:`1px solid ${c.border}`, borderRadius:10, padding:"0.7rem 1rem" }}>
                <div style={{ width:8, height:8, borderRadius:"50%", background:c.color, flexShrink:0 }}/>
                <span style={{ flex:1, fontFamily:"monospace", fontSize:12, color:"#e0e0ff", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{e.url}</span>
                <VBadge verdict={e.threat_type}/>
                <span style={{ fontSize:11, color:"#8b8fa8", whiteSpace:"nowrap" }}>{e.timestamp_human}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: URL Scanner
// ═══════════════════════════════════════════════════════════════════════════
function TabScanner() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const scan = async () => {
    if (!url.trim()) return;
    setLoading(true); setResult(null);
    try {
      const d = await api("/scan", { method:"POST", body: JSON.stringify({ url }) });
      setResult(d);
    } catch {
      const conf = 0.5 + Math.random() * 0.5;
      setResult({ url, verdict: conf>0.8?"phishing":conf>0.6?"suspicious":"clean",
        confidence:conf, action:conf>0.8?"blocked":"alerted",
        severity:conf>0.8?"critical":"high", matched_rules:["R001"],
        saved_to_db: conf>0.6,
        features:{ suspicious_keyword_count:3, url_length:url.length, domain_entropy:3.2, brand_abuse:1, heuristic_risk_score:0.75 } });
    }
    setLoading(false);
  };

  const vc = result ? (VERDICT[result.verdict] || VERDICT.clean) : null;
  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
      <div style={S.card}>
        <div style={{ display:"flex", gap:10 }}>
          <input style={S.input} value={url} onChange={e=>setUrl(e.target.value)}
            onKeyDown={e=>e.key==="Enter"&&scan()}
            placeholder="Enter any URL to scan — e.g. hdfc-kyc.xyz/login"/>
          <button style={S.btn()} onClick={scan} disabled={loading}>
            {loading ? "Scanning…" : "Scan URL"}
          </button>
        </div>
      </div>

      {result && (
        <div style={{ ...S.card, border:`1px solid ${vc.color}44` }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start", marginBottom:14 }}>
            <div>
              <div style={{ fontSize:11, color:"#8b8fa8", marginBottom:4 }}>Result for</div>
              <div style={{ fontFamily:"monospace", fontSize:13, color:"#e0e0ff" }}>{result.url}</div>
            </div>
            <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", gap:6 }}>
              <VBadge verdict={result.verdict}/>
              {result.saved_to_db && (
                <span style={{ fontSize:10, color:"#4caf50", display:"flex", alignItems:"center", gap:4 }}>
                  <Database size={10}/> Saved to DB
                </span>
              )}
            </div>
          </div>

          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12, marginBottom:14 }}>
            <div style={{ background:"rgba(0,0,0,0.3)", borderRadius:8, padding:"0.8rem" }}>
              <div style={{ fontSize:10, color:"#8b8fa8", marginBottom:4 }}>Confidence</div>
              <ConfBar val={result.confidence}/>
            </div>
            <div style={{ background:"rgba(0,0,0,0.3)", borderRadius:8, padding:"0.8rem" }}>
              <div style={{ fontSize:10, color:"#8b8fa8", marginBottom:4 }}>Action Taken</div>
              <SBadge severity={result.severity}/>
            </div>
          </div>

          {result.features && (
            <div>
              <div style={{ fontSize:10, color:"#8b8fa8", marginBottom:8 }}>KEY FEATURES</div>
              <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(180px,1fr))", gap:6 }}>
                {Object.entries(result.features)
                  .filter(([,v]) => typeof v === "number" && v !== 0)
                  .slice(0, 12)
                  .map(([k,v]) => (
                    <div key={k} style={{ background:"rgba(0,0,0,0.3)", borderRadius:6, padding:"0.5rem 0.7rem",
                      display:"flex", justifyContent:"space-between", alignItems:"center" }}>
                      <span style={{ fontSize:10, color:"#8b8fa8" }}>{k.replace(/_/g," ")}</span>
                      <span style={{ fontSize:11, color:"#e0e0ff", fontFamily:"monospace" }}>
                        {typeof v === "number" ? v.toFixed(v < 1 && v > 0 ? 3 : 0) : v}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: Malicious URL Database
// ═══════════════════════════════════════════════════════════════════════════
function TabDatabase() {
  const [rows, setRows]         = useState([]);
  const [total, setTotal]       = useState(0);
  const [dbStats, setDbStats]   = useState({});
  const [loading, setLoading]   = useState(false);
  const [search, setSearch]     = useState("");
  const [filterVerdict, setFV]  = useState("");
  const [filterSev, setFS]      = useState("");
  const [page, setPage]         = useState(0);
  const [toast, setToast]       = useState(null);
  const LIMIT = 20;

  const showToast = (msg, col="#4caf50") => {
    setToast({ msg, col });
    setTimeout(() => setToast(null), 2500);
  };

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        limit: LIMIT, offset: page * LIMIT, active_only: true,
        ...(filterVerdict && { verdict: filterVerdict }),
        ...(filterSev     && { severity: filterSev }),
        ...(search        && { search }),
      });
      const d = await api(`/db/malicious?${params}`);
      setRows(d.items || []);
      setTotal(d.total || 0);

      const s = await api("/db/malicious/stats");
      setDbStats(s);
    } catch {
      setRows(DEMO_MALICIOUS);
      setTotal(DEMO_MALICIOUS.length);
      setDbStats({ total:6, phishing:4, suspicious:2, critical_count:2, active:6, total_detections:38 });
    }
    setLoading(false);
  }, [page, filterVerdict, filterSev, search]);

  useEffect(() => { load(); }, [load]);

  const resolve = async (norm) => {
    try {
      await api("/db/malicious/resolve", { method:"POST", body: JSON.stringify({ url_normalized: norm }) });
      showToast("Marked as resolved");
      load();
    } catch { showToast("Failed (demo mode)", "#ff2244"); }
  };

  const remove = async (norm) => {
    if (!confirm("Permanently delete this record?")) return;
    try {
      await api(`/db/malicious/${encodeURIComponent(norm)}`, { method:"DELETE" });
      showToast("Deleted");
      load();
    } catch { showToast("Failed (demo mode)", "#ff2244"); }
  };

  const exportCSV = async () => {
    try {
      const r = await fetch(`${API}/db/malicious/export`);
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "malicious_urls.csv";
      a.click();
    } catch { showToast("Export failed (demo mode)", "#ff2244"); }
  };

  const totalPages = Math.ceil(total / LIMIT);

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16, position:"relative" }}>

      {/* Toast */}
      {toast && (
        <div style={{ position:"fixed", top:24, right:24, background:toast.col,
          color:"#fff", borderRadius:10, padding:"0.7rem 1.2rem", fontSize:13,
          fontWeight:600, zIndex:1000, boxShadow:"0 4px 20px rgba(0,0,0,0.4)" }}>
          {toast.msg}
        </div>
      )}

      {/* DB stat cards */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(140px,1fr))", gap:10 }}>
        <StatCard label="Total in DB"    value={dbStats.total           || 0} color="#7c6aff" icon={<Database size={15}/>}/>
        <StatCard label="Phishing"       value={dbStats.phishing        || 0} color="#ff2244" icon={<XCircle  size={15}/>}/>
        <StatCard label="Suspicious"     value={dbStats.suspicious      || 0} color="#ff8c00" icon={<AlertTriangle size={15}/>}/>
        <StatCard label="Critical"       value={dbStats.critical_count  || 0} color="#ff2244" icon={<Shield   size={15}/>}/>
        <StatCard label="Active Threats" value={dbStats.active          || 0} color="#e040fb" icon={<Eye      size={15}/>}/>
        <StatCard label="Total Hits"     value={dbStats.total_detections|| 0} color="#4caf50" icon={<Activity size={15}/>}/>
      </div>

      {/* Toolbar */}
      <div style={{ display:"flex", gap:10, flexWrap:"wrap", alignItems:"center" }}>
        <input style={{ ...S.input, flex:"1 1 200px", padding:"0.6rem 1rem" }}
          placeholder="Search URL keyword…"
          value={search} onChange={e => { setSearch(e.target.value); setPage(0); }}/>
        <select style={{ ...S.input, flex:"none", width:140 }}
          value={filterVerdict} onChange={e => { setFV(e.target.value); setPage(0); }}>
          <option value="">All verdicts</option>
          <option value="phishing">Phishing</option>
          <option value="suspicious">Suspicious</option>
        </select>
        <select style={{ ...S.input, flex:"none", width:140 }}
          value={filterSev} onChange={e => { setFS(e.target.value); setPage(0); }}>
          <option value="">All severity</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
        </select>
        <button style={S.btn("#4caf50")} onClick={exportCSV}>
          <Download size={13} style={{ marginRight:6, verticalAlign:"middle" }}/>Export CSV
        </button>
        <button style={S.btnSm("#7c6aff")} onClick={load}>
          <RefreshCw size={12} style={{ marginRight:4, verticalAlign:"middle" }}/>Refresh
        </button>
      </div>

      {/* Table */}
      <div style={{ ...S.card, padding:0, overflow:"hidden" }}>
        <div style={{ overflowX:"auto" }}>
          <table style={{ width:"100%", borderCollapse:"collapse", fontSize:12 }}>
            <thead>
              <tr style={{ background:"rgba(0,0,0,0.4)" }}>
                {["URL","Verdict","Confidence","Severity","Times Seen","First Seen","Last Seen","Source IP","Actions"]
                  .map(h => (
                    <th key={h} style={{ padding:"0.75rem 1rem", textAlign:"left",
                      color:"#8b8fa8", fontWeight:600, whiteSpace:"nowrap",
                      borderBottom:"1px solid rgba(255,255,255,0.06)" }}>
                      {h}
                    </th>
                  ))}
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr><td colSpan={9} style={{ padding:"2rem", textAlign:"center", color:"#8b8fa8" }}>Loading…</td></tr>
              ) : rows.length === 0 ? (
                <tr><td colSpan={9} style={{ padding:"2rem", textAlign:"center", color:"#8b8fa8" }}>No records found</td></tr>
              ) : rows.map((r, i) => (
                <tr key={r.id || i}
                  style={{ borderBottom:"1px solid rgba(255,255,255,0.04)",
                    background: i%2===0 ? "transparent" : "rgba(255,255,255,0.01)" }}>
                  <td style={{ padding:"0.7rem 1rem", maxWidth:260 }}>
                    <span style={{ fontFamily:"monospace", color:"#e0e0ff", fontSize:11,
                      overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", display:"block" }}
                      title={r.url}>
                      {r.url}
                    </span>
                  </td>
                  <td style={{ padding:"0.7rem 1rem" }}><VBadge verdict={r.verdict}/></td>
                  <td style={{ padding:"0.7rem 1rem", minWidth:120 }}><ConfBar val={r.confidence}/></td>
                  <td style={{ padding:"0.7rem 1rem" }}><SBadge severity={r.severity}/></td>
                  <td style={{ padding:"0.7rem 1rem", textAlign:"center" }}>
                    <span style={{ background:"rgba(124,106,255,0.15)", color:"#7c6aff",
                      borderRadius:6, padding:"2px 10px", fontFamily:"monospace", fontWeight:700 }}>
                      {r.seen_count}×
                    </span>
                  </td>
                  <td style={{ padding:"0.7rem 1rem", ...S.mono, whiteSpace:"nowrap" }}>{r.first_seen_human}</td>
                  <td style={{ padding:"0.7rem 1rem", ...S.mono, whiteSpace:"nowrap" }}>{r.last_seen_human}</td>
                  <td style={{ padding:"0.7rem 1rem", ...S.mono }}>{r.source_ip || "—"}</td>
                  <td style={{ padding:"0.7rem 1rem" }}>
                    <div style={{ display:"flex", gap:6 }}>
                      <button style={S.btnSm("#4caf50")} onClick={() => resolve(r.url_normalized)}
                        title="Mark resolved">
                        <CheckSquare size={11}/>
                      </button>
                      <button style={S.btnSm("#ff2244")} onClick={() => remove(r.url_normalized)}
                        title="Delete record">
                        <Trash2 size={11}/>
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center",
            padding:"0.8rem 1.2rem", borderTop:"1px solid rgba(255,255,255,0.06)" }}>
            <span style={{ ...S.mono }}>
              Showing {page*LIMIT+1}–{Math.min((page+1)*LIMIT, total)} of {total}
            </span>
            <div style={{ display:"flex", gap:8 }}>
              <button style={S.btnSm("#7c6aff")} onClick={() => setPage(p=>Math.max(0,p-1))} disabled={page===0}>
                ← Prev
              </button>
              <span style={{ ...S.mono, padding:"0 8px", lineHeight:"2rem" }}>
                Page {page+1}/{totalPages}
              </span>
              <button style={S.btnSm("#7c6aff")} onClick={() => setPage(p=>Math.min(totalPages-1,p+1))} disabled={page>=totalPages-1}>
                Next →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: Bulk Scan
// ═══════════════════════════════════════════════════════════════════════════
function TabBulk() {
  const [text, setText]     = useState("");
  const [loading, setLoad]  = useState(false);
  const [summary, setSumm]  = useState(null);
  const [results, setRes]   = useState([]);

  const scan = async () => {
    const urls = text.split("\n").map(u=>u.trim()).filter(Boolean);
    if (!urls.length) return;
    setLoad(true); setSumm(null); setRes([]);
    try {
      const d = await api("/scan/bulk", { method:"POST", body: JSON.stringify({ urls }) });
      setSumm(d.summary); setRes(d.results);
    } catch {
      const r = urls.map(u => {
        const c = Math.random();
        return { url:u, verdict:c>0.7?"phishing":c>0.5?"suspicious":"clean", confidence:c, action:c>0.7?"blocked":"alerted", severity:c>0.7?"critical":"medium" };
      });
      setSumm({ total:r.length, phishing:r.filter(x=>x.verdict==="phishing").length,
        suspicious:r.filter(x=>x.verdict==="suspicious").length,
        clean:r.filter(x=>x.verdict==="clean").length,
        saved_to_db:r.filter(x=>x.verdict!=="clean").length });
      setRes(r);
    }
    setLoad(false);
  };

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
      <div style={S.card}>
        <div style={{ fontSize:11, color:"#8b8fa8", marginBottom:10 }}>PASTE URLs — ONE PER LINE (max 500)</div>
        <textarea style={{ ...S.input, width:"100%", minHeight:140, resize:"vertical", display:"block", boxSizing:"border-box" }}
          value={text} onChange={e=>setText(e.target.value)}
          placeholder={"hdfc-kyc.xyz\nsbi-secure.in\nhttps://www.google.com\nairtelrecharge.co.in"}/>
        <button style={{ ...S.btn(), marginTop:12 }} onClick={scan} disabled={loading}>
          {loading ? "Scanning…" : `Scan ${text.split("\n").filter(s=>s.trim()).length} URLs`}
        </button>
      </div>

      {summary && (
        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(130px,1fr))", gap:10 }}>
          <StatCard label="Total"      value={summary.total}      color="#7c6aff" icon={<Activity size={14}/>}/>
          <StatCard label="Phishing"   value={summary.phishing}   color="#ff2244" icon={<XCircle  size={14}/>}/>
          <StatCard label="Suspicious" value={summary.suspicious} color="#ff8c00" icon={<AlertTriangle size={14}/>}/>
          <StatCard label="Clean"      value={summary.clean}      color="#4caf50" icon={<CheckCircle size={14}/>}/>
          <StatCard label="Saved to DB"value={summary.saved_to_db||0} color="#e040fb" icon={<Database size={14}/>}/>
        </div>
      )}

      {results.length > 0 && (
        <div style={S.card}>
          <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
            {results.map((r,i) => {
              const vc = VERDICT[r.verdict] || VERDICT.clean;
              return (
                <div key={i} style={{ display:"flex", alignItems:"center", gap:12,
                  background:`${vc.color}10`, border:`1px solid ${vc.color}22`,
                  borderRadius:8, padding:"0.55rem 0.9rem" }}>
                  <VBadge verdict={r.verdict}/>
                  <span style={{ flex:1, fontFamily:"monospace", fontSize:12, color:"#e0e0ff",
                    overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{r.url}</span>
                  <span style={{ fontSize:11, color:vc.color, fontFamily:"monospace" }}>
                    {Math.round(r.confidence*100)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// TAB: IPS/IDS
// ═══════════════════════════════════════════════════════════════════════════
function TabIDS({ events }) {
  const [blocked, setBlocked]   = useState([]);
  const [blockIP, setBlockIP]   = useState("");
  const [blockReason, setBR]    = useState("");

  useEffect(() => {
    api("/ids/blocked-ips").then(setBlocked).catch(() =>
      setBlocked([{ ip:"103.21.244.82", threat_score:92, phishing_hits:7, is_blocked:true, block_reason:"Auto: threshold" },
                  { ip:"185.220.101.12", threat_score:88, phishing_hits:5, is_blocked:true, block_reason:"Manual" }])
    );
  }, []);

  const doBlock = async () => {
    if (!blockIP.trim()) return;
    try {
      await api("/ids/block-ip", { method:"POST", body: JSON.stringify({ ip:blockIP, reason:blockReason||"Manual" }) });
      setBlocked(b => [...b, { ip:blockIP, threat_score:100, is_blocked:true, block_reason:blockReason||"Manual" }]);
      setBlockIP(""); setBR("");
    } catch { /* demo */ }
  };

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
      {/* Manual block form */}
      <div style={S.card}>
        <div style={{ fontSize:11, color:"#8b8fa8", marginBottom:10 }}>MANUAL IP BLOCK</div>
        <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
          <input style={{ ...S.input, flex:"1 1 160px" }} value={blockIP}
            onChange={e=>setBlockIP(e.target.value)} placeholder="IP address e.g. 1.2.3.4"/>
          <input style={{ ...S.input, flex:"2 1 200px" }} value={blockReason}
            onChange={e=>setBR(e.target.value)} placeholder="Reason (optional)"/>
          <button style={S.btn("#ff2244")} onClick={doBlock}>Block IP</button>
        </div>
      </div>

      {/* Blocked IPs */}
      <div style={S.card}>
        <div style={{ fontSize:11, color:"#8b8fa8", marginBottom:12, display:"flex", alignItems:"center", gap:6 }}>
          <ShieldOff size={13}/> BLOCKED IPs ({blocked.length})
        </div>
        {blocked.length === 0
          ? <div style={{ color:"#8b8fa8", fontSize:13 }}>No IPs currently blocked</div>
          : blocked.map((b,i) => (
            <div key={i} style={{ display:"flex", alignItems:"center", gap:12, marginBottom:8,
              background:"rgba(255,34,68,0.06)", border:"1px solid rgba(255,34,68,0.15)",
              borderRadius:8, padding:"0.65rem 1rem" }}>
              <Terminal size={14} color="#ff2244"/>
              <span style={{ fontFamily:"monospace", color:"#ff6680", flex:"none", minWidth:130 }}>{b.ip}</span>
              <span style={{ fontSize:11, color:"#8b8fa8", flex:1 }}>{b.block_reason}</span>
              <span style={{ fontSize:11, color:"#ff2244", fontFamily:"monospace" }}>
                Score: {b.threat_score?.toFixed?.(0) ?? b.threat_score}
              </span>
            </div>
          ))
        }
      </div>

      {/* Recent events */}
      <div style={S.card}>
        <div style={{ fontSize:11, color:"#8b8fa8", marginBottom:12, display:"flex", alignItems:"center", gap:6 }}>
          <Activity size={13}/> RECENT EVENTS
        </div>
        <div style={{ display:"flex", flexDirection:"column", gap:7 }}>
          {events.slice(0,10).map((e,i) => {
            const c = SEV[e.severity] || SEV.low;
            return (
              <div key={i} style={{ display:"flex", gap:12, alignItems:"center",
                background:c.bg, border:`1px solid ${c.border}`, borderRadius:8, padding:"0.6rem 0.9rem" }}>
                <div style={{ width:7, height:7, borderRadius:"50%", background:c.color, flexShrink:0 }}/>
                <span style={{ fontFamily:"monospace", fontSize:11, color:"#e0e0ff", flex:1,
                  overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>{e.url}</span>
                <VBadge verdict={e.threat_type}/>
                <SBadge severity={e.severity}/>
                <span style={{ ...S.mono, whiteSpace:"nowrap" }}>{e.timestamp_human}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Root App
// ═══════════════════════════════════════════════════════════════════════════
export default function CognitiveFirewall() {
  const [tab, setTab]       = useState("dashboard");
  const [stats, setStats]   = useState(DEMO_STATS);
  const [events, setEvents] = useState(DEMO_EVENTS);

  const refresh = useCallback(async () => {
    try {
      const [s, e] = await Promise.all([
        api("/db/stats"),
        api("/ids/events?limit=20"),
      ]);
      setStats(s);
      setEvents(Array.isArray(e) ? e : DEMO_EVENTS);
    } catch { /* keep demo data */ }
  }, []);

  useEffect(() => { refresh(); const t = setInterval(refresh, 15000); return () => clearInterval(t); }, [refresh]);

  const TABS = [
    { id:"dashboard", label:"Dashboard",   icon:<Activity size={14}/> },
    { id:"scanner",   label:"URL Scanner", icon:<Search   size={14}/> },
    { id:"database",  label:"Threat DB",   icon:<Database size={14}/> },
    { id:"bulk",      label:"Bulk Scan",   icon:<Upload   size={14}/> },
    { id:"ids",       label:"IPS / IDS",   icon:<Shield   size={14}/> },
  ];

  return (
    <div style={{ minHeight:"100vh", background:"#0d0d1a",
      fontFamily:"'Segoe UI',system-ui,sans-serif", color:"#e0e0ff" }}>

      {/* Header */}
      <div style={{ background:"rgba(0,0,0,0.6)", borderBottom:"1px solid rgba(255,255,255,0.07)",
        padding:"0 2rem", display:"flex", alignItems:"center", justifyContent:"space-between",
        height:56, position:"sticky", top:0, zIndex:100, backdropFilter:"blur(12px)" }}>
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          <Shield size={22} color="#7c6aff"/>
          <span style={{ fontSize:17, fontWeight:700, letterSpacing:"0.04em" }}>
            Cognitive<span style={{ color:"#7c6aff" }}>Firewall</span>
          </span>
          <span style={{ fontSize:10, color:"#8b8fa8", background:"rgba(124,106,255,0.12)",
            border:"1px solid rgba(124,106,255,0.25)", borderRadius:5, padding:"2px 7px" }}>
            v2.1 + DB
          </span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:6 }}>
          <div style={{ width:7, height:7, borderRadius:"50%", background:"#4caf50",
            boxShadow:"0 0 6px #4caf50" }}/>
          <span style={{ fontSize:11, color:"#8b8fa8" }}>Live</span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background:"rgba(0,0,0,0.3)", borderBottom:"1px solid rgba(255,255,255,0.06)",
        padding:"0 2rem", display:"flex", gap:4 }}>
        {TABS.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{ background:"none", border:"none", cursor:"pointer",
              padding:"0.85rem 1.1rem",
              color: tab===t.id ? "#7c6aff" : "#8b8fa8",
              borderBottom: tab===t.id ? "2px solid #7c6aff" : "2px solid transparent",
              fontSize:13, fontWeight:600, display:"flex", alignItems:"center", gap:6,
              transition:"color 0.2s" }}>
            {t.icon}{t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ maxWidth:1200, margin:"0 auto", padding:"1.6rem 1.5rem" }}>
        {tab === "dashboard" && <TabDashboard stats={stats} events={events}/>}
        {tab === "scanner"   && <TabScanner/>}
        {tab === "database"  && <TabDatabase/>}
        {tab === "bulk"      && <TabBulk/>}
        {tab === "ids"       && <TabIDS events={events}/>}
      </div>
    </div>
  );
}