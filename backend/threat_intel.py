import os
import requests
from typing import Dict, Any
from page_scanner import scan_html_for_phishing

GSB_API_KEY = os.environ.get("GSB_API_KEY", "")
VT_API_KEY = os.environ.get("VT_API_KEY", "")

def check_google_safe_browsing(url: str) -> dict:
    if not GSB_API_KEY:
        return {"flagged": False, "confidence": 0.0, "reason": "gsb_not_configured"}
    
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GSB_API_KEY}"
    payload = {
        "client": {
            "clientId": "cognitive-firewall",
            "clientVersion": "3.0"
        },
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    
    try:
        res = requests.post(endpoint, json=payload, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if "matches" in data and len(data["matches"]) > 0:
                threat_type = data["matches"][0]["threatType"]
                return {"flagged": True, "confidence": 1.0, "reason": f"GSB:{threat_type}"}
    except Exception:
        pass
    
    return {"flagged": False, "confidence": 0.0, "reason": "clean"}

def check_virustotal(url: str) -> dict:
    if not VT_API_KEY:
        return {"flagged": False, "confidence": 0.0, "reason": "vt_not_configured"}
        
    try:
        import base64
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
        headers = {
            "x-apikey": VT_API_KEY
        }
        res = requests.get(endpoint, headers=headers, timeout=5)
        if res.status_code == 200:
            stats = res.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            malicious = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)
            total = sum(stats.values())
            
            if malicious >= 3 or suspicious > 3:
                return {
                    "flagged": True, 
                    "confidence": min(1.0, (malicious + suspicious) / max(total, 1) + 0.8),
                    "reason": f"VirusTotal: {malicious} engines detected"
                }
    except Exception:
        pass
    
    return {"flagged": False, "confidence": 0.0, "reason": "clean"}

def analyze_threats(url: str, enable_dynamic: bool = True) -> Dict[str, Any]:
    """
    Main aggregator for all external threat intelligence sources.
    Evaluates APIs and dynamically scans HTML as a fallback.
    """
    # 1. Google Safe Browsing
    gsb_res = check_google_safe_browsing(url)
    if gsb_res["flagged"]:
        return gsb_res
        
    # 2. Virus Total
    vt_res = check_virustotal(url)
    if vt_res["flagged"]:
        return vt_res
        
    # 3. Dynamic Page Scanner API (Zero-Day catch-all)
    if enable_dynamic:
        dyn_res = scan_html_for_phishing(url)
        if dyn_res["flagged"]:
            return dyn_res
            
    return {"flagged": False, "confidence": 0.0, "reason": "clean"}
