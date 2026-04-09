import requests
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def scan_html_for_phishing(urlStr: str) -> dict:
    """
    Downloads page content using requests to detect malicious intents like:
    - Unexpected password forms
    - Login prompts on non-traditional endpoints
    """
    if not urlStr.startswith("http"):
        urlStr = f"http://{urlStr}"
    try:
        # Use a real user-agent to bypass basic blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        res = requests.get(urlStr, headers=headers, timeout=5, verify=False)
        html = res.text.lower()

        # Check for phishing characteristics in the HTML
        has_pwd = bool(re.search(r'<input[^>]+type=[\'"]?password[\'"]?', html))
        has_cc = bool(re.search(r'(credit card|card number|cvv|expiry date|social security|ssn)', html))
        
        score = 0.0
        reasons = []

        if has_pwd:
            score += 0.5
            reasons.append("contains_password_input")
        if has_cc:
            score += 0.8
            reasons.append("requests_sensitive_financial_info")
            
        # Example heuristic: if a page asks for CC or password but isn't on a trusted domain => flagged
        if score > 0:
            return {
                "flagged": True if score >= 0.8 else False, 
                "confidence": min(1.0, score + 0.5), # Boost confidence
                "reason": "dynamic_page_analysis",
                "details": reasons
            }

        return {"flagged": False, "confidence": 0.0, "reason": "clean", "details": []}
    except Exception as e:
        return {"flagged": False, "confidence": 0.0, "reason": f"fetch_failed: {e}", "details": []}
