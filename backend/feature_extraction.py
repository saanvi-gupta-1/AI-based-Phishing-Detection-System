"""
Cognitive Firewall v2 - Feature Extraction
80+ hand-engineered features covering URL structure, domain patterns,
lexical analysis, brand similarity, and India-specific threat signals.
No external network calls required — pure offline feature engineering.
"""

import re
import math
import warnings
from urllib.parse import urlparse, parse_qs
from collections import Counter
from functools import lru_cache

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = {
    "login", "signin", "verify", "secure", "account", "update", "banking",
    "paypal", "ebay", "amazon", "google", "facebook", "apple", "microsoft",
    "confirm", "password", "credential", "support", "helpdesk", "admin",
    "billing", "invoice", "alert", "warning", "suspended", "locked",
    "urgent", "important", "act-now", "limited", "free", "prize",
    "kyc", "reward", "offer", "claim", "winner", "lucky", "bonus",
    "refund", "cashback", "recharge", "topup", "wallet", "upi",
    "netbank", "netbanking", "onlinebank", "mobilebank"
}

BRAND_KEYWORDS = {
    "sbi", "statebank", "hdfc", "icici", "axis", "kotak", "pnb",
    "bankofbaroda", "bob", "unionbank", "canarabank",
    "airtel", "jio", "vodafone", "bsnl", "tata",
    "nic", "gov", "npci", "uidai", "aadhar", "irctc",
    "iocl", "bpcl", "hpcl", "ongc",
    "paytm", "phonepe", "googlepay", "bhim", "rupay",
    "amazon", "flipkart", "myntra", "snapdeal"
}

# Trusted legitimate domains (whitelist)
TRUSTED_DOMAINS = {
    "sbi.co.in", "onlinesbi.sbi", "yonobusiness.sbi", "sbicard.com",
    "hdfcbank.com", "hdfc.com", "hdfclife.com", "hdfcergo.com",
    "icicibank.com", "icicidirect.com", "iciciprulife.com",
    "axisbank.com", "axisdirect.in",
    "kotakbank.com", "kotak.com",
    "pnbindia.in", "netpnb.com",
    "bankofbaroda.in",
    "airtel.in", "airtel.com",
    "jio.com", "jiomart.com",
    "irctc.co.in", "irctc.com",
    "iocl.com",
    "nic.gov.in", "gov.in",
    "paytm.com", "phonepe.com",
    "amazon.in", "flipkart.com",
    "google.com", "microsoft.com", "apple.com", "github.com",
    "wikipedia.org", "linkedin.com", "youtube.com"
}

# Known URL shorteners
URL_SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "cli.gs", "migre.me", "ff.im", "tiny.cc", "tr.im", "su.pr",
    "twurl.nl", "snipurl.com", "short.to", "budurl.com", "ping.fm",
    "rb.gy", "cutt.ly", "shorturl.at", "tiny.one"
}

# Suspicious TLDs commonly used in phishing
SUSPICIOUS_TLDS = {
    "xyz", "top", "shop", "club", "site", "online", "store", "fun",
    "world", "buzz", "info", "biz", "mobi", "ws", "cc", "tk", "ml",
    "ga", "cf", "gq", "pw", "bid", "win", "download", "click",
    "link", "email", "live", "party", "review", "trade", "loan",
    "racing", "stream", "vip", "icu", "work"
}

# Common legitimate TLDs
TRUSTED_TLDS = {"com", "org", "net", "edu", "gov", "mil", "int", "co"}

# Known brand names for typosquat detection
TRUSTED_BRANDS = [
    "sbi", "statebank", "hdfc", "hdfcbank", "icici", "icicibank",
    "axisbank", "kotakbank", "airtel",
    "google", "facebook", "amazon", "apple", "microsoft", "netflix",
    "paypal", "paytm", "phonepe", "flipkart", "irctc",
    "bankofbaroda", "pnb", "canara", "unionbank"
]


# ── Helpers ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=100000)
def _entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _levenshtein(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost)
    return dp[n]


def _brand_similarity(domain: str) -> float:
    """Max normalised similarity of domain to any trusted brand."""
    domain_low = domain.lower()
    best = 0.0
    for brand in TRUSTED_BRANDS:
        dist = _levenshtein(domain_low, brand)
        max_len = max(len(domain_low), len(brand))
        sim = 1.0 - dist / max_len
        if sim > best:
            best = sim
    return round(best, 4)


def _parse_url(url: str):
    """Safely parse a URL, adding scheme if missing."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url
    try:
        return urlparse(url)
    except Exception:
        return urlparse("")


def _extract_domain_parts(url: str):
    """Return (subdomain, domain, suffix) without tldextract."""
    parsed = _parse_url(url)
    netloc = parsed.netloc.lower()
    # strip port
    netloc = re.sub(r":\d+$", "", netloc)
    # strip leading www
    parts = netloc.split(".")
    if not parts:
        return "", "", ""

    # Detect suffix (handle 2-part like co.in)
    two_part = {"co.in", "co.uk", "com.au", "org.in", "net.in", "gov.in", "ac.in", "nic.in"}
    if len(parts) >= 3 and ".".join(parts[-2:]) in two_part:
        suffix = ".".join(parts[-2:])
        domain = parts[-3]
        subdomain = ".".join(parts[:-3])
    elif len(parts) >= 2:
        suffix = parts[-1]
        domain = parts[-2]
        subdomain = ".".join(parts[:-2])
    else:
        suffix = ""
        domain = parts[0]
        subdomain = ""

    return subdomain, domain, suffix


# ── Main feature extractor ────────────────────────────────────────────────────

def extract_features(url: str) -> dict:
    """
    Extract 80+ features from a URL.
    Returns an ordered dict of float/int values.
    All features are offline — no HTTP requests made.
    """
    url = str(url).strip()
    f = {}

    parsed = _parse_url(url)
    subdomain, domain, suffix = _extract_domain_parts(url)
    path = parsed.path or ""
    query = parsed.query or ""
    netloc = parsed.netloc or ""
    full_domain = f"{domain}.{suffix}" if suffix else domain
    full_domain_low = full_domain.lower()
    url_low = url.lower()

    # ── 1. Raw URL character features ────────────────────────────────────────
    f["url_length"] = len(url)
    f["num_dots"] = url.count(".")
    f["num_hyphens"] = url.count("-")
    f["num_underscores"] = url.count("_")
    f["num_slashes"] = url.count("/")
    f["num_question_marks"] = url.count("?")
    f["num_equals"] = url.count("=")
    f["num_ampersands"] = url.count("&")
    f["num_at_signs"] = url.count("@")
    f["num_percent"] = url.count("%")
    f["num_hash"] = url.count("#")
    f["num_exclamation"] = url.count("!")
    f["num_dollar"] = url.count("$")
    f["num_digits"] = sum(c.isdigit() for c in url)
    f["num_letters"] = sum(c.isalpha() for c in url)
    f["num_uppercase"] = sum(c.isupper() for c in url)
    f["digit_ratio"] = f["num_digits"] / max(len(url), 1)
    f["letter_ratio"] = f["num_letters"] / max(len(url), 1)
    f["special_char_ratio"] = (
        len(url) - f["num_digits"] - f["num_letters"]
    ) / max(len(url), 1)

    # ── 2. URL entropy ────────────────────────────────────────────────────────
    f["url_entropy"] = _entropy(url)
    f["has_repeated_digits"] = int(bool(re.search(r"(\d)\1{2,}", url)))
    f["has_repeated_chars"] = int(bool(re.search(r"(.)\1{3,}", url)))

    # ── 3. Protocol / scheme ──────────────────────────────────────────────────
    f["has_https"] = int(url_low.startswith("https"))
    f["has_http"] = int(url_low.startswith("http"))
    f["has_ftp"] = int(url_low.startswith("ftp"))
    f["has_port"] = int(bool(parsed.port))
    f["port_number"] = parsed.port or 0
    # Suspicious ports
    f["suspicious_port"] = int(parsed.port in {8080, 8443, 8888, 3000, 4444, 1337}
                               if parsed.port else False)

    # ── 4. Domain features ────────────────────────────────────────────────────
    f["domain_length"] = len(domain)
    f["domain_entropy"] = _entropy(domain)
    f["domain_has_digits"] = int(bool(re.search(r"\d", domain)))
    f["domain_digit_count"] = sum(c.isdigit() for c in domain)
    f["domain_hyphen_count"] = domain.count("-")
    f["domain_starts_with_digit"] = int(bool(domain) and domain[0].isdigit())
    f["domain_is_ip"] = int(bool(re.match(
        r"^\d{1,3}(\.\d{1,3}){3}$", netloc.split(":")[0]
    )))
    f["has_ip_in_url"] = int(bool(re.search(
        r"(^|[/\@])\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url
    )))

    # ── 5. TLD features ───────────────────────────────────────────────────────
    f["tld_length"] = len(suffix)
    f["is_suspicious_tld"] = int(suffix in SUSPICIOUS_TLDS)
    f["is_trusted_tld"] = int(suffix in TRUSTED_TLDS)
    f["tld_is_country_code"] = int(len(suffix) == 2 and suffix.isalpha())
    f["is_trusted_domain"] = int(full_domain_low in TRUSTED_DOMAINS)
    f["uses_shortener"] = int(full_domain_low in URL_SHORTENERS)

    # ── 6. Subdomain features ─────────────────────────────────────────────────
    subdomains = [s for s in subdomain.split(".") if s] if subdomain else []
    f["num_subdomains"] = len(subdomains)
    f["subdomain_length"] = len(subdomain)
    f["subdomain_entropy"] = _entropy(subdomain) if subdomain else 0.0
    f["has_www"] = int(subdomain.lower().startswith("www"))
    f["subdomain_has_digits"] = int(bool(re.search(r"\d", subdomain)))
    f["subdomain_has_hyphen"] = int("-" in subdomain)
    # Deep subdomain (more than 2 levels = suspicious)
    f["deep_subdomain"] = int(len(subdomains) > 2)

    # ── 7. Path features ──────────────────────────────────────────────────────
    path_low = path.lower()
    f["path_length"] = len(path)
    f["path_depth"] = path.count("/")
    f["path_entropy"] = _entropy(path) if path else 0.0
    f["path_has_exe"] = int(bool(re.search(r"\.(exe|bat|sh|cmd|msi|apk|dmg|ps1)$", path_low)))
    f["path_has_login"] = int(bool(re.search(r"(login|signin|logon|auth|authenticate)", path_low)))
    f["path_has_admin"] = int(bool(re.search(r"(admin|administrator|wp-admin|cpanel)", path_low)))
    f["path_has_redirect"] = int(bool(re.search(r"(redirect|redir|forward|goto|url=)", path_low)))
    f["path_has_encoded"] = int("%" in path)
    f["file_ext_suspicious"] = int(bool(re.search(
        r"\.(php|asp|aspx|jsp|cgi|pl|py|rb|cfm)$", path_low
    )))
    f["has_double_slash_in_path"] = int("//" in path)

    # ── 8. Query string features ──────────────────────────────────────────────
    f["query_length"] = len(query)
    f["num_query_params"] = len(parse_qs(query))
    f["query_has_url"] = int("http" in query.lower())
    f["query_has_ip"] = int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", query)))

    # ── 9. Suspicious keyword features ───────────────────────────────────────
    f["suspicious_keyword_count"] = sum(kw in url_low for kw in SUSPICIOUS_KEYWORDS)
    f["has_brand_keyword"] = int(any(bk in url_low for bk in BRAND_KEYWORDS))
    f["brand_in_subdomain"] = int(any(bk in subdomain.lower() for bk in BRAND_KEYWORDS))
    f["brand_as_subdomain_fqdn"] = int(
    any(td in subdomain.lower() for td in TRUSTED_DOMAINS)
    )
    f["brand_in_path"] = int(any(bk in path_low for bk in BRAND_KEYWORDS))

    # ── 10. Typosquatting / brand abuse ──────────────────────────────────────
    f["brand_similarity_score"] = _brand_similarity(domain)
    # Domain contains brand name but is NOT a trusted domain
    f["brand_abuse"] = int(
        any(bk in domain.lower() for bk in BRAND_KEYWORDS)
        and not f["is_trusted_domain"]
    )

    # ── 11. Encoding / obfuscation ────────────────────────────────────────────
    f["has_hex_encoding"] = int(bool(re.search(r"%[0-9a-fA-F]{2}", url)))
    f["has_unicode_escape"] = int(bool(re.search(r"\\u[0-9a-fA-F]{4}", url)))
    f["is_punycode"] = int("xn--" in url_low)
    f["num_encoded_chars"] = len(re.findall(r"%[0-9a-fA-F]{2}", url))

    # ── 12. India-specific threat signals ────────────────────────────────────
    # Common phishing TLDs used against Indian brands
    # Note: co.in is a legitimate Indian TLD — excluded intentionally
    f["indian_phishing_tld"] = int(suffix in {
        "xyz", "top", "shop", "site", "online", "world", "buzz"
    })
    # Legitimate Indian domain
    f["legitimate_indian_domain"] = int(
        suffix in {"gov.in", "nic.in", "ac.in"} or
        full_domain_low in TRUSTED_DOMAINS
    )
    # Indian bank keywords in a non-trusted domain
    INDIAN_BANKS = {"sbi", "hdfc", "icici", "axis", "kotak", "pnb", "bob", "canara"}
    f["indian_bank_keyword"] = int(any(b in domain.lower() for b in INDIAN_BANKS))
    f["indian_bank_phishing"] = int(
        f["indian_bank_keyword"] == 1 and f["is_trusted_domain"] == 0
    )
    GOV_BRANDS_SET = {"irs", "usps", "ssa", "fbi", "cdc", "uidai", "epfindia", "incometax"}
    f["gov_brand_wrong_tld"] = int(
    any(b in domain.lower() for b in GOV_BRANDS_SET)
    and suffix not in {"gov", "gov.in", "nic.in"}
    and not f["is_trusted_domain"]
)

    # ── 13. Lexical / token features ─────────────────────────────────────────
    tokens = re.split(r"[.\-_/=?&]", url_low)
    tokens = [t for t in tokens if t]
    f["token_count"] = len(tokens)
    f["avg_token_length"] = sum(len(t) for t in tokens) / max(len(tokens), 1)
    f["max_token_length"] = max((len(t) for t in tokens), default=0)
    f["longest_word"] = max(
        (len(w) for w in re.findall(r"[a-zA-Z]+", url)), default=0
    )
    # Number of numeric tokens
    f["numeric_token_count"] = sum(1 for t in tokens if t.isdigit())

    # ── 14. Structural red flags ──────────────────────────────────────────────
    f["has_at_sign"] = int("@" in url)
    f["multiple_subdomains"] = int(len(subdomains) > 1)
    f["excessive_dots"] = int(f["num_dots"] > 5)
    f["excessive_hyphens"] = int(f["num_hyphens"] > 3)
    f["tld_in_path"] = int(bool(re.search(
        r"\.(com|net|org|gov|edu|in)/", path_low
    )))
    f["url_length_gt_75"] = int(len(url) > 75)
    f["url_length_gt_100"] = int(len(url) > 100)

    # ── 15. Composite risk score (heuristic) ──────────────────────────────────
    # Useful as an additional feature for the ML model
    risk = 0.0
    risk += f["suspicious_keyword_count"] * 0.08
    risk += f["has_ip_in_url"] * 0.25
    risk += f["is_suspicious_tld"] * 0.15
    risk += f["brand_abuse"] * 0.3
    risk += f["indian_bank_phishing"] * 0.35
    risk += f["domain_entropy"] / 10
    risk += f["uses_shortener"] * 0.15
    risk += f["has_hex_encoding"] * 0.1
    risk += f["deep_subdomain"] * 0.1
    risk += f["brand_similarity_score"] * 0.2
    risk -= f["is_trusted_domain"] * 0.5
    risk -= f["has_https"] * 0.05
    risk += f["brand_as_subdomain_fqdn"] * 0.40   # very strong signal
    risk += f["gov_brand_wrong_tld"] * 0.35
    f["heuristic_risk_score"] = round(min(max(risk, 0.0), 1.0), 4)

    return f


# ── Batch extraction ──────────────────────────────────────────────────────────

def extract_features_batch(urls: list) -> "pd.DataFrame":
    """Extract features for a list of URLs. Returns a DataFrame."""
    import pandas as pd
    rows = []
    for url in urls:
        try:
            rows.append(extract_features(url))
        except Exception:
            rows.append({k: 0 for k in FEATURE_NAMES})
    return pd.DataFrame(rows).fillna(0)


# Build canonical feature name list once
FEATURE_NAMES: list = list(extract_features("http://example.com").keys())
NUM_FEATURES: int = len(FEATURE_NAMES)


if __name__ == "__main__":
    tests = [
        "https://www.sbi.co.in",
        "http://sbi-netbanking-verify.xyz/login",
        "http://103.21.244.82/hdfc/update",
        "https://airtelrecharge.co.in/kyc",
        "https://www.google.com",
        "http://paypa1-secure-login.ru/auth",
    ]
    import pandas as pd
    df = extract_features_batch(tests)
    print(f"Features extracted: {NUM_FEATURES}")
    print(df[["url_length", "domain_entropy", "brand_abuse", "indian_bank_phishing",
              "heuristic_risk_score", "is_trusted_domain"]].to_string())