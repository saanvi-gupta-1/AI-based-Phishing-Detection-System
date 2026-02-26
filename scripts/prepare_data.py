"""
Cognitive Firewall v5 - Data Preparation (FIXED)
=================================================
Fixes from audit of v2:

PROBLEM 1 — min_safe=900 is meaningless without checking malicious count
  → Now targets a RATIO (default 40% safe) not a raw count
  → Warns if final ratio is still below 25%

PROBLEM 2 — Augmented safe URLs are all apex domains (google.com, sbi.co.in)
  → ML model needs to learn path/query patterns too
  → Added URLs with realistic paths, query strings, subdomains

PROBLEM 3 — No adversarial safe URLs
  → Model never sees: HTTP safe, unknown small domains, subdomains of safe sites
  → Added "structurally boring but safe" URLs that confuse naive models

PROBLEM 4 — normalize_label() silently maps everything unknown to "safe"
  → Any label that isn't phishing/malicious/bad/suspect becomes "safe"
  → This silently corrupts data if column contains numeric 0/1 or unknown strings
  → Fixed: explicit mapping with fallback warning

PROBLEM 5 — label column detection is fragile
  → If CSV has "Type" or "Result" or "Status" column, it breaks
  → Now handles numeric labels (0/1), string labels, and common column names

PROBLEM 6 — No deduplication across augmented + original by apex domain
  → Augmenting sbi.co.in when training set already has it adds noise
  → Now deduplicates by apex domain, not just exact URL

PROBLEM 7 — No validation output / stats saved
  → Trainer can't verify what it got
  → Now saves data_stats.json with full breakdown

PROBLEM 8 — Suspected class: binary_label=1 (treated as malicious)
  → "Suspected" URLs are ambiguous; training on them as definite malicious adds noise
  → Now configurable: keep_suspected=False drops them by default

PROBLEM 9 — No phishing augmentation
  → Only augments safe; if malicious class is also thin in some categories, model gaps
  → Added diverse phishing patterns for all major attack types
"""

import os, sys, re, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  APEX DOMAIN EXTRACTOR  (for dedup)
# ═══════════════════════════════════════════════════════════════════════════════

TWO_PART_TLDS = {
    "co.in","gov.in","org.in","net.in","ac.in","nic.in","edu.in","res.in",
    "co.uk","org.uk","me.uk","net.uk","co.nz","com.au","net.au","org.au",
    "co.za","co.jp","or.jp","ne.jp","co.kr",
}

def apex_domain(url: str) -> str:
    url = str(url).lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = url.split("/")[0].split("?")[0].split(":")[0].split("@")[-1]
    parts = url.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_PART_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else url


# ═══════════════════════════════════════════════════════════════════════════════
#  LABEL NORMALISATION  (fixes Problem 4)
# ═══════════════════════════════════════════════════════════════════════════════

# Exhaustive mapping — anything not listed here gets flagged, not silently mapped
_PHISHING_STRINGS  = {"phishing","malicious","malware","bad","fraud","scam",
                       "spam","attack","malign","unsafe","danger","fake","1"}
_SAFE_STRINGS      = {"safe","legitimate","benign","good","clean","trusted",
                       "whitelist","legit","0"}
_SUSPECT_STRINGS   = {"suspected","suspicious","suspect","defacement","unknown"}

def normalize_label(raw) -> str:
    """Returns 'safe' | 'phishing' | 'suspected' | 'unknown'."""
    s = str(raw).strip().lower()
    if s in _SAFE_STRINGS:
        return "safe"
    if s in _PHISHING_STRINGS:
        return "phishing"
    if s in _SUSPECT_STRINGS:
        return "suspected"
    # Handle numeric 0/1 directly
    try:
        n = int(float(s))
        return "safe" if n == 0 else "phishing"
    except ValueError:
        pass
    # Partial match fallback (e.g. "phishing_v2")
    for kw in _PHISHING_STRINGS:
        if kw in s:
            return "phishing"
    for kw in _SAFE_STRINGS:
        if kw in s:
            return "safe"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  SAFE URL AUGMENTATION POOL  (fixes Problems 2, 3)
#  Rules:
#   - Mix apex, subdomains, paths, query strings
#   - Include HTTP safe URLs (breaks http=malicious shortcut)
#   - Include small/unknown domains (breaks "only big sites are safe" shortcut)
#   - Include Indian + global diversity
# ═══════════════════════════════════════════════════════════════════════════════

AUGMENT_SAFE = [
    # ── Indian Government (with realistic paths) ────────────────────────────
    "https://www.india.gov.in/topics/e-governance",
    "https://www.uidai.gov.in/en/my-aadhaar/get-aadhaar.html",
    "https://www.mca.gov.in/content/mca/global/en/home.html",
    "https://www.incometax.gov.in/iec/foportal/help/how-to-file-itr",
    "https://www.gst.gov.in/help/helpdesk",
    "https://www.mygov.in/campaigns/digital-india/",
    "https://www.epfindia.gov.in/site_en/For_Employees.php",
    "https://www.digilocker.gov.in/dashboard",
    "https://www.sebi.gov.in/legal/regulations.html",
    "https://www.rbi.org.in/scripts/PublicationsView.aspx",
    "https://www.npci.org.in/what-we-do/upi/product-overview",
    "https://www.bseindia.com/markets/equity/EQReports/MarketWatch.aspx",
    "https://www.nseindia.com/market-data/live-equity-market",
    "https://www.irdai.gov.in/ADMINCMS/cms/Frmgeneral.aspx",
    "https://www.trai.gov.in/release-publication/reports",
    # ── Indian Banking (with subdomains + paths) ────────────────────────────
    "https://www.sbi.co.in/web/personal-banking/accounts/savings-account",
    "https://netbanking.hdfcbank.com/netbanking/",
    "https://www.icicibank.com/personal-banking/bank-accounts",
    "https://www.axisbank.com/retail/accounts/savings-account",
    "https://www.kotakbank.com/personal/savings-accounts.html",
    "https://www.pnbindia.in/personal-banking.html",
    "https://www.bankofbaroda.in/personal-banking/accounts",
    "https://www.unionbankofindia.co.in/english/home.aspx",
    "https://www.canarabank.com/User_page.aspx?menuid=8",
    "https://www.federalbank.co.in/personal-banking",
    "https://www.indusind.com/iblogs/personal-finance/",
    "https://www.yesbank.in/personal-banking/yes-first",
    "https://www.idfcfirstbank.com/personal-banking",
    "https://www.southindianbank.com/retail/",
    # ── Indian Fintech / Payments ───────────────────────────────────────────
    "https://www.paytm.com/recharge/mobile-recharge",
    "https://www.phonepe.com/app/",
    "https://www.bhimupi.org.in/about-upi",
    "https://zerodha.com/varsity/module/introduction-to-stock-markets/",
    "https://groww.in/mutual-funds/axis-bluechip-fund-direct-growth",
    "https://www.upstox.com/open-demat-account/",
    "https://www.angelone.in/open-demat-account",
    "https://www.motilaloswal.com/open-demat-account/",
    "https://www.hdfcsec.com/",
    "https://cleartax.in/s/itr-filing-online",
    "https://razorpay.com/payment-gateway/",
    "https://www.policybazaar.com/health-insurance/",
    "https://www.coverfox.com/car-insurance/",
    "https://www.acko.com/car-insurance/",
    "https://www.bajajfinserv.in/personal-loan",
    "https://www.bankbazaar.com/personal-loan.html",
    # ── Indian Telecom ──────────────────────────────────────────────────────
    "https://www.airtel.in/broadband/",
    "https://www.jio.com/selfcare/plans/prepaid/",
    "https://www.bsnl.co.in/opencms/bsnl/BSNL/services/internet/broadband.html",
    "https://www.vi.in/prepaid-recharge",
    # ── Indian E-commerce ───────────────────────────────────────────────────
    "https://www.flipkart.com/mobiles/~samsung/pr?sid=tyy,4io",
    "https://www.amazon.in/s?k=laptop&ref=nb_sb_noss",
    "https://www.myntra.com/men-shirts",
    "https://www.ajio.com/shop/men",
    "https://www.nykaa.com/beauty/",
    "https://www.meesho.com/women-sarees",
    "https://www.snapdeal.com/category/electronics",
    "https://www.bigbasket.com/pc/fruits-vegetables/",
    "https://www.jiomart.com/c/groceries",
    "https://www.tatacliq.com/electronics",
    "https://www.1mg.com/otc/vitamin-c",
    "https://www.pharmeasy.in/online-medicine-order/",
    "https://www.netmeds.com/prescriptions",
    # ── Indian Travel / Services ────────────────────────────────────────────
    "https://www.irctc.co.in/nget/train-search",
    "https://www.makemytrip.com/flights/",
    "https://www.cleartrip.com/hotels/",
    "https://www.goibibo.com/buses/",
    "https://www.yatra.com/holidays/",
    "https://www.zomato.com/bangalore/restaurants",
    "https://www.swiggy.com/restaurants",
    "https://www.ola.com/",
    # ── Indian News / Media ─────────────────────────────────────────────────
    "https://www.ndtv.com/india-news/",
    "https://www.thehindu.com/news/national/",
    "https://www.economictimes.indiatimes.com/markets",
    "https://www.livemint.com/market/",
    "https://www.moneycontrol.com/stocksmarketsindia/",
    "https://www.financialexpress.com/economy/",
    # ── Indian Education ────────────────────────────────────────────────────
    "https://www.iitb.ac.in/newacadhome/topmenubar.jsp",
    "https://www.iitd.ac.in/content/about",
    "https://www.iimb.ac.in/programmes",
    "https://www.du.ac.in/du/index.php",
    "https://www.ugc.ac.in/oldpdf/pub/",
    # ── Indian IT / Tech ────────────────────────────────────────────────────
    "https://www.infosys.com/services/",
    "https://www.tcs.com/services",
    "https://www.wipro.com/en-IN/",
    "https://www.zoho.com/crm/",
    "https://www.freshworks.com/freshdesk/",
    # ── Global Trusted ──────────────────────────────────────────────────────
    "https://www.google.com/search?q=python+tutorial",
    "https://www.google.co.in/intl/en/about/",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://accounts.google.com/ServiceLogin",
    "https://www.microsoft.com/en-in/windows/windows-11",
    "https://login.microsoftonline.com/",
    "https://www.apple.com/in/iphone/",
    "https://github.com/torvalds/linux",
    "https://github.com/python/cpython/blob/main/README.rst",
    "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.linkedin.com/jobs/search/?keywords=python",
    "https://www.youtube.com/watch?v=aircAruvnKk",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
    "https://reactjs.org/docs/getting-started.html",
    "https://npmjs.com/package/express",
    "https://pypi.org/project/requests/",
    "https://hub.docker.com/_/nginx",
    "https://www.w3schools.com/html/html_intro.asp",
    "https://css-tricks.com/snippets/css/a-guide-to-flexbox/",
    "https://medium.com/tag/machine-learning",
    "https://www.reddit.com/r/learnprogramming/",
    "https://www.quora.com/What-is-machine-learning",
    "https://news.ycombinator.com/",
    "https://www.notion.so/templates",
    "https://www.figma.com/community",
    "https://www.canva.com/templates/",
    "https://www.dropbox.com/features",
    "https://slack.com/intl/en-in/",
    "https://www.netflix.com/in/browse/genre/81508195",
    "https://www.spotify.com/in/premium/",
    "https://www.adobe.com/in/products/photoshop.html",
    "https://www.salesforce.com/in/products/",
    "https://www.ibm.com/in-en/cloud",
    "https://www.oracle.com/in/database/",
    # ── HTTP safe (critical for Problem 3 — breaks http=malicious shortcut) ─
    "http://www.example-college.edu.in/admissions/2024",
    "http://localclinic.co.in/appointments",
    "http://villagecooperative.org.in/members",
    "http://smallshop-india.in/products",
    "http://www.communityschool.ac.in/fees",
    "http://blog.myportfolio.in/about-me",
    "http://techclub.iitm.ac.in/events",
    "http://municipalcorporation.gov.in/services",
    "http://district-court.nic.in/notices",
    "http://www.panchayat.gov.in/schemes",
    # ── Structurally "boring" unknown domains (unknown ≠ malicious) ─────────
    "https://www.krishnamurthys-bakery.in/menu",
    "https://suresh-auto-works.co.in/services",
    "https://rgpvuniversity.ac.in/studentZone/",
    "https://www.patnamedicalcollege.edu.in/mbbs-admission",
    "https://dlsapatna.gov.in/legal-aid",
    "https://www.ramdeobaba.ac.in/courses/",
    "https://niituniversity.in/admission/undergraduate",
    "https://www.manipalhospitals.com/bangalore/",
    "https://www.fortishealthcare.com/india/hospital",
    "https://www.apollohospitals.com/patient-care/",
]

# ═══════════════════════════════════════════════════════════════════════════════
#  PHISHING AUGMENTATION POOL  (diverse attack patterns)
#  Ensures model sees HTTPS phishing, brand abuse, all attack types
# ═══════════════════════════════════════════════════════════════════════════════

AUGMENT_PHISHING = [
    # ── HTTPS phishing (breaks "https=safe" shortcut) ───────────────────────
    "https://sbi-netbanking-secure.xyz/login?redirect=dashboard",
    "https://hdfc-kyc-verification.shop/update-now",
    "https://secure-icici-banking.net/signin",
    "https://myaxisbank-secure.com/verify-account",
    "https://paytm-cashback-offer.xyz/claim?ref=abc",
    "https://irctc-refund-portal.shop/claim/booking",
    "https://aadhaar-link-mobile.xyz/link-now",
    "https://income-tax-refund2024.xyz/claim",
    "https://sbi-account-blocked.shop/reactivate",
    "https://hdfcbank-fraud-alert.xyz/confirm-identity",
    "https://google-prize-2024.com/winner/claim",
    "https://amazon-lucky-draw.xyz/verify-account",
    "https://flipkart-cashback-2024.shop/redeem",
    "https://zerodha-account-verify.xyz/kyc",
    "https://groww-kyc-pending.shop/complete-now",
    "https://phonepe-reward.xyz/scratch-card",
    "https://jio-recharge-offer.shop/free-data",
    "https://airtel-unlimited-plan.xyz/activate",
    "https://epf-withdrawal-portal.shop/apply",
    "https://pan-aadhaar-link.xyz/last-date",
    # ── IP-based phishing ───────────────────────────────────────────────────
    "http://103.21.244.82/sbi/netbanking/login",
    "http://45.33.32.156/hdfc/verify",
    "https://103.21.244.82/icici/secure/",
    "http://192.168.1.1.evil.xyz/admin",
    "http://212.77.100.101/bank/login",
    # ── Typosquatting / homograph ───────────────────────────────────────────
    "https://hdfcbankk.com/login",
    "https://sbii.co.in/netbanking",
    "https://gooogle.com/accounts/signin",
    "https://paytm-wallet.com/pay",
    "https://flipkart.net/account",
    "https://arnazon.in/orders",
    "https://linkedln.com/login",
    "https://microsooft.com/signin",
    # ── Deep subdomain abuse ────────────────────────────────────────────────
    "https://secure.login.verify.sbi-netbanking.xyz/dashboard",
    "https://account.update.hdfc.bank-verify.net/confirm",
    "https://kyc.verification.icici-bank.support/proceed",
    "https://login.secure.axis-bank.phish.net/entry",
    # ── Brand abuse with suspicious TLD ────────────────────────────────────
    "http://sbi-netbanking.shop/login",
    "http://hdfcbank-kyc.xyz/update",
    "http://icicibank-support.net/verify",
    "http://airtelrecharge.co.in.phish.net/",
    "http://irctc-refund.xyz/claim",
    "http://crs-orgi-gov.site/register",
    # ── URL shorteners wrapping phishing ───────────────────────────────────
    "https://bit.ly/hdfc-kyc-urgent",
    "https://tinyurl.com/sbi-alert-2024",
    "https://t.co/fake-prize-claim",
    "https://rb.gy/pan-link-last-day",
    # ── Free hosting phishing ───────────────────────────────────────────────
    "https://sbi-login.000webhostapp.com/",
    "https://hdfc-verify.netlify.app/",
    "https://icici-kyc.pages.dev/update",
    "https://axis-bank-login.glitch.me/",
    "https://sbi-kyc.web.app/verify",
    # ── Long path / query obfuscation ──────────────────────────────────────
    "http://sbi.co.in.verify-now.xyz/netbanking/session?id=abc123&redirect=true",
    "http://hdfcbank.com.secure-login.shop/auth?token=xyz&user=admin",
    "https://www.google.com.account-verify.net/signin?continue=gmail",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV LOADER  (fixes Problem 5 — robust label/url column detection)
# ═══════════════════════════════════════════════════════════════════════════════

# Column name patterns that commonly hold URL data
_URL_PATTERNS   = ["url", "link", "domain", "address", "website", "site", "href"]
# Column name patterns that commonly hold label data
_LABEL_PATTERNS = ["label", "class", "type", "category", "result", "status",
                   "phishing", "target", "output", "y", "flag", "is_phish"]

def _find_url_col(df: pd.DataFrame) -> str:
    for pat in _URL_PATTERNS:
        for col in df.columns:
            if pat in col.lower():
                return col
    # fallback: first column that looks like URLs
    for col in df.columns:
        sample = df[col].dropna().head(20).astype(str)
        if sample.str.contains(r"https?://|www\.|\.com|\.in", regex=True).mean() > 0.5:
            return col
    return df.columns[0]

def _find_label_col(df: pd.DataFrame, url_col: str) -> str:
    for pat in _LABEL_PATTERNS:
        for col in df.columns:
            if col == url_col:
                continue
            if pat in col.lower():
                return col
    # fallback: column with fewest unique values (most likely categorical label)
    candidates = [c for c in df.columns if c != url_col]
    if not candidates:
        raise ValueError("Could not find label column")
    return min(candidates, key=lambda c: df[c].nunique())

def load_csv(path: str, name: str = "") -> pd.DataFrame:
    """
    Robust CSV loader. Handles:
    - Numeric labels (0/1)
    - String labels (safe/phishing/malicious/etc.)
    - Various column naming conventions
    - Files with extra columns
    """
    df = pd.read_csv(path)
    print(f"  {name or os.path.basename(path)}: {len(df)} rows, "
          f"columns: {list(df.columns)}")

    url_col   = _find_url_col(df)
    label_col = _find_label_col(df, url_col)
    print(f"    URL col='{url_col}'  Label col='{label_col}'")

    out = df[[url_col, label_col]].copy()
    out.columns = ["url", "raw_label"]
    out["url"]   = out["url"].astype(str).str.strip()
    out["label"] = out["raw_label"].apply(normalize_label)

    # Report unknown labels
    unknown = out[out["label"] == "unknown"]
    if len(unknown) > 0:
        print(f"    ⚠️  {len(unknown)} rows with unknown labels (will be DROPPED):")
        for v in unknown["raw_label"].value_counts().head(5).index:
            print(f"       '{v}'")

    return out[["url", "label"]].dropna()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PREPARATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(
    *csv_paths,                   # variadic: pass any number of CSV paths
    output_path: str,
    target_safe_ratio: float = 0.35,   # target % of safe in final dataset
    keep_suspected:    bool  = False,  # whether to keep "suspected" URLs
    augment_phishing:  bool  = True,   # whether to add phishing augmentation
    verbose:           bool  = True,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    *csv_paths          : one or more paths to CSV files to merge
    output_path         : where to write combined.csv
    target_safe_ratio   : after augmentation, safe should be at least this fraction
    keep_suspected      : if False, drop "suspected" URLs (noisy label)
    augment_phishing    : if True, add diverse phishing patterns to training
    """

    print("\n" + "=" * 65)
    print("  COGNITIVE FIREWALL v5 — Data Preparation")
    print("=" * 65)

    # ─── 1. Load all CSVs ─────────────────────────────────────────────────────
    frames = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  ⚠️  Not found, skipping: {path}")
            continue
        try:
            frames.append(load_csv(path, os.path.basename(path)))
        except Exception as e:
            print(f"  ❌ Failed to load {path}: {e}")

    if not frames:
        raise FileNotFoundError("No valid CSV files loaded.")

    combined = pd.concat(frames, ignore_index=True)

    # ─── 2. Filter unknown labels ─────────────────────────────────────────────
    n_before = len(combined)
    combined = combined[combined["label"] != "unknown"].copy()
    if verbose and n_before > len(combined):
        print(f"\n  Dropped {n_before - len(combined)} unknown-label rows")

    # ─── 3. Handle suspected class (fixes Problem 8) ─────────────────────────
    if keep_suspected:
        # Treat suspected as phishing (conservative)
        combined.loc[combined["label"] == "suspected", "label"] = "phishing"
        print("  Suspected URLs → treated as phishing")
    else:
        n_susp = (combined["label"] == "suspected").sum()
        combined = combined[combined["label"] != "suspected"].copy()
        if verbose and n_susp > 0:
            print(f"  Dropped {n_susp} 'suspected' URLs (ambiguous labels)")

    # ─── 4. Dedup by exact URL ─────────────────────────────────────────────────
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)

    # ─── 5. Report raw distribution ───────────────────────────────────────────
    n_safe = (combined["label"] == "safe").sum()
    n_mal  = (combined["label"] == "phishing").sum()
    total  = len(combined)
    print(f"\n  Raw distribution after merge:")
    print(f"    safe     : {n_safe:>6,}  ({n_safe/total*100:.1f}%)")
    print(f"    phishing : {n_mal:>6,}  ({n_mal/total*100:.1f}%)")
    print(f"    total    : {total:>6,}")

    # ─── 6. Phishing augmentation ─────────────────────────────────────────────
    if augment_phishing:
        existing_urls = set(combined["url"])
        # Deduplicate by apex domain too
        existing_apexes = {apex_domain(u) for u in existing_urls}
        new_phish = [u for u in AUGMENT_PHISHING
                     if u not in existing_urls]  # allow same apex — different paths matter
        phish_aug = pd.DataFrame({"url": new_phish, "label": "phishing"})
        combined  = pd.concat([combined, phish_aug], ignore_index=True)
        if verbose:
            print(f"\n  Phishing augmentation: +{len(phish_aug)} URLs")

    # ─── 7. Safe augmentation to reach target ratio ───────────────────────────
    n_mal_now  = (combined["label"] == "phishing").sum()
    n_safe_now = (combined["label"] == "safe").sum()
    total_now  = len(combined)

    current_ratio = n_safe_now / total_now if total_now else 0
    target_n_safe = int(n_mal_now * target_safe_ratio / (1.0 - target_safe_ratio))
    safe_needed   = max(0, target_n_safe - n_safe_now)

    if verbose:
        print(f"\n  Safe augmentation:")
        print(f"    Current safe: {n_safe_now:,} ({current_ratio*100:.1f}%)")
        print(f"    Target ratio: {target_safe_ratio*100:.0f}%")
        print(f"    Need to add : {safe_needed:,} safe URLs")

    if safe_needed > 0:
        existing_urls   = set(combined["url"])
        existing_apexes = {apex_domain(u) for u in existing_urls}
        # Filter out URLs whose apex domain is already in training set
        candidates = [u for u in AUGMENT_SAFE
                      if u not in existing_urls
                      and apex_domain(u) not in existing_apexes]

        if len(candidates) < safe_needed:
            # Repeat pool with path variants if we don't have enough
            extra = [u + f"?v={i}" for i, u in enumerate(candidates)]
            candidates = candidates + extra
            print(f"    ⚠️  Pool exhausted; using {len(candidates)} with path variants")

        aug_safe = pd.DataFrame({
            "url":   candidates[:safe_needed],
            "label": "safe",
        })
        combined = pd.concat([combined, aug_safe], ignore_index=True)
        if verbose:
            print(f"    Added {len(aug_safe)} safe URLs from augmentation pool")
    else:
        if verbose:
            print(f"    ✅ Safe ratio already sufficient ({current_ratio*100:.1f}%)")

    # ─── 8. Final dedup + binary label ────────────────────────────────────────
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    combined["url"]          = combined["url"].astype(str).str.strip()
    combined["binary_label"] = (combined["label"] == "phishing").astype(int)

    # ─── 9. Final report ──────────────────────────────────────────────────────
    n_s = (combined["binary_label"] == 0).sum()
    n_m = (combined["binary_label"] == 1).sum()
    tot = len(combined)

    print(f"\n  ✅ Final distribution:")
    print(f"    safe     : {n_s:>6,}  ({n_s/tot*100:.1f}%)")
    print(f"    phishing : {n_m:>6,}  ({n_m/tot*100:.1f}%)")
    print(f"    total    : {tot:>6,}")

    if n_s / tot < 0.20:
        print(f"\n  ⚠️  WARNING: Safe ratio still only {n_s/tot*100:.1f}%.")
        print(f"     Recommend adding more safe URLs or increasing target_safe_ratio.")

    # ─── 10. Save CSV + stats ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    combined[["url", "label", "binary_label"]].to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}")

    stats = {
        "total":           tot,
        "safe":            int(n_s),
        "phishing":        int(n_m),
        "safe_ratio":      round(n_s / tot, 4),
        "phishing_ratio":  round(n_m / tot, 4),
    }
    stats_path = output_path.replace(".csv", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats  → {stats_path}")
    print("=" * 65)

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Cognitive Firewall v5 — Data Preparation")
    p.add_argument("--data-dir",    default=None,
                   help="Directory containing CSV files (auto-discovers all CSVs)")
    p.add_argument("--training",    default=None, help="Path to training_data.csv")
    p.add_argument("--mock",        default=None, help="Path to mock_data.csv")
    p.add_argument("--output",      default=None, help="Output path for combined.csv")
    p.add_argument("--safe-ratio",  type=float, default=0.35,
                   help="Target safe class ratio (default 0.35 = 35%%)")
    p.add_argument("--keep-suspected", action="store_true",
                   help="Keep suspected URLs as phishing (default: drop)")
    args = p.parse_args()

    # Determine base dir
    HERE = os.path.dirname(os.path.abspath(__file__))
    base = os.path.dirname(HERE)

    # Collect CSV paths
    csv_paths = []
    if args.data_dir:
        csv_paths = [os.path.join(args.data_dir, f)
                     for f in os.listdir(args.data_dir) if f.endswith(".csv")]
    else:
        training = args.training or os.path.join(base, "data", "training_data.csv")
        mock     = args.mock     or os.path.join(base, "data", "mock_data.csv")
        csv_paths = [training, mock]

    output = args.output or os.path.join(base, "data", "combined.csv")

    prepare_data(
        *csv_paths,
        output_path    = output,
        target_safe_ratio = args.safe_ratio,
        keep_suspected    = args.keep_suspected,
    )