"""
Cognitive Firewall v6 - Data Preparation
=========================================
Supports merging your India-specific CSVs with the PhiUSIIL Kaggle dataset.

KEY CHANGE — PhiUSIIL label convention is INVERTED:
  PhiUSIIL:  Label=1 → legitimate (safe),  Label=0 → phishing
  Your data: label="safe" → safe,           label="Phishing" → phishing
  This file detects PhiUSIIL automatically and flips its labels correctly.

PhiUSIIL dataset columns:
  URL, Label, FILENAME (ignore), + 54 feature columns (ignored, we re-extract)

Usage:
  # Auto-discover all CSVs in data/   (just drop PhiUSIIL CSV in data/ folder)
  python prepare_data.py

  # Explicit files
  python prepare_data.py --csvs data/training_data.csv data/mock_data.csv data/PhiUSIIL_Phishing_URL_Dataset.csv

  # Control how many rows to sample from PhiUSIIL
  python prepare_data.py --phiusiil-safe 8000 --phiusiil-phishing 5000
"""

import os, sys, re, json, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

TWO_PART_TLDS = {
    "co.in","gov.in","org.in","net.in","ac.in","nic.in","edu.in","res.in",
    "co.uk","org.uk","me.uk","net.uk","co.nz","com.au","net.au","org.au",
    "co.za","co.jp","or.jp","ne.jp","co.kr",
}

# Columns unique to PhiUSIIL — used to auto-detect this dataset
_PHIUSIIL_MARKERS = {"URLTitleMatchScore", "CharContinuationRate",
                     "URLCharProb", "TLDLegitimateProb", "FILENAME"}

_PHISHING_STRINGS = {"phishing","malicious","malware","bad","fraud","scam",
                     "spam","attack","malign","unsafe","danger","fake"}
_SAFE_STRINGS     = {"safe","legitimate","benign","good","clean","trusted",
                     "whitelist","legit"}
_SUSPECT_STRINGS  = {"suspected","suspicious","suspect","defacement"}


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def apex_domain(url: str) -> str:
    url = str(url).lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = url.split("/")[0].split("?")[0].split(":")[0].split("@")[-1]
    parts = url.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in TWO_PART_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else url


def normalize_label(raw, phiusiil_mode: bool = False) -> str:
    """
    Returns 'safe' | 'phishing' | 'suspected' | 'unknown'.

    phiusiil_mode=True  → PhiUSIIL convention: 1=legitimate/safe, 0=phishing
    phiusiil_mode=False → standard convention:  0=safe, 1=phishing, or string
    """
    s = str(raw).strip().lower()

    # Numeric fast path
    try:
        n = int(float(s))
        if phiusiil_mode:
            return "safe" if n == 1 else "phishing"   # ← INVERTED for PhiUSIIL
        else:
            return "safe" if n == 0 else "phishing"
    except ValueError:
        pass

    # String mapping
    if s in _SAFE_STRINGS:      return "safe"
    if s in _PHISHING_STRINGS:  return "phishing"
    if s in _SUSPECT_STRINGS:   return "suspected"
    for kw in _PHISHING_STRINGS:
        if kw in s: return "phishing"
    for kw in _SAFE_STRINGS:
        if kw in s: return "safe"
    return "unknown"


def _find_url_col(df: pd.DataFrame) -> str:
    patterns = ["url","link","domain","address","website","site","href"]
    for pat in patterns:
        for col in df.columns:
            if pat in col.lower():
                return col
    for col in df.columns:
        sample = df[col].dropna().head(30).astype(str)
        if sample.str.contains(r"https?://|www\.|\.com|\.in|\.org", regex=True).mean() > 0.4:
            return col
    return df.columns[0]


def _find_label_col(df: pd.DataFrame, url_col: str) -> str:
    patterns = ["label","class","type","category","result","status",
                "phishing","target","output","y","flag","is_phish"]
    for pat in patterns:
        for col in df.columns:
            if col == url_col: continue
            if pat in col.lower(): return col
    candidates = [c for c in df.columns if c != url_col]
    if not candidates:
        raise ValueError("Cannot find label column")
    return min(candidates, key=lambda c: df[c].nunique())


def is_phiusiil(df: pd.DataFrame) -> bool:
    """Detect PhiUSIIL dataset by its unique column names."""
    return bool(_PHIUSIIL_MARKERS.intersection(set(df.columns)))


def _report_unknowns(df: pd.DataFrame, name: str) -> None:
    unknown = df[df["label"] == "unknown"]
    if len(unknown) > 0:
        vals = unknown["raw_label"].value_counts().head(5).to_dict()
        print(f"    ⚠️  {len(unknown)} unknown labels will be dropped: {vals}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_india_csv(path: str) -> pd.DataFrame:
    """Load India-specific CSVs (training_data.csv, mock_data.csv)."""
    df        = pd.read_csv(path)
    name      = os.path.basename(path)
    url_col   = _find_url_col(df)
    label_col = _find_label_col(df, url_col)

    print(f"\n  [{name}]  {len(df):,} rows")
    print(f"    URL col   : '{url_col}'")
    print(f"    Label col : '{label_col}'")
    print(f"    Labels    : {df[label_col].value_counts().to_dict()}")

    out = df[[url_col, label_col]].copy()
    out.columns = ["url", "raw_label"]
    out["url"]   = out["url"].astype(str).str.strip()
    out["label"] = out["raw_label"].apply(lambda x: normalize_label(x, phiusiil_mode=False))

    _report_unknowns(out, name)
    return out[["url", "label"]].dropna()


def load_phiusiil_csv(path:       str,
                      n_safe:     int = 5000,
                      n_phishing: int = 3000,
                      seed:       int = 42) -> pd.DataFrame:
    """
    Load PhiUSIIL dataset with controlled sampling.

    IMPORTANT — PhiUSIIL label convention (opposite of standard):
      Label = 1  →  legitimate / safe
      Label = 0  →  phishing

    We read only URL + Label columns for speed (ignore the 54 feature columns
    since our pipeline re-extracts features from raw URLs).
    """
    name = os.path.basename(path)
    print(f"\n  [{name}]  PhiUSIIL dataset detected")

    # Try reading only URL + Label for speed
    try:
        df = pd.read_csv(path, usecols=["URL", "Label"])
    except ValueError:
        df        = pd.read_csv(path)
        url_col   = _find_url_col(df)
        label_col = _find_label_col(df, url_col)
        df        = df.rename(columns={url_col: "URL", label_col: "Label"})[["URL","Label"]]

    df = df.dropna(subset=["URL","Label"])
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"].str.len() > 5]   # drop garbage/empty rows

    total_safe  = (df["Label"] == 1).sum()
    total_phish = (df["Label"] == 0).sum()
    print(f"    Full dataset : {len(df):,} rows")
    print(f"    Label=1 (legitimate/safe): {total_safe:,}")
    print(f"    Label=0 (phishing)       : {total_phish:,}")
    print(f"    Sampling → safe: {n_safe:,}  |  phishing: {n_phishing:,}")

    # Sample safe (Label=1)
    safe_pool    = df[df["Label"] == 1]["URL"]
    n_s          = min(n_safe, len(safe_pool))
    safe_sample  = safe_pool.sample(n=n_s, random_state=seed)

    # Sample phishing (Label=0)
    phish_pool   = df[df["Label"] == 0]["URL"]
    n_p          = min(n_phishing, len(phish_pool))
    phish_sample = phish_pool.sample(n=n_p, random_state=seed)

    result = pd.DataFrame({
        "url":   pd.concat([safe_sample, phish_sample]).values,
        "label": ["safe"] * n_s + ["phishing"] * n_p,
    })
    print(f"    Loaded  → safe: {n_s:,}  |  phishing: {n_p:,}")
    return result.reset_index(drop=True)


def load_any_csv(path: str) -> pd.DataFrame:
    """Generic loader for any other CSV."""
    df   = pd.read_csv(path)
    name = os.path.basename(path)
    print(f"\n  [{name}]  {len(df):,} rows — generic loader")

    url_col   = _find_url_col(df)
    label_col = _find_label_col(df, url_col)
    print(f"    URL col : '{url_col}'  |  Label col : '{label_col}'")
    print(f"    Labels  : {df[label_col].value_counts().to_dict()}")

    out = df[[url_col, label_col]].copy()
    out.columns = ["url", "raw_label"]
    out["url"]   = out["url"].astype(str).str.strip()
    out["label"] = out["raw_label"].apply(lambda x: normalize_label(x, phiusiil_mode=False))
    _report_unknowns(out, name)
    return out[["url", "label"]].dropna()


# ═══════════════════════════════════════════════════════════════════════════════
#  ADVERSARIAL AUGMENTATION  (Indian-specific patterns missing from PhiUSIIL)
# ═══════════════════════════════════════════════════════════════════════════════

AUGMENT_PHISHING = [
    # Indian bank brand abuse over HTTPS (breaks "https = safe" shortcut)
    "https://sbi-netbanking-secure.xyz/login",
    "https://hdfc-kyc-verification.shop/update-now",
    "https://secure-icici-banking.net/signin",
    "https://myaxisbank-secure.com/verify-account",
    "https://paytm-cashback-offer.xyz/claim",
    "https://irctc-refund-portal.shop/claim",
    "https://aadhaar-link-mobile.xyz/link-now",
    "https://income-tax-refund2024.xyz/claim",
    "https://sbi-account-blocked.shop/reactivate",
    "https://hdfcbank-fraud-alert.xyz/confirm-identity",
    "https://zerodha-account-verify.xyz/kyc",
    "https://groww-kyc-pending.shop/complete-now",
    "https://phonepe-reward.xyz/scratch-card",
    "https://jio-recharge-offer.shop/free-data",
    "https://airtel-unlimited-plan.xyz/activate",
    "https://epf-withdrawal-portal.shop/apply",
    "https://pan-aadhaar-link.xyz/last-date",
    # IP-based
    "http://103.21.244.82/sbi/netbanking/login",
    "http://45.33.32.156/hdfc/verify",
    "https://103.21.244.82/icici/secure/",
    # Typosquatting
    "https://hdfcbankk.com/login",
    "https://sbii.co.in/netbanking",
    "https://paytm-wallet.com/pay",
    "https://flipkart.net/account",
    "https://arnazon.in/orders",
    # Deep subdomain abuse
    "https://secure.login.verify.sbi-netbanking.xyz/dashboard",
    "https://account.update.hdfc.bank-verify.net/confirm",
    # Free hosting phishing
    "https://sbi-login.000webhostapp.com/",
    "https://hdfc-verify.netlify.app/",
    "https://icici-kyc.pages.dev/update",
    "https://sbi-kyc.web.app/verify",
    # URL shorteners
    "https://bit.ly/hdfc-kyc-urgent",
    "https://tinyurl.com/sbi-alert-2024",
    # Long obfuscated phishing
    "http://sbi.co.in.verify-now.xyz/netbanking/session?id=abc123&redirect=true",
    "http://hdfcbank.com.secure-login.shop/auth?token=xyz&user=admin",
]

AUGMENT_SAFE = [
    # Indian govt with realistic paths
    "https://www.india.gov.in/topics/e-governance",
    "https://www.uidai.gov.in/en/my-aadhaar/get-aadhaar.html",
    "https://www.incometax.gov.in/iec/foportal/help/how-to-file-itr",
    "https://www.epfindia.gov.in/site_en/For_Employees.php",
    "https://www.digilocker.gov.in/dashboard",
    "https://www.sebi.gov.in/legal/regulations.html",
    "https://www.rbi.org.in/scripts/PublicationsView.aspx",
    "https://www.npci.org.in/what-we-do/upi/product-overview",
    "https://www.nseindia.com/market-data/live-equity-market",
    # Indian banking with subdomains + paths
    "https://www.sbi.co.in/web/personal-banking/accounts/savings-account",
    "https://netbanking.hdfcbank.com/netbanking/",
    "https://www.icicibank.com/personal-banking/bank-accounts",
    "https://www.axisbank.com/retail/accounts/savings-account",
    "https://www.kotakbank.com/personal/savings-accounts.html",
    "https://www.pnbindia.in/personal-banking.html",
    "https://www.bankofbaroda.in/personal-banking/accounts",
    "https://www.federalbank.co.in/personal-banking",
    "https://www.yesbank.in/personal-banking/yes-first",
    # Indian fintech
    "https://zerodha.com/varsity/module/introduction-to-stock-markets/",
    "https://groww.in/mutual-funds/axis-bluechip-fund-direct-growth",
    "https://www.upstox.com/open-demat-account/",
    "https://cleartax.in/s/itr-filing-online",
    "https://razorpay.com/payment-gateway/",
    "https://www.policybazaar.com/health-insurance/",
    "https://www.bajajfinserv.in/personal-loan",
    # Indian telecom
    "https://www.airtel.in/broadband/",
    "https://www.jio.com/selfcare/plans/prepaid/",
    "https://www.vi.in/prepaid-recharge",
    # Indian ecommerce + travel
    "https://www.flipkart.com/mobiles/~samsung/pr?sid=tyy,4io",
    "https://www.amazon.in/s?k=laptop&ref=nb_sb_noss",
    "https://www.bigbasket.com/pc/fruits-vegetables/",
    "https://www.irctc.co.in/nget/train-search",
    "https://www.makemytrip.com/flights/",
    "https://www.zomato.com/bangalore/restaurants",
    # HTTP safe (teaches: HTTP ≠ malicious)
    "http://www.example-college.edu.in/admissions/2024",
    "http://localclinic.co.in/appointments",
    "http://municipalcorporation.gov.in/services",
    "http://district-court.nic.in/notices",
    "http://techclub.iitm.ac.in/events",
    "http://smallbusiness-india.in/products",
    # Global trusted with realistic paths
    "https://www.google.com/search?q=python+tutorial",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://accounts.google.com/ServiceLogin",
    "https://github.com/torvalds/linux",
    "https://stackoverflow.com/questions/11227809/",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide",
    "https://www.linkedin.com/jobs/search/?keywords=python",
    "https://www.microsoft.com/en-in/windows/windows-11",
    "https://login.microsoftonline.com/",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(
    *csv_paths,
    output_path:       str   = "data/combined.csv",
    phiusiil_safe:     int   = 5000,
    phiusiil_phishing: int   = 3000,
    target_safe_ratio: float = 0.40,
    keep_suspected:    bool  = False,
    seed:              int   = 42,
) -> pd.DataFrame:

    print("\n" + "═" * 65)
    print("  COGNITIVE FIREWALL v6 — Data Preparation")
    print("═" * 65)

    # ─── 1. Load each CSV ──────────────────────────────────────────────────────
    print("\n[1/6] Loading CSVs …")
    frames = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"  ⚠️  NOT FOUND — skipping: {path}")
            continue
        try:
            peek = pd.read_csv(path, nrows=3)
            if is_phiusiil(peek):
                df = load_phiusiil_csv(path,
                                       n_safe=phiusiil_safe,
                                       n_phishing=phiusiil_phishing,
                                       seed=seed)
            else:
                india_cols = {"critical","sector","evidence","source of detection"}
                is_india   = any(any(kw in c.lower() for kw in india_cols)
                                 for c in peek.columns)
                df = load_india_csv(path) if is_india else load_any_csv(path)
            frames.append(df)
        except Exception as e:
            print(f"  ❌ Failed to load {os.path.basename(path)}: {e}")

    if not frames:
        raise FileNotFoundError("No valid CSV files loaded.")

    combined = pd.concat(frames, ignore_index=True)

    # ─── 2. Clean labels ──────────────────────────────────────────────────────
    print("\n[2/6] Cleaning labels …")
    n_before = len(combined)
    combined = combined[combined["label"] != "unknown"].copy()
    if n_before > len(combined):
        print(f"  Dropped {n_before - len(combined):,} unknown-label rows")

    if keep_suspected:
        combined.loc[combined["label"] == "suspected", "label"] = "phishing"
        print("  Suspected → phishing")
    else:
        n_susp = (combined["label"] == "suspected").sum()
        combined = combined[combined["label"] != "suspected"].copy()
        if n_susp: print(f"  Dropped {n_susp:,} suspected URLs")

    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"  Removed {before - len(combined):,} duplicate URLs")

    # ─── 3. Raw distribution ──────────────────────────────────────────────────
    n_s = (combined["label"] == "safe").sum()
    n_m = (combined["label"] == "phishing").sum()
    tot = len(combined)
    print(f"\n[3/6] After merge:")
    print(f"  safe     : {n_s:>7,}  ({n_s/tot*100:.1f}%)")
    print(f"  phishing : {n_m:>7,}  ({n_m/tot*100:.1f}%)")
    print(f"  total    : {tot:>7,}")

    # ─── 4. Adversarial augmentation ─────────────────────────────────────────
    print("\n[4/6] Indian adversarial augmentation …")
    existing  = set(combined["url"])
    new_phish = [u for u in AUGMENT_PHISHING if u not in existing]
    new_safe  = [u for u in AUGMENT_SAFE  if u not in existing]
    aug       = pd.DataFrame({
        "url":   new_phish + new_safe,
        "label": ["phishing"] * len(new_phish) + ["safe"] * len(new_safe),
    })
    combined = pd.concat([combined, aug], ignore_index=True)
    print(f"  +{len(new_phish)} phishing  +{len(new_safe)} safe")

    # ─── 5. Balance check & recommendation ────────────────────────────────────
    print("\n[5/6] Balance check …")
    n_m   = (combined["label"] == "phishing").sum()
    n_s   = (combined["label"] == "safe").sum()
    tot   = len(combined)
    ratio = n_s / tot

    target_safe   = int(n_m * target_safe_ratio / (1.0 - target_safe_ratio))
    still_needed  = max(0, target_safe - n_s)

    print(f"  Current  : safe={n_s:,} ({ratio*100:.1f}%)  phishing={n_m:,}")
    print(f"  Target   : {target_safe_ratio*100:.0f}% safe  →  need {target_safe:,} safe URLs total")

    if still_needed > 0:
        recommended = phiusiil_safe + still_needed + 200
        print(f"  ⚠️  Short by {still_needed:,} safe URLs.")
        print(f"     → Re-run with: --phiusiil-safe {recommended:,}")
    else:
        print(f"  ✅ Target met!")

    # ─── 6. Finalise & save ────────────────────────────────────────────────────
    print("\n[6/6] Saving …")
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    combined["binary_label"] = (combined["label"] == "phishing").astype(int)

    n_s  = (combined["binary_label"] == 0).sum()
    n_m  = (combined["binary_label"] == 1).sum()
    tot  = len(combined)
    r    = n_s / tot

    print(f"\n  ✅ Final dataset:")
    print(f"  safe     : {n_s:>7,}  ({r*100:.1f}%)")
    print(f"  phishing : {n_m:>7,}  ({(1-r)*100:.1f}%)")
    print(f"  total    : {tot:>7,}")

    if r < 0.30:
        print(f"\n  ⚠️  Still imbalanced ({r*100:.1f}% safe).")
        print(f"     Run with: --phiusiil-safe {int(n_m*0.45/0.55)+500:,}")
    else:
        print(f"\n  ✅ Good balance — ready for training.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    combined[["url", "label", "binary_label"]].to_csv(output_path, index=False)
    print(f"\n  CSV   → {output_path}")

    stats = {
        "total": tot, "safe": int(n_s), "phishing": int(n_m),
        "safe_ratio": round(r, 4), "phishing_ratio": round(1-r, 4),
        "phiusiil_safe_sampled": phiusiil_safe,
        "phiusiil_phishing_sampled": phiusiil_phishing,
    }
    stats_path = output_path.replace(".csv", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats → {stats_path}")
    print("═" * 65)
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Cognitive Firewall v6 — Data Preparation")
    p.add_argument("--csvs",              nargs="+", default=None)
    p.add_argument("--data-dir",          default=None)
    p.add_argument("--output",            default=None)
    p.add_argument("--phiusiil-safe",     type=int, default=5000,
                   help="Safe URLs to sample from PhiUSIIL (default: 5000)")
    p.add_argument("--phiusiil-phishing", type=int, default=3000,
                   help="Phishing URLs to sample from PhiUSIIL (default: 3000)")
    p.add_argument("--safe-ratio",        type=float, default=0.40)
    p.add_argument("--keep-suspected",    action="store_true")
    p.add_argument("--seed",              type=int, default=42)
    args = p.parse_args()

    HERE     = os.path.dirname(os.path.abspath(__file__))
    base     = os.path.dirname(HERE)
    data_dir = args.data_dir or os.path.join(base, "data")
    output   = args.output   or os.path.join(data_dir, "combined.csv")

    if args.csvs:
        csv_paths = args.csvs
    elif os.path.isdir(data_dir):
        csv_paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv") and f != "combined.csv"
        ])
    else:
        csv_paths = []

    if not csv_paths:
        print("❌ No CSVs found. Use --csvs or --data-dir.")
        sys.exit(1)

    print(f"\n  Found {len(csv_paths)} CSV file(s):")
    for c in csv_paths:
        print(f"    {'✅' if os.path.exists(c) else '❌'}  {c}")

    prepare_data(
        *csv_paths,
        output_path        = output,
        phiusiil_safe      = args.phiusiil_safe,
        phiusiil_phishing  = args.phiusiil_phishing,
        target_safe_ratio  = args.safe_ratio,
        keep_suspected     = args.keep_suspected,
        seed               = args.seed,
    )