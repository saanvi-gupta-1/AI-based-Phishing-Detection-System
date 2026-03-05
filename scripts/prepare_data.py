"""
prepare_data.py  -  Cognitive Firewall v3
==========================================
Auto-finds training_data.csv and Mock_Data_01_08_2025.csv anywhere in
the repo tree, so it works regardless of which folder you run from.

Usage (from repo root):
    python scripts/prepare_data.py

Usage (from scripts/ folder):
    python prepare_data.py

Output: <repo_root>/data/combined.csv
"""

import re, os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

def _find_root(start):
    current = start
    for _ in range(8):
        for name in ["training_data.csv", "Mock_Data_01_08_2025.csv"]:
            if os.path.exists(os.path.join(current, name)):
                return current
            if os.path.exists(os.path.join(current, "data", name)):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return start

REPO_ROOT = _find_root(HERE)
DATA_OUT  = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_OUT, exist_ok=True)


def _find_csv(name):
    candidates = [
        os.path.join(REPO_ROOT, "data", name),
        os.path.join(REPO_ROOT, name),
        os.path.join(HERE, name),
        os.path.join(HERE, "..", "data", name),
        os.path.join(HERE, "..", name),
        os.path.join(HERE, "..", "..", "data", name),
        os.path.join(HERE, "..", "..", name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


SAFE_URLS = [
    "claude.ai","anthropic.com","openai.com","chat.openai.com",
    "gemini.google.com","copilot.microsoft.com","huggingface.co",
    "cohere.ai","mistral.ai","perplexity.ai","docs.anthropic.com",
    "api.anthropic.com","platform.openai.com",
    "google.com","google.co.in","gmail.com","youtube.com","maps.google.com",
    "drive.google.com","docs.google.com","sheets.google.com","meet.google.com",
    "microsoft.com","outlook.com","office.com","azure.microsoft.com",
    "teams.microsoft.com","live.com","hotmail.com",
    "github.com","gitlab.com","bitbucket.org","stackoverflow.com",
    "apple.com","icloud.com","developer.apple.com",
    "amazon.com","aws.amazon.com","amazon.in",
    "linkedin.com","twitter.com","x.com","facebook.com",
    "instagram.com","whatsapp.com","telegram.org","discord.com","slack.com",
    "netflix.com","spotify.com","twitch.tv",
    "wikipedia.org","medium.com","reddit.com","quora.com",
    "dropbox.com","notion.so","zoom.us","webex.com",
    "cloudflare.com","vercel.com","netlify.com","digitalocean.com",
    "stripe.com","paypal.com","shopify.com","wordpress.com",
    "adobe.com","canva.com","figma.com",
    "ibm.com","oracle.com","salesforce.com","sap.com",
    "nvidia.com","amd.com","intel.com","tesla.com",
    "npmjs.com","pypi.org","rubygems.org",
    "sbi.co.in","onlinesbi.sbi","onlinesbi.com","yonobusiness.sbi","sbicard.com",
    "hdfcbank.com","hdfc.com","hdfclife.com","hdfcsec.com",
    "icicibank.com","icicidirect.com","iciciprulife.com",
    "axisbank.com","axisdirect.in","kotakbank.com","kotak.com",
    "pnbindia.in","bankofbaroda.in","canarabank.in",
    "unionbankofindia.co.in","indusind.com","yesbank.in",
    "idfcfirstbank.com","federalbank.co.in","rbl.co.in",
    "paytm.com","paytmbank.com","phonepe.com","gpay.app",
    "razorpay.com","cashfree.com","freecharge.in","mobikwik.com",
    "zerodha.com","kite.zerodha.com","groww.in","upstox.com","angelone.in",
    "motilaloswal.com","policybazaar.com","bankbazaar.com",
    "cleartax.in","bajajfinserv.in","bajajfinance.in",
    "airtel.in","airtel.com","airtelbank.com","jio.com","jiomart.com",
    "jiofiber.com","vi.in","vodafone.in","bsnl.co.in",
    "gov.in","india.gov.in","nic.in","irctc.co.in","irctc.com",
    "incometax.gov.in","uidai.gov.in","digilocker.gov.in","mca.gov.in",
    "epfindia.gov.in","sebi.gov.in","rbi.org.in","npci.org.in",
    "gst.gov.in","bseindia.com","nseindia.com","cbse.gov.in",
    "iitb.ac.in","iitd.ac.in","iitm.ac.in","meity.gov.in",
    "flipkart.com","myntra.com","meesho.com","bigbasket.com","blinkit.com",
    "swiggy.com","zomato.com","makemytrip.com","goibibo.com",
    "cleartrip.com","redbus.in","naukri.com","99acres.com","housing.com",
    "practo.com","1mg.com","netmeds.com","pharmeasy.in",
    "ajio.com","snapdeal.com","nykaa.com","tatacliq.com",
    "bookmyshow.com","byju.com","unacademy.com","croma.com",
    "tcs.com","infosys.com","wipro.com","zoho.com","freshworks.com",
    "ndtv.com","thehindu.com","moneycontrol.com","livemint.com",
    "businessstandard.com","economictimes.indiatimes.com",
    "bbc.com","cnn.com","reuters.com","bloomberg.com","nytimes.com",
    "techcrunch.com","wired.com","theverge.com",
    "coursera.org","udemy.com","edx.org","khanacademy.org",
    "mit.edu","stanford.edu","harvard.edu",
    "visa.com","mastercard.com","americanexpress.com",
    "booking.com","airbnb.com","expedia.com","tripadvisor.com",
    "ebay.com","walmart.com","etsy.com","archive.org",
    "grammarly.com","trello.com","asana.com","mailchimp.com",
    "godaddy.com","namecheap.com",
]


def norm(url):
    return re.sub(r"^https?://(www\.)?", "", str(url).strip(), flags=re.I).rstrip("/").lower()


def to_bin(v):
    s = str(v).strip().lower()
    try:   return int(float(s))
    except: pass
    return 0 if s in {"safe","legitimate","benign","good","clean","0"} else 1


def build(out_csv):
    print("=" * 60)
    print("  prepare_data.py")
    print("=" * 60)
    print(f"  Repo root  : {REPO_ROOT}")
    print(f"  Output dir : {DATA_OUT}")
    print()

    frames = []

    p1 = _find_csv("training_data.csv")
    if p1:
        print(f"  [ok] training_data.csv  found")
        print(f"       {p1}")
        t1 = pd.read_csv(p1)
        uc = next((c for c in t1.columns if "url"   in c.lower()), None)
        lc = next((c for c in t1.columns if "label" in c.lower()), None)
        if uc and lc:
            t1 = t1[[uc, lc]].dropna()
            t1.columns = ["url","label"]
            t1["label"] = t1["label"].apply(to_bin)
            frames.append(t1)
            print(f"       {len(t1)} rows  safe={(t1['label']==0).sum()}  mal={(t1['label']==1).sum()}")
    else:
        print("  [!] training_data.csv NOT FOUND - skipping")

    p2 = _find_csv("Mock_Data_01_08_2025.csv")
    if p2:
        print(f"  [ok] Mock_Data  found")
        print(f"       {p2}")
        t2 = pd.read_csv(p2)
        uc = next((c for c in t2.columns if "url"   in c.lower()),                          None)
        lc = next((c for c in t2.columns if "class" in c.lower() or "phish" in c.lower()), None)
        if uc and lc:
            t2 = t2[[uc, lc]].dropna()
            t2.columns = ["url","label"]
            t2["label"] = t2["label"].apply(to_bin)
            frames.append(t2)
            print(f"       {len(t2)} rows  safe={(t2['label']==0).sum()}  mal={(t2['label']==1).sum()}")
    else:
        print("  [!] Mock_Data_01_08_2025.csv NOT FOUND - skipping")

    safe_df = pd.DataFrame({"url": SAFE_URLS, "label": 0})
    frames.append(safe_df)
    print(f"  [ok] Safe seed list  : {len(SAFE_URLS)} URLs")

    if len(frames) == 1:
        print()
        print("  ERROR: No source CSVs found.")
        print("  Copy your CSV files to the data/ folder:")
        print(f"    {os.path.join(REPO_ROOT, 'data', 'training_data.csv')}")
        print(f"    {os.path.join(REPO_ROOT, 'data', 'Mock_Data_01_08_2025.csv')}")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    df["url"] = df["url"].apply(norm)
    df = df[df["url"].str.len() > 4]
    df = (df.sort_values("label", ascending=False)
            .drop_duplicates("url", keep="first")
            .reset_index(drop=True))

    n0 = (df["label"] == 0).sum()
    n1 = (df["label"] == 1).sum()
    print()
    print(f"  Final: Safe={n0}  Malicious={n1}  Total={len(df)}")
    df.to_csv(out_csv, index=False)
    print(f"  Saved to {out_csv}")
    return df


if __name__ == "__main__":
    build(os.path.join(DATA_OUT, "combined.csv"))