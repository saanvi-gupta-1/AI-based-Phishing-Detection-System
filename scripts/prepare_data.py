"""
Cognitive Firewall v2 - Data Preparation
Merges your two CSVs, augments the under-represented safe class,
and produces a clean balanced training CSV ready for model_trainer.py
"""

import pandas as pd
import numpy as np
import os
import re
import sys

# ── Safe URL augmentation pool ────────────────────────────────────────────────
# These are real legitimate Indian and global domains to balance the safe class.
# Curated to be genuinely diverse — not just big-tech but also small legit sites.

EXTRA_SAFE_URLS = [
    # Indian government
    "https://www.india.gov.in", "https://www.uidai.gov.in", "https://www.mca.gov.in",
    "https://www.incometax.gov.in", "https://www.gst.gov.in", "https://www.mygov.in",
    "https://www.epfindia.gov.in", "https://www.digilocker.gov.in",
    "https://www.sebi.gov.in", "https://www.rbi.org.in",
    "https://www.npci.org.in", "https://www.bseindia.com", "https://www.nseindia.com",
    "https://www.irdai.gov.in", "https://www.trai.gov.in",
    # Indian banking (trusted)
    "https://www.sbi.co.in", "https://www.hdfcbank.com", "https://www.icicibank.com",
    "https://www.axisbank.com", "https://www.kotakbank.com", "https://www.pnbindia.in",
    "https://www.bankofbaroda.in", "https://www.unionbankofindia.co.in",
    "https://www.canarabank.com", "https://www.federalbank.co.in",
    "https://www.indusind.com", "https://www.yesbank.in",
    # Indian fintech / payments
    "https://www.paytm.com", "https://www.phonepe.com", "https://www.googlepay.com",
    "https://www.bhimupi.org.in", "https://www.freecharge.in",
    # Indian telecom
    "https://www.airtel.in", "https://www.jio.com", "https://www.bsnl.co.in",
    "https://www.vodafone.in", "https://www.idea.in",
    # Indian e-commerce
    "https://www.flipkart.com", "https://www.amazon.in", "https://www.myntra.com",
    "https://www.ajio.com", "https://www.nykaa.com", "https://www.meesho.com",
    "https://www.snapdeal.com", "https://www.bigbasket.com", "https://www.jiomart.com",
    "https://www.tatacliq.com", "https://www.pepperfry.com", "https://www.1mg.com",
    "https://www.pharmeasy.in", "https://www.netmeds.com",
    # Indian news / media
    "https://www.ndtv.com", "https://www.thehindu.com", "https://www.indiatimes.com",
    "https://www.economictimes.indiatimes.com", "https://www.livemint.com",
    "https://www.hindustantimes.com", "https://www.timesofindia.indiatimes.com",
    "https://www.moneycontrol.com", "https://www.financialexpress.com",
    # Indian education
    "https://www.iitb.ac.in", "https://www.iitd.ac.in", "https://www.iitm.ac.in",
    "https://www.iimb.ac.in", "https://www.du.ac.in", "https://www.mu.ac.in",
    "https://www.ugc.ac.in", "https://www.aicte-india.org",
    # Indian small business / services (the critical missing class)
    "https://www.justdial.com", "https://www.sulekha.com", "https://www.zomato.com",
    "https://www.swiggy.com", "https://www.ola.com", "https://www.uber.com",
    "https://www.makemytrip.com", "https://www.cleartrip.com", "https://www.goibibo.com",
    "https://www.yatra.com", "https://www.irctc.co.in",
    "https://www.policybazaar.com", "https://www.coverfox.com",
    "https://www.acko.com", "https://www.digit.in",
    "https://www.naukri.com", "https://www.shine.com", "https://www.monster.com",
    "https://www.timesjobs.com", "https://www.indeed.com",
    "https://www.housing.com", "https://www.magicbricks.com", "https://www.99acres.com",
    "https://www.commonfloor.com", "https://www.olx.in",
    # Indian IT / tech
    "https://www.infosys.com", "https://www.tcs.com", "https://www.wipro.com",
    "https://www.hcltech.com", "https://www.techm.com",
    "https://www.zoho.com", "https://www.freshworks.com",
    # Global trusted
    "https://www.google.com", "https://www.microsoft.com", "https://www.apple.com",
    "https://www.github.com", "https://www.stackoverflow.com", "https://www.wikipedia.org",
    "https://www.mozilla.org", "https://www.python.org", "https://www.linux.org",
    "https://www.adobe.com", "https://www.ibm.com", "https://www.oracle.com",
    "https://www.salesforce.com", "https://www.dropbox.com", "https://www.slack.com",
    "https://www.notion.so", "https://www.figma.com", "https://www.canva.com",
    "https://www.medium.com", "https://www.reddit.com", "https://www.quora.com",
    "https://www.twitter.com", "https://www.instagram.com", "https://www.linkedin.com",
    "https://www.youtube.com", "https://www.netflix.com", "https://www.spotify.com",
    # Legit Indian .in and .co.in domains
    "https://www.rediff.com", "https://www.sify.com",
    "https://www.bajajfinserv.in", "https://www.hdfcsec.com",
    "https://www.icicisec.com", "https://www.kite.zerodha.com",
    "https://www.groww.in", "https://www.upstox.com", "https://www.angelbroking.com",
]


def normalize_label(lbl: str) -> str:
    lbl = str(lbl).strip().lower()
    if any(x in lbl for x in ["phishing", "malicious", "bad"]):
        return "Phishing"
    if "suspect" in lbl:
        return "Suspected"
    return "safe"


def binary_label(lbl: str) -> int:
    return 0 if lbl == "safe" else 1


def load_training_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    url_col = next((c for c in df.columns if c.lower() == "url"), None)
    if url_col is None:
        url_col = df.columns[1]  # fallback: second column

    label_col = None
    for c in df.columns:
        cl = c.lower()
        if "label" in cl or "class" in cl or "phishing" in cl:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"Cannot find label column in {path}")

    out = df[[url_col, label_col]].copy()
    out.columns = ["url", "label"]
    out["label"] = out["label"].apply(normalize_label)
    return out.dropna(subset=["url"])


def prepare_data(
    training_path: str,
    mock_path: str,
    output_path: str,
    min_safe: int = 900,
) -> pd.DataFrame:

    frames = []

    if os.path.exists(training_path):
        df1 = load_training_csv(training_path)
        frames.append(df1)
        print(f"  Loaded training_data.csv  → {len(df1)} rows")

    if os.path.exists(mock_path):
        df2 = load_training_csv(mock_path)
        frames.append(df2)
        print(f"  Loaded mock_data.csv      → {len(df2)} rows")

    if not frames:
        raise FileNotFoundError("No CSV files found.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["url"]).dropna(subset=["url"])
    combined["url"] = combined["url"].astype(str).str.strip()

    # ── Augment safe class ────────────────────────────────────────────────────
    existing_safe = combined[combined["label"] == "safe"]["url"].tolist()
    safe_needed = max(0, min_safe - len(existing_safe))

    if safe_needed > 0:
        new_safe_urls = [u for u in EXTRA_SAFE_URLS if u not in set(combined["url"])]
        aug_df = pd.DataFrame({
            "url": new_safe_urls[:safe_needed],
            "label": "safe"
        })
        combined = pd.concat([combined, aug_df], ignore_index=True)
        print(f"  Augmented safe class: +{len(aug_df)} URLs")

    combined = combined.drop_duplicates(subset=["url"])
    combined["binary_label"] = combined["label"].apply(binary_label)

    dist = combined["label"].value_counts()
    print("\n  Final label distribution:")
    for lbl, cnt in dist.items():
        pct = cnt / len(combined) * 100
        print(f"    {lbl:15s}: {cnt:5d}  ({pct:.1f}%)")
    print(f"  Total URLs: {len(combined)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\n  Saved to: {output_path}")
    return combined


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 60)
    print("COGNITIVE FIREWALL - Data Preparation")
    print("=" * 60)
    prepare_data(
        training_path=os.path.join(base, "data", "training_data.csv"),
        mock_path=os.path.join(base, "data", "mock_data.csv"),
        output_path=os.path.join(base, "data", "combined.csv"),
        min_safe=900,
    )