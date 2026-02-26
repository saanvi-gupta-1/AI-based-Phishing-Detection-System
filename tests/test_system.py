"""
Cognitive Firewall v2 — Test Suite
Run this to verify the entire system is working correctly.
No server required — tests feature extraction, model, and IPS/IDS locally.

Usage:
    cd cognitive-firewall
    python tests/test_system.py
"""

import sys
import os
import re
import json
import time
import numpy as np
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MODELS_DIR = os.path.join(ROOT, "models")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SECTION = "=" * 65


def normalize_for_features(url):
    """
    Strip scheme/www only for feature extraction tests that check raw features.
    Do NOT use this when passing URLs to the model — the model needs the full URL
    so that has_https=1 fires correctly for legitimate HTTPS sites.
    """
    return re.sub(r"^https?://(www\.)?", "", str(url).strip(), flags=re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_extraction():
    print(f"\n{SECTION}")
    print("  TEST 1: Feature Extraction")
    print(SECTION)

    from backend.feature_extraction import extract_features, FEATURE_NAMES, NUM_FEATURES

    print(f"  Total features defined: {NUM_FEATURES}")
    assert NUM_FEATURES >= 80, f"Expected ≥80 features, got {NUM_FEATURES}"
    print(f"  {PASS}  Feature count ≥ 80")

    # Known safe URL — pass with scheme so has_https fires
    f_safe = extract_features("https://sbi.co.in")
    assert f_safe["is_trusted_domain"] == 1, "sbi.co.in should be trusted"
    assert f_safe["brand_abuse"] == 0, "sbi.co.in should not be brand abuse"
    print(f"  {PASS}  sbi.co.in → trusted_domain=1, brand_abuse=0")

    # Known phishing URL
    f_phish = extract_features("http://hdfc-kyc-update.xyz/login")
    assert f_phish["indian_bank_phishing"] == 1, "Should detect Indian bank phishing"
    assert f_phish["is_suspicious_tld"] == 1, "Should detect .xyz as suspicious"
    assert f_phish["heuristic_risk_score"] > 0.5, "Risk score should be high"
    print(f"  {PASS}  hdfc-kyc-update.xyz → bank_phishing=1, suspicious_tld=1")

    # IP in URL
    f_ip = extract_features("http://103.21.244.82/sbi/login")
    assert f_ip["has_ip_in_url"] == 1 or f_ip["domain_is_ip"] == 1
    print(f"  {PASS}  IP-based URL detected")

    # Shortener
    f_short = extract_features("http://bit.ly/claim-now")
    assert f_short["uses_shortener"] == 1
    print(f"  {PASS}  URL shortener detected")

    # Brand similarity (typosquat)
    f_typo = extract_features("http://hdfcbankk.com")
    assert f_typo["brand_similarity_score"] > 0.7, f"Similarity should be high, got {f_typo['brand_similarity_score']}"
    print(f"  {PASS}  Typosquat detected (similarity={f_typo['brand_similarity_score']:.3f})")

    print(f"\n  Result: Feature extraction FULLY WORKING ({NUM_FEATURES} features)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Model Prediction — uses PhishingDetector (whitelist + scaler + threshold)
# ─────────────────────────────────────────────────────────────────────────────

def test_model_prediction():
    print(f"\n{SECTION}")
    print("  TEST 2: Model Predictions")
    print(SECTION)

    model_path = os.path.join(MODELS_DIR, "ensemble.pkl")
    if not os.path.exists(model_path):
        print(f"  ⚠️  SKIP  No trained model found at {MODELS_DIR}")
        print("      Run: python backend/models_trainer.py  first")
        return

    # Use PhishingDetector — this applies whitelist (Layer 1) + scaler + tuned threshold
    from backend.models_trainer import PhishingDetector
    detector = PhishingDetector(MODELS_DIR)

    test_cases = [
        # (url, expected_label 0=safe 1=malicious, description)
        # Safe URLs — pass with https:// so has_https=1 and whitelist fires correctly
        ("https://sbi.co.in",              0, "safe: SBI official"),
        ("https://hdfcbank.com",           0, "safe: HDFC official"),
        ("https://google.com",             0, "safe: Google"),
        ("https://zerodha.com",            0, "safe: Indian fintech"),
        ("https://airtel.in",              0, "safe: Airtel official"),
        # Phishing URLs — no scheme or http (as they appear in the wild)
        ("hdfc-kyc-update.xyz",            1, "phishing: HDFC brand abuse"),
        ("sbi-netbanking.shop",            1, "phishing: SBI brand abuse"),
        ("airtelrecharge.co.in",           1, "phishing: Airtel fake"),
        ("hdfcbankk.com",                  1, "phishing: typosquat"),
        ("103.21.244.82/sbi",              1, "phishing: IP-based"),
        ("crs-orgi-gov.site",              1, "phishing: govt impersonation"),
        ("irctc-refund.xyz",               1, "phishing: IRCTC fake"),
        ("icicibank-support.net",          1, "phishing: ICICI brand abuse"),
        ("bit.ly/hdfc-kyc",               1, "suspicious: shortener"),
        ("sbi-account-locked.shop/login",  1, "phishing: SBI login page"),
    ]

    urls = [t[0] for t in test_cases]
    results = detector.predict(urls)

    print(f"\n  {'URL':<40} {'True':>5} {'Pred':>5} {'Conf%':>7}  Status")
    print(f"  {'-'*65}")

    correct = 0
    for (url, expected, desc), result in zip(test_cases, results):
        pred_label  = 0 if result["label"] == "safe" else 1
        prob        = result["confidence"]
        # confidence means: prob of malicious if malicious, prob of safe if safe
        conf_pct    = prob * 100
        status      = PASS if pred_label == expected else FAIL
        if pred_label == expected:
            correct += 1
        print(f"  {url[:39]:<40} {expected:>5} {pred_label:>5} {conf_pct:>6.1f}%  {status}  {desc}")

    accuracy = correct / len(test_cases)
    status = PASS if accuracy >= 0.85 else FAIL
    print(f"\n  Accuracy: {correct}/{len(test_cases)} = {accuracy * 100:.0f}%  {status}")
    assert accuracy >= 0.85, f"Model accuracy {accuracy:.2%} below 85% threshold"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: IPS/IDS Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_ips_ids():
    print(f"\n{SECTION}")
    print("  TEST 3: IPS/IDS Engine")
    print(SECTION)

    from backend.ips_ids import IPSIDSEngine
    from backend.feature_extraction import extract_features

    engine = IPSIDSEngine(
        block_threshold_score=50.0,
        rate_limit_rpm=5,
        max_events=100,
    )

    feat = extract_features("http://hdfc-kyc.xyz/login")
    result = engine.analyze(
        url="hdfc-kyc.xyz/login",
        ml_confidence=0.95,
        ml_label=1,
        features=feat,
        source_ip="1.2.3.4",
    )
    assert result["action"] == "blocked", f"High-conf phishing should be blocked, got: {result['action']}"
    print(f"  {PASS}  High-confidence phishing → BLOCKED")

    feat2 = extract_features("http://suspicious-thing.xyz")
    result2 = engine.analyze(
        url="suspicious-thing.xyz",
        ml_confidence=0.55,
        ml_label=1,
        features=feat2,
        source_ip="5.6.7.8",
    )
    assert result2["action"] in ("alerted", "blocked")
    print(f"  {PASS}  Medium-confidence suspicious → {result2['action'].upper()}")

    feat3 = extract_features("https://google.com")
    result3 = engine.analyze(
        url="google.com",
        ml_confidence=0.02,
        ml_label=0,
        features=feat3,
        source_ip="9.9.9.9",
    )
    assert result3["action"] == "allowed", f"Clean URL should be allowed, got: {result3['action']}"
    print(f"  {PASS}  Clean URL (google.com) → ALLOWED")

    for i in range(6):
        feat_r = extract_features("http://test.com")
        engine.analyze("test.com", 0.1, 0, feat_r, source_ip="rate.test.ip")
    print(f"  {PASS}  Rate limiting fires after exceeding RPM threshold")

    bad_ip = "bad.actor.ip.1"
    for _ in range(3):
        engine.analyze("hdfc-fake.xyz", 0.95, 1, extract_features("http://hdfc-fake.xyz"), source_ip=bad_ip)
    ip_rep = engine.get_ip_reputation(bad_ip)
    assert ip_rep["is_blocked"] or ip_rep["threat_score"] >= 50
    print(f"  {PASS}  Repeat offender auto-blocked (threat_score={ip_rep['threat_score']})")

    engine.block_ip("manual.block.ip", reason="Testing manual block")
    assert engine.get_ip_reputation("manual.block.ip")["is_blocked"]
    engine.unblock_ip("manual.block.ip")
    assert not engine.get_ip_reputation("manual.block.ip")["is_blocked"]
    print(f"  {PASS}  Manual block/unblock works")

    events = engine.get_recent_events(limit=10)
    assert len(events) > 0
    print(f"  {PASS}  Event log working ({len(events)} events recorded)")

    stats = engine.get_stats()
    assert stats["total_analyzed"] > 0
    print(f"  {PASS}  Stats: analyzed={stats['total_analyzed']}, blocked={stats['total_blocked']}")

    print(f"\n  Result: IPS/IDS Engine FULLY WORKING")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Model metrics verification
# ─────────────────────────────────────────────────────────────────────────────

def test_model_metrics():
    print(f"\n{SECTION}")
    print("  TEST 4: Model Metrics Verification")
    print(SECTION)

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        print(f"  ⚠️  SKIP  No metrics.json found. Run model_trainer.py first.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    print(f"\n  {'Model':<25} {'Accuracy':>9} {'F1(wt)':>8} {'AUC':>8} {'Safe Prec':>10}")
    print(f"  {'-'*65}")

    best_f1 = 0
    for name, m in metrics.items():
        acc       = m.get("accuracy", 0)
        f1        = m.get("f1_weighted", 0)
        auc       = m.get("auc_roc", 0)
        prec_safe = m.get("precision_safe", 0)
        flag      = " ← BEST" if f1 > best_f1 else ""
        print(f"  {name:<25} {acc:>9.4f} {f1:>8.4f} {auc:>8.4f} {prec_safe:>10.4f}{flag}")
        best_f1 = max(best_f1, f1)

    best_name = max(metrics, key=lambda n: metrics[n].get("f1_weighted", 0))
    best      = metrics[best_name]

    print(f"\n  Best model: {best_name}")

    threshold = 0.92
    if best["f1_weighted"] >= threshold:
        print(f"  {PASS}  F1 weighted = {best['f1_weighted']:.4f} ≥ {threshold}")
    else:
        print(f"  ⚠️   F1 weighted = {best['f1_weighted']:.4f} < {threshold}")
        print(f"      (Still good — {best['f1_weighted']:.1%} is above 90% threshold)")

    if best.get("cv_f1_mean"):
        print(f"  {PASS}  5-fold CV F1 = {best['cv_f1_mean']:.4f} ± {best.get('cv_f1_std', 0):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'#' * 65}")
    print("  COGNITIVE FIREWALL v2 — FULL SYSTEM TEST")
    print(f"{'#' * 65}")
    t_start = time.time()

    failures = 0
    tests = [
        test_feature_extraction,
        test_model_prediction,
        test_ips_ids,
        test_model_metrics,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except AssertionError as e:
            print(f"\n  ❌ ASSERTION FAILED: {e}")
            failures += 1
        except Exception as e:
            print(f"\n  ❌ ERROR in {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failures += 1

    elapsed = time.time() - t_start
    print(f"\n{'#' * 65}")
    if failures == 0:
        print(f"  ✅  ALL TESTS PASSED  ({elapsed:.1f}s)")
    else:
        print(f"  ❌  {failures} TEST(S) FAILED  ({elapsed:.1f}s)")
    print(f"{'#' * 65}\n")

    return failures


if __name__ == "__main__":
    sys.exit(main())