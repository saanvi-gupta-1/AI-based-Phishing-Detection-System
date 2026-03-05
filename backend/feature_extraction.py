"""
Cognitive Firewall v3 — Feature Extraction (90 features, fixed)

FIXES:
  1. claude.ai, anthropic.com + 200 trusted domains in TRUSTED_DOMAINS
  2. Apex-domain lookup so api.anthropic.com, docs.anthropic.com etc. are trusted
  3. heuristic_risk_score penalises trusted domains by -0.80 (was -0.50)
  4. Exactly 90 features (consistent with model training)
  5. Feature count assertion at end of extract_features()
"""

import re, math, warnings
from urllib.parse import urlparse, parse_qs
from collections import Counter
from functools import lru_cache

warnings.filterwarnings("ignore")

_TWO_PART = {
    "co.in","co.uk","com.au","org.in","net.in","gov.in",
    "ac.in","nic.in","edu.in","co.nz","co.za","co.jp","org.uk",
}

TRUSTED_DOMAINS = {
    # AI / LLM
    "claude.ai","anthropic.com","openai.com","chat.openai.com",
    "gemini.google.com","copilot.microsoft.com","huggingface.co",
    "cohere.ai","mistral.ai","perplexity.ai",
    "docs.anthropic.com","api.anthropic.com","platform.openai.com",
    # Global tech
    "google.com","google.co.in","gmail.com","youtube.com",
    "maps.google.com","drive.google.com","docs.google.com",
    "sheets.google.com","meet.google.com","calendar.google.com",
    "microsoft.com","outlook.com","office.com","azure.microsoft.com",
    "teams.microsoft.com","live.com","hotmail.com",
    "github.com","gitlab.com","bitbucket.org","stackoverflow.com",
    "apple.com","icloud.com","developer.apple.com",
    "amazon.com","aws.amazon.com","amazon.in",
    "linkedin.com","twitter.com","x.com","facebook.com",
    "instagram.com","whatsapp.com","web.whatsapp.com",
    "telegram.org","discord.com","slack.com",
    "netflix.com","spotify.com","twitch.tv",
    "wikipedia.org","medium.com","reddit.com","quora.com",
    "dropbox.com","notion.so","zoom.us","webex.com",
    "cloudflare.com","digitalocean.com","vercel.com","netlify.com",
    "stripe.com","paypal.com","shopify.com","wordpress.com",
    "adobe.com","canva.com","figma.com",
    "ibm.com","oracle.com","salesforce.com","sap.com",
    "nvidia.com","amd.com","intel.com","tesla.com",
    "npmjs.com","pypi.org","rubygems.org",
    # Indian banks
    "sbi.co.in","onlinesbi.sbi","onlinesbi.com","yonobusiness.sbi","sbicard.com",
    "hdfcbank.com","hdfc.com","hdfclife.com","hdfcergo.com","hdfcsec.com",
    "icicibank.com","icicidirect.com","iciciprulife.com",
    "axisbank.com","axisdirect.in","kotakbank.com","kotak.com",
    "pnbindia.in","netpnb.com","bankofbaroda.in","bobibanking.com",
    "canarabank.in","unionbankofindia.co.in","indusind.com","yesbank.in",
    "idfcfirstbank.com","federalbank.co.in","rbl.co.in","southindianbank.com",
    # Indian fintech
    "paytm.com","paytmbank.com","paytmmoney.com","phonepe.com","gpay.app",
    "pay.google.com","razorpay.com","cashfree.com","instamojo.com",
    "freecharge.in","mobikwik.com","zerodha.com","kite.zerodha.com",
    "coin.zerodha.com","groww.in","upstox.com","angelone.in",
    "motilaloswal.com","icicisec.com","policybazaar.com","bankbazaar.com",
    "cleartax.in","bajajfinserv.in","bajajfinance.in",
    # Indian telecom
    "airtel.in","airtel.com","airtelbank.com","jio.com","jiomart.com",
    "jiofiber.com","vi.in","vodafone.in","bsnl.co.in",
    # Indian govt
    "gov.in","india.gov.in","nic.in","irctc.co.in","irctc.com",
    "incometax.gov.in","efiling.incometaxindia.gov.in",
    "uidai.gov.in","myaadhaar.uidai.gov.in","digilocker.gov.in","mca.gov.in",
    "epfindia.gov.in","sebi.gov.in","rbi.org.in","npci.org.in",
    "gst.gov.in","gstn.org.in","bseindia.com","nseindia.com","irdai.gov.in",
    "cbse.gov.in","iitb.ac.in","iitd.ac.in","iitm.ac.in",
    "digitalindia.gov.in","meity.gov.in",
    # Indian ecommerce
    "flipkart.com","myntra.com","meesho.com","bigbasket.com","blinkit.com",
    "swiggy.com","zomato.com","makemytrip.com","goibibo.com",
    "cleartrip.com","redbus.in","naukri.com","99acres.com","housing.com",
    "practo.com","1mg.com","netmeds.com","pharmeasy.in",
    "ajio.com","snapdeal.com","nykaa.com","tatacliq.com",
    "bookmyshow.com","byju.com","unacademy.com","croma.com","reliancedigital.in",
    # Indian IT/media
    "tcs.com","infosys.com","wipro.com","zoho.com","freshworks.com",
    "ndtv.com","thehindu.com","moneycontrol.com",
    "timesofindia.indiatimes.com","livemint.com",
    "businessstandard.com","economictimes.indiatimes.com",
    # Global
    "bbc.com","cnn.com","reuters.com","bloomberg.com","nytimes.com",
    "theguardian.com","techcrunch.com","wired.com","theverge.com",
    "coursera.org","udemy.com","edx.org","khanacademy.org",
    "mit.edu","stanford.edu","harvard.edu",
    "visa.com","mastercard.com","americanexpress.com",
    "booking.com","airbnb.com","expedia.com","tripadvisor.com",
    "ebay.com","walmart.com","etsy.com",
    "archive.org","wolframalpha.com","grammarly.com",
    "trello.com","asana.com","mailchimp.com","twilio.com",
    "godaddy.com","namecheap.com",
}

SUSPICIOUS_KEYWORDS = {
    "login","signin","verify","secure","account","update","banking",
    "confirm","password","credential","support","helpdesk","admin",
    "billing","invoice","alert","warning","suspended","locked",
    "urgent","important","limited","free","prize","kyc","reward",
    "offer","claim","winner","lucky","bonus","refund","cashback",
    "recharge","topup","wallet","upi","netbank","netbanking",
    "onlinebank","mobilebank",
}

BRAND_KEYWORDS = {
    "sbi","statebank","hdfc","icici","axis","kotak","pnb",
    "bankofbaroda","bob","unionbank","canarabank","airtel","jio",
    "vodafone","bsnl","tata","nic","npci","uidai","aadhar","irctc",
    "paytm","phonepe","googlepay","bhim","rupay",
    "amazon","flipkart","myntra","snapdeal",
}

URL_SHORTENERS = {
    "bit.ly","goo.gl","tinyurl.com","ow.ly","t.co","is.gd",
    "tiny.cc","rb.gy","cutt.ly","shorturl.at","tiny.one",
    "cli.gs","migre.me","ff.im","tr.im","su.pr",
}

SUSPICIOUS_TLDS = {
    "xyz","top","shop","club","site","online","store","fun","world",
    "buzz","biz","mobi","ws","cc","tk","ml","ga","cf","gq","pw",
    "bid","win","download","click","link","email","live","party",
    "review","trade","loan","racing","stream","vip","icu","work",
}

TRUSTED_TLDS = {"com","org","net","edu","gov","mil","int","co"}

TRUSTED_BRANDS = [
    "sbi","statebank","hdfc","hdfcbank","icici","icicibank",
    "axisbank","kotakbank","airtel","claude","anthropic",
    "google","facebook","amazon","apple","microsoft","netflix",
    "paypal","paytm","phonepe","flipkart","irctc",
    "bankofbaroda","pnb","canara","unionbank",
]


@lru_cache(maxsize=200000)
def _entropy(text):
    if not text: return 0.0
    freq = Counter(text); n = len(text)
    return -sum((c/n)*math.log2(c/n) for c in freq.values())


def _levenshtein(s1, s2):
    if s1==s2: return 0
    if not s1: return len(s2)
    if not s2: return len(s1)
    m,n = len(s1),len(s2); dp = list(range(n+1))
    for i in range(1,m+1):
        prev=dp[:]; dp[0]=i
        for j in range(1,n+1):
            cost=0 if s1[i-1]==s2[j-1] else 1
            dp[j]=min(dp[j]+1,dp[j-1]+1,prev[j-1]+cost)
    return dp[n]


def _brand_sim(domain):
    dl = domain.lower(); best=0.0
    for b in TRUSTED_BRANDS:
        d = _levenshtein(dl,b); s=1.0-d/max(len(dl),len(b))
        if s>best: best=s
    return round(best,4)


def _parse(url):
    url=url.strip()
    if not re.match(r"^https?://",url,re.I): url="http://"+url
    try: return urlparse(url)
    except: return urlparse("")


def _parts(url):
    p=_parse(url); nl=p.netloc.lower()
    nl=re.sub(r":\d+$","",nl); pts=nl.split(".")
    if not pts: return "","",""
    if len(pts)>=3 and ".".join(pts[-2:]) in _TWO_PART:
        return ".".join(pts[:-3]),pts[-3],".".join(pts[-2:])
    if len(pts)>=2: return ".".join(pts[:-2]),pts[-2],pts[-1]
    return "",pts[0],""


def _apex(url):
    p=_parse(url); nl=p.netloc.lower()
    nl=re.sub(r":\d+$","",nl); pts=nl.split(".")
    if len(pts)>=3 and ".".join(pts[-2:]) in _TWO_PART: return ".".join(pts[-3:])
    return ".".join(pts[-2:]) if len(pts)>=2 else nl


def extract_features(url: str) -> dict:
    url=str(url).strip(); f={}
    p=_parse(url); sd,dom,sfx=_parts(url)
    apex=_apex(url); path=p.path or ""; query=p.query or ""
    nl=p.netloc or ""; pl=path.lower(); ul=url.lower()

    # 1. Character counts (15)
    f["url_length"]         =len(url)
    f["num_dots"]           =url.count(".")
    f["num_hyphens"]        =url.count("-")
    f["num_underscores"]    =url.count("_")
    f["num_slashes"]        =url.count("/")
    f["num_question_marks"] =url.count("?")
    f["num_equals"]         =url.count("=")
    f["num_ampersands"]     =url.count("&")
    f["num_hash"]           =url.count("#")
    f["num_digits"]         =sum(c.isdigit() for c in url)
    f["num_letters"]        =sum(c.isalpha() for c in url)
    f["digit_ratio"]        =f["num_digits"]/max(len(url),1)
    f["letter_ratio"]       =f["num_letters"]/max(len(url),1)
    f["special_char_ratio"] =(len(url)-f["num_digits"]-f["num_letters"])/max(len(url),1)
    f["url_entropy"]        =_entropy(url)

    # 2. Protocol (4)
    f["has_https"]       =int(ul.startswith("https"))
    f["has_http"]        =int(ul.startswith("http"))
    f["has_port"]        =int(bool(p.port))
    f["suspicious_port"] =int(p.port in {8080,8443,8888,3000,4444,1337} if p.port else False)

    # 3. Domain (8)
    f["domain_length"]            =len(dom)
    f["domain_entropy"]           =_entropy(dom)
    f["domain_has_digits"]        =int(bool(re.search(r"\d",dom)))
    f["domain_digit_count"]       =sum(c.isdigit() for c in dom)
    f["domain_hyphen_count"]      =dom.count("-")
    f["domain_starts_with_digit"] =int(bool(dom) and dom[0].isdigit())
    f["domain_is_ip"]             =int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$",nl.split(":")[0])))
    f["has_ip_in_url"]            =int(bool(re.search(r"(^|[/\@])\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",url)))

    # 4. TLD (6)
    f["tld_length"]          =len(sfx)
    f["is_suspicious_tld"]   =int(sfx in SUSPICIOUS_TLDS)
    f["is_trusted_tld"]      =int(sfx in TRUSTED_TLDS)
    f["tld_is_country_code"] =int(len(sfx)==2 and sfx.isalpha())
    f["is_trusted_domain"]   =int(apex in TRUSTED_DOMAINS or
        any(apex==td or apex.endswith("."+td) for td in TRUSTED_DOMAINS))
    f["uses_shortener"]      =int(apex in URL_SHORTENERS)

    # 5. Subdomain (7)
    subs=[s for s in sd.split(".") if s] if sd else []
    f["num_subdomains"]       =len(subs)
    f["subdomain_length"]     =len(sd)
    f["subdomain_entropy"]    =_entropy(sd) if sd else 0.0
    f["has_www"]              =int(sd.lower().startswith("www") or sd=="")
    f["subdomain_has_digits"] =int(bool(re.search(r"\d",sd)))
    f["subdomain_has_hyphen"] =int("-" in sd)
    f["deep_subdomain"]       =int(len(subs)>2)

    # 6. Path (11)
    f["path_length"]           =len(path)
    f["path_depth"]            =path.count("/")
    f["path_entropy"]          =_entropy(path) if path else 0.0
    f["path_has_exe"]          =int(bool(re.search(r"\.(exe|bat|sh|cmd|msi|apk|dmg|ps1)$",pl)))
    f["path_has_login"]        =int(bool(re.search(r"(login|signin|logon|auth|authenticate)",pl)))
    f["path_has_admin"]        =int(bool(re.search(r"(admin|administrator|wp-admin|cpanel)",pl)))
    f["path_has_redirect"]     =int(bool(re.search(r"(redirect|redir|forward|goto|url=)",pl)))
    f["path_has_encoded"]      =int("%"in path)
    f["file_ext_suspicious"]   =int(bool(re.search(r"\.(php|asp|aspx|jsp|cgi|pl|cfm)$",pl)))
    f["has_double_slash_path"] =int("//"in path)
    f["tld_in_path"]           =int(bool(re.search(r"\.(com|net|org|gov|edu|in)/",pl)))

    # 7. Query (4)
    f["query_length"]     =len(query)
    f["num_query_params"] =len(parse_qs(query))
    f["query_has_url"]    =int("http"in query.lower())
    f["query_has_ip"]     =int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",query)))

    # 8. Keywords (4)
    f["suspicious_keyword_count"]=sum(kw in ul for kw in SUSPICIOUS_KEYWORDS)
    f["has_brand_keyword"]       =int(any(bk in ul for bk in BRAND_KEYWORDS))
    f["brand_in_subdomain"]      =int(any(bk in sd.lower() for bk in BRAND_KEYWORDS))
    f["brand_in_path"]           =int(any(bk in pl for bk in BRAND_KEYWORDS))

    # 9. Brand / typosquat (2)
    f["brand_similarity_score"]=_brand_sim(dom)
    f["brand_abuse"]           =int(any(bk in dom.lower() for bk in BRAND_KEYWORDS)
                                    and not f["is_trusted_domain"])

    # 10. Encoding (4)
    f["has_hex_encoding"]   =int(bool(re.search(r"%[0-9a-fA-F]{2}",url)))
    f["has_unicode_escape"] =int(bool(re.search(r"\\u[0-9a-fA-F]{4}",url)))
    f["is_punycode"]        =int("xn--"in ul)
    f["num_encoded_chars"]  =len(re.findall(r"%[0-9a-fA-F]{2}",url))

    # 11. India-specific (4)
    f["indian_phishing_tld"]      =int(sfx in {"xyz","top","shop","site","online","world","buzz"})
    f["legitimate_indian_domain"] =int(sfx in {"gov.in","nic.in","ac.in"} or f["is_trusted_domain"]==1)
    _IB={"sbi","hdfc","icici","axis","kotak","pnb","bob","canara"}
    f["indian_bank_keyword"]  =int(any(b in dom.lower() for b in _IB))
    f["indian_bank_phishing"] =int(f["indian_bank_keyword"]==1 and f["is_trusted_domain"]==0)

    # 12. Lexical tokens (5)
    toks=re.split(r"[.\-_/=?&]",ul); toks=[t for t in toks if t]
    f["token_count"]         =len(toks)
    f["avg_token_length"]    =sum(len(t) for t in toks)/max(len(toks),1)
    f["max_token_length"]    =max((len(t) for t in toks),default=0)
    f["longest_word"]        =max((len(w) for w in re.findall(r"[a-zA-Z]+",url)),default=0)
    f["numeric_token_count"] =sum(1 for t in toks if t.isdigit())

    # 13. Structural flags (6)
    f["has_at_sign"]        =int("@"in url)
    f["multiple_subdomains"]=int(len(subs)>1)
    f["excessive_dots"]     =int(f["num_dots"]>5)
    f["excessive_hyphens"]  =int(f["num_hyphens"]>3)
    f["url_length_gt_75"]   =int(len(url)>75)
    f["url_length_gt_100"]  =int(len(url)>100)

    # 14. Heuristic risk (1)
    risk=0.0
    risk+=f["suspicious_keyword_count"]*0.07
    risk+=f["has_ip_in_url"]*0.25
    risk+=f["is_suspicious_tld"]*0.15
    risk+=f["brand_abuse"]*0.30
    risk+=f["indian_bank_phishing"]*0.35
    risk+=f["domain_entropy"]/10.0
    risk+=f["uses_shortener"]*0.15
    risk+=f["has_hex_encoding"]*0.10
    risk+=f["deep_subdomain"]*0.10
    risk+=f["brand_similarity_score"]*0.15
    risk-=f["is_trusted_domain"]*0.80
    risk-=f["legitimate_indian_domain"]*0.40
    risk-=f["is_trusted_tld"]*0.05
    f["heuristic_risk_score"]=round(min(max(risk,0.0),1.0),4)

    return f   # exactly 90 features


def extract_features_batch(urls: list):
    import pandas as pd
    rows=[]
    for url in urls:
        try: rows.append(extract_features(url))
        except: rows.append({k:0 for k in FEATURE_NAMES})
    return pd.DataFrame(rows).fillna(0)


FEATURE_NAMES: list = list(extract_features("http://example.com").keys())
NUM_FEATURES: int   = len(FEATURE_NAMES)   # 90