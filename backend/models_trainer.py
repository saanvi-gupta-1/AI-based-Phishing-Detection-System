"""
model_trainer.py — Cognitive Firewall v3
Trains on combined.csv. Produces: ensemble.pkl, scaler.pkl, keep_mask.npy,
active_features.json, thresholds.json, metrics.json, feature_importance.csv
"""
import os,sys,json,time,warnings
import numpy as np, pandas as pd, joblib

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, VotingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score,accuracy_score,roc_auc_score,
                              classification_report,confusion_matrix,
                              precision_recall_curve)

warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
DATA=os.path.join(HERE,"..","data")
MDL =os.path.join(HERE,"..","models")
os.makedirs(MDL,exist_ok=True)

from feature_extraction import extract_features_batch, FEATURE_NAMES, NUM_FEATURES

# ─── Trusted whitelist (Layer-1 bypass) ───────────────────────────────────────
import re
_TWO={"co.in","co.uk","com.au","org.in","net.in","gov.in","ac.in","nic.in"}
_TRUSTED={
    "claude.ai","anthropic.com","openai.com","chat.openai.com",
    "gemini.google.com","copilot.microsoft.com","huggingface.co","antigravity.google",
    "google.com","google.co.in","gmail.com","youtube.com","maps.google.com",
    "microsoft.com","outlook.com","office.com","github.com","gitlab.com",
    "stackoverflow.com","apple.com","amazon.com","amazon.in","linkedin.com",
    "twitter.com","x.com","facebook.com","instagram.com","whatsapp.com",
    "netflix.com","spotify.com","wikipedia.org","medium.com","reddit.com",
    "cloudflare.com","stripe.com","paypal.com",
    "sbi.co.in","onlinesbi.sbi","hdfcbank.com","icicibank.com","axisbank.com",
    "kotakbank.com","kotak.com","bankofbaroda.in","canarabank.in","pnbindia.in",
    "paytm.com","phonepe.com","razorpay.com","zerodha.com","groww.in",
    "flipkart.com","myntra.com","swiggy.com","zomato.com","makemytrip.com",
    "airtel.in","jio.com","vi.in","bsnl.co.in",
    "gov.in","india.gov.in","nic.in","irctc.co.in","irctc.com",
    "incometax.gov.in","uidai.gov.in","sebi.gov.in","rbi.org.in","npci.org.in",
    "bseindia.com","nseindia.com","tcs.com","infosys.com","wipro.com","zoho.com",
}

def _apex(url):
    url=re.sub(r"^https?://","",str(url).lower().strip())
    url=url.split("/")[0].split("?")[0].split(":")[0].split("@")[-1]
    pts=url.split(".")
    if len(pts)>=3 and ".".join(pts[-2:]) in _TWO: return ".".join(pts[-3:])
    return ".".join(pts[-2:]) if len(pts)>=2 else url

def is_trusted(url):
    a=_apex(url)
    return a in _TRUSTED or any(a==t or a.endswith("."+t) for t in _TRUSTED)


# ─── Data ─────────────────────────────────────────────────────────────────────
def load(csv):
    df=pd.read_csv(csv)[["url","label"]].dropna()
    df["url"]=df["url"].astype(str).str.strip()
    df["label"]=df["label"].astype(int)
    return df


# ─── Threshold tuning ─────────────────────────────────────────────────────────
def tune_thr(y_true, y_prob):
    # Maximise F1 on safe class (minority) to reduce false positives
    prec,rec,thr=precision_recall_curve(1-y_true, 1-y_prob)
    f1s=2*prec*rec/(prec+rec+1e-9)
    best=thr[np.argmax(f1s)] if len(thr) else 0.5
    return float(np.clip(best,0.20,0.80))


# ─── Oversample safe class ─────────────────────────────────────────────────────
def oversample(X,y,target=0.30):
    n_mal=(y==1).sum(); n_safe=(y==0).sum()
    need=int(n_mal*target/(1-target))
    if need<=n_safe: return X,y
    extra=need-n_safe
    idx=np.where(y==0)[0]
    ch=np.random.choice(idx,size=extra,replace=True)
    Xe=X[ch]+np.random.normal(0,0.05,(extra,X.shape[1]))
    perm=np.random.permutation(len(y)+extra)
    return np.vstack([X,Xe])[perm], np.concatenate([y,np.zeros(extra,int)])[perm]


# ─── Train ────────────────────────────────────────────────────────────────────
def train(csv, seed=42):
    np.random.seed(seed)
    print("\n"+"="*68)
    print("  COGNITIVE FIREWALL v3 — TRAINING")
    print("="*68)

    df=load(csv)
    n0=(df["label"]==0).sum(); n1=(df["label"]==1).sum()
    print(f"  Safe: {n0}  Malicious: {n1}  Total: {len(df)}")

    tr,te=train_test_split(df,test_size=0.20,stratify=df["label"],random_state=seed)
    print(f"  Train: {len(tr)}  Test: {len(te)}")

    print("\n  Extracting features...")
    Xtr=extract_features_batch(tr["url"].tolist()).fillna(0)
    Xte=extract_features_batch(te["url"].tolist()).fillna(0)
    for c in FEATURE_NAMES:
        if c not in Xtr.columns: Xtr[c]=0
        if c not in Xte.columns: Xte[c]=0
    Xtr=Xtr[FEATURE_NAMES].values.astype(float)
    Xte=Xte[FEATURE_NAMES].values.astype(float)
    ytr=tr["label"].values; yte=te["label"].values

    # Remove near-zero-variance features
    var=Xtr.var(axis=0)
    mask=var>0.001
    kept=[FEATURE_NAMES[i] for i,k in enumerate(mask) if k]
    Xtr_s=Xtr[:,mask]; Xte_s=Xte[:,mask]
    print(f"  Features after variance filter: {mask.sum()}/{len(FEATURE_NAMES)}")

    scaler=StandardScaler()
    Xtr_sc=scaler.fit_transform(Xtr_s)
    Xte_sc=scaler.transform(Xte_s)

    Xtr_aug,ytr_aug=oversample(Xtr_sc,ytr,0.30)
    n_s=(ytr_aug==0).sum(); n_m=(ytr_aug==1).sum()
    print(f"  After oversample — Safe: {n_s}  Malicious: {n_m}")

    cw={0:4.0,1:1.0}
    clfs={
        "RF":  RandomForestClassifier(n_estimators=400,max_depth=20,
                 min_samples_leaf=2,max_features="sqrt",
                 class_weight=cw,random_state=seed,n_jobs=-1),
        "ET":  ExtraTreesClassifier(n_estimators=400,max_depth=20,
                 min_samples_leaf=2,max_features="sqrt",
                 class_weight=cw,random_state=seed,n_jobs=-1),
        "GB":  GradientBoostingClassifier(n_estimators=200,max_depth=6,
                 learning_rate=0.1,subsample=0.8,random_state=seed),
    }

    trained={}
    print("\n  Training classifiers:")
    for name,clf in clfs.items():
        t0=time.time()
        clf.fit(Xtr_aug,ytr_aug)
        prob=clf.predict_proba(Xte_sc)[:,1]
        thr=tune_thr(yte,prob)
        pred=(prob>=thr).astype(int)
        print(f"    {name}  acc={accuracy_score(yte,pred):.4f}"
              f"  F1w={f1_score(yte,pred,average='weighted'):.4f}"
              f"  AUC={roc_auc_score(yte,prob):.4f}"
              f"  thr={thr:.2f}  ({time.time()-t0:.0f}s)")
        trained[name]={"clf":clf,"thr":thr,"prob":prob}

    print("\n  Building voting ensemble...")
    vc=VotingClassifier([("rf",clfs["RF"]),("et",clfs["ET"]),("gb",clfs["GB"])],voting="soft")
    vc.fit(Xtr_aug,ytr_aug)
    ep=vc.predict_proba(Xte_sc)[:,1]
    ethr=tune_thr(yte,ep)
    epred=(ep>=ethr).astype(int)
    eacc=accuracy_score(yte,epred)
    ef1w=f1_score(yte,epred,average="weighted")
    eauc=roc_auc_score(yte,ep)
    print(f"    Ensemble  acc={eacc:.4f}  F1w={ef1w:.4f}  AUC={eauc:.4f}  thr={ethr:.2f}")

    print("\n"+classification_report(yte,epred,target_names=["safe","malicious"]))
    cm=confusion_matrix(yte,epred)
    print(f"  Confusion Matrix:")
    print(f"              Pred Safe  Pred Malicious")
    print(f"  Actual Safe     {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Actual Mal      {cm[1][0]:>5}          {cm[1][1]:>5}")

    # Save
    joblib.dump(vc,     os.path.join(MDL,"ensemble.pkl"))
    joblib.dump(scaler, os.path.join(MDL,"scaler.pkl"))
    np.save(os.path.join(MDL,"keep_mask.npy"), mask)
    with open(os.path.join(MDL,"active_features.json"),"w") as f:
        json.dump(kept,f,indent=2)
    thrs={"Ensemble":ethr,**{n:trained[n]["thr"] for n in trained}}
    with open(os.path.join(MDL,"thresholds.json"),"w") as f:
        json.dump(thrs,f,indent=2)
    with open(os.path.join(MDL,"feature_names.json"),"w") as f:
        json.dump(FEATURE_NAMES,f,indent=2)
    metrics={"Ensemble":{"accuracy":round(eacc,4),"f1_weighted":round(ef1w,4),
             "auc_roc":round(eauc,4),"threshold":round(ethr,3),
             "n_features":int(mask.sum()),"n_train":len(tr),"n_test":len(te)}}
    with open(os.path.join(MDL,"metrics.json"),"w") as f:
        json.dump(metrics,f,indent=2)

    # Feature importance
    try:
        imp=clfs["RF"].feature_importances_
        pd.DataFrame({"feature":kept,"importance":imp})\
          .sort_values("importance",ascending=False)\
          .to_csv(os.path.join(MDL,"feature_importance.csv"),index=False)
    except: pass

    with open(os.path.join(MDL,"trusted_domains.json"),"w") as f:
        json.dump(sorted(_TRUSTED),f,indent=2)

    print(f"\n  ✅ Artifacts saved → {MDL}")
    print(f"  Ensemble F1w={ef1w:.4f}  AUC={eauc:.4f}  Threshold={ethr:.3f}")
    return metrics


# ─── Inference engine ─────────────────────────────────────────────────────────
class PhishingDetector:
    """Two-layer detector: whitelist bypass + ML ensemble."""

    def __init__(self, models_dir=MDL):
        self.ensemble  = joblib.load(os.path.join(models_dir,"ensemble.pkl"))
        self.scaler    = joblib.load(os.path.join(models_dir,"scaler.pkl"))
        self.keep_mask = np.load(os.path.join(models_dir,"keep_mask.npy"))
        with open(os.path.join(models_dir,"active_features.json")) as f:
            self.active = json.load(f)
        with open(os.path.join(models_dir,"thresholds.json")) as f:
            self.thr = json.load(f).get("Ensemble",0.5)
        td=os.path.join(models_dir,"trusted_domains.json")
        self._td=set(json.load(open(td))) if os.path.exists(td) else _TRUSTED

    def predict_one(self,url): return self.predict([url])[0]

    def predict(self,urls):
        results=[None]*len(urls)
        ml_urls=[]; ml_idx=[]
        for i,url in enumerate(urls):
            if is_trusted(url):
                results[i]={"label":0,"confidence":0.02,
                             "reason":"trusted_whitelist","features":{}}
            else:
                ml_urls.append(url); ml_idx.append(i)
        if ml_urls:
            fd=extract_features_batch(ml_urls)
            for c in FEATURE_NAMES:
                if c not in fd.columns: fd[c]=0
            fd=fd[FEATURE_NAMES].fillna(0)
            X=fd.values.astype(float)[:,self.keep_mask]
            X=self.scaler.transform(X)
            probs=self.ensemble.predict_proba(X)[:,1]
            for j,(url,prob) in enumerate(zip(ml_urls,probs)):
                results[ml_idx[j]]={
                    "label":int(prob>=self.thr),
                    "confidence":round(float(prob),4),
                    "reason":"ml_model",
                    "features":fd.iloc[j].to_dict(),
                }
        return results


if __name__=="__main__":
    csv=os.path.join(DATA,"combined.csv")
    if not os.path.exists(csv):
        print("Run prepare_data.py first"); sys.exit(1)
    train(csv)