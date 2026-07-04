!pip install requests scikit-learn pandas numpy scipy -q
!pip install xgboost lightgbm catboost -q
!pip install transformers torch -q

#Phase 1: Registry & Version-Pair Retrieval Setup
import csv, json, logging, os, time, io, re, ast, math, tarfile, zipfile
from dataclasses import dataclass, asdict
from typing import Optional
import requests, numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pipeline")
NPM_URL="https://registry.npmjs.org/{p}"; PYPI_URL="https://pypi.org/pypi/{p}/json"
HEADERS={"User-Agent":"supply-chain-research/1.0 (academic)"}; DELAY=0.4

def _get(url):
    try:
        r=requests.get(url,headers=HEADERS,timeout=15); time.sleep(DELAY)
        return r.json() if r.status_code==200 else None
    except requests.RequestException: return None

def fetch_npm_metadata(p): return _get(NPM_URL.format(p=p))
def fetch_pypi_metadata(p): return _get(PYPI_URL.format(p=p))

def npm_versions(m):
    t=m.get("time",{}); vs=[v for v in m.get("versions",{}) if v not in ("created","modified") and v!="0.0.1-security"]
    pr=[(v,t.get(v)) for v in vs if t.get(v)]; pr.sort(key=lambda x:x[1]); return pr

def pypi_versions(m):
    pr=[]
    for v,files in m.get("releases",{}).items():
        if not files: continue
        ut=files[0].get("upload_time_iso_8601") or files[0].get("upload_time")
        if ut: pr.append((v,ut))
    pr.sort(key=lambda x:x[1]); return pr

print("Phase 1 setup ready.")

#Phase 2: Feature Extraction Setup
SUSPICIOUS = {
  "net_http": re.compile(r"\b(https?://|requests\.(get|post)|urllib|fetch\(|XMLHttpRequest|http\.request|axios)", re.I),
  "subprocess_exec": re.compile(r"\b(subprocess|child_process|os\.system|exec\(|eval\(|spawn\(|popen)", re.I),
  "env_cred_access": re.compile(r"(process\.env|os\.environ|getenv|\.aws/credentials|id_rsa|\.npmrc|\.pypirc|SSH_|API_KEY|SECRET)", re.I),
  "filesystem": re.compile(r"\b(fs\.(write|read|unlink)|open\([^)]*['\"]w|shutil|os\.remove|writeFile)", re.I),
  "encoded_blob": re.compile(r"(base64|atob|btoa|fromCharCode|eval\(.*decode|exec\(.*decode)", re.I),
  "install_hook": re.compile(r"(preinstall|postinstall|setup\.py|cmdclass|__import__)", re.I),
  "obfuscation": re.compile(r"(_0x[0-9a-f]{4,}|[A-Za-z0-9+/]{120,})", re.I),
}

def entropy(s):
    if not s: return 0.0
    from collections import Counter as C; n=len(s)
    return -sum((c/n)*math.log2(c/n) for c in C(s).values())

def fetch_tarball_url(package,ecosystem,version):
    if ecosystem=="npm":
        m=fetch_npm_metadata(package)
        try: return m["versions"][version]["dist"]["tarball"]
        except (KeyError,TypeError): return None
    m=fetch_pypi_metadata(package)
    try:
        files=m["releases"].get(version,[])
        if not files: return None
        sd=[f for f in files if f["url"].endswith((".tar.gz",".zip"))]
        return (sd[0] if sd else files[0])["url"]
    except (KeyError,TypeError): return None

def download_bytes(url):
    if not url: return None
    try:
        r=requests.get(url,headers=HEADERS,timeout=30); time.sleep(DELAY)
        return r.content if r.status_code==200 else None
    except requests.RequestException: return None

def extract_code(archive,ecosystem):
    if not archive: return ""
    chunks=[]
    try:
        if ecosystem=="npm":
            with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as t:
                for mem in t.getmembers():
                    if mem.isfile() and mem.name.endswith((".js",".ts",".json",".mjs",".cjs")):
                        fo=t.extractfile(mem)
                        if fo: chunks.append(fo.read().decode("utf-8","ignore"))
        else:
            try:
                with tarfile.open(fileobj=io.BytesIO(archive),mode="r:gz") as t:
                    for mem in t.getmembers():
                        if mem.isfile() and mem.name.endswith((".py",".cfg",".toml",".txt")):
                            fo=t.extractfile(mem)
                            if fo: chunks.append(fo.read().decode("utf-8","ignore"))
            except tarfile.ReadError:
                with zipfile.ZipFile(io.BytesIO(archive)) as z:
                    for nm in z.namelist():
                        if nm.endswith((".py",".cfg",".toml",".txt")):
                            chunks.append(z.read(nm).decode("utf-8","ignore"))
    except Exception: return ""
    return "\n".join(chunks)

def pcounts(code): return {k:len(rx.findall(code)) for k,rx in SUSPICIOUS.items()}

def diff_features(cb,ca):
    bc,ac=pcounts(cb),pcounts(ca); f={}
    for k in SUSPICIOUS: f[f"delta_{k}"]=ac[k]-bc[k]; f[f"after_{k}"]=ac[k]
    f["delta_loc"]=ca.count("\n")-cb.count("\n"); f["delta_char"]=len(ca)-len(cb)
    f["entropy_before"]=round(entropy(cb[:50000]),3); f["entropy_after"]=round(entropy(ca[:50000]),3)
    f["delta_entropy"]=round(f["entropy_after"]-f["entropy_before"],3)
    lt=lambda c: max((len(t) for t in re.findall(r"\S+",c)),default=0)
    f["delta_longest_token"]=lt(ca)-lt(cb)
    try: f["ast_nodes_after"]=sum(1 for _ in ast.walk(ast.parse(ca)))
    except SyntaxError: f["ast_nodes_after"]=-1
    return f

MIN_CODE_CHARS=30

def features_for_pair(package,ecosystem,vbefore,vafter):
    ca=extract_code(download_bytes(fetch_tarball_url(package,ecosystem,vafter)),ecosystem)
    if not ca or len(ca.strip())<MIN_CODE_CHARS: return None,"after_download_failed"
    cb=extract_code(download_bytes(fetch_tarball_url(package,ecosystem,vbefore)),ecosystem) if vbefore else ""
    if not cb or len(cb.strip())<MIN_CODE_CHARS: return None,"before_download_failed"
    return diff_features(cb,ca),"ok"

print("Phase 2 setup ready.")

#Phase 3: Dataset Preparation & Ecosystem Balancing
usable_df = pd.read_csv("merged_dataset_usable.csv")
from collections import Counter
print("malicious usable by ecosystem:", dict(Counter(usable_df["ecosystem"])))

RECOVER_EXTRA_PYPI = True
extra_pypi = []
if RECOVER_EXTRA_PYPI:
    try:
        api="https://api.github.com/repos/lxyeternal/pypi_malregistry/contents/malware"
        r=requests.get(api,headers=HEADERS,timeout=20); time.sleep(0.3)
        if r.status_code==200:
            for entry in r.json():
                if entry["type"]=="dir":
                    extra_pypi.append(entry["name"])
            log.info(f"pypi_malregistry: found {len(extra_pypi)} candidate malicious PyPI package names")
        else:
            log.warning(f"pypi_malregistry index HTTP {r.status_code} - skipping recovery")
    except Exception as e:
        log.warning(f"pypi recovery failed: {e}")
print(f"extra PyPI candidates to try: {len(extra_pypi)}")

#Phase 4: Feature Matrix Construction
ONLY_BOTH_AVAILABLE=True
CHECKPOINT_EVERY=200
N_BENIGN_TARGET=1500

def get_top_npm(n):
    names=[]
    try:
        for off in range(0,n,250):
            u=f"https://registry.npmjs.org/-/v1/search?text=boost-exact:false&popularity=1.0&size=250&from={off}"
            r=requests.get(u,headers=HEADERS,timeout=20); time.sleep(0.3)
            if r.status_code!=200: break
            objs=r.json().get("objects",[])
            if not objs: break
            names+=[o["package"]["name"] for o in objs]
            if len(names)>=n: break
    except Exception as e: log.warning(f"npm top list: {e}")
    return list(dict.fromkeys(names))[:n]

def get_top_pypi(n):
    try:
        u="https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
        r=requests.get(u,headers=HEADERS,timeout=20); time.sleep(0.3)
        if r.status_code==200: return [row["project"] for row in r.json()["rows"][:n]]
    except Exception as e: log.warning(f"pypi top list: {e}")
    return []

if ONLY_BOTH_AVAILABLE and "download_status" in usable_df.columns:
    mal_df=usable_df[usable_df["download_status"]=="both_available"].reset_index(drop=True)
else:
    mal_df=usable_df.reset_index(drop=True)

rows=[]; done=set()
if os.path.exists("features_partial.csv"):
    prev=pd.read_csv("features_partial.csv"); rows=prev.to_dict("records")
    done=set((r["package"],r["ecosystem"]) for r in rows)

skipped={"after_download_failed":0,"before_download_failed":0,"error":0}

def add_checkpoint():
    pd.DataFrame(rows).to_csv("features_partial.csv",index=False)

proc=0
for _,r in mal_df.iterrows():
    if (r["package"],r["ecosystem"]) in done: continue
    try:
        feats,st=features_for_pair(r["package"],r["ecosystem"],r["version_before"],r["version_after"])
        if st!="ok": skipped[st]=skipped.get(st,0)+1; continue
        feats.update({"package":r["package"],"ecosystem":r["ecosystem"],
            "version_after_date":r.get("version_after_date"),"label":1}); rows.append(feats)
    except Exception as e: skipped["error"]+=1
    proc+=1
    if proc%CHECKPOINT_EVERY==0: add_checkpoint()

for pkg in (extra_pypi if 'extra_pypi' in dir() else []):
    if (pkg,"pypi") in done: continue
    try:
        vds=pypi_versions(fetch_pypi_metadata(pkg) or {})
        if len(vds)<2: continue
        feats,st=features_for_pair(pkg,"pypi",vds[-2][0],vds[-1][0])
        if st!="ok": continue
        feats.update({"package":pkg,"ecosystem":"pypi","version_after_date":vds[-1][1] or "2025-01-01T00:00:00Z","label":1})
        rows.append(feats)
    except Exception: pass
    proc+=1
    if proc%CHECKPOINT_EVERY==0: add_checkpoint()

half=N_BENIGN_TARGET//2
top_npm=get_top_npm(half); top_pypi=get_top_pypi(N_BENIGN_TARGET-half)

proc=0
for pkg in top_npm:
    if (pkg,"npm") in done: continue
    try:
        vds=npm_versions(fetch_npm_metadata(pkg) or {})
        if len(vds)<2: continue
        feats,st=features_for_pair(pkg,"npm",vds[-2][0],vds[-1][0])
        if st!="ok": continue
        feats.update({"package":pkg,"ecosystem":"npm","version_after_date":vds[-1][1] or "2026-06-01T00:00:00Z","label":0})
        rows.append(feats)
    except Exception: pass
    proc+=1
    if proc%CHECKPOINT_EVERY==0: add_checkpoint()

for pkg in top_pypi:
    if (pkg,"pypi") in done: continue
    try:
        vds=pypi_versions(fetch_pypi_metadata(pkg) or {})
        if len(vds)<2: continue
        feats,st=features_for_pair(pkg,"pypi",vds[-2][0],vds[-1][0])
        if st!="ok": continue
        feats.update({"package":pkg,"ecosystem":"pypi","version_after_date":vds[-1][1] or "2026-06-01T00:00:00Z","label":0})
        rows.append(feats)
    except Exception: pass
    proc+=1
    if proc%CHECKPOINT_EVERY==0: add_checkpoint()

feat_df=pd.DataFrame(rows); feat_df.to_csv("features.csv",index=False)
if os.path.exists("features_partial.csv"): os.remove("features_partial.csv")

#Phase 5: Semantic Layer (CodeBERT - Optional)
USE_CODEBERT=False
if USE_CODEBERT:
    from transformers import AutoTokenizer, AutoModel
    import torch
    tok=AutoTokenizer.from_pretrained("microsoft/codebert-base")
    cb=AutoModel.from_pretrained("microsoft/codebert-base"); cb.eval()
    def emb(text):
        if not text: return np.zeros(768)
        with torch.no_grad():
            i=tok(text[:512],return_tensors="pt",truncation=True,max_length=512)
            return cb(**i).last_hidden_state.mean(dim=1).squeeze().numpy()

#Phase 6: Model Panel Setup
import pandas as pd, numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score,recall_score,f1_score,roc_auc_score,accuracy_score,confusion_matrix)

feat_df=pd.read_csv("features.csv")
meta={"package","ecosystem","version_after_date","label"}
feature_names=[c for c in feat_df.columns if c not in meta]
Xraw=feat_df[feature_names].fillna(0).values.astype(float)
eco=(feat_df["ecosystem"]=="npm").astype(float).values.reshape(-1,1)
X=np.hstack([Xraw,eco]); feature_names_full=feature_names+["ecosystem_is_npm"]
y=feat_df["label"].values.astype(int)
Xs=StandardScaler().fit_transform(X)

def get_models():
    spw=float((y==0).sum())/max(int((y==1).sum()),1)
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    M={"LogisticRegression":LogisticRegression(max_iter=5000,class_weight="balanced"),
       "RandomForest":RandomForestClassifier(n_estimators=300,random_state=42,class_weight="balanced"),
       "GradientBoosting":GradientBoostingClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=42)}
    notes={}
    try:
        from xgboost import XGBClassifier
        M["XGBoost"]=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",random_state=42,scale_pos_weight=spw)
    except Exception as e: notes["XGBoost"]=str(e)
    try:
        from lightgbm import LGBMClassifier
        M["LightGBM"]=LGBMClassifier(n_estimators=300,learning_rate=0.05,subsample=0.8,random_state=42,verbose=-1,class_weight="balanced")
    except Exception as e: notes["LightGBM"]=str(e)
    try:
        from catboost import CatBoostClassifier
        M["CatBoost"]=CatBoostClassifier(iterations=300,depth=5,learning_rate=0.05,random_state=42,verbose=0,auto_class_weights="Balanced")
    except Exception as e: notes["CatBoost"]=str(e)
    return M,notes

def metric_row(yt,yp,proba=None):
    tn,fp,fn,tp=confusion_matrix(yt,yp,labels=[0,1]).ravel()
    auc=None
    if proba is not None:
        try: auc=roc_auc_score(yt,proba)
        except ValueError: auc=None
    return {"accuracy":accuracy_score(yt,yp),"precision":precision_score(yt,yp,zero_division=0),
        "recall":recall_score(yt,yp,zero_division=0),"f1":f1_score(yt,yp,zero_division=0),
        "auc_roc":auc,"fpr":fp/(fp+tn) if (fp+tn) else 0.0}

#Phase 7: Cross-Validation & Statistical Evaluation
from scipy import stats as st
models,notes=get_models()
min_class=int(np.bincount(y).min()); n_splits=max(2,min(5,min_class))
skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)

def ci95(vals):
    vals=[v for v in vals if v is not None]
    if len(vals)<2: return (np.mean(vals) if vals else float('nan'),0.0)
    m=np.mean(vals); se=st.sem(vals); h=se*1.96
    return m,h

per_model_f1={}
summary=[]
for name,model in models.items():
    folds={k:[] for k in ["accuracy","precision","recall","f1","auc_roc","fpr"]}
    for tr,te in skf.split(X,y):
        Xtr = Xs[tr] if name=="LogisticRegression" else X[tr]
        Xte = Xs[te] if name=="LogisticRegression" else X[te]
        try: m=clone(model)
        except Exception: m=model
        m.fit(Xtr,y[tr]); yp=m.predict(Xte)
        try: proba=m.predict_proba(Xte)[:,1]
        except Exception: proba=None
        mr=metric_row(y[te],yp,proba)
        for k in folds: folds[k].append(mr[k])
    per_model_f1[name]=folds["f1"]
    row={"model":name}
    for k in folds:
        m,h=ci95(folds[k]); row[k]=round(m,4); row[f"{k}_ci95"]=round(h,4)
    summary.append(row)

comp=pd.DataFrame(summary).set_index("model").sort_values("f1",ascending=False)
comp.to_csv("model_comparison_ci.csv")
best_model_name=comp.index[0]

#Phase 8: Significance Testing
top2=list(comp.index[:2])
if len(top2)==2:
    a,b=per_model_f1[top2[0]],per_model_f1[top2[1]]
    t,p=st.ttest_rel(a,b)

#Phase 9: Temporal Split (Generalization Test)
CUTOFF="2026-01-01"
dates=feat_df["version_after_date"].fillna("").astype(str).values
tr=[i for i,d in enumerate(dates) if d and d<CUTOFF]
te=[i for i,d in enumerate(dates) if d>=CUTOFF]

if len(tr)>10 and len(te)>10 and len(set(y[tr]))==2 and len(set(y[te]))==2:
    m=clone(models[best_model_name])
    useS = best_model_name=="LogisticRegression"
    m.fit((Xs[tr] if useS else X[tr]), y[tr])
    yp=m.predict(Xs[te] if useS else X[te])
    try: proba=m.predict_proba(Xs[te] if useS else X[te])[:,1]
    except Exception: proba=None
    tmetrics=metric_row(y[te],yp,proba)

#Phase 10: Ablation Study
groups={
  "static_signal_deltas":[c for c in feature_names if c.startswith("delta_") and c not in ("delta_loc","delta_char","delta_entropy","delta_longest_token")],
  "size_deltas":["delta_loc","delta_char","delta_longest_token"],
  "entropy":["entropy_before","entropy_after","delta_entropy"],
  "absolute_after_counts":[c for c in feature_names if c.startswith("after_")],
  "ast":["ast_nodes_after"],
}
def eval_subset(cols):
    idx=[feature_names_full.index(c) for c in cols if c in feature_names_full]
    Xsub=X[:,idx]; f1s=[]
    for tr,te in skf.split(Xsub,y):
        m=clone(models[best_model_name])
        m.fit(Xsub[tr],y[tr]); f1s.append(f1_score(y[te],m.predict(Xsub[te]),zero_division=0))
    return float(np.mean(f1s))

full_cols=list(feature_names_full)
base=eval_subset(full_cols)
abl=[{"config":"ALL features","f1":round(base,4),"drop":0.0}]
for g,cols in groups.items():
    keep=[c for c in full_cols if c not in cols]
    if not keep: continue
    s=eval_subset(keep)
    abl.append({"config":f"without {g}","f1":round(s,4),"drop":round(base-s,4)})
abl_df=pd.DataFrame(abl).sort_values("drop",ascending=False)
abl_df.to_csv("ablation.csv",index=False)

#Phase 11: Per-Ecosystem Evaluation
def eval_on_mask(mask):
    if mask.sum()<10 or len(set(y[mask]))<2: return None
    from sklearn.model_selection import StratifiedKFold as SKF
    idx=np.where(mask)[0]; ym=y[idx]; Xm=X[idx]
    if len(set(ym))<2 or np.bincount(ym).min()<2: return None
    sk=SKF(n_splits=min(3,np.bincount(ym).min()),shuffle=True,random_state=42)
    f1s=[];fprs=[]
    for tr,te in sk.split(Xm,ym):
        m=clone(models[best_model_name]); m.fit(Xm[tr],ym[tr]); yp=m.predict(Xm[te])
        mr=metric_row(ym[te],yp); f1s.append(mr["f1"]); fprs.append(mr["fpr"])
    return {"n":int(mask.sum()),"f1":round(float(np.mean(f1s)),4),"fpr":round(float(np.mean(fprs)),4)}

for eco_name in ["npm","pypi"]:
    mask=(feat_df["ecosystem"]==eco_name).values
    res=eval_on_mask(mask)

Phase 12: Download Results
from google.colab import files
for fn in ["features.csv","model_comparison_ci.csv","ablation.csv"]:
    if os.path.exists(fn): files.download(fn)


