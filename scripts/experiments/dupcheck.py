import warnings; warnings.filterwarnings("ignore")
import pandas as pd, os, collections
from PIL import Image
import imagehash
df=pd.read_csv("data/extracted/train_labels.csv"); df["id"]=df["id"].astype(str)
idx={p:os.path.join(r,p) for r,_,fs in os.walk("data/extracted/train") for p in fs}
def ph(p):
    try: return imagehash.phash(Image.open(p))
    except: return None
print("hashing ALL train (69k)...", flush=True)
trdict=collections.defaultdict(list)   # phash-str -> [labels]
trlist=[]
for _,x in df.iterrows():
    p=idx.get(x["id"]+".jpeg")
    if p:
        h=ph(p)
        if h is not None:
            trdict[str(h)].append(x["label"]); trlist.append((h,x["label"]))
print("train hashed:", len(trlist), flush=True)
pubdir="data/extracted/public_test/public_test"
pub=[f[:-5] for f in os.listdir(pubdir) if f.endswith(".jpeg")]
print("hashing public_test:", len(pub), flush=True)
exact=0; exlab=collections.Counter(); near=0; nearlab=collections.Counter()
import random; random.seed(0); trsample=random.sample(trlist, min(len(trlist),15000))
for i,s in enumerate(pub):
    h=ph(f"{pubdir}/{s}.jpeg")
    if h is None: continue
    k=str(h)
    if k in trdict:
        exact+=1; 
        for l in trdict[k]: exlab[l]+=1
    else:
        best=99; bl=None
        for th,l in trsample:
            d=h-th
            if d<best: best=d; bl=l
            if d<=2: break
        if best<=6: near+=1; nearlab[bl]+=1
print("=== RESULTADO ===")
print(f"EXACT phash dup public<->train: {exact}/{len(pub)} | labels: {dict(exlab)}")
print(f"NEAR dup (hamming<=6, vs muestra 15k): {near}/{len(pub)} | labels: {dict(nearlab)}")
