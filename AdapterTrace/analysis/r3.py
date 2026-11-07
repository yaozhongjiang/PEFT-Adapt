import numpy as np, glob
from scipy import stats
import os
A=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","raw")+"/"
N=2000; PER=16; E=5; STEPS=int(np.ceil(N/PER))*E; W0=int(np.ceil(STEPS*0.03))
def eta(pos):
    st=pos//PER; return np.where(st<W0,(st+1)/W0,np.maximum(0,(STEPS-st)/(STEPS-W0)))
def feats(pos):
    w=eta(pos); return {"w_O":w.sum(0),"eta_last":w[-1]}
def indep(pos):
    sig=[{np.argsort(e).tobytes() for e in p} for p in pos]; k=len(pos)
    return np.array([[i!=j and not(sig[i]&sig[j]) for j in range(k)] for i in range(k)])
z=np.load(A+"llama31_tofu_grid_palimpsest_loglik.npz"); P=z["positions"]; L=z["adapter_loglik"]; seeds=z["seeds"]
ind=indep(P); k=len(P)
# MC nulls: rank-correlation of random residual ranks with feature of a random order, for subset size m
rng=np.random.default_rng(123)
def mc_null(fname,m,B=200000):
    out=np.empty(B); c=0
    while c<B:
        b=min(2000,B-c)
        # random 5-epoch order
        pos=np.stack([np.stack([rng.permutation(N)+e*N for e in range(E)]) for _ in range(1)])[0]
        f=feats(pos)[fname]
        idx=rng.choice(N,m,replace=False) if m<N else np.arange(N)
        fr=stats.rankdata(f[idx]); fr=(fr-fr.mean())/fr.std()
        R=np.argsort(rng.random((b,m)),axis=1).argsort(axis=1).astype(float); R=(R-R.mean(1,keepdims=True))/R.std(1,keepdims=True)
        out[c:c+b]=(R@fr)/m; c+=b
    return np.sort(out)
NULL={}
def pval(rho,fname,m):
    key=(fname,m)
    if key not in NULL: NULL[key]=mc_null(fname,m)
    nl=NULL[key]; return (1+len(nl)-np.searchsorted(nl,rho,side="left"))/(len(nl)+1)
F=[feats(P[i]) for i in range(k)]
def test(ll,i,ref,idx=None,fname="w_O"):
    f=F[i][fname] if idx is None else F[i][fname][idx]
    r=stats.spearmanr(ll-ref,f).statistic; return r,pval(r,fname,len(f))
for fname in ["w_O","eta_last"]:
    print("=== feature",fname)
    # owners & null siblings, full n
    det=[test(L[i],i,L[[l for l in range(k) if ind[i,l]]].mean(0),fname=fname)[1] for i in range(k)]
    nul=[test(L[j],i,L[[l for l in range(k) if ind[i,l] and ind[j,l] and l!=j]].mean(0),fname=fname)[1] for i in range(k) for j in range(k) if ind[i,j]]
    nul=np.array(nul); print(f"owners det@1e-3 {np.mean(np.array(det)<1e-3):.2f}; siblings {len(nul)} pairs FPR@.05={np.mean(nul<.05):.3f} @1e-3={np.mean(nul<1e-3):.4f}")
    # query budget
    for m in [250,500,1000]:
        h=[]
        for i in range(k):
            ref=L[[l for l in range(k) if ind[i,l]]].mean(0)
            for _ in range(20):
                idx=rng.choice(N,m,replace=False); h.append(test(L[i][idx],i,ref[idx],idx,fname)[1]<1e-3)
        print(f"  owners det@1e-3 with {m} queries: {np.mean(h):.2f}")
    # distillation (keep 1000)
    for tag in ["ownerprompts","otherdata"]:
        d=np.load(A+f"llama31_tofu_grid_distill_{tag}_scores.npz"); keep=d["keep"]
        ps=[]
        for o,sc in zip(d["owners"],d["scores"]):
            i=int(np.where(seeds==o)[0][0]); ref=L[[l for l in range(k) if ind[i,l]]][:,keep].mean(0)
            ps.append(test(sc,i,ref,keep,fname)[1])
        ps=np.array(ps); print(f"  distill {tag}: @.05 {np.mean(ps<.05):.2f} @1e-3 {np.mean(ps<1e-3):.2f}  p={np.round(ps,5)}")
    # continued FT, unmodified shadows
    for tag in ["thief","h_thief"]:
        fs=sorted(glob.glob(A+f"llama31_tofu_grid_contft_{tag}_[ab].npz"))
        idxs=np.concatenate([np.load(f)["indices"] for f in fs]); ll=np.concatenate([np.load(f)["ll"] for f in fs])
        steps=np.load(fs[0])["steps"]
        for s_i,s in enumerate(steps):
            ps=np.array([test(ll[c,s_i],int(i),L[[l for l in range(k) if ind[int(i),l]]].mean(0),fname=fname)[1] for c,i in enumerate(idxs)])
            print(f"  contft {tag} {s} steps: @.05 {int((ps<.05).sum())}/{len(ps)} @1e-3 {int((ps<1e-3).sum())}/{len(ps)}")
