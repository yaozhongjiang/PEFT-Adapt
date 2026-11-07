import numpy as np, sys
from scipy import stats
import os
A=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","raw")+"/"
def lrw(pos, per_step, epochs, n, warm=0.03, kind="linear"):
    steps=int(np.ceil(n/per_step))*epochs
    w0=int(np.ceil(steps*warm)); st=pos//per_step
    if kind=="linear":
        w=np.where(st<w0,(st+1)/max(w0,1),np.maximum(0,(steps-st)/max(steps-w0,1)))
    return w
def indep(pos):
    sig=[{np.argsort(e).tobytes() for e in p} for p in pos]
    k=len(pos); return np.array([[i!=j and not(sig[i]&sig[j]) for j in range(k)] for i in range(k)])
def tp(r,n):
    r=np.clip(r,-.999999,.999999); t=r*np.sqrt((n-2)/(1-r*r)); return stats.t.sf(t,n-2)
def run(name,epochs,budget,reps=50,seed=0):
    z=np.load(A+name+"_palimpsest_loglik.npz"); P=z["positions"]; L=z["adapter_loglik"]; B=z["base_loglik"]
    if P.ndim==2: P=P[:,None,:]
    k,E,n=P.shape; ind=indep(P)
    feats={}
    W=np.stack([lrw(P[i],16,epochs,n) for i in range(k)])  # k,E,n
    G=P+np.arange(E)[None,:,None]*n  # global position
    feats["position (mean, published)"]=G.mean(1)
    feats["last occurrence"]=G[:,-1]
    feats["-position (sign flip)"]=-G.mean(1)
    feats["-last occurrence"]=-G[:,-1]
    feats["-first occurrence"]=-G[:,0]
    feats["schedule weight w_O"]=W.sum(1)
    rng=np.random.default_rng(seed)
    out={}
    for ref in ["none","base","shadow"]:
        for fn,F in feats.items():
            if ref!="shadow" and fn not in ("position (mean, published)","schedule weight w_O"): continue
            rhos=[];det_full=0;det_b=[]
            for i in range(k):
                others=[l for l in range(k) if ind[i,l]]
                res=L[i]-(0 if ref=="none" else B if ref=="base" else L[others].mean(0))
                r=stats.spearmanr(res,F[i]).statistic; rhos.append(r); det_full+=tp(r,n)<1e-3
                hits=0
                for _ in range(reps):
                    idx=rng.choice(n,budget,replace=False)
                    hits+=tp(stats.spearmanr(res[idx],F[i][idx]).statistic,budget)<1e-3
                det_b.append(hits/reps)
            out[(ref,fn)]=(np.median(rhos),det_full,k,np.mean(det_b))
            print(f"{name:22s} ref={ref:6s} {fn:28s} med_rho={np.median(rhos):+.3f} full={det_full}/{k} det@{budget}={np.mean(det_b):.2f}",flush=True)
    # correlation among features (owner 0)
    f=list(feats); C=np.array([[stats.spearmanr(feats[a][0],feats[b][0]).statistic for b in f] for a in f])
    print("feature corr with w_O:",{a:round(C[i,f.index('schedule weight w_O')],3) for i,a in enumerate(f)})
run("llama31_dolly_grid",1,100)
run("qwen25_dolly_grid",1,100)
run("llama31_tofu_grid",5,1000)
