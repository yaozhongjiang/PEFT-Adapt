import numpy as np, glob
from scipy import stats
exec(open("tab1.py").read().split("import sys")[0])
def logp(rho,n):
    rho=np.clip(rho,-.9999999,.9999999); t=rho*np.sqrt((n-2)/(1-rho**2)); return stats.t.logsf(t,n-2)/np.log(10)
def rz(x): r=stats.rankdata(x,axis=-1); return (r-r.mean(-1,keepdims=True))/r.std(-1,keepdims=True)
for f,E,orders in [("llama31_tofu_grid",5,np.concatenate([np.load(g)["positions"] for g in sorted(glob.glob(A+"llama31_tofu_grid_replayed_extra_*.npz"))])),
                   ("llama31_dolly_grid",1,np.stack([rng.permutation(4000)[None,:] for _ in range(2000)])),
                   ("qwen25_dolly_grid",1,np.stack([rng.permutation(4000)[None,:] for _ in range(2000)]))]:
    P,L,Bl=load(f); k,_,n=P.shape; ind=indep(P)
    for name in (["w_O","eta_last"] if E>1 else ["w_O"]):
        Fz=rz(np.stack([featfun(name,o,n,E) for o in orders]))
        true,best=[],[]
        for i in range(k):
            r=rz(L[i]-L[[l for l in range(k) if ind[i,l]]].mean(0))
            true.append(logp(stats.spearmanr(r,featfun(name,P[i],n,E)).statistic,n))
            best.append(logp((Fz@r/n).max(),n))
        print(f"{f:20s} {name:8s} own order log10 p: max(worst owner)={max(true):.1f} median={np.median(true):.1f} | best of {len(orders)} foreign orders: min={min(best):.1f} median={np.median(best):.1f}")
