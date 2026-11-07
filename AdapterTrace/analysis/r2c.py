import numpy as np
from scipy import stats
exec(open("r2b.py").read().split("for f,E,orders")[0])
P,L,Bl=load("llama31_dolly_grid"); k,_,n=P.shape; ind=indep(P)
res=[]
for i in range(10):
    for j in [l for l in range(k) if ind[i,l]][:5]:
        rO=L[i]-L[[l for l in range(k) if ind[i,l]]].mean(0)
        rS=L[j]-L[[l for l in range(k) if ind[i,l] and ind[j,l] and l!=j]].mean(0)
        for frac in [0.1,0.25,0.5]:
            pos=P[i,0].copy()
            sel=rng.choice(n,int(frac*n),replace=False)       # examples whose slots get reassigned
            slots=np.sort(pos[sel])                            # available step slots (early = high eta)
            order=sel[np.argsort(-rS[sel])]                    # suspect-favoured examples get earliest slots
            pos[order]=slots
            w=featfun("w_O",pos[None,:],n,1)
            Q=rng.choice(n,250,replace=False)
            res.append((frac,logp(stats.spearmanr(rO,w).statistic,n),logp(stats.spearmanr(rS[Q],w[Q]).statistic,250)))
res=np.array(res)
for frac in [0.1,0.25,0.5]:
    r=res[res[:,0]==frac]; print(f"frac={frac}: owner V3 log10p median={np.median(r[:,1]):.0f} max={r[:,1].max():.0f}; innocent suspect p<1e-3 in {np.mean(r[:,2]<-3):.2f}, p<0.05 in {np.mean(r[:,2]<np.log10(.05)):.2f}")
