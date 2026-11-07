import numpy as np
from scipy import stats
exec(open("mech.py").read().split("def run(")[0])
def prep(name,epochs):
    z=np.load(A+name+"_palimpsest_loglik.npz"); P=z["positions"]; L=z["adapter_loglik"]
    if P.ndim==2: P=P[:,None,:]
    k,E,n=P.shape; ind=indep(P)
    W=np.stack([lrw(P[i],16,epochs,n) for i in range(k)])
    G=P+np.arange(E)[None,:,None]*n
    R=np.stack([L[i]-L[[l for l in range(k) if ind[i,l]]].mean(0) for i in range(k)])
    return P,W,G,R,k,E,n
def power(R,F,budget,reps=200,seed=1):
    rng=np.random.default_rng(seed); k,n=R.shape; h=0
    for i in range(k):
        for _ in range(reps):
            idx=rng.choice(n,budget,replace=False); h+=tp(stats.spearmanr(R[i,idx],F[i,idx]).statistic,budget)<1e-3
    return h/(k*reps)
for name,ep,buds in [("llama31_dolly_grid",1,[30,50]),("qwen25_dolly_grid",1,[30,50]),("llama31_tofu_grid",5,[250,500])]:
    P,W,G,R,k,E,n=prep(name,ep)
    feats={"-position":-G.mean(1),"w_O":W.sum(1),"eta(last occ.)":W[:,-1]}
    T=E*n
    for g in [0.3,0.5,0.7]:
        if E>1: feats[f"sum eta*{g}^(E-e)"]=(W*np.array([g**(E-1-e) for e in range(E)])[None,:,None]).sum(1)
    for fn,F in feats.items():
        med=np.median([stats.spearmanr(R[i],F[i]).statistic for i in range(k)])
        print(f"{name:20s} {fn:22s} med_rho={med:+.3f} "+" ".join(f"det@{b}={power(R,F,b):.2f}" for b in buds),flush=True)
