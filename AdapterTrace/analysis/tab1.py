import numpy as np
from scipy import stats
import os
A=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results","raw")+"/"
rng=np.random.default_rng(7)
def lrw(pos,n,E):
    steps=int(np.ceil(n/16))*E; w0=int(np.ceil(steps*0.03)); st=pos//16
    return np.where(st<w0,(st+1)/w0,np.maximum(0,(steps-st)/(steps-w0)))
def featfun(name,pos,n,E):
    w=lrw(pos,n,E)
    if name=="w_O": return w.sum(0)
    if name=="eta_last": return w[-1]
    if name=="pos": return pos.mean(0)
    if name=="negpos": return -pos.mean(0)
    if name=="neglast": return -pos[-1]
def mcnull(name,n,E,m,B=100000):
    out=[];c=0
    while c<B:
        b=min(1000,B-c); pos=np.stack([rng.permutation(n)+e*n for e in range(E)])
        f=featfun(name,pos,n,E); idx=rng.choice(n,m,replace=False) if m<n else np.arange(n)
        fr=stats.rankdata(f[idx]); fr=(fr-fr.mean())/fr.std()
        R=np.argsort(rng.random((b,m)),axis=1).argsort(axis=1).astype(float); R=(R-R.mean(1,keepdims=True))/R.std(1,keepdims=True)
        out.append(R@fr/m); c+=b
    return np.sort(np.concatenate(out))
NL={}
def p(rho,name,n,E,m):
    k=(name,n,E,m)
    if k not in NL: NL[k]=mcnull(name,n,E,m)
    nl=NL[k]; return (1+len(nl)-np.searchsorted(nl,rho,side="left"))/(len(nl)+1)
def indep(P):
    sig=[{np.argsort(e).tobytes() for e in q} for q in P]; k=len(P)
    return np.array([[i!=j and not(sig[i]&sig[j]) for j in range(k)] for i in range(k)])
def load(f):
    z=np.load(A+f+"_palimpsest_loglik.npz"); P=z["positions"]
    if P.ndim==2: P=P[:,None,:]
    return P,z["adapter_loglik"],z["base_loglik"]
import sys
mode=sys.argv[1]
if mode=="tab1":
  for f,E in [("llama31_dolly_grid",1),("qwen25_dolly_grid",1),("llama31_tofu_grid",5),("llama31_lineage_victims",1)]:
    P,L,Bl=load(f); k,_,n=P.shape; ind=indep(P)
    for ref in ["none","base","shadow"]:
        for name in ["pos","w_O"]:
            d=0
            for i in range(k):
                res=L[i]-(0 if ref=="none" else Bl if ref=="base" else L[[l for l in range(k) if ind[i,l]]].mean(0))
                d+=p(stats.spearmanr(res,featfun(name,P[i],n,E)).statistic,name,n,E,n)<1e-3
            print(f"{f:24s} {ref:6s} {name:5s} {d}/{k}",flush=True)
if mode=="mech":
  for f,E,buds in [("llama31_dolly_grid",1,[50]),("qwen25_dolly_grid",1,[50]),("llama31_tofu_grid",5,[250])]:
    P,L,Bl=load(f); k,_,n=P.shape; ind=indep(P)
    R=np.stack([L[i]-L[[l for l in range(k) if ind[i,l]]].mean(0) for i in range(k)])
    names=["pos","negpos","neglast","w_O","eta_last"] if E>1 else ["pos","negpos","w_O"]
    for name in names:
        med=np.median([stats.spearmanr(R[i],featfun(name,P[i],n,E)).statistic for i in range(k)])
        s=f"{f:22s} {name:9s} med_rho={med:+.3f}"
        for m in buds:
            h=[]
            for i in range(k):
                F=featfun(name,P[i],n,E)
                for _ in range(25):
                    idx=rng.choice(n,m,replace=False); h.append(p(stats.spearmanr(R[i,idx],F[idx]).statistic,name,n,E,m)<1e-3)
            s+=f" det@{m}={np.mean(h):.2f}"
        print(s,flush=True)
