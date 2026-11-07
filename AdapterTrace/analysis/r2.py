import numpy as np, glob
from scipy import stats
exec(open("tab1.py").read().split("import sys")[0])
def ranks_z(x): r=stats.rankdata(x,axis=-1); return (r-r.mean(-1,keepdims=True))/r.std(-1,keepdims=True)
def run(f,E,orders,label,m):
    P,L,Bl=load(f); k,_,n=P.shape; ind=indep(P)
    # residuals: owner i itself (V3) and innocent sibling j, reference = shadows independent of i (and j)
    nl=NL.setdefault(("w_O",n,E,m),mcnull("w_O",n,E,m,B=200000))
    pv=lambda rho:(1+len(nl)-np.searchsorted(nl,rho,side="left"))/(len(nl)+1)
    idx=np.sort(rng.choice(n,m,replace=False)) if m<n else np.arange(n)
    Fz=ranks_z(np.stack([featfun("w_O",o,n,E) for o in orders])[:,idx])  # M,m
    res={}
    for tau in [1e-3,1e-4]:
        for a in [0.05,1e-3]:
            res[(tau,a)]=[0,0,0]
    v3_fp=[]
    for i in range(k):
        refi=L[[l for l in range(k) if ind[i,l]]].mean(0)
        po=pv(Fz@ranks_z((L[i]-refi)[idx])/m)   # owner adapter under foreign orders
        v3_fp.append(np.mean(po<1e-3))
        for j in range(k):
            if not ind[i,j]: continue
            refij=L[[l for l in range(k) if ind[i,l] and ind[j,l] and l!=j]].mean(0)
            ps=pv(Fz@ranks_z((L[j]-refij)[idx])/m)
            for (tau,a),c in res.items():
                c[0]+= (ps<a).any()            # grinding without binding
                c[1]+= ((ps<a)&(po<tau)).any() # grinding with V3 binding to theta_O
                c[2]+=1
    print(f"{label}: M={len(orders)} orders, n={m}; owner adapter detected under a foreign order (p<1e-3): {np.mean(v3_fp):.4f}")
    for (tau,a),c in res.items():
        print(f"   V3 tau={tau:g} alpha={a:g}: accuse w/o binding {c[0]/c[2]:.3f}  with binding {c[1]/c[2]:.4f}  ({c[2]} pairs)")
tofu=np.concatenate([np.load(f)["positions"] for f in sorted(glob.glob(A+"llama31_tofu_grid_replayed_extra_*.npz"))])
run("llama31_tofu_grid",5,tofu,"TOFU replayed trainer orders",1000)
dolly=[ (rng.permutation(4000))[None,:] for _ in range(1000)]
run("llama31_dolly_grid",1,dolly,"Dolly fresh shuffles",250)
