exec(open("r3.py").read().split("for fname in")[0])
def both(ll,i,ref,idx=None):
    p1=test(ll,i,ref,idx,"w_O")[1]; p2=test(ll,i,ref,idx,"eta_last")[1]; return min(1,2*min(p1,p2)),p1,p2
ref_i=lambda i: L[[l for l in range(k) if ind[i,l]]].mean(0)
det=[both(L[i],i,ref_i(i))[0] for i in range(k)]
nul=np.array([both(L[j],i,L[[l for l in range(k) if ind[i,l] and ind[j,l] and l!=j]].mean(0))[0] for i in range(k) for j in range(k) if ind[i,j]])
print(f"combined owners@1e-3 {np.mean(np.array(det)<1e-3):.2f}  siblings FPR .05={np.mean(nul<.05):.3f} 1e-3={np.mean(nul<1e-3):.4f}")
for m in [250,500]:
    h=[]
    for i in range(k):
        r=ref_i(i)
        for _ in range(20):
            idx=rng.choice(N,m,replace=False); h.append(both(L[i][idx],i,r[idx],idx)[0]<1e-3)
    print(f"combined owners@1e-3 {m} queries {np.mean(h):.2f}")
for tag in ["ownerprompts","otherdata"]:
    d=np.load(A+f"llama31_tofu_grid_distill_{tag}_scores.npz"); keep=d["keep"]
    ps=np.array([both(sc,int(np.where(seeds==o)[0][0]),ref_i(int(np.where(seeds==o)[0][0]))[keep],keep)[0] for o,sc in zip(d["owners"],d["scores"])])
    print(f"combined distill {tag}: @.05 {np.mean(ps<.05):.2f} @1e-3 {np.mean(ps<1e-3):.2f}")
for tag in ["thief","h_thief"]:
    fs=sorted(glob.glob(A+f"llama31_tofu_grid_contft_{tag}_[ab].npz"))
    idxs=np.concatenate([np.load(f)["indices"] for f in fs]); ll=np.concatenate([np.load(f)["ll"] for f in fs]); steps=np.load(fs[0])["steps"]
    for s_i,s in enumerate(steps):
        ps=np.array([both(ll[c,s_i],int(i),ref_i(int(i)))[0] for c,i in enumerate(idxs)])
        print(f"combined contft {tag} {s}: @.05 {int((ps<.05).sum())}/7 @1e-3 {int((ps<1e-3).sum())}/7")
# crosspop: 2300 replayed real orders as fictitious owners
orders=np.concatenate([np.load(f)["positions"] for f in sorted(glob.glob(A+"llama31_tofu_grid_replayed_extra_*.npz"))])
own=set(map(bytes,[np.argsort(e).tobytes() for p in P for e in p]))
keepo=[o for o in orders if not ({np.argsort(e).tobytes() for e in o} & own)]
print("independent replayed orders:",len(keepo),"of",len(orders))
res={"w_O":[], "eta_last":[], "both":[]}
nl={f:NULL.setdefault((f,N),mc_null(f,N)) for f in ["w_O","eta_last"]}
R=np.stack([L[j]-L[[l for l in range(k) if l!=j]].mean(0) for j in range(k)])
Rr=np.apply_along_axis(stats.rankdata,1,R); Rr=(Rr-Rr.mean(1,keepdims=True))/Rr.std(1,keepdims=True)
for o in keepo:
    fe=feats(o); pp={}
    for f in ["w_O","eta_last"]:
        fr=stats.rankdata(fe[f]); fr=(fr-fr.mean())/fr.std(); rho=Rr@fr/N
        pp[f]=(1+len(nl[f])-np.searchsorted(nl[f],rho,side="left"))/(len(nl[f])+1); res[f].extend(pp[f])
    res["both"].extend(np.minimum(1,2*np.minimum(pp["w_O"],pp["eta_last"])))
for f,v in res.items():
    v=np.array(v); print(f"crosspop {f}: {len(v)} tests FPR .05={np.mean(v<.05):.4f} 1e-3={np.mean(v<1e-3):.5f}")
