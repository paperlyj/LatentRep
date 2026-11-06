"""Matched sensitivity analysis. The input NPZ contains X, y and g in manifest order."""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

GRID = np.round(np.arange(1,6.01,.1),2)
GRAPH = None

def grouped_matrix(x):
    sd=x.std(axis=1,keepdims=True)
    sd[sd==0]=1
    return normalize((x-x.mean(axis=1,keepdims=True))/sd).astype(np.float32)

def make_graph(indices,similarities):
    nb=[set(map(int,row)) for row in indices]
    sim=[dict(zip(row.tolist(),w.tolist())) for row,w in zip(indices,similarities)]
    edges={(min(i,j),max(i,j)) for i,neigh in enumerate(indices) for j in map(int,neigh)}
    graph=nx.Graph()
    graph.add_nodes_from(range(len(indices)))
    for i,j in edges:
        direct=[w for w in [sim[i].get(j),sim[j].get(i)] if w is not None]
        jac=len(nb[i]&nb[j])/len(nb[i]|nb[j])
        graph.add_edge(i,j,weight=.6*float(np.mean(direct))+.4*jac)
    return graph

def run_resolution(gamma):
    labels=np.empty(len(GRAPH),np.int32)
    for k,c in enumerate(nx.community.louvain_communities(GRAPH,weight='weight',resolution=float(gamma),seed=42)):
        labels[list(c)]=k
    return labels

def infer(z,mode,workers,out):
    global GRAPH
    nn=NearestNeighbors(n_neighbors=11,metric='cosine',algorithm='brute',n_jobs=-1).fit(z)
    dist,idx=nn.kneighbors(z)
    if mode=='index':
        keep=np.array([np.flatnonzero(row!=i)[:10] for i,row in enumerate(idx)])
    else:
        keep=np.broadcast_to(np.arange(1,11),(len(z),10))
    idx=np.take_along_axis(idx,keep,axis=1)
    sim=1.-np.take_along_axis(dist,keep,axis=1)
    GRAPH=make_graph(idx,sim)
    loops=nx.number_of_selfloops(GRAPH)
    print(mode,'edges',GRAPH.number_of_edges(),'loops',loops,flush=True)
    with mp.get_context('fork').Pool(workers) as pool:
        parts=pool.map(run_resolution,GRID,chunksize=1)
    aris={}
    for i in range(len(GRID)):
        for j in range(i+1,min(i+4,len(GRID))):
            aris[i,j]=adjusted_rand_score(parts[i],parts[j])
    stats=[]
    for i,gamma in enumerate(GRID):
        def mean_near(span):
            return np.mean([aris[min(i,j),max(i,j)] for j in range(max(0,i-span),min(len(GRID),i+span+1)) if j!=i])
        stats.append((gamma,mean_near(3),mean_near(1),len(np.unique(parts[i]))))
    df=pd.DataFrame(stats,columns=['resolution','persistence','adjacent','n_groups'])
    j=df.sort_values(['persistence','adjacent'],ascending=False).index[0]
    np.save(out/f'{mode}_groups.npy',parts[j])
    df.to_csv(out/f'{mode}_grid.csv',index=False)
    rep=run_resolution(GRID[j])
    assert np.array_equal(parts[j],rep)
    return parts[j].astype(str),dict(resolution=float(GRID[j]),groups=int(df.n_groups[j]),persistence=float(df.persistence[j]),loops=int(loops),edges=GRAPH.number_of_edges())

def evaluate(job):
    strategy,seed=job
    groups=GROUPS[strategy]
    if groups is None:
        cv=StratifiedKFold(5,shuffle=True,random_state=seed).split(X,Y)
    else:
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed).split(X,Y,groups)
    pred=np.full(len(X),-1)
    fold=np.full(len(X),-1)
    for k,(train,test) in enumerate(cv):
        model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,random_state=42))
        model.fit(X[train],Y[train])
        pred[test]=model.predict(X[test])
        fold[test]=k
    assert np.all(pred>=0)
    return (strategy,seed,f1_score(Y,pred,average='macro'),fold,pred)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('npz');ap.add_argument('out');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'.run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (out/'f1_by_seed.csv').exists():
        raise SystemExit('Output directory already holds a completed run; use an empty directory to rerun.')
    arr=np.load(args.npz)
    X=np.ascontiguousarray(arr['X'],dtype=np.float32)
    # Canonicalize physical array layout for preprocessing. Class and group vectors enter only evaluation.
    z=grouped_matrix(X)
    assert np.array_equal(z,grouped_matrix(np.ascontiguousarray(np.asfortranarray(X))))
    threadpool_limits(2)
    reports={}
    GROUPS={}
    for mode in ['legacy','index']:
        GROUPS[mode],reports[mode]=infer(z,mode,args.workers,out)
    k=reports['index']['groups']
    km=KMeans(n_clusters=k,random_state=42,n_init=10,algorithm='lloyd')
    GROUPS['kmeans']=km.fit_predict(z).astype(str)
    np.save(out/'kmeans_groups.npy',GROUPS['kmeans'].astype(int))
    Y=arr['y'].astype(int)
    GROUPS.update(random=None,oracle=arr['g'].astype(str))
    for mode in ['legacy','index','kmeans']:
        reports.setdefault(mode,{})['ARI_unit']=adjusted_rand_score(GROUPS['oracle'],GROUPS[mode])
    reports['ARI_legacy_index']=adjusted_rand_score(GROUPS['legacy'],GROUPS['index'])
    reports['input_X_sha256']=hashlib.sha256(X.tobytes()).hexdigest()
    json.dump(reports,open(out/'diagnostics.json','w'),indent=2)
    print(reports,flush=True)
    threadpool_limits(1)
    seeds=list(range(10))+[2026]
    jobs=[(name,seed) for name in GROUPS for seed in seeds]
    with mp.get_context('fork').Pool(args.workers) as pool:
        outputs=pool.map(evaluate,jobs,chunksize=1)
    rows=[]
    for name,seed,score,fold,pred in outputs:
        rows.append((name,seed,score))
        np.savez_compressed(out/f'{name}_{seed}.npz',fold=fold,pred=pred)
    f=pd.DataFrame(rows,columns=['design','seed','f1']).pivot(index='seed',columns='design',values='f1')
    f.to_csv(out/'f1_by_seed.csv')
    e=f.sub(f.oracle,axis=0).abs()
    print('F1 at 2026',f.loc[2026].to_dict(),'error at 2026',e.loc[2026].to_dict(), 'mean abs 0-9',e.loc[0:9].mean().to_dict(),flush=True)
