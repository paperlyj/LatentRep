"""Offline checks of stored predictions, group isolation and Table 3.
No spectral files, network access or model fitting are needed.
This does NOT reconstruct the historical 68-group prostate partition.
"""
from pathlib import Path
import json, platform, hashlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, adjusted_rand_score

base=Path(__file__).resolve().parent
report={"scope":"Stored post hoc outputs; not a rerun of graph inference or historical benchmarks", "python":platform.python_version(),"numpy":np.__version__,"scikit_learn":sklearn.__version__,"datasets":[]}
expected={"covid19_salvia":[.0724,.0473,.0473,.0601],"parkinson":[.0604,.0266,.0266,.0502],"serum_prostate_cancer":[.2661,.0965,.0965,.1359],"alzheimer":[.0435,.0217,.0217,.0234]}
for folder in sorted((base/'results').iterdir()):
    a=np.load(folder/'evaluation_labels.npz');y,g=a['y'],a['g']
    h=json.loads((folder/'input_hashes.json').read_text())
    assert hashlib.sha256(y.tobytes()).hexdigest()==h['y']
    assert hashlib.sha256(g.tobytes()).hexdigest()==h['g']
    groups={s:np.load(folder/f'{s}_groups.npy') for s in ['legacy','index','kmeans']}
    groups['oracle']=g
    saved=pd.read_csv(folder/'f1_by_seed.csv').set_index('seed')
    scores={}; largest=0.; count=0
    for seed in list(range(10))+[2026]:
        scores[seed]={}
        for strategy in ['random','legacy','index','kmeans','oracle']:
            p=np.load(folder/f'{strategy}_{seed}.npz');fold,pred=p['fold'],p['pred']
            assert len(fold)==len(pred)==len(y)
            assert len(np.unique(fold))==5
            if strategy!='random':
                for unit in np.unique(groups[strategy]):
                    assert len(np.unique(fold[groups[strategy]==unit]))==1,(folder.name,strategy,seed)
            score=f1_score(y,pred,average='macro');scores[seed][strategy]=score
            delta=abs(score-float(saved.loc[seed,strategy]));largest=max(largest,delta)
            assert delta<1e-12,(folder.name,strategy,seed,delta)
            count+=1
    frame=pd.DataFrame.from_dict(scores,orient='index').loc[list(range(10))]
    errors=frame.sub(frame.oracle,axis=0).abs().mean()
    order=['random','legacy','index','kmeans']
    assert [round(errors[s],4) for s in order]==expected[folder.name]
    ari=adjusted_rand_score(groups['legacy'],groups['index'])
    diag=json.loads((folder/'diagnostics.json').read_text())
    assert abs(ari-diag['ARI_legacy_index'])<1e-12
    for strategy in ['legacy','index']:
        assert len(np.unique(groups[strategy]))==diag[strategy]['groups']
        assert abs(adjusted_rand_score(g,groups[strategy])-diag[strategy]['ARI_unit'])<1e-12
    report['datasets'].append({'task':folder.name,'prediction_vectors_checked':count,'max_F1_discrepancy':largest,'group_isolation_passed':True,'ARI_between_rules':ari,'table3_mean_errors':{s:float(errors[s]) for s in order}})
report['status']='PASS'
print(json.dumps(report,indent=2))
