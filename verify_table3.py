"""Recalculate Table 3 from the fold-wise predictions and reconstructed inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
base=Path(__file__).resolve().parent
for d in sorted((base/'results').iterdir()):
    y=np.load(base/'inputs'/(d.name+'.npz'))['y']
    scores=pd.DataFrame(index=range(10))
    for strategy in ('random','legacy','index','kmeans','oracle'):
        scores[strategy]=[f1_score(y,np.load(d/f'{strategy}_{seed}.npz')['pred'],average='macro') for seed in scores.index]
    e=scores.sub(scores.oracle,axis=0).abs().mean()
    print(d.name,e.drop('oracle').round(4).to_dict())
