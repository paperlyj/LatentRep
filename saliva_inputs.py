"""Read the public MIND archive using the v1.4.0 folder/category rules."""
from pathlib import Path
import argparse, io, hashlib, json, zipfile
import numpy as np
import pandas as pd
def load_tasks(archive,out):
    results={}
    with zipfile.ZipFile(archive) as z:
        files=z.namelist()
        for key,folder,categories,expected in [
            ('covid19_salvia','covid_dataset',{'CTRL','COV+','COV-'},(2501,101)),
            ('parkinson','pd_ad_dataset',{'PD','CTRL'},(1476,60)),
            ('alzheimer','pd_ad_dataset',{'AD','CTRL'},(1151,46))]:
            xs=[]; ys=[]; gs=[]; axes=[]; cats={}
            for name in sorted(n for n in files if '/'+folder+'/' in n and n.endswith('/user_information.csv')):
                base=name.rsplit('/',1)[0]+'/'
                if base+'spectra.csv' not in files or base+'raman_shift.csv' not in files: continue
                meta=pd.read_csv(io.BytesIO(z.read(name)))
                catcol=next((c for c in meta if c.lower()=='category'),meta.columns[1])
                cat=str(meta[catcol].iloc[0])
                if cat not in categories: continue
                x=pd.read_csv(io.BytesIO(z.read(base+'spectra.csv')),header=None).to_numpy(dtype=float)
                shift=pd.read_csv(io.BytesIO(z.read(base+'raman_shift.csv')),header=None).to_numpy().squeeze()
                axes.append(shift)
                gs.extend([len(xs)]*len(x))
                xs.append(x);ys.extend([cat]*len(x))
                cats[cat]=cats.get(cat,0)+1
            aligned=not all(np.allclose(axes[0],a) for a in axes)
            if aligned:
                axis=np.arange(max(a[0] for a in axes),min(a[-1] for a in axes),min(a[1]-a[0] for a in axes))
                xs=[np.stack([np.interp(axis,a,row) for row in mat]) for mat,a in zip(xs,axes)]
            x=np.ascontiguousarray(np.vstack(xs),dtype=np.float32)
            _,y=np.unique(ys,return_inverse=True)
            assert (len(x),len(xs))==expected and x.shape[1]==885
            np.savez(out/(key+'.npz'),X=x,y=y,g=np.asarray(gs).astype(str))
            results[key]={'aligned':aligned,'shape':list(x.shape),'recorded_units_by_class':cats,'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
    return results
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('out');a=p.parse_args()
    o=Path(a.out);o.mkdir(exist_ok=True,parents=True)
    print(json.dumps(load_tasks(a.archive,o),indent=2))
