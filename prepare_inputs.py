"""Rebuild the external task matrices with the recorded source order and loader conventions."""
from pathlib import Path
import io, collections, json, pickle, re, zipfile, hashlib, urllib.request
import numpy as np
from saliva_inputs import load_tasks
root=Path(__file__).resolve().parent
sources=json.loads((root/'sources.json').read_text())
data=root/'inputs'
data.mkdir(exist_ok=True)
cache=data/'source_cache'
cache.mkdir(exist_ok=True)
def fetch(path,url,key,expected):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=120) as response:
            path.write_bytes(response.read())
    if hashlib.new(key,path.read_bytes()).hexdigest()!=expected:
        raise ValueError(f'Checksum mismatch: {path}. Verify the pinned source before continuing.')
    return path
class Reader(pickle.Unpickler):
 def __init__(self,z,n):self.z=z;self.root=n.rsplit('/',1)[0];super().__init__(io.BytesIO(z.read(n)))
 def find_class(self,m,n):
  if (m,n)==('torch','FloatStorage'):return 'float32'
  if (m,n)==('collections','OrderedDict'):return collections.OrderedDict
  if (m,n)==('torch._utils','_rebuild_tensor_v2'):return lambda storage,offset,size,stride,*a:np.ndarray(shape=size,dtype='float32',buffer=storage,offset=offset*4,strides=tuple(v*4 for v in stride)).copy()
  raise ValueError((m,n))
 def persistent_load(self,pid):return self.z.read(self.root+'/data/'+pid[2])

if __name__=='__main__':
    archive=fetch(cache/'Raman-Spectra-Data.zip',sources['saliva_source'],'sha256',sources['saliva_sha256'])
    load_tasks(archive,data)
    spectra=[]; labels=[]; groups=[]; ids={}
    for record in sources['prostate_files']:
        path=fetch(cache/(str(record['id'])+'.pt'),record['download_url'],'md5',record['computed_md5'])
        with zipfile.ZipFile(path) as z:
            p=next(n for n in z.namelist() if n.endswith('/data.pkl'))
            x=np.atleast_2d(Reader(z,p).load())
        name=record['name']; end=name.index(']')
        sample=re.sub(r'\s*[Nn]o[Bb]ase','',name[end+1:]).removesuffix('.pt').strip()
        ids.setdefault(sample,len(ids))
        spectra.append(x)
        labels.extend([int(name[1:end])-1]*len(x))
        groups.extend([ids[sample]]*len(x))
    np.savez(data/'serum_prostate_cancer.npz',X=np.concatenate(spectra).astype(np.float32),
             y=np.asarray(labels),g=np.asarray(groups).astype(str))
    for f in sorted(data.glob('*.npz')):
        with np.load(f) as a:
            digest=hashlib.sha256(np.ascontiguousarray(a['X']).tobytes()).hexdigest()
        expected=json.loads((root/'results'/f.stem/'diagnostics.json').read_text())['input_X_sha256']
        if digest!=expected: raise ValueError(f'Input matrix differs: {f.stem}')
        print(f.stem,digest)
