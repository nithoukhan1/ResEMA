"""Private-only D4-E R3 candidate: lossless, weights_only-compatible Albumentations RNG preservation.

Operates on the TRAIN dataset.transforms tree (workers=0 only). Never uses B-TEST.
The original Chat20 snapshot did NOT contain this state and is intentionally incompatible.
"""
from __future__ import annotations
import copy
import random
from typing import Any
import numpy as np

SCHEMA='D4E_TRAIN_ALBUMENTATIONS_RNG_R2'


def _iter_rng_nodes(root: Any):
    seen=set()
    def walk(obj,path):
        if obj is None or id(obj) in seen:return
        seen.add(id(obj))
        pr=getattr(obj,'py_random',None)
        ng=getattr(obj,'random_generator',None)
        py=isinstance(pr,random.Random)
        npg=isinstance(ng,np.random.Generator)
        if py or npg:
            yield (path,obj,py,npg)
        for attr in ('transforms','transform'):
            ch=getattr(obj,attr,None)
            if isinstance(ch,(tuple,list)):
                for i,v in enumerate(ch):
                    yield from walk(v, f'{path}/{attr}/{i}')
            elif attr=='transform' and ch is not None:
                yield from walk(ch,f'{path}/transform')
    return list(walk(root,'train_transforms'))


def _plain(v: Any):
    """Only safe primitive structures are permitted inside an authenticated weights_only capsule."""
    if v is None or isinstance(v,(str,bool,int,float)):
        return v
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return float(v)
    if isinstance(v,np.ndarray):
        return {'__numpy_array__':True,'dtype':str(v.dtype),'shape':list(v.shape),'values':v.tolist()}
    if isinstance(v,tuple):return tuple(_plain(x) for x in v)
    if isinstance(v,list):return [_plain(x) for x in v]
    if isinstance(v,dict):
        if not all(isinstance(k,str) for k in v):
            raise TypeError('AUGMENT_RNG_NONSTRING_DICT_KEY')
        return {k:_plain(x) for k,x in v.items()}
    raise TypeError('AUGMENT_RNG_UNSUPPORTED_STATE_TYPE:'+type(v).__name__)


def _unplain(v: Any):
    if isinstance(v,dict) and v.get('__numpy_array__') is True:
        return np.asarray(v['values'], dtype=np.dtype(v['dtype'])).reshape(v['shape'])
    if isinstance(v,dict):return {k:_unplain(x) for k,x in v.items()}
    if isinstance(v,list):return [_unplain(x) for x in v]
    if isinstance(v,tuple):return tuple(_unplain(x) for x in v)
    return v


def _describe(root:Any):
    nodes=_iter_rng_nodes(root)
    if not nodes:
        raise RuntimeError('AUGMENT_RNG_NO_INTERNAL_GENERATORS_FAIL_CLOSED')
    if not any('albumentations' in type(obj).__module__.lower() for _,obj,_,_ in nodes):
        raise RuntimeError('AUGMENT_RNG_NO_REAL_ALBUMENTATIONS_PIPELINE')
    return nodes


def capture_augmentation_rng(root: Any) -> dict:
    entries=[]
    for path,obj,haspy,hasnp in _describe(root):
        entries.append({
            'path':path,
            'type':type(obj).__module__+'.'+type(obj).__qualname__,
            'python_state':_plain(copy.deepcopy(obj.py_random.getstate())) if haspy else None,
            'numpy_state':_plain(copy.deepcopy(obj.random_generator.bit_generator.state)) if hasnp else None,
        })
    return {'schema':SCHEMA,'entries':entries}


def restore_augmentation_rng(root: Any, snapshot:dict) -> None:
    if not isinstance(snapshot,dict) or snapshot.get('schema')!=SCHEMA:
        raise RuntimeError('AUGMENT_RNG_SCHEMA_MISMATCH')
    entries=snapshot.get('entries')
    current=_describe(root)
    if not isinstance(entries,list) or len(entries)!=len(current):
        raise RuntimeError('AUGMENT_RNG_PIPELINE_LENGTH_DRIFT')
    # Validate the ENTIRE structure before mutating any active generator.
    for snap,(path,obj,haspy,hasnp) in zip(entries,current):
        if (snap['path']!=path or snap['type']!=type(obj).__module__+'.'+type(obj).__qualname__
            or (snap['python_state'] is None)==haspy or (snap['numpy_state'] is None)==hasnp):
            raise RuntimeError('AUGMENT_RNG_PIPELINE_LAYOUT_DRIFT')
    for snap,(_,obj,haspy,hasnp) in zip(entries,current):
        if haspy:obj.py_random.setstate(_unplain(copy.deepcopy(snap['python_state'])))
        if hasnp:obj.random_generator.bit_generator.state=_unplain(copy.deepcopy(snap['numpy_state']))
    # Strict check that every RNG state was restored, not just accepted by its setter.
    if capture_augmentation_rng(root)!=snapshot:
        raise RuntimeError('AUGMENT_RNG_POSTRESTORE_VERIFICATION_FAILED')


def test_digest(snapshot:dict)->str:
    """Portable fingerprint for local testing only; not an external authentication pin."""
    import hashlib, json
    return hashlib.sha256(json.dumps(snapshot,sort_keys=True,default=str).encode()).hexdigest()
