from __future__ import annotations
import hashlib,json,random
from pathlib import Path
import yaml

def load_config(path):
    data=yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data,dict): raise ValueError("Configuration must be a mapping")
    return data
def save_json(path,payload):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")
def stable_hash(payload):
    raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",",":"),default=str).encode()
    return hashlib.sha256(raw).hexdigest()
def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
    except Exception: pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception: pass
def resolve_stage_order(config):
    c=config.get("curriculum",{}); mode=c.get("mode","ordered"); stages=list(c.get("stages",[]))
    if mode=="single_stage_mixed": return ["mixed"]
    if mode=="ordered": return stages
    if mode=="reverse": return list(reversed(stages))
    if mode=="random":
        out=list(stages); random.Random(int(config.get("seed",0))).shuffle(out); return out
    if mode=="partial":
        excluded=set(c.get("excluded_stages",[])); return [s for s in stages if s not in excluded]
    raise ValueError(f"Unknown curriculum mode: {mode}")
