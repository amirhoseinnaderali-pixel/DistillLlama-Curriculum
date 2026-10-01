import argparse,json
from pathlib import Path
from cgkd_cr.config import load_config
from cgkd_cr.data import STAGES,load_stage_dataset
from cgkd_cr.distillation import generate_records
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--stage",required=True); p.add_argument("--teacher",required=True); p.add_argument("--max-samples",type=int,default=None); p.add_argument("--output",required=True); a=p.parse_args(); cfg=load_config(a.config); ds=load_stage_dataset(a.stage,cfg); override=cfg.get("stage_overrides",{}).get(a.stage,{})
    rows=generate_records(ds,stage=a.stage,teacher_model=a.teacher,source_dataset=override.get("dataset",STAGES[a.stage]["dataset"]),max_samples=a.max_samples,max_tokens=int(cfg.get("distillation",{}).get("max_tokens",1024)),temperature=float(cfg.get("distillation",{}).get("temperature",0.7)))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(f"Wrote {len(rows)} records to {a.output}")
if __name__=="__main__": main()
