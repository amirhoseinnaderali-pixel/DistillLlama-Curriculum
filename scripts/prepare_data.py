import argparse,json
from cgkd_cr.config import load_config
from cgkd_cr.data import dataset_fingerprint,load_stage_dataset
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=load_config(a.config); rows=[]
    for stage in cfg["curriculum"]["stages"]:
        ds=load_stage_dataset(stage,cfg); rows.append({"stage":stage,"examples":len(ds),"fingerprint":dataset_fingerprint(ds)})
    print(json.dumps(rows,indent=2))
if __name__=="__main__": main()
