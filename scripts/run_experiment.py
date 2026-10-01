import argparse,json
from pathlib import Path
from cgkd_cr.config import load_config,resolve_stage_order,save_json
from cgkd_cr.results import ExperimentResult
from cgkd_cr.training import train_experiment
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--dry-run",action="store_true"); a=p.parse_args(); cfg=load_config(a.config); cfg["config_path"]=a.config
    if a.dry_run:
        print(json.dumps({"experiment_id":cfg["experiment_id"],"method":cfg["method"],"stage_order":resolve_stage_order(cfg),"teacher":cfg.get("distillation",{}).get("teacher_model"),"training":cfg["training"]},indent=2)); return
    manifest=train_experiment(cfg); result=ExperimentResult(cfg["experiment_id"],cfg["model"]["base_model"],cfg["method"],manifest["stage_order"],[cfg["distillation"]["teacher_model"]] if cfg.get("distillation",{}).get("enabled") else [],sum(x["examples"] for x in manifest["stage_records"]),0,0,training_time_hours=manifest["training_time_hours"])
    save_json(Path(cfg["runtime"]["output_root"])/"result.json",result.to_dict())
if __name__=="__main__": main()
