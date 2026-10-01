import argparse
from cgkd_cr.config import load_config
from cgkd_cr.training import train_experiment
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=load_config(a.config); cfg["config_path"]=a.config; print(train_experiment(cfg))
if __name__=="__main__": main()
