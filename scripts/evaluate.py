import argparse,json
from pathlib import Path
from cgkd_cr.evaluation import evaluate_predictions
def main():
    p=argparse.ArgumentParser(); p.add_argument("--predictions",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    with open(a.predictions,encoding="utf-8") as f: rows=[json.loads(x) for x in f if x.strip()]
    metrics=evaluate_predictions(rows); Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(metrics,indent=2),encoding="utf-8"); print(json.dumps(metrics,indent=2))
if __name__=="__main__": main()
