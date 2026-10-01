import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--runs",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    rows=[json.loads(x.read_text()) for x in Path(args.runs).glob("*/result.json")]
    rows.sort(key=lambda x:x["experiment_id"])
    lines=["| Method | Curriculum | KD | Eval Problems | Solve Rate | Compile Rate | Train Hours | Inference Latency |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        method=r.get("method",""); curriculum="Yes" if method in {"curriculum","curriculum_random","curriculum_reverse","cgkd_cr"} else "No"; kd="Yes" if r.get("teacher_models") else "No"
        def fmt(x): return "Not yet evaluated" if x is None else str(x)
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (method,curriculum,kd,fmt(r.get("eval_examples")),fmt(r.get("pass_rate")),fmt(r.get("compile_rate")),fmt(r.get("training_time_hours")),fmt(r.get("inference_latency_ms"))))
    Path(args.output).write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(Path(args.output))
if __name__=="__main__": main()
