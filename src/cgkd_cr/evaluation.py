from __future__ import annotations
import ast,re,subprocess,tempfile,time
from pathlib import Path
def extract_python(text):
    fence=chr(96)*3; match=re.findall(fence+r"python\s*(.*?)"+fence,text,flags=re.S|re.I)
    if match: return match[-1].strip()
    match=re.findall(fence+r"\s*(.*?)"+fence,text,flags=re.S)
    return match[-1].strip() if match else text.strip()
def compile_ok(code):
    try: ast.parse(code); compile(code,"<generated>","exec"); return True
    except Exception: return False
def run_case(code,harness,timeout_s=5.0):
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/"eval.py"; path.write_text(code+"\n\n"+harness,encoding="utf-8"); start=time.perf_counter()
        try: proc=subprocess.run(["python",str(path)],capture_output=True,text=True,timeout=timeout_s)
        except subprocess.TimeoutExpired: return {"passed":False,"status":"timeout","latency_ms":(time.perf_counter()-start)*1000}
        return {"passed":proc.returncode==0,"status":"ok" if proc.returncode==0 else "runtime_error","latency_ms":(time.perf_counter()-start)*1000}
def evaluate_predictions(predictions):
    total=len(predictions)
    if not total: return {"eval_examples":0,"compile_rate":None,"pass_rate":None,"inference_latency_ms":None}
    compiled=passed=0; latencies=[]
    for item in predictions:
        if compile_ok(extract_python(item.get("prediction",""))):
            compiled+=1
            if item.get("harness"):
                result=run_case(extract_python(item.get("prediction","")),item["harness"])
                if result["passed"]: passed+=1
                latencies.append(result["latency_ms"])
    return {"eval_examples":total,"compile_rate":compiled/total,"pass_rate":passed/total if any(x.get("harness") for x in predictions) else None,"inference_latency_ms":sum(latencies)/len(latencies) if latencies else None}
