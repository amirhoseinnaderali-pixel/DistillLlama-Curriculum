from __future__ import annotations
import hashlib,json

STAGES={
 "foundation":{"dataset":"iamtarun/python_code_instructions_18k_alpaca","task_type":"python_instruction_following"},
 "algorithms":{"dataset":"codeparrot/apps","task_type":"algorithmic_problem_solving"},
 "debugging":{"dataset":"m-a-p/Code-Feedback","task_type":"debugging"},
 "advanced":{"dataset":"ise-uiuc/Magicoder-Evol-Instruct-110K","task_type":"advanced_code_reasoning"},
}
def is_ioi_like(example):
    return "ioi" in " ".join(str(example.get(k,"")) for k in ("question","title","source","tags","url")).lower()
def load_stage_dataset(stage,config):
    from datasets import load_dataset
    if stage not in STAGES: raise KeyError(stage)
    override=config.get("stage_overrides",{}).get(stage,{})
    if stage=="algorithms" and override.get("use_local_ioi"):
        ds=load_dataset("json",data_files=override.get("ioi_json_path","ioi_multi_view.json"),split="train")
    else:
        ds=load_dataset(override.get("dataset",STAGES[stage]["dataset"]),split=override.get("split","train"))
    if stage=="algorithms" and override.get("difficulty_filter",True) and not override.get("use_local_ioi"):
        ds=ds.filter(lambda x:x.get("difficulty") in ["introductory","interview"])
    if stage=="algorithms" and override.get("ioi_only",False) and not override.get("use_local_ioi"):
        ds=ds.filter(is_ioi_like)
    return ds
def algorithm_view_text(example):
    view=example.get("algorithm_view")
    if not isinstance(view,dict): return f"Problem:\n{example.get('question','')}\nSolve this problem."
    parts=[]
    for label,key in [("Title","title"),("Summary","one_line"),("Goal","goal"),("Mechanism","mechanism")]:
        if view.get(key): parts.append(f"{label}: {view[key]}")
    if view.get("given"):
        given="\n".join(view["given"]) if isinstance(view["given"],list) else str(view["given"])
        parts.append(f"Given:\n{given}")
    if isinstance(view.get("constraints"),dict) and view["constraints"].get("critical"):
        parts.append("Key constraints:\n- " + "\n- ".join(map(str,view["constraints"]["critical"])))
    return "\n\n".join(parts)
def format_supervised_example(stage,example,target=None):
    if stage=="foundation":
        user,answer,system=example.get("instruction",""),target if target is not None else example.get("output",""),"You are a Python programming instructor."
    elif stage=="algorithms":
        user=algorithm_view_text(example); answer=target if target is not None else (example.get("solutions",["# Solution unavailable"])[0] if example.get("solutions") else "# Solution unavailable"); system="You are an algorithms expert who solves programming problems rigorously."
    elif stage=="debugging":
        fence=chr(96)*3; user=f"Review this code:\n{fence}python\n{example.get('code','')}\n{fence}\nIssue: {example.get('query','')}"; answer=target if target is not None else f"Issues:\n{example.get('feedback','')}\n\nCorrected code:\n{fence}python\n{example.get('corrected_code','')}\n{fence}"; system="You are a code review and debugging expert."
    elif stage=="advanced":
        user,answer,system=example.get("instruction",""),target if target is not None else example.get("response",""),"You are an expert programmer with strong code reasoning ability."
    else: raise KeyError(stage)
    return {"text":"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"+system+"<|eot_id|>\n"+"<|start_header_id|>user<|end_header_id|>\n"+user+"<|eot_id|>\n"+"<|start_header_id|>assistant<|end_header_id|>\n"+answer+"<|eot_id|>"}
def dataset_fingerprint(ds,sample_limit=1000):
    n=len(ds) if sample_limit is None else min(len(ds),sample_limit); h=hashlib.sha256()
    for row in ds.select(range(n)):
        h.update(json.dumps(row,ensure_ascii=False,sort_keys=True,default=str).encode()); h.update(b"\n")
    h.update(str(len(ds)).encode()); return h.hexdigest()
