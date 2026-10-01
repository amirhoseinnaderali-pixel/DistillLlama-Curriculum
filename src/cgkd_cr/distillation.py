from __future__ import annotations
import os,time
from .data import format_supervised_example

class TeacherClient:
    """Response-level teacher client; not logit-level KD."""
    def __init__(self):
        self.google_api_key=os.getenv("GOOGLE_API_KEY"); self.openai_api_key=os.getenv("OPENAI_API_KEY"); self.openai_base_url=os.getenv("OPENAI_BASE_URL")
        self._gemini=None; self._openai=None
        if self.google_api_key:
            try:
                from google import genai; self._gemini=genai.Client(api_key=self.google_api_key)
            except Exception: self._gemini=None
        if self.openai_api_key and self.openai_base_url:
            from openai import OpenAI; self._openai=OpenAI(api_key=self.openai_api_key,base_url=self.openai_base_url)
    def generate(self,model,prompt,max_tokens=1024,temperature=0.7,retries=3):
        last=None
        for attempt in range(retries):
            try:
                if model.startswith("gemini"):
                    if self._gemini is None: raise RuntimeError("GOOGLE_API_KEY is required for Gemini.")
                    response=self._gemini.models.generate_content(model=model,contents=prompt,config={"max_output_tokens":max_tokens,"temperature":temperature}); text=getattr(response,"text","") or ""
                else:
                    if self._openai is None: raise RuntimeError("OPENAI_API_KEY and OPENAI_BASE_URL are required for non-Gemini teachers.")
                    response=self._openai.chat.completions.create(model=model,messages=[{"role":"system","content":"You are an expert coding and algorithmic reasoning teacher."},{"role":"user","content":prompt}],max_tokens=max_tokens,temperature=temperature); text=response.choices[0].message.content or ""
                if text.strip(): return text.strip()
                raise RuntimeError("Empty teacher response")
            except Exception as exc:
                last=exc
                if attempt+1<retries: time.sleep(2**attempt)
        raise RuntimeError(f"Teacher generation failed after {retries} attempts: {last}")
def make_prompt(stage,example):
    return "Produce only the high-quality student target response for the following task.\n\n"+format_supervised_example(stage,example,target="")["text"]
def generate_records(rows,stage,teacher_model,source_dataset,max_samples,max_tokens,temperature):
    client=TeacherClient(); records=[]
    for i,row in enumerate(rows):
        if max_samples is not None and i>=max_samples: break
        answer=client.generate(teacher_model,make_prompt(stage,row),max_tokens=max_tokens,temperature=temperature); example_id=str(row.get("id") or row.get("problem_id") or i)
        records.append({"problem_id":example_id,"stage":stage,"teacher_model":teacher_model,"teacher_response":answer,"generation_parameters":{"max_tokens":max_tokens,"temperature":temperature},"source_dataset":source_dataset,"quality_filter":"non_empty_response","student_target":answer,"source_fields":{k:row.get(k) for k in ("question","title","instruction","code")},"text":format_supervised_example(stage,row,target=answer)["text"]})
    return records
