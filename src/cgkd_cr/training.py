from __future__ import annotations
from pathlib import Path
import json
import time
from .checkpoints import write_lineage
from .config import resolve_stage_order,save_json,set_seed
from .data import STAGES,format_supervised_example,load_stage_dataset
from .distillation import generate_records
def load_model(model_name,max_seq_length):
    try: from unsloth import FastLanguageModel
    except ImportError as exc: raise RuntimeError("Unsloth is required for training.") from exc
    import torch
    model,tokenizer=FastLanguageModel.from_pretrained(model_name=model_name,max_seq_length=max_seq_length,dtype=torch.float16,load_in_4bit=True,device_map="auto"); return FastLanguageModel,model,tokenizer
def attach_lora(FastLanguageModel,model,cfg):
    return FastLanguageModel.get_peft_model(model,r=cfg["lora_r"],lora_alpha=cfg.get("lora_alpha",cfg["lora_r"]*2),lora_dropout=cfg.get("lora_dropout",0.05),target_modules=cfg.get("target_modules",["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),use_gradient_checkpointing="unsloth")
def make_trainer(model,tokenizer,dataset,cfg):
    from transformers import TrainingArguments
    from trl import SFTTrainer
    try:
        from trl import SFTConfig
        args=SFTConfig(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            warmup_steps=cfg.get("warmup_steps",0),
            fp16=cfg.get("fp16",True),
            logging_steps=cfg.get("logging_steps",10),
            optim=cfg.get("optim","adamw_8bit"),
            weight_decay=cfg.get("weight_decay",0.01),
            lr_scheduler_type=cfg.get("lr_scheduler_type","cosine"),
            save_strategy=cfg.get("save_strategy","epoch"),
            save_total_limit=cfg.get("save_total_limit",2),
            report_to="none",
            dataset_text_field="text",
            max_length=cfg["max_seq_length"],
        )
        return SFTTrainer(model=model,train_dataset=dataset,args=args,processing_class=tokenizer)
    except (ImportError,TypeError):
        args=TrainingArguments(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            num_train_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            warmup_steps=cfg.get("warmup_steps",0),
            fp16=cfg.get("fp16",True),
            logging_steps=cfg.get("logging_steps",10),
            optim=cfg.get("optim","adamw_8bit"),
            weight_decay=cfg.get("weight_decay",0.01),
            lr_scheduler_type=cfg.get("lr_scheduler_type","cosine"),
            save_strategy=cfg.get("save_strategy","epoch"),
            save_total_limit=cfg.get("save_total_limit",2),
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            report_to="none",
        )
        return SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=args,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=cfg["max_seq_length"],
        )
def _token_count(dataset, tokenizer, batch_size=32):
    total=0
    for start in range(0,len(dataset),batch_size):
        batch=dataset[start:start+batch_size]["text"]
        encoded=tokenizer(batch, add_special_tokens=True, truncation=False)
        total += sum(len(ids) for ids in encoded["input_ids"])
    return int(total)


def _teacher_token_count(records, tokenizer):
    total=0
    for row in records:
        text=str(row.get("teacher_response",""))
        if text:
            total += len(tokenizer(text, add_special_tokens=True)["input_ids"])
    return int(total)


def _write_jsonl(rows, path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+"\n")


def train_experiment(config):
    set_seed(int(config.get("seed",0))); base_model=config["model"]["base_model"]; order=resolve_stage_order(config); teacher_enabled=bool(config.get("distillation",{}).get("enabled")); teacher=config.get("distillation",{}).get("teacher_model"); root=Path(config["runtime"]["output_root"]); previous=None; records=[]; started=time.perf_counter()
    distill_root=root/"distillation"
    teacher_rows_by_stage={}
    def prepared(stage,ds):
        if teacher_enabled:
            override=config.get("stage_overrides",{}).get(stage,{})
            rows=generate_records(ds,stage=stage,teacher_model=teacher,source_dataset=override.get("dataset",STAGES[stage]["dataset"]),max_samples=config["distillation"].get("max_samples"),max_tokens=int(config["distillation"].get("max_tokens",1024)),temperature=float(config["distillation"].get("temperature",0.7)))
            _write_jsonl(rows,distill_root/f"{stage}.jsonl")
            teacher_rows_by_stage[stage]=rows
            from datasets import Dataset
            return Dataset.from_list(rows)
        teacher_rows_by_stage[stage]=[]
        return ds.map(lambda x:format_supervised_example(stage,x))
    if order==["mixed"]:
        from datasets import concatenate_datasets
        parts=[prepared(stage,load_stage_dataset(stage,config)) for stage in config["curriculum"]["stages"]]; train_ds=concatenate_datasets(parts)
        cfg=dict(config["training"]["stage_defaults"]); cfg["output_dir"]=str(root/"mixed"); FastLanguageModel,model,tokenizer=load_model(base_model,cfg["max_seq_length"]); model=attach_lora(FastLanguageModel,model,cfg); result=make_trainer(model,tokenizer,train_ds,cfg).train()
        Path(cfg["output_dir"]).mkdir(parents=True,exist_ok=True); model.save_pretrained(cfg["output_dir"]); tokenizer.save_pretrained(cfg["output_dir"]); records.append({"stage":"mixed","examples":len(train_ds),"metrics":result.metrics})
    else:
        for idx,stage in enumerate(order):
            cfg=dict(config["training"]["stage_defaults"]); cfg.update(config.get("stage_training",{}).get(stage,{})); cfg["output_dir"]=str(root/f"stage{idx+1}-{stage}"); ds=prepared(stage,load_stage_dataset(stage,config))
            FastLanguageModel,model,tokenizer=load_model(previous or base_model,cfg["max_seq_length"]); model=attach_lora(FastLanguageModel,model,cfg); result=make_trainer(model,tokenizer,ds,cfg).train()
            Path(cfg["output_dir"]).mkdir(parents=True,exist_ok=True); model.save_pretrained(cfg["output_dir"]); tokenizer.save_pretrained(cfg["output_dir"]); write_lineage(Path(cfg["output_dir"])/"checkpoint_lineage.json",base_model=base_model,previous_checkpoint=previous,stage=stage,dataset=config.get("stage_overrides",{}).get(stage,{}).get("dataset",STAGES[stage]["dataset"]),teacher=teacher if teacher_enabled else None,training_config=config.get("config_path","unknown"),seed=int(config.get("seed",0))); previous=cfg["output_dir"]; records.append({"stage":stage,"examples":len(ds),"metrics":result.metrics})
    manifest={"stage_order":order,"final_model":previous,"stage_records":records,"train_examples":sum(x["examples"] for x in records),"train_tokens":sum(int(x.get("train_tokens",0) or 0) for x in records),"teacher_tokens":sum(int(x.get("teacher_tokens",0) or 0) for x in records) if teacher_enabled else None,"optimizer_steps":sum(int(x.get("optimizer_steps",0) or 0) for x in records),"training_time_hours":(time.perf_counter()-started)/3600,"teacher_models":[teacher] if teacher_enabled else []}; save_json(root/"run_manifest.json",manifest); return manifest
