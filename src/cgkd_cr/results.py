from dataclasses import asdict,dataclass
@dataclass
class ExperimentResult:
    experiment_id:str; model:str; method:str; stage_order:list[str]; teacher_models:list[str]; train_examples:int; train_tokens:int|None; eval_examples:int|None
    accuracy:float|None=None; pass_rate:float|None=None; compile_rate:float|None=None; training_time_hours:float|None=None; inference_latency_ms:float|None=None; gpu_memory_gb:float|None=None
    def to_dict(self): return asdict(self)
