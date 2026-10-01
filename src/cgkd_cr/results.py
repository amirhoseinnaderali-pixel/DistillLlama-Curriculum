from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentResult:
    experiment_id: str
    model: str
    method: str
    stage_order: list[str]
    teacher_models: list[str]
    train_examples: int
    train_tokens: int | None
    eval_examples: int | None
    teacher_tokens: int | None = None
    optimizer_steps: int | None = None
    accuracy: float | None = None
    solve_rate: float | None = None
    pass_rate: float | None = None
    compile_rate: float | None = None
    training_time_hours: float | None = None
    inference_latency_ms: float | None = None
    tokens_per_second: float | None = None
    gpu_memory_gb: float | None = None
    parameter_count: int | None = None
    trainable_parameter_count: int | None = None
    seed: int | None = None
    config_hash: str | None = None
    git_commit: str | None = None
    status: str = "pending_experiment"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
