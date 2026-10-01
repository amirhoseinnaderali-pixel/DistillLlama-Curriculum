# Experimental Matrix

| Method | Curriculum | Teacher supervision | LoRA |
|---|---:|---:|---:|
| SFT baseline | No | No | Yes |
| Distillation baseline | No | Yes | Yes |
| Curriculum | Yes | No | Yes |
| CGKD-CR | Yes | Yes | Yes |

Planned ablations:
- random stage order
- reverse stage order
- remove one stage
- three-stage curriculum
- one teacher vs multiple teachers
- teacher prompt changes
- teacher strength changes
- LoRA rank, learning rate, epochs, data-volume controls

Every unrun cell remains Pending evaluation.
