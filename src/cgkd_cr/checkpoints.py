from .config import save_json
def write_lineage(path,*,base_model,previous_checkpoint,stage,dataset,teacher,training_config,seed):
    save_json(path,{"base_model":base_model,"previous_checkpoint":previous_checkpoint,"stage":stage,"dataset":dataset,"teacher":teacher,"training_config":training_config,"seed":seed})
