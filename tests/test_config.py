from cgkd_cr.config import resolve_stage_order

def test_modes():
    cfg={"seed":0,"curriculum":{"stages":["a","b","c"],"mode":"ordered"}}
    assert resolve_stage_order(cfg)==["a","b","c"]
    cfg["curriculum"]["mode"]="reverse"
    assert resolve_stage_order(cfg)==["c","b","a"]

def test_random_is_reproducible():
    cfg={"seed":7,"curriculum":{"stages":["a","b","c"],"mode":"random"}}
    assert resolve_stage_order(cfg)==resolve_stage_order(cfg)
