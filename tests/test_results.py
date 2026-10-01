from cgkd_cr.results import ExperimentResult

def test_result():
    x=ExperimentResult("e","m","sft",["mixed"],[],10,0,5)
    assert x.to_dict()["eval_examples"]==5
