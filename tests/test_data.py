from cgkd_cr.data import format_supervised_example

def test_algorithm():
    out=format_supervised_example("algorithms",{"question":"Q","solutions":["def f(): pass"]})
    assert "Q" in out["text"] and "def f" in out["text"]

def test_debugging():
    out=format_supervised_example("debugging",{"code":"x=1","query":"bug","feedback":"bad","corrected_code":"x=2"})
    assert "Corrected code" in out["text"]
