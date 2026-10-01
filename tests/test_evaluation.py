from cgkd_cr.evaluation import compile_ok,extract_python

def test_extract_python():
    fence=chr(96)*3
    text="answer\n"+fence+"python\ndef f():\n    return 1\n"+fence
    assert "def f" in extract_python(text)

def test_compile():
    assert compile_ok("def f():\n    return 1")
    assert not compile_ok("def f(:")
