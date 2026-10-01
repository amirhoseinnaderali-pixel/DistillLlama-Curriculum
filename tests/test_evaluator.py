from scripts.evaluate_checkpoints import extract_code


def test_extract_code():
    fence = chr(96) * 3
    src = fence + "python\nprint('ok')\n" + fence
    assert extract_code(src) == "print('ok')"
