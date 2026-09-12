from search import bsearch


def test_finds_first():
    assert bsearch([1, 3, 5, 7, 9], 1) == 0


def test_finds_last():
    assert bsearch([1, 3, 5, 7, 9], 9) == 4


def test_finds_middle():
    assert bsearch([1, 3, 5, 7, 9], 5) == 2


def test_absent():
    assert bsearch([1, 3, 5, 7, 9], 4) == -1


def test_empty():
    assert bsearch([], 1) == -1
