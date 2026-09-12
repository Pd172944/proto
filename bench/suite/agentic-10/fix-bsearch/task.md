`search.py` implements binary search over a sorted list, but it returns the wrong
index in several cases and loops forever on some inputs.

Fix `bsearch` so it returns the index of `target` when present and `-1` when absent.
`test_search.py` has the cases the caller depends on.
