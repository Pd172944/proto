def bsearch(items, target):
    """Return the index of target in the sorted list items, or -1 if absent."""
    lo = 0
    hi = len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid] == target:
            return mid
        if items[mid] < target:
            lo = mid
        else:
            hi = mid
    return -1
