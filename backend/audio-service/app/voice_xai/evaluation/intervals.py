"""Interval operations on the shared source-audio timeline."""

import math


# Merge overlapping/touching half-open spans without filling unknown gaps.
def union_spans(spans):
    ordered = sorted((float(start), float(end)) for start, end in spans)
    result = []
    for start, end in ordered:
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError('Expected finite, positive source-audio intervals.')
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return tuple(result)


# Intersect interval unions so evidence is compared only on common coverage.
def intersect_spans(left, right):
    left, right = union_spans(left), union_spans(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return union_spans(result)


# Remove failed or unprocessed window supports, including overlapping portions.
def subtract_spans(spans, excluded):
    result = []
    excluded = union_spans(excluded)
    for start, end in union_spans(spans):
        cursor = start
        for cut_start, cut_end in excluded:
            if cut_end <= cursor:
                continue
            if cut_start >= end:
                break
            if cut_start > cursor:
                result.append((cursor, min(cut_start, end)))
            cursor = max(cursor, cut_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return tuple(result)


# Measure union duration once, avoiding double-counted overlapping windows.
def span_duration(spans):
    return sum(end - start for start, end in union_spans(spans))
