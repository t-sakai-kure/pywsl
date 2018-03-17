"""
Check utility
"""


def same_dim(x1, x2):
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("""Dimension must be the same.
        Expected: x1.shape[1] == x2.shape[1]
        Actual: x1.shape[1] != x2.shape[1]""")
