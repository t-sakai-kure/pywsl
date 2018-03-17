"""
Check utility
"""


def same_dim(x1, x2):
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("""Dimension must be the same.
        Expected: x1.shape[1] == x2.shape[1]
        Actual: x1.shape[1] != x2.shape[1]""")


def in_range(x, a, b, name):
    if x <= a or b <= x:
        raise ValueError("{0} must satisfy {1} < {0} < {2}.".format(
            name, a, b))

