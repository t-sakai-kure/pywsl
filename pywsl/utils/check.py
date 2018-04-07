"""
Check utility
"""

import numpy as np


def same_dim(x1, x2):
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("""Dimension must be the same.
        Expected: x1.shape[1] == x2.shape[1]
        Actual: x1.shape[1] != x2.shape[1]""")


def in_range(x, a, b, name):
    if x <= a or b <= x:
        raise ValueError("{0} must satisfy {1} < {0} < {2}.".format(
            name, a, b))


def check_bags(input):
    """Input validation on a list of bags.

    The input must be list of bags. By default, each bag is converted to
    at least 2D numpy array.
    """
    if not isinstance(input, list):
        raise TypeError("Input type of bags must be list.")

    bags = [np.array(array) for array in input]
    for i in range(len(bags)):
        if bags[i].ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\nbag={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(bags[i]))

        bags[i] = np.atleast_2d(bags[i])

    return bags
