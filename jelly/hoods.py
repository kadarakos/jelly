import numpy as np
from enum import Enum
from functools import partial

from .util import EnumChoices


def moore(C: np.ndarray, i: int, j: int, size: int) -> np.ndarray:
    """
        + + + + +
        + + + + +
        + + x + +
        + + + + +
        + + + + +
    """
    n = sum(8 * s for s in range(1, size + 1))
    p = 0
    out = np.empty(n, dtype="int8")
    for i_ in range(i - size, i + size + 1):
        for j_ in range(j - size, j + size + 1):
            if i_ == i and j_ == j:
                continue
            out[p] = C[i_, j_]
            p += 1
    return out


def moore_rim(C: np.ndarray, i: int, j: int, size: int) -> np.ndarray:
    """
        + + + + +
        +       +
        +   x   +
        +       +
        + + + + +
    """
    n = 8 * size
    p = 0
    out = np.empty(n, dtype="int8")
    for row in (i - size, i + size):
        for j_ in range(j - size, j + size + 1):
            out[p] = C[row, j_]
            p += 1
    for column in (j - size, j + size):
        for i_ in range(i - size + 1, i + size):
            out[p] = C[i_, column]
            p += 1
    return out


# TODO
def neumann(C: np.ndarray, i: int, j: int, size: int) -> np.ndarray:
    """
            +
          + + +
        + + x + +
          + + +
            +
    """
    raise NotImplementedError


# TODO
def neumann_rim(C: np.ndarray, i: int, j: int, size: int) -> np.ndarray:
    """
            +
          +   +
        +   x   +
          +   +
            +
    """
    raise NotImplementedError


def cross(C: np.ndarray, i: int, j: int, size: int) -> np.ndarray:
    """
            +
            +
        + + x + +
            +
            +
    """
    n = size * 4
    p = 0
    out = np.empty(n, dtype="int8")
    for i_ in range(i - size, i + size + 1):
        if i_ == i:
            continue
        out[p] = C[i_, j]
        p += 1
    for j_ in range(j - size, j + size + 1):
        if j_ == j:
            continue
        out[p] = C[i, j_]
        p += 1
    return out


class HoodFuncs(Enum):
    moore = partial(moore)
    moore_rim = partial(moore_rim)
    neumann = partial(neumann)
    neumann_rim = partial(neumann_rim)
    cross = partial(cross)


HoodChoices = EnumChoices("HoodChoices", HoodFuncs)
