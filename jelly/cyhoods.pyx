# cython: binding=True, infer_types=True
# distutils: language = c++
import numpy as np
cimport numpy as cnp
import cython
from .typedefs cimport DTYPE_t, hoodfunc_t
from .ty import DTYPE
from enum import Enum


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void moore(const DTYPE_t[:, :] C, int i, int j, int size, DTYPE_t[:] out) noexcept nogil:
    """
        + + + + +
        + + + + +
        + + x + +
        + + + + +
        + + + + +
    """
    cdef int p = 0
    for i_ in range(i - size, i + size + 1):
        for j_ in range(j - size, j + size + 1):
            if i_ == i and j_ == j:
                continue
            out[p] = C[i_, j_]
            p += 1



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void moore_rim(const DTYPE_t[:, :] C, int i, int j, int size, DTYPE_t[:] out) noexcept nogil:
    """
        + + + + +
        +       +
        +   x   +
        +       +
        + + + + +
    """
    cdef int p = 0
    for j_ in range(j - size, j + size + 1):
        out[p] = C[i - size, j_]
        p += 1
    for j_ in range(j - size, j + size + 1):
        out[p] = C[i + size, j_]
        p += 1
    for i_ in range(i - size + 1, i + size):
        out[p] = C[i_, j - size]
        p += 1
    for i_ in range(i - size + 1, i + size):
        out[p] = C[i_, j + size]
        p += 1



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cross(const DTYPE_t[:, :] C, int i, int j, int size, DTYPE_t[:] out) noexcept nogil:
    """
            +
            +
        + + x + +
            +
            +
    """
    cdef int p = 0
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


cdef hoodfunc_t hood_by_name(str name):
    if name == "moore":
        return moore
    elif name == "moore_rim":
        return moore_rim
    elif name == "cross":
        return cross


cpdef int hood_size(str name, int size):
    cdef int n = 0
    if name == "moore":
        for s in range(1, size + 1):
            n += 8 * s
    elif name == "moore_rim":
        n = 8 * size
    elif name == "cross":
        n = 4 * size
    return n


class HoodOptions(str, Enum):
    moore = "moore"
    moore_rim = "moore_rim"
    cross = "cross"
