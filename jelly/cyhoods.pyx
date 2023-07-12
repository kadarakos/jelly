# cython: binding=True, infer_types=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as cnp
import cython
from .typedefs cimport DTYPE_t, hoodfunc_t
from .typedefs import DTYPE
# This should be enough for cells.


#XXX NOTHING HAS RETURN TYPES?
@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[DTYPE_t, ndim=1] moore(const DTYPE_t[:, :] C, int i, int j, int size):
    cdef int n = 0
    for s in range(1, size + 1):
        n += 8 * s
    cdef int p = 0
    out = np.empty(n, dtype=DTYPE)
    cdef DTYPE_t[:] out_view = out
    for i_ in range(i - size, i + size + 1):
        for j_ in range(j - size, j + size + 1):
            if i_ == i and j_ == j:
                continue
            out_view[p] = C[i_, j_]
            p += 1
    return out



@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[DTYPE_t, ndim=1] moore_rim(const DTYPE_t[:, :] C, int i, int j, int size):
    """
        + + + + +
        +       +
        +   x   +
        +       +
        + + + + +
    """
    cdef int n = 8 * size
    cdef int p = 0
    out = np.empty(n, dtype="int8")
    cdef DTYPE_t[:] out_view = out
    for row in (i - size, i + size):
        for j_ in range(j - size, j + size + 1):
            out_view[p] = C[row, j_]
            p += 1
    for column in (j - size, j + size):
        for i_ in range(i - size + 1, i + size):
            out_view[p] = C[i_, column]
            p += 1
    return out



@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[DTYPE_t, ndim=1] cross(const DTYPE_t[:, :] C, int i, int j, int size):
    """
            +
            +
        + + x + +
            +
            +
    """
    cdef int n = size * 4
    cdef int p = 0
    out = np.empty(n, dtype="int8")
    cdef DTYPE_t[:] out_view = out
    for i_ in range(i - size, i + size + 1):
        if i_ == i:
            continue
        out_view[p] = C[i_, j]
        p += 1
    for j_ in range(j - size, j + size + 1):
        if j_ == j:
            continue
        out_view[p] = C[i, j_]
        p += 1
    return out


cdef hoodfunc_t hood_by_name(str name):
    if name == "moore":
        return moore
    elif name == "moore_rim":
        return moore_rim
    elif name == "cross":
        return cross
