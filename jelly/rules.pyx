# cython: binding=True, infer_types=True
cimport numpy as cnp
import numpy as np
from .typedefs cimport DTYPE_t
import cython



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_t cyclic_rule(int value, DTYPE_t[:] neighbors, int n_states, int threshold) nogil:
    successor = (value + 1) % n_states
    cdef Py_ssize_t size = neighbors.shape[0]
    cdef int n_successors = 0
    for i in range(size):
        if neighbors[i] == successor:
            n_successors += 1
    if n_successors >= threshold:
        return successor
    else:
        return value


@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t lifelike_rule(int value, DTYPE_t[:] neighbors, DTYPE_t[:] B, DTYPE_t[:] S) nogil:
    # Padding
    if value == -1:
        return -1
    cdef int alive = 0
    cdef Py_ssize_t neighbors_size = neighbors.shape[0]
    cdef Py_ssize_t B_size = B.shape[0]
    cdef Py_ssize_t S_size = S.shape[0]
    for i in range(neighbors_size):
        if neighbors[i] == 1:
            alive += 1
    # Cell is dead.
    if value == 0:
        for i in range(B_size):
            if alive == B[i]:
                return 1
        return 0
    # Cell is alive.
    else:
        for i in range(S_size):
            if alive == S[i]:
                return 1
        return 0
