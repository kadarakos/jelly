# cython: binding=True, infer_types=True
cimport numpy as cnp
import numpy as np
from .typedefs cimport DTYPE_t
import cython
from libc.stdlib cimport malloc, free



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef DTYPE_t cyclic_rule(int value, DTYPE_t[:] neighbors, int n_states, int threshold) nogil:
    successor = (value + 1) % n_states
    cdef int n_successors = 0
    for i in range(neighbors.shape[0]):
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
    for i in range(neighbors.shape[0]):
        if neighbors[i] == 1:
            alive += 1
    # Cell is dead.
    if value == 0:
        for i in range(B.shape[0]):
            if alive == B[i]:
                return 1
        return 0
    # Cell is alive.
    else:
        for i in range(S.shape[0]):
            if alive == S[i]:
                return 1
        return 0



@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t majority_rule(int value, DTYPE_t[:] neighbors, int n_states) nogil:
    if value == -1:
        return value
    cdef int *counts = <int *> malloc(n_states * sizeof(int))
    cdef int maxi = -1
    cdef DTYPE_t argmaxi = -1
    cdef int i
    cdef DTYPE_t neighbor
    for i in range(neighbors.shape[0]):
        neighbor = neighbors[i]
        counts[neighbor] += 1
    for i in range(n_states):
        count = counts[i]
        if count > maxi:
            maxi = count
            argmaxi = i
    free(counts)
    return argmaxi
