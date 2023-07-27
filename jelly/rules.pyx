# cython: binding=True, infer_types=True
# distutils: language = c++
cimport numpy as cnp
import numpy as np
from .typedefs cimport DTYPE_t
import cython
from libc.stdlib cimport malloc, free
from libcpp.unordered_map cimport unordered_map, pair



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
cdef DTYPE_t majority_rule(int value, DTYPE_t[:] neighbors) nogil:
    cdef unordered_map[DTYPE_t, int] counter
    cdef int max_count = -1
    cdef DTYPE_t max_item
    cdef int neighbor
    cdef pair[DTYPE_t, int] count_pair
    for i in range(neighbors.shape[0]):
        neighbor = neighbors[i]
        if neighbor != -1:
            counter[neighbor] += 1
    for count_pair in counter:
        if count_pair.second > max_count:
            max_count = count_pair.second
            max_item = count_pair.first
    return max_item
