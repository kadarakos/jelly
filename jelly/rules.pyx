# cython: binding=True, infer_types=True
cimport numpy as cnp
from .typedefs cimport DTYPE_t
import cython



@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t cyclic_rule(int value, DTYPE_t[:] neighbors, int n_states, int threshold):
    successor = (value + 1) % n_states
    cdef Py_ssize_t size = neighbors.shape[0]
    cdef Py_ssize_t i
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
def game_of_life():
    ...
