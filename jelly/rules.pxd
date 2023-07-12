from .typedefs cimport DTYPE_t

cdef cyclic_rule(int value, DTYPE_t[:] neighbors, int n_states, int threshold)
