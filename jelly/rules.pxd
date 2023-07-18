from .typedefs cimport DTYPE_t

cdef DTYPE_t cyclic_rule(int value, DTYPE_t[:] neighbors, int n_states, int threshold) nogil
cdef DTYPE_t lifelike_rule(int value, DTYPE_t[:] neighbors, DTYPE_t[:] B, DTYPE_t[:] S) nogil
cdef DTYPE_t majority_rule(int value, DTYPE_t[:] neighbors, int n_states) nogil
