from .typedefs cimport DTYPE_t, hoodfunc_t
cimport numpy as cnp

cdef void moore(const DTYPE_t[:, :] C, int i, int j, int size, DTYPE_t[:] out) nogil except +
cdef void moore_rim(const DTYPE_t[:, :] C, int i, int j, int size, DTYPE_t[:] out) nogil except +
cdef void cross(const DTYPE_t[:, :] C, int i, int j, int size, DTYPE_t[:] out) nogil except +
cdef hoodfunc_t hood_by_name(str name)
cpdef int hood_size(str name, int size)
