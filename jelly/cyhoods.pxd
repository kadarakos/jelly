from .typedefs cimport DTYPE_t, hoodfunc_t
cimport numpy as cnp

cdef cnp.ndarray[DTYPE_t, ndim=1] moore(const DTYPE_t[:, :] C, int i, int j, int size)
cdef cnp.ndarray[DTYPE_t, ndim=1] moore_rim(const DTYPE_t[:, :] C, int i, int j, int size)
cdef cnp.ndarray[DTYPE_t, ndim=1] cross(const DTYPE_t[:, :] C, int i, int j, int size)
cdef hoodfunc_t hood_by_name(str name)
