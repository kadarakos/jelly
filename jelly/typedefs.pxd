cimport numpy as cnp
import numpy as np


ctypedef cnp.int8_t DTYPE_t
ctypedef cnp.uint8_t IMGTYPE_t
ctypedef cnp.ndarray[DTYPE_t, ndim=1] (*hoodfunc_t)(const DTYPE_t[:, :], int i, int j, int size)
