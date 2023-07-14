# cython: binding=True, infer_types=True
# distutils: language = c++
import numpy as np

from .rules cimport cyclic_rule
from .typedefs cimport DTYPE_t, IMGTYPE_t
from .ty import DTYPE
from .cyhoods cimport hood_by_name, hood_size, moore
cimport numpy as cnp
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyclic_step(
    const DTYPE_t[:, :] C_padded,
    int n_states,
    str neighborhood,
    int neighborhood_size,
    const IMGTYPE_t[:, :] cmap,
    int threshold,
    int size,
    const IMGTYPE_t[:, :, :] out_rgb
):
    neighborhood_func = hood_by_name(neighborhood)
    height = C_padded.shape[0]
    width = C_padded.shape[1]
    out = np.empty((height, width), dtype=DTYPE)
    neighbors = np.empty((neighborhood_size, ), dtype=DTYPE)
    cdef DTYPE_t[:, :] out_view = out
    cdef DTYPE_t[:] neighbors_view = neighbors
    cdef DTYPE_t value
    cdef DTYPE_t new_value
    cdef int cmap_size = cmap.shape[0]
    cdef int i, j
    for i in range(height):
        for j in range(width):
            value = C_padded[i, j]
            # Padding
            if value == -1:
                new_value = -1
                out_view[i, j] = new_value
            else:
                moore(C_padded, i, j, size, neighbors_view)
                new_value = cyclic_rule(value, neighbors_view, n_states, threshold)
                out_view[i, j] = new_value
                out_rgb[i - size, j - size, :] = cmap[new_value % cmap_size]
    return out
