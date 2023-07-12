# cython: binding=True, infer_types=True
import numpy as np

from .rules cimport cyclic_rule
from .typedefs cimport DTYPE_t, IMGTYPE_t
from .ty import DTYPE
from .cyhoods cimport hood_by_name
cimport numpy as cnp


def cyclic_step(
    const DTYPE_t[:, :] C_padded,
    int n_states,
    str neighborhood,
    const IMGTYPE_t[:, :] cmap,
    int threshold,
    int size,
):
    neighborhood_func = hood_by_name(neighborhood)
    height = C_padded.shape[0]
    width = C_padded.shape[1]
    out = np.empty((height, width), dtype="int8")
    out_rgb = np.empty((height - size * 2, width - size * 2, 3), dtype="uint8")
    cdef DTYPE_t[:, :] out_view = out
    cdef IMGTYPE_t[:, :, :] out_rgb_view = out_rgb
    cdef cnp.ndarray[DTYPE_t, ndim=1] neighbors
    cdef DTYPE_t value
    for i in range(height):
        for j in range(width):
            value = C_padded[i, j]
            # Padding
            if value == -1:
                new_value = -1
                out_view[i, j] = new_value
            else:
                neighbors = neighborhood_func(C_padded, i, j, size)
                new_value = cyclic_rule(value, neighbors, n_states, threshold)
                out_view[i, j] = new_value
                out_rgb_view[i - size, j - size, :] = cmap[new_value % len(cmap)]
    return out, out_rgb
