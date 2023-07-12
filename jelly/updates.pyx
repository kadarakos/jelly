# cython: binding=True, infer_types=True
import numpy as np
from .rules cimport cyclic_rule
from .typedefs cimport DTYPE_t, IMGTYPE_t
from .typedefs import DTYPE
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
    height = C_padded.shape[0] - size
    width = C_padded.shape[1] - size
    out = np.empty((height, width), dtype="int8")
    out_rgb = np.empty((height, width, 3), dtype="uint8")
    cdef DTYPE_t[:, :] out_view = out
    cdef IMGTYPE_t[:, :, :] out_rgb_view = out_rgb
    cdef cnp.ndarray[DTYPE_t, ndim=1] neighbors
    for i in range(size, height):
        for j in range(size, width):
            value = C_padded[i, j]
            neighbors = neighborhood_func(C_padded, i, j, size)
            new_value = cyclic_rule(value, neighbors, n_states, threshold)
            out_view[i - size, j - size] = new_value
            out_rgb_view[i, j, :] = cmap[new_value % len(cmap)]
    return out, out_rgb
