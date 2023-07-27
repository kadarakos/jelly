# cython: binding=True, infer_types=True
# distutils: language = c++

import numpy as np
from enum import Enum
from functools import partial


from .rules cimport cyclic_rule, lifelike_rule, majority_rule
from .typedefs cimport DTYPE_t, IMGTYPE_t
from .ty import DTYPE
from .cyhoods cimport hood_by_name, moore
cimport numpy as cnp
import cython
from libcpp.unordered_map cimport unordered_map



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cyclic_step(
    const DTYPE_t[:, :] C_padded,
    DTYPE_t[:, :] out,
    IMGTYPE_t[:, :, :] out_rgb,
    DTYPE_t[:] neighbors,
    IMGTYPE_t[:, :] cmap,
    *,
    int states,
    str neighborhood,
    int threshold,
    int size
):
    neighborhood_func = hood_by_name(neighborhood)
    height = C_padded.shape[0]
    width = C_padded.shape[1]
    cdef DTYPE_t value
    cdef DTYPE_t new_value
    cdef int cmap_size = cmap.shape[0]
    cdef int i, j, rgb_i, rgb_j
    cdef int color
    for i in range(height):
        for j in range(width):
            value = C_padded[i, j]
            if value == -1:
                out[i, j] = value
            else:
                rgb_i = i - size
                rgb_j = j - size
                neighborhood_func(C_padded, i, j, size, neighbors)
                new_value = cyclic_rule(value, neighbors, states, threshold)
                out[i, j] = new_value
                # Stupid way of writing because slicing seems slow?
                color = new_value % cmap_size
                out_rgb[rgb_i, rgb_j, 0] = cmap[color][0]
                out_rgb[rgb_i, rgb_j, 1] = cmap[color][1]
                out_rgb[rgb_i, rgb_j, 2] = cmap[color][2]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lifelike_step(
    const DTYPE_t[:, :] C_padded,
    DTYPE_t[:, :] out,
    IMGTYPE_t[:, :, :] out_rgb,
    DTYPE_t[:] neighbors,
    IMGTYPE_t[:, :] cmap,
    *,
    DTYPE_t[:] B,
    DTYPE_t[:] S,
):
    height = C_padded.shape[0]
    width = C_padded.shape[1]
    cdef DTYPE_t value
    cdef DTYPE_t new_value
    cdef int cmap_size = cmap.shape[0]
    cdef int i, j, rgb_i, rgb_j
    cdef int color
    for i in range(height):
        for j in range(width):
            value = C_padded[i, j]
            if value == -1:
                out[i, j] = value
            else:
                rgb_i = i - 1
                rgb_j = j - 1
                moore(C_padded, i, j, 1, neighbors)
                new_value = lifelike_rule(value, neighbors, B, S)
                out[i, j] = new_value
                # Stupid way of writing because slicing seems slow?
                color = new_value % cmap_size
                out_rgb[rgb_i, rgb_j, 0] = cmap[color][0]
                out_rgb[rgb_i, rgb_j, 1] = cmap[color][1]
                out_rgb[rgb_i, rgb_j, 2] = cmap[color][2]
    return out


def majority_step(
    const DTYPE_t[:, :] C_padded,
    DTYPE_t[:, :] out,
    IMGTYPE_t[:, :, :] out_rgb,
    DTYPE_t[:] neighbors,
    IMGTYPE_t[:, :] cmap,
    *,
    int states,
    str neighborhood,
    int size
):
    height = C_padded.shape[0]
    width = C_padded.shape[1]
    neighborhood_func = hood_by_name(neighborhood)
    cdef DTYPE_t value
    cdef DTYPE_t new_value
    cdef int cmap_size = cmap.shape[0]
    cdef int i, j, rgb_i, rgb_j
    cdef int color
    for i in range(height):
        for j in range(width):
            value = C_padded[i, j]
            if value == -1:
                out[i, j] = value
            else:
                rgb_i = i - size
                rgb_j = j - size
                neighborhood_func(C_padded, i, j, 1, neighbors)
                new_value = majority_rule(value, neighbors)
                out[i, j] = new_value
                color = new_value % cmap_size
                out_rgb[rgb_i, rgb_j, 0] = cmap[color][0]
                out_rgb[rgb_i, rgb_j, 1] = cmap[color][1]
                out_rgb[rgb_i, rgb_j, 2] = cmap[color][2]
    return out


class AutomataOptions(Enum):
    cca = partial(cyclic_step)
    lifelike = partial(lifelike_step)
    majority = partial(majority_step)
