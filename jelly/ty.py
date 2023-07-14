import numpy as np

from typing import Callable, Tuple, List


# TODO this typing is pretty shit and its just a sketch.
HoodFunc = Callable[[np.ndarray, int, int, int], np.ndarray]
StepFunc = Callable[[np.ndarray, ...], np.ndarray]
RGB = Tuple[int, int, int]
CMAP = List[RGB]
DTYPE = np.int8
IMGTYPE = np.uint8
