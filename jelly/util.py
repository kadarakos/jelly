import numpy as np
from .ty import IMGTYPE


def enlarge_img(arr: np.ndarray, ratio: int) -> np.ndarray:
    big = np.kron(arr, np.ones((ratio, ratio, 1), dtype=IMGTYPE))
    return big
