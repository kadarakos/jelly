from enum import Enum

import numpy as np

from .ty import IMGTYPE


class EnumChoices:
    enumeration: Enum
    options: Enum

    def __init__(self, name: str, enumeration: Enum):
        self.enumeration = enumeration
        options = {x.name: x.name for x in enumeration}
        self.options = Enum(name, options)

    def resolve(self, option: str):
        return getattr(self.enumeration, option).value


def enlarge_img(arr: np.ndarray, ratio: int):
    big = np.kron(arr, np.ones((ratio, ratio, 1), dtype=IMGTYPE))
    return big
