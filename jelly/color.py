from enum import Enum
import numpy as np

from .util import EnumChoices


class ColorMaps(Enum):
    """
    unspoken = [
        (148, 162, 111),
        (120, 126, 88),
        (34, 45, 38),
        (138, 148, 148),
        (220, 222, 181),
    ]

    juju = np.array([
        (29, 44, 55),
        (22, 76, 55),
        (81, 132, 133),
        (137, 223, 196),
        (212, 231, 213),
    ], dtype="uint8")
    """

    blues = np.array([
        (0, 150, 255),
        (39, 166, 255),
        (77, 181, 255),
        (105, 193, 255),
        (206, 235, 255),
    ], dtype="uint8")


ColorChoices = EnumChoices("ColorChoices", ColorMaps)
