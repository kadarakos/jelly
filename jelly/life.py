import numpy as np
import typer
import imageio
import tqdm

from typing import List, Optional

from .color import ColorChoices
from .ty import DTYPE, IMGTYPE
from .cyhoods import hood_size
from .updates import lifelike_step
from .util import enlarge_img


def life_video(
    output_file: str,
    steps: int,
    height: int,
    width: int,
    ratio: int,
    colormap: ColorChoices.options,
    B: Optional[List[int]] = None,
    S: Optional[List[int]] = None,
):
    colormap = ColorChoices.resolve(colormap.name)
    C = np.random.randint(0, 2, (height, width), dtype=DTYPE)
    B = np.array(B, dtype=DTYPE)
    S = np.array(S, dtype=DTYPE)
    C_padded = np.pad(C, (1, 1), constant_values=(-1, -1))
    C_rgb = np.empty((height, width, 3), dtype=IMGTYPE)
    neighborhood_size = hood_size("moore", 1)
    neighbors = np.empty((neighborhood_size, ), dtype=DTYPE)
    with imageio.get_writer(output_file, mode='I') as writer:
        for step in tqdm.tqdm(range(steps)):
            out = np.empty(C_padded.shape, dtype=DTYPE)
            C_padded = lifelike_step(
                C_padded,
                neighbors,
                colormap,
                B,
                S,
                out,
                C_rgb,
            )
            writer.append_data(enlarge_img(C_rgb, ratio))


if __name__ == "__main__":
    typer.run(life_video)
