import numpy as np
import imageio
import tqdm
import typer

from typing import Optional

from .cyhoods import HoodChoices
from .color import ColorChoices
from .updates import cyclic_step


def cca_video(
    output_file: str,
    steps: int,
    height: int,
    width: int,
    states: int,
    neighborhood: HoodChoices,
    colormap: ColorChoices.options,
    *,
    threshold: Optional[int] = 1,
    size: Optional[int] = 1
):
    colormap = ColorChoices.resolve(colormap.name)
    C = np.random.randint(0, states, (height, width), dtype="int8")
    C_padded = np.pad(C, (size, size), constant_values=(-1, -1))
    # XXX I don't like it but its faster to allocate C_rgb once and mutate.
    C_rgb = np.empty((height, width, 3), dtype="uint8")
    with imageio.get_writer(output_file, mode='I') as writer:
        for step in tqdm.tqdm(range(steps)):
            C_padded = cyclic_step(
                C_padded,
                states,
                neighborhood.name,
                colormap,
                threshold,
                size,
                C_rgb,
            )
            writer.append_data(C_rgb)


if __name__ == "__main__":
    typer.run(cca_video)
