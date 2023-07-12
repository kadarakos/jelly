import numpy as np
import imageio
import tqdm
import typer

from typing import Optional, Tuple
from functools import partial

from .ty import HoodFunc, CMAP
from .hoods import HoodChoices
from .color import ColorChoices


def cca_step(
    C: np.ndarray,
    n_states: int,
    neighborhood: HoodFunc,
    cmap: CMAP,
    *,
    threshold: Optional[int] = 1,
    size: Optional[int] = 1,
    padding: Optional[int] = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = C.shape
    # XXX Padding is expensive to do at each step.
    C_padded = np.pad(C, (size, size), constant_values=(padding, padding))
    out = np.empty((height, width), dtype="uint8")
    out_rgb = np.empty((height, width, 3), dtype="uint8")
    for i in range(size, height):
        for j in range(size, width):
            value = C_padded[i, j]
            successor = (value + 1) % n_states
            neighbors = neighborhood(C_padded, i, j, size)
            n_successors = (neighbors == successor).sum()
            if n_successors >= threshold:
                new_val = successor
            else:
                new_val = C_padded[i, j]
            out[i - size, j - size] = new_val
            out_rgb[i, j, :] = cmap[new_val % len(cmap)]
    return out, out_rgb


def cca_video(
    output_file: str,
    steps: int,
    height: int,
    width: int,
    states: int,
    neighborhood: HoodChoices.options,
    colormap: ColorChoices.options,
    *,
    threshold: Optional[int] = 1,
    size: Optional[int] = 1
):
    neighborhood = HoodChoices.resolve(neighborhood.name)
    colormap = ColorChoices.resolve(colormap.name)
    step_func = partial(
        cca_step,
        n_states=states,
        neighborhood=neighborhood,
        cmap=colormap,
        threshold=threshold,
        size=size,
        padding=-1
    )
    with imageio.get_writer(output_file, mode='I') as writer:
        C = np.random.randint(0, states, (height, width))
        for step in tqdm.tqdm(range(steps)):
            C, C_rgb = step_func(C)
            writer.append_data(C_rgb)


if __name__ == "__main__":
    typer.run(cca_video)
