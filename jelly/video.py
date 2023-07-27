import tqdm
import yaml
import typer
import imageio
import numpy as np

from functools import partial

from .ty import DTYPE, IMGTYPE
from .cyhoods import hood_size
from .config import parse_config, prune_params
from .util import enlarge_img


def generate_video(config_yaml: str, output_file: str):
    with open(config_yaml, "r") as fin:
        args = yaml.load(fin, Loader=yaml.FullLoader)
    params = parse_config(args)
    # Unpack required arguments
    params["output_file"] = output_file
    states, size = params["states"], params["size"]
    height, width = params["height"], params["width"]
    ratio, steps = params["ratio"], params["steps"]
    neighborhood = params["neighborhood"]
    automaton = params["automaton"]
    # Initialize arrays
    C = np.random.randint(0, states, (height, width), dtype=DTYPE)
    C_padded = np.pad(C, (size, size), constant_values=(-1, -1))
    C_rgb = np.empty((height, width, 3), dtype=IMGTYPE)
    neighborhood_size = hood_size(neighborhood, size)
    neighbors = np.empty((neighborhood_size, ), dtype=DTYPE)
    # Initialize update_func
    params = prune_params(automaton, params)
    update_func = partial(automaton, **params)
    with imageio.get_writer(output_file, mode='I') as writer:
        for step in tqdm.tqdm(range(steps)):
            out = np.empty(C_padded.shape, dtype=DTYPE)
            C_padded = update_func(
                C_padded=C_padded,
                neighbors=neighbors,
                out=out,
                out_rgb=C_rgb,
            )
            if ratio != 1:
                writer.append_data(enlarge_img(C_rgb, ratio))
            else:
                writer.append_data(C_rgb)


if __name__ == "__main__":
    typer.run(generate_video)
