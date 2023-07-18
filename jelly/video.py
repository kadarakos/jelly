import yaml
import typer

from enum import Enum
from functools import partial

from .life import life_video
from .cca import cca_video


class AutomataOptions(Enum):
    lifelike = partial(life_video)
    cca = partial(cca_video)


def generate_video(config_yaml: str, output_file: str):
    with open(config_yaml, "r") as fin:
        config = yaml.load(fin, Loader=yaml.FullLoader)
    if "automaton" not in config:
        raise ValueError(
            "Config has to contain key 'automaton'."
        )
    producer = getattr(AutomataOptions, config["automaton"]["name"]).value
    del config["automaton"]["name"]
    config["automaton"]["output_file"] = output_file
    config = config["automaton"]
    producer(**config)


if __name__ == "__main__":
    typer.run(generate_video)
