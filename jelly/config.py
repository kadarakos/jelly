import inspect

import numpy as np

from typing import Callable
from enum import Enum

from .color import ColorOptions
from .cyhoods import HoodOptions
from .updates import AutomataOptions
from .ty import ConfigDict, DTYPE, IMGTYPE


VideoFields = ["steps", "height", "width", "ratio", "cmap"]
AutomataFields = ["automaton", "neighborhood", "size", "states"]
RequiredFields = VideoFields + AutomataFields

DEFAULT_CONFIG = {
    "automaton": "lifelike",
    "neighborhood": "moore",
    "size": 1,
    "cmap": "blues",
    "states": 2,
    "height": 100,
    "width": 100,
    "steps": 100,
    "ratio": 1
}


class Options(Enum):
    cmap = ColorOptions
    automaton = AutomataOptions
    neighborhood = HoodOptions


def parse_config(config: ConfigDict) -> ConfigDict:
    params = {}
    filled = []
    for field in RequiredFields:
        if field not in config:
            value = DEFAULT_CONFIG[field]
            filled.append((field, value))
            if field in Options.__members__:
                value = getattr(getattr(Options, field).value, value).value
                if field == "cmap":
                    value = np.array(value, dtype=IMGTYPE)
                elif field == "automaton":
                    value = value.func
            params[field] = value
    for field, value in config.items():
        if field in Options.__members__:
            choices = getattr(Options, field).value
            choice = config[field]
            if choice in choices.__members__:
                value = getattr(choices, choice).value
                if field == "cmap":
                    value = np.array(value, dtype=IMGTYPE)
                elif field == "automaton":
                    value = value.func
                params[field] = value
            else:
                available = [x.name for x in list(choices)]
                raise KeyError(
                    f"Could not find '{choice}' "
                    f"in the options choices for '{field}'. "
                    f"Available options are: {available}."
                )
        else:
            if isinstance(value, list):
                value = np.array(value, dtype=DTYPE)
            params[field] = value
    if filled:
        print(
            "The following arguments were missing in the config"
            " and were filled in with default values."
        )
        for field, value in filled:
            print(f"{field}: {value}")
    return params


def prune_params(foo: Callable, config: ConfigDict) -> ConfigDict:
    """
    Remove entries from 'config' that are not arguments of 'foo'.
    """
    params = inspect.signature(foo).parameters
    new_dict = {}
    for key, value in config.items():
        if key in params:
            new_dict[key] = value
    return new_dict
