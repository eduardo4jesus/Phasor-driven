#!/usr/bin/env python
import logging
from pathlib import Path

import torch
from phd_spectral.conv2d_rfft import (
    Conv2dRFFTFunction,
    Conv2dRFFTPhasorFunction,
)

_TEST_FUNCTIONS = [
    Conv2dRFFTFunction,
    Conv2dRFFTPhasorFunction,
]


def load_profile(
    *,
    function,
    profile_name="VGG16_CIFAR100",
):
    function_name = function.__name__ if function else "Conv2d_date"
    profile_name += f"_{function_name}"

    # Loading The Latest Profiler
    profile_path = Path(__file__).parent / ".profiles"
    profile_path = list(profile_path.glob(f"{profile_name}*.pt"))
    profile_path = profile_path[-1]  # Hoping `-1` is the lastest.

    profile_keys_average = torch.load(str(profile_path))

    logging.info("-" * 125)
    logging.info(f"Showing {profile_path.name}")

    logging.info(profile_keys_average.table(sort_by="cpu_time_total"))


if __name__ == "__main__":
    from phd_spectral.logging_formatter import set_logging_formatter
    from phd_spectral.utilities import set_deterministic, set_print_options

    # Configuring logging
    logging.basicConfig(level=logging.DEBUG)
    set_logging_formatter()

    set_deterministic()
    set_print_options()

    for function in _TEST_FUNCTIONS:
        load_profile(function=function)
