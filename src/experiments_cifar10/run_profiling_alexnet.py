#!/usr/bin/env python
import logging
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from phd_spectral.conv2d_rfft import (
    Conv2dRFFTCompactFunction,
    Conv2dRFFTFunction,
    Conv2dRFFTPhasorFunction,
)
from phd_spectral.ctx_manager import OverrideConv2d

_TEST_FUNCTIONS = [
    # None,
    Conv2dRFFTFunction,
    # Conv2dRFFTCompactFunction,
    Conv2dRFFTPhasorFunction,
]


def profile(
    *,
    function: torch.autograd.Function,
    timestamp: str,
    batch_size: int = 64,
    profile_name: str = "AlexNet_CIFAR10",
    num_classes: int = 10,
    skip_first: int = 0,
    wait: int = 4,
    warmup: int = 4,
    active: int = 4,
    repeat: int = 1,
):
    """_summary_

    Args:
        `function` (required):
        `batch_size` (int, optional): Defaults to 64.
        `profile_name` (str, optional): Defaults to "VGG16_Targets10".
        `num_classes` (int, optional): Defaults to 10.
        `skip_first` (int, optional): Defaults to 0.
        `wait` (int, optional): Defaults to 32.
        `warmup` (int, optional): Defaults to 32.
        `active` (int, optional): Defaults to 4.
        `repeat` (int, optional): Repeats scheduler cycle. Defaults to 2.

        The scheduler `skip_first` then the following cycle in run `repeat` times:
            - `wait`;
            - `warmup;
            - `active`.
    """
    function_name = function.__name__ if function else "Conv2d"
    profile_name += f"_{function_name}"

    logging.info(
        "Provided parameters: "
        f"profile("
        f"function={function_name} "
        f"timestamp={timestamp} "
        f"batch_size={batch_size} "
        f"profile_name={profile_name} "
        f"num_classes={num_classes} "
        f"skip_first={skip_first} "
        f"wait={wait} "
        f"warmup={warmup} "
        f"active={active} "
        f"repeat={repeat})"
    )

    inputs_shape = (batch_size, 3, 224, 224)
    gradients_shape = (batch_size, num_classes)

    model_weights = torchvision.models.AlexNet_Weights.DEFAULT

    model = torchvision.models.alexnet(weights=model_weights, progress=True)

    logging.info(f"Original model architecture is {model}")

    model.classifier[-1] = torch.nn.Linear(
        in_features=model.classifier[-1].in_features,
        out_features=num_classes,
        bias=True,
    )

    model = model.cuda()

    logging.info(f"Adopted model architecture is {model}")

    for i, parameter in enumerate(model.parameters()):
        logging.debug(f"({i}) parameter.requires_grad = {parameter.requires_grad}")

    logging.info(f"Profiling Model {profile_name}.")

    schedule = torch.profiler.schedule(
        skip_first=skip_first, wait=wait, warmup=warmup, active=active, repeat=repeat
    )

    steps = skip_first + repeat * (wait + warmup + active)

    # NOTE: `profile_memory` and `record_shapes` are very very slow for lots of steps.
    # TODO: Is this from experience of documentation?

    assert torch.cuda.is_available()
    with torch.profiler.profile(
        schedule=schedule,
        profile_memory=True,
        record_shapes=True,
        with_flops=True,
    ) as profile:
        for step in range(steps):
            with torch.profiler.record_function(f"{function_name} Step#{step}"):
                logging.info("#" * 80)
                logging.info(
                    f"Running {function_name} "
                    f"step={step}/{steps} ({100*step/steps:.2f}%)"
                )
                logging.info("#" * 80)
                inputs = torch.randn(inputs_shape, requires_grad=True).cuda()
                gradients = torch.randn(gradients_shape, requires_grad=True).cuda()

                with OverrideConv2d(function):
                    outputs = model(inputs)
                    outputs.backward(gradients)

            profile.step()

        logging.info("#" * 80)

    # Saving and Exporting Profiler
    profile_path = Path(__file__).parent / ".profiles"
    profile_path /= f"{profile_name}_date-{timestamp}"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(f"{profile_path}.json")
    logging.info(f"Profile trace saved to {profile_path}.json.")

    torch.save(profile.key_averages(), f"{profile_path}.pt")
    logging.info(f"Profile saved to {profile_path}.pt.")


if __name__ == "__main__":
    from phd_spectral.logging_formatter import set_logging_formatter
    from phd_spectral.utilities import set_deterministic, set_print_options

    # Configuring logging
    logging.basicConfig(level=logging.DEBUG)
    set_logging_formatter()

    set_deterministic()
    set_print_options()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"{'#' * 16} Profiling started at {timestamp} {'#' * 16}.")

    for function in _TEST_FUNCTIONS:
        profile(function=function, timestamp=timestamp)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.info(f"{'#' * 16} Profiling ended at {timestamp} {'#' * 16}.")
