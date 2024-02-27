#!/usr/bin/env python
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import torchvision
from phd_spectral.conv2d_rfft import Conv2dRFFTFunction, Conv2dRFFTPhasorFunction
from phd_spectral.ctx_manager import OverrideConv2d
from phd_spectral.logging_formatter import set_logging_formatter
from phd_spectral.utilities import (
    get_pretrained_model_state_dict,
    set_deterministic,
    set_print_options,
    train_model,
    validate_model,
)

_TEST_FUNCTIONS = [
    Conv2dRFFTFunction,
    Conv2dRFFTPhasorFunction,
]


def main(
    *,
    function_name: str,
    total_num_epochs: int = 1,
    freeze_features_epochs: int = 0,
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    model_name="ResNet18_CIFAR10",
    timestamp: str = "",
):
    # Output paths
    file_path = Path(__file__)
    log_folder = file_path.parent / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_path_csv = log_folder / f"{file_path.stem}_{function_name}_date-{timestamp}.csv"
    log_path_log = log_folder / f"{file_path.stem}_{function_name}_date-{timestamp}.log"

    pretrained_model_folder = Path(__file__).parent / ".pretrained_models"
    pretrained_model_folder.mkdir(parents=True, exist_ok=True)

    trained_model_folder = Path(__file__).parent / ".trained_models"
    trained_model_folder.mkdir(parents=True, exist_ok=True)

    # Clean up any previous log configuration.
    logging.flush()
    logging.close()

    # Configuring logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path_log), logging.StreamHandler()],
    )
    logging.getLogger().name = function_name

    set_logging_formatter()

    logging.info(
        "Provided parameters: "
        f"main("
        f"function_name={function_name}, "
        f"total_num_epochs={total_num_epochs},"
        f"freeze_features_epochs={freeze_features_epochs}, "
        f"batch_size={batch_size})"
    )

    ###########################################################################
    # TRAINING PREPARATION
    ###########################################################################

    set_deterministic()
    set_print_options()

    model_weights = torchvision.models.ResNet18_Weights.DEFAULT
    transform = model_weights.transforms()

    training_data = torchvision.datasets.CIFAR10(
        "~/Datasets/CIFAR10",
        train=True,
        download=True,
        transform=transform,
    )
    validation_data = torchvision.datasets.CIFAR10(
        "~/Datasets/CIFAR10",
        train=False,
        download=True,
        transform=transform,
    )

    num_classes = len(training_data.classes)

    training_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = torchvision.models.resnet18(progress=True)

    logging.info(f"Model orginal architecture is {model}")

    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,
        bias=True,
    )

    pretrained_state_dict = get_pretrained_model_state_dict(
        pretrained_model_folder, model_name
    )
    model.load_state_dict(pretrained_state_dict)

    model = model.cuda()

    logging.info(f"Model modified architecture is {model}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    # CUDA is mandatory for our application
    assert torch.cuda.is_available()

    train_df_log = pd.DataFrame()
    validation_df_log = pd.DataFrame()

    best_validation_loss = float("inf")
    training_step = 0

    ###########################################################################
    # TRAINING
    ###########################################################################

    for epoch in range(total_num_epochs):
        freeze_features = epoch < freeze_features_epochs

        if freeze_features:
            logging.info(f"Features frozen on epoch {epoch}")
        else:
            logging.info(f"Features not frozen on epoch {epoch}")

        # Freeze all layers except the fully connected (fc) layers
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = not freeze_features

        for i, parameter in enumerate(model.parameters()):
            logging.debug(f"({i}) parameter.requires_grad = {parameter.requires_grad}")

        t0 = time.perf_counter()
        training_loss, training_accuracy, training_step, df_log = train_model(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            training_loader=training_loader,
            training_step=training_step,
        )
        training_time = time.perf_counter() - t0

        train_df_log = pd.concat([train_df_log, df_log])

        t0 = time.perf_counter()
        validation_loss, validation_accuracy, df_log = validate_model(
            model=model,
            loss_fn=loss_fn,
            validation_loader=validation_loader,
            epoch=epoch,
            training_step=training_step,
        )
        validation_time = time.perf_counter() - t0

        validation_df_log = pd.concat([validation_df_log, df_log])

        logging.info(
            f"--- End of Epoch: "
            f"epoch={epoch} "
            f"duration_training={training_time} "
            f"duration_validation={validation_time} "
            f"training_step={training_step} "
            f"training_loss={training_loss:.5f} "
            f"training_accuracy={training_accuracy:.5f} "
            f"validation_loss={validation_loss:.5f} "
            f"validation_accuracy={validation_accuracy:.5f}."
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model_path = trained_model_folder / (
                f"{model_name}_{function_name}_date-{timestamp}_"
                f"epoch-{epoch}_training_step-{training_step}_"
                f"accuracy-{validation_accuracy:.3f}.pt"
            )

            torch.save(model.state_dict(), model_path)

    # Saving CSV Log
    df_log = pd.merge(
        train_df_log, validation_df_log, how="outer", left_index=True, right_index=True
    )
    df_log.to_csv(log_path_csv)
    logging.shutdown()


if __name__ == "__main__":
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    for function in _TEST_FUNCTIONS:
        function_name = function.__name__ if function else "Conv2d"
        with OverrideConv2d(function):
            main(function_name=function_name, timestamp=timestamp)
