#!/usr/bin/env python
import logging
from pathlib import Path

import pandas as pd
import torch
import torchvision
from phd_spectral.logging_formatter import set_logging_formatter
from phd_spectral.utilities import (
    set_deterministic,
    set_print_options,
    train_model,
    validate_model,
)


def main(
    *,
    total_num_epochs: int = 4,
    freeze_features_epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    model_name="ResNet18_CIFAR100",
    timestamp: str = "",
):
    # Output paths
    file_path = Path(__file__)
    log_folder = file_path.parent / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)

    log_path_csv = log_folder / f"{file_path.stem}_date-{timestamp}.csv"
    log_path_log = log_folder / f"{file_path.stem}_date-{timestamp}.log"

    pretrained_model_folder = Path(__file__).parent / ".pretrained_models"
    pretrained_model_folder.mkdir(parents=True, exist_ok=True)

    # Configuring logging
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging.FileHandler(log_path_log), logging.StreamHandler()],
    )
    set_logging_formatter()

    logging.info(
        "Provided parameters: "
        f"main(total_num_epochs={total_num_epochs},"
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

    training_data = torchvision.datasets.CIFAR100(
        "~/Datasets/CIFAR100",
        train=True,
        download=True,
        transform=transform,
    )
    validation_data = torchvision.datasets.CIFAR100(
        "~/Datasets/CIFAR100",
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

    model = torchvision.models.resnet18(weights=model_weights, progress=True)

    logging.info(f"Model orginal architecture is {model}")

    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes,
        bias=True,
    )

    model = model.cuda()

    logging.info(f"Model classifier architecture is {model}")

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

        training_loss, training_accuracy, training_step, df_log = train_model(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            training_loader=training_loader,
            training_step=training_step,
        )

        train_df_log = pd.concat([train_df_log, df_log])

        validation_loss, validation_accuracy, df_log = validate_model(
            model=model,
            loss_fn=loss_fn,
            validation_loader=validation_loader,
            epoch=epoch,
            training_step=training_step,
        )

        validation_df_log = pd.concat([validation_df_log, df_log])

        logging.info(
            f"--- End of Epoch: "
            f"epoch={epoch} "
            f"training_step={training_step} "
            f"training_loss={training_loss:.5f} "
            f"training_accuracy={training_accuracy:.5f} "
            f"validation_loss={validation_loss:.5f} "
            f"validation_accuracy={validation_accuracy:.5f}."
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model_path = pretrained_model_folder / (
                f"{model_name}_date-{timestamp}_"
                f"epoch-{epoch}_training_step-{training_step}_"
                f"accuracy-{validation_accuracy:.3f}.pt"
            )

            torch.save(model.state_dict(), model_path)

    # Saving CSV Log
    df_log = pd.merge(
        train_df_log, validation_df_log, how="outer", left_index=True, right_index=True
    )
    df_log.to_csv(log_path_csv)


if __name__ == "__main__":
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    main(timestamp=timestamp)
