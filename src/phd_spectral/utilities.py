import logging
import random
import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
from torchvision.models import inception


def set_print_options() -> None:
    torch.set_printoptions(linewidth=200, sci_mode=False)


def set_deterministic(seed: int = 0) -> None:
    seed = int(seed) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_model(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_loader: torch.utils.data.DataLoader,
    epoch: int,
    training_step: int,
    total_steps: int = None,
    log_info_step: Union[int, float] = 0.01,
) -> Tuple[float, float, int, pd.DataFrame]:
    """Train the model

    Args:
        model (torch.nn.Module): _description_
        loss_fn (torch.nn.Module): _description_
        optimizer (torch.optim.Optimizer): _description_
        training_loader (torch.utils.data.DataLoader): _description_
        epoch (int): _description_
        training_step (int, optional): _description_. Defaults to 0.
        total_steps (int, optional): _description_. Defaults to None.
        log_info_step (Union[int, float], optional):
            Determines how ofter the training will be evaluated and logged.
            If greater than 1, meatrics will be computed and logged at every
            `log_info_step` batchs.
            If smaller than 1, it represent a *period* in epochs, in this it will serve
            as a parcentage to compute the log_info_step based on the number of batches
            in the epoch.
            Defaults to 0.025.

    Returns:
        Tuple[float, float, int, pd.DataFrame]: _description_
    """

    sum_correct = 0
    accuracy = None

    sum_loss = 0.0
    loss = None

    sum_runtime_forward_s = 0.0
    batch_runtime_forward_ms = None

    sum_runtime_backward_s = 0.0
    batch_runtime_backward_ms = None

    if total_steps is None or total_steps == 0 or total_steps > len(training_loader):
        total_steps = len(training_loader)

    if type(log_info_step) is float:
        if log_info_step > 1:
            log_info_step = int(log_info_step)
        else:
            log_info_step = int(np.ceil(log_info_step * len(training_loader) + 0.1))

    log_data = []
    model.train()
    for i, (inputs, labels) in enumerate(training_loader):
        if i == total_steps:
            break

        inputs, labels = (inputs.cuda(), labels.cuda())

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(inputs)

        if isinstance(outputs, inception.InceptionOutputs):
            outputs = outputs[0]

        # Compute the loss and its gradients
        model_loss = loss_fn(outputs, labels)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        model_loss.backward()

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        sum_loss += model_loss

        prediction = outputs.argmax(dim=1)
        sum_correct += (prediction == labels).to(torch.float).sum()

        sum_runtime_forward_s += t1 - t0
        sum_runtime_backward_s += t2 - t1

        if i % log_info_step == (log_info_step - 1):
            loss = sum_loss / log_info_step
            sum_loss = 0.0

            num_samples = len(inputs) * log_info_step
            accuracy = 100 * sum_correct / num_samples
            sum_correct = 0.0

            batch_runtime_forward_ms = 1e3 * sum_runtime_forward_s / log_info_step
            sum_runtime_forward_s = 0

            batch_runtime_backward_ms = 1e3 * sum_runtime_backward_s / log_info_step
            sum_runtime_backward_s = 0

            logging.info(
                f"Training: "
                f"epoch={epoch} "
                f"step={i}/{total_steps} "
                f"loss={loss:.5f} "
                f"accuracy={accuracy:.5f} "
                f"runtime_forward_ms={batch_runtime_forward_ms:.5f} "
                f"runtime_backward_ms={batch_runtime_backward_ms:.5f}."
            )

            log_data.append(
                [
                    training_step + i,
                    loss.item(),
                    accuracy.item(),
                    batch_runtime_forward_ms,
                    batch_runtime_backward_ms,
                ]
            )

    df_log = pd.DataFrame(
        log_data,
        columns=[
            "training_step",
            "training_loss",
            "training_accuracy",
            "training_runtime_forward_ms",
            "training_runtime_backward_ms",
        ],
    ).set_index("training_step")

    return loss, accuracy, training_step + i, df_log


def validate_model(
    *,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    validation_loader: torch.utils.data.DataLoader,
    epoch: int,
    training_step: int,
) -> Tuple[float, float, pd.DataFrame]:
    sum_correct = 0
    sum_num_samples = 0
    accuracy = None

    sum_loss = 0.0
    loss = None

    sum_runtime_forward_s = 0.0
    batch_runtime_forward_ms = None

    model.eval()
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = (inputs.cuda(), labels.cuda())

            t0 = time.perf_counter()
            outputs = model(inputs)

            # Compute the loss and its gradients
            model_loss = loss_fn(outputs, labels)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            sum_loss += model_loss

            prediction = outputs.argmax(dim=1)
            sum_correct += (prediction == labels).to(torch.float).sum()
            sum_num_samples += len(inputs)

            sum_runtime_forward_s += t1 - t0

    loss = sum_loss / len(validation_loader)
    accuracy = 100 * sum_correct / sum_num_samples

    batch_runtime_forward_ms = 1e3 * sum_runtime_forward_s / len(validation_loader)

    logging.info(
        f"Validation: "
        f"epoch={epoch} "
        f"training_step={training_step} "
        f"loss={loss:.5f} "
        f"accuracy={accuracy:.5f} "
        f"runtime_forward_ms={batch_runtime_forward_ms:.5f}."
    )

    df_log = pd.DataFrame(
        [
            [
                training_step,
                loss.item(),
                accuracy.item(),
                batch_runtime_forward_ms,
            ]
        ],
        columns=[
            "training_step",
            "validation_loss",
            "validation_accuracy",
            "validation_runtime_forward_ms",
        ],
    ).set_index("training_step")

    return loss, accuracy, df_log


def pair_with_accuracy(path) -> Tuple[Path, float]:
    accuracy = float(str(path.stem).split(sep="accuracy-")[1])
    return path, accuracy


def get_pretrained_model_state_dict(model_folder: Path, model_name: str):
    """Returns the pretained model that has the highest (validation) accuracy"""
    if not model_folder.exists():
        return None

    saved_models = list(model_folder.glob(f"{model_name}*.pt"))

    if len(saved_models) == 0:
        return None

    saved_models = list(map(pair_with_accuracy, saved_models))
    saved_models = sorted(saved_models, key=lambda x: x[1])

    best_model = saved_models[-1]
    best_model_path = best_model[0]
    state_dict = torch.load(best_model_path)
    return state_dict
