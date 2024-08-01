from typing import Optional, Union, Literal
import torch
from pydantic import BaseModel
import mlflow
import mlflow.pytorch
import mlflow.tracking.fluent
from tap import Tap
import config
from utils import TorchDeviceManager
from loguru import logger
from tqdm.auto import tqdm


class TrainTask(BaseModel):
    learning_rate: float = 0.01  # Learning rate for the optimizer
    epochs: int = 10  # Number of epochs to train
    gpu_id: int = -1  # GPU ID to use, -1 for automatic allocation
    run_name: Optional[str] = None  # Optional run name for MLFlow
    exp_name: Optional[str] = None  # Optional experiment name for MLFlow
    # NOTE: with `bool` you activate this as a flag like `--save_every_epoch`
    save_every_epoch: bool = False  # Whether to save state_dict at every epoch
    # NOTE: with `Literal[False, True]` you activate this as a flag like normal argument `--save_model True` or `--save_model False`
    save_model: Literal[False, True] = True  # Whether to save model at the end


class TrainArgs(Tap):
    learning_rate: float = 0.01  # Learning rate for the optimizer
    epochs: int = 10  # Number of epochs to train
    gpu_id: int = -1  # GPU ID to use, -1 for automatic allocation
    run_name: str = None  # Optional name for the MLFlow run
    exp_name: Optional[str] = None  # Optional experiment name for MLFlow
    save_every_epoch: bool = False  # Whether to save state_dict at every epoch
    save_model: Literal[False, True] = True  # Whether to save model at the end


def get_args_from_model(param: TrainTask) -> TrainArgs:
    # https://docs.pydantic.dev/latest/concepts/serialization/#modelmodel_dump
    # https://docs.pydantic.dev/1.10/usage/exporting_models/
    # legacy syntax
    # return TrainArgs().from_dict(param.dict())
    return TrainArgs().from_dict(param.model_dump())


def get_exp_id(exp_name: Optional[str] = None) -> str:
    if not exp_name:
        exp_id = mlflow.tracking.fluent._get_experiment_id()
    else:
        if (exp := mlflow.get_experiment_by_name(exp_name)) is None:
            exp_id = mlflow.create_experiment(exp_name)
        else:
            exp_id = exp.experiment_id
    return exp_id


def train_model(
    task: Union[TrainTask, TrainArgs],
    run_id: Optional[str] = None,
    resume_state_dict: dict = {},
):
    try:

        device, lock = TorchDeviceManager().get_device_and_lock(task.gpu_id)

        logger.info(f"Using device {device}")

        # Example model and training loop
        init_epoch = resume_state_dict.get("epoch", -1) + 1
        model = torch.nn.Linear(10, 1).to(device)
        if model_state := resume_state_dict.get("model_state_dict"):
            logger.info("Loading checkpoint model state...")
            model.load_state_dict(model_state)

        optimizer = torch.optim.SGD(model.parameters(), lr=task.learning_rate)
        if optimizer_state := resume_state_dict.get("optimizer_state_dict"):
            logger.info("Loading checkpoint optimizer state...")
            optimizer.load_state_dict(optimizer_state)

        criterion = torch.nn.MSELoss()

        with lock:
            try:
                with mlflow.start_run(
                    run_id=run_id,
                    experiment_id=get_exp_id(task.exp_name) if task.exp_name else None,
                    run_name=task.run_name,
                    tags={
                        "Device": str(device),
                    },
                    # Currently set nested can by pass MLFlow multi-thread
                    nested=config.USE_THREAD,
                ):
                    if isinstance(task, BaseModel):
                        task = get_args_from_model(task)
                    mlflow.log_dict(task.as_dict(), "TrainArgs.json")

                    # Log parameters
                    mlflow.log_param("learning_rate", task.learning_rate)
                    mlflow.log_param("epochs", task.epochs)

                    # Dummy data
                    data = torch.randn(100, 10).to(device)
                    target = torch.randn(100, 1).to(device)

                    # Training loop
                    pbar = tqdm(range(init_epoch, task.epochs), desc="Train")
                    for epoch in pbar:
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        # logger.info(f"Epoch {epoch + 1}, Loss: {loss.item()}")
                        pbar.set_description(f"Train Epoch {epoch + 1}")
                        pbar.set_postfix(loss=loss.item())

                        # Log metrics
                        mlflow.log_metric("loss", loss.item(), step=epoch)
                        # All the information needed for resuming goes here
                        if task.save_every_epoch:
                            mlflow.pytorch.log_state_dict(
                                {
                                    "epoch": epoch,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(),
                                },
                                # NOTE: this path is a "folder name"
                                f"checkpoint/state_dict_epoch_{epoch}",
                            )
                        mlflow.pytorch.log_state_dict(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                            },
                            # NOTE: this path is a "folder name"
                            f"checkpoint/latest",
                        )
                    if task.save_model:
                        mlflow.pytorch.log_model(model, f"model/latest")
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                mlflow.log_param("error", str(e))
            finally:
                if lock:
                    logger.info("Released lock for GPU")
                else:
                    logger.info(
                        "No lock to be released. We don't create lock when we are using CPU."
                    )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
