from typing import Tuple, Optional, Union
import GPUtil
import torch
import time
import os
from filelock import FileLock
from pydantic import BaseModel
import mlflow
import mlflow.pytorch
from tap import Tap
import config

# Constants
LOCK_DIR = os.path.expanduser("~/.gpu_locks")
LOCK_EXTENSION = ".lock"
WAIT_TIME = 10


def create_lock_dir() -> None:
    if not os.path.exists(LOCK_DIR):
        os.makedirs(LOCK_DIR)


def get_lock_file_path(gpu_id: int) -> str:
    return os.path.join(LOCK_DIR, f"gpu_{gpu_id}{LOCK_EXTENSION}")


def get_available_gpu() -> Tuple[int, FileLock]:
    create_lock_dir()

    while True:
        available_gpus = GPUtil.getAvailable(
            order="first", limit=1, maxLoad=0.05, maxMemory=0.05, includeNan=False
        )
        for gpu_id in available_gpus:
            lock_file = get_lock_file_path(gpu_id)
            lock = FileLock(lock_file)
            try:
                lock.acquire(timeout=0)  # Try to acquire the lock without waiting
                return gpu_id, lock
            except:
                continue

        print("No available GPUs. Waiting...")
        time.sleep(WAIT_TIME)


class TrainTask(BaseModel):
    learning_rate: float = 0.01  # Learning rate for the optimizer
    epochs: int = 10  # Number of epochs to train
    gpu_id: int = -1  # GPU ID to use, -1 for automatic allocation
    run_name: Optional[str] = None  # Optional run name for MLFlow
    exp_name: Optional[str] = None  # Optional experiment name for MLFlow


class TrainArgs(Tap):
    learning_rate: float = 0.01  # Learning rate for the optimizer
    epochs: int = 10  # Number of epochs to train
    gpu_id: int = -1  # GPU ID to use, -1 for automatic allocation
    run_name: str = None  # Optional name for the MLFlow run
    exp_name: Optional[str] = None  # Optional experiment name for MLFlow


def get_args_from_model(param: TrainTask) -> TrainArgs:
    # https://docs.pydantic.dev/latest/concepts/serialization/#modelmodel_dump
    # https://docs.pydantic.dev/1.10/usage/exporting_models/
    # legacy syntax
    # return TrainArgs().from_dict(param.dict())
    return TrainArgs().from_dict(param.model_dump())


class DummyContextManager:
    def __enter__(self):
        pass  # No setup needed

    def __exit__(self, exc_type, exc_value, traceback):
        pass  # No cleanup needed

    def __bool__(self) -> bool:
        return False  # Make this object a False means "no lock"


def train_model(
    task: Union[TrainTask, TrainArgs],
    run_id: Optional[str] = None,
    resume_state_dict: dict = {},
):
    try:
        if torch.cuda.is_available():
            if task.gpu_id == -1:
                gpu_id, lock = get_available_gpu()
            else:
                gpu_id = task.gpu_id
                lock_file = get_lock_file_path(gpu_id)
                lock = FileLock(lock_file)
                try:
                    lock.acquire(timeout=0)
                except:
                    print(f"GPU {gpu_id} is currently occupied. Waiting...")
                    while True:
                        try:
                            lock.acquire(timeout=0)
                            break
                        except:
                            time.sleep(WAIT_TIME)

            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
            lock = DummyContextManager()

        print(f"Using device {device}")

        # Example model and training loop
        init_epoch = resume_state_dict.get("epoch", -1) + 1
        model = torch.nn.Linear(10, 1).to(device)
        if model_state := resume_state_dict.get("model_state_dict"):
            print("Loading checkpoint model state...")
            model.load_state_dict(model_state)

        optimizer = torch.optim.SGD(model.parameters(), lr=task.learning_rate)
        if optimizer_state := resume_state_dict.get("optimizer_state_dict"):
            print("Loading checkpoint optimizer state...")
            optimizer.load_state_dict(optimizer_state)

        criterion = torch.nn.MSELoss()

        with lock:
            try:
                with mlflow.start_run(
                    run_id=run_id,
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
                    for epoch in range(init_epoch, task.epochs):
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

                        # Log metrics
                        mlflow.log_metric("loss", loss.item(), step=epoch)
                        # All the information needed for resuming goes here
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
                    mlflow.pytorch.log_model(model, f"model/latest")
            except Exception as e:
                print(f"An error occurred: {e}")
                mlflow.log_param("error", str(e))
            finally:
                if lock:
                    print("Released lock for GPU")
                else:
                    print(
                        "No lock to be released. We don't create lock when we are using CPU."
                    )
    except Exception as e:
        print(f"An error occurred: {e}")
