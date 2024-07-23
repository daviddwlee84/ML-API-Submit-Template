from typing import Tuple, Optional, Union
import GPUtil
import torch
import time
import os
from filelock import FileLock
from pydantic import BaseModel
import mlflow
from tap import Tap

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


class TrainArgs(Tap):
    learning_rate: float = 0.01  # Learning rate for the optimizer
    epochs: int = 10  # Number of epochs to train
    gpu_id: int = -1  # GPU ID to use, -1 for automatic allocation
    run_name: str = None  # Optional name for the MLFlow run


def get_args_from_model(param: TrainTask) -> TrainArgs:
    return TrainArgs().from_dict(param.model_dump())


class DummyContextManager:
    def __enter__(self):
        pass  # No setup needed

    def __exit__(self, exc_type, exc_value, traceback):
        pass  # No cleanup needed

    def __bool__(self) -> bool:
        return False  # Make this object a False means "no lock"


def train_model(task: Union[TrainTask, TrainArgs], run_id: Optional[str] = None):
    try:
        args = task

        if torch.cuda.is_available():
            if args.gpu_id == -1:
                gpu_id, lock = get_available_gpu()
            else:
                gpu_id = args.gpu_id
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

        with lock:
            try:
                with mlflow.start_run(
                    run_id=run_id,
                    tags={
                        "Device": str(device),
                    },
                ):
                    # Example model and training loop
                    model = torch.nn.Linear(10, 1).to(device)
                    optimizer = torch.optim.SGD(
                        model.parameters(), lr=args.learning_rate
                    )
                    criterion = torch.nn.MSELoss()

                    # Log parameters
                    mlflow.log_param("learning_rate", args.learning_rate)
                    mlflow.log_param("epochs", args.epochs)

                    # Dummy data
                    data = torch.randn(100, 10).to(device)
                    target = torch.randn(100, 1).to(device)

                    # Training loop
                    for epoch in range(args.epochs):
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

                        # Log metrics
                        mlflow.log_metric("loss", loss.item(), step=epoch)
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
