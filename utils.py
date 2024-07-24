from typing import Tuple, Union, Optional
import config
import os
import time
import GPUtil
from filelock import FileLock
import torch


def create_lock_dir() -> None:
    if not os.path.exists(config.LOCK_DIR):
        os.makedirs(config.LOCK_DIR)


def get_lock_file_path(gpu_id: int) -> str:
    return os.path.join(config.LOCK_DIR, f"gpu_{gpu_id}{config.LOCK_EXTENSION}")


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
        time.sleep(config.WAIT_TIME)


class DummyContextManager:
    def __enter__(self):
        pass  # No setup needed

    def __exit__(self, exc_type, exc_value, traceback):
        pass  # No cleanup needed

    def __bool__(self) -> bool:
        return False  # Make this object a False means "no lock"


def get_device_and_lock(
    gpu_id: int = -1,
) -> Tuple[torch.device, Union[FileLock, DummyContextManager]]:
    if torch.cuda.is_available():
        if gpu_id == -1:
            gpu_id, lock = get_available_gpu()
        else:
            gpu_id = gpu_id
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
                        time.sleep(config.WAIT_TIME)

        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
        lock = DummyContextManager()

    return device, lock


def get_parallel_num() -> Optional[int]:
    if torch.cuda.is_available():
        return len(GPUtil.getGPUs())
    else:
        return None if not config.MAX_PARALLEL_NUM else config.MAX_PARALLEL_NUM
