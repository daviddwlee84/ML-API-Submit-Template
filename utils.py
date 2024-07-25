from typing import Tuple, Union, Optional, Literal, ContextManager
import config
import os
import time
import GPUtil
from filelock import FileLock
import torch
from loguru import logger


class DummyContextManager:
    def __enter__(self):
        pass  # No setup needed

    def __exit__(self, exc_type, exc_value, traceback):
        pass  # No cleanup needed

    def __bool__(self) -> bool:
        return False  # Make this object a False means "no lock"

    def acquire(self) -> None:
        return

    def release(self) -> None:
        return


class TorchDeviceManager:

    def __init__(
        self,
        lock_dir: str = config.LOCK_DIR,
        lock_extension: str = config.LOCK_EXTENSION,
        wait_time: int = config.WAIT_TIME,
    ):
        self._lock_dir = lock_dir
        self._lock_extension = lock_extension
        self._wait_time = wait_time

    def _create_lock_dir(self) -> None:
        if not os.path.exists(self._lock_dir):
            os.makedirs(self._lock_dir)

    def _get_lock_file_path(self, gpu_id: int) -> str:
        return os.path.join(self._lock_dir, f"gpu_{gpu_id}{self._lock_extension}")

    @staticmethod
    def is_gpu_available(mode: Literal["torch", "gputils"] = "torch") -> bool:
        if mode == "torch":
            return torch.cuda.is_available()
        elif mode == "gputils":
            try:
                GPUtil.getGPUs()
                return True
            except:
                return False
        else:
            raise NotImplementedError(f"Invalid Mode {mode}")

    def get_gpu_number(self, mode: Literal["torch", "gputils"] = "torch") -> int:
        return len(GPUtil.getGPUs()) if self.is_gpu_available(mode=mode) else 0

    def _get_available_gpu(self) -> Optional[Tuple[int, FileLock]]:
        if not self.is_gpu_available():
            logger.warning("No valid GPU.")
            return None
        self._create_lock_dir()
        while True:
            available_gpus = GPUtil.getAvailable(
                order="first",
                limit=self.get_gpu_number(),
                maxLoad=0.05,
                maxMemory=0.05,
                includeNan=False,
            )
            for gpu_id in available_gpus:
                lock_file = self._get_lock_file_path(gpu_id)
                lock = FileLock(lock_file)
                try:
                    lock.acquire(timeout=0)  # Try to acquire the lock without waiting
                    return gpu_id, lock
                except:
                    continue

            logger.info("No available GPUs. Waiting...")
            time.sleep(self._wait_time)

    @staticmethod
    def _get_dummy_lock() -> ContextManager:
        return DummyContextManager()

    def get_device_and_lock(
        self,
        gpu_id: int = -1,
        return_str: bool = False,
    ) -> Tuple[Union[torch.device], Union[FileLock, DummyContextManager]]:
        if self.is_gpu_available():
            if gpu_id == -1:
                gpu_id, lock = self._get_available_gpu()
            else:
                gpu_id = gpu_id
                lock_file = self._get_lock_file_path(gpu_id)
                lock = FileLock(lock_file)
                try:
                    lock.acquire(timeout=0)
                except:
                    logger.info(f"GPU {gpu_id} is currently occupied. Waiting...")
                    while True:
                        try:
                            lock.acquire(timeout=0)
                            break
                        except:
                            time.sleep(self._wait_time)

            device = (
                torch.device(f"cuda:{gpu_id}") if not return_str else f"cuda:{gpu_id}"
            )
        else:
            device = torch.device("cpu") if not return_str else "cpu"
            lock = self._get_dummy_lock()

        return device, lock


def get_parallel_num() -> Optional[int]:
    if torch.cuda.is_available():
        return len(GPUtil.getGPUs())
    else:
        return None if not config.MAX_PARALLEL_NUM else config.MAX_PARALLEL_NUM


if __name__ == "__main__":
    manager = TorchDeviceManager()
    print(manager.get_gpu_number())
    device, lock = manager.get_device_and_lock()
    lock.acquire()
    print(device, lock)
    import ipdb

    ipdb.set_trace()
    lock.release()
