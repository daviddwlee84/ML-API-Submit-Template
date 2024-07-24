import os

USE_THREAD = False

# Constants
LOCK_DIR = os.path.expanduser("~/.gpu_locks")
LOCK_EXTENSION = ".lock"
WAIT_TIME = 10

MAX_PARALLEL_NUM = os.cpu_count()
