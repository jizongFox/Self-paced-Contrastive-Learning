import subprocess
from functools import lru_cache
from pathlib import Path

PROJECT_PATH = str(Path(__file__).parents[1])
DATA_PATH = str(Path(PROJECT_PATH) / ".data")
Path(DATA_PATH).mkdir(exist_ok=True, parents=True)
CONFIG_PATH = str(Path(PROJECT_PATH, "config"))

try:
    __git_hash__ = (
        subprocess.check_output([f"cd {PROJECT_PATH}; git rev-parse HEAD"], shell=True)
            .strip()
            .decode()
    )
except:
    __git_hash__ = None


@lru_cache()
def get_cc_data_path():
    import os
    possible_path = os.environ.get("SLURM_TMPDIR", None)
    if possible_path:
        possible_folders = os.listdir(possible_path)
        if len(possible_folders) > 0:
            print("cc_data_path is {}".format(possible_path))
            return possible_path
    print("cc_data_path is {}".format(DATA_PATH))
    return DATA_PATH


@lru_cache()
def on_cc() -> bool:
    import socket
    hostname = socket.gethostname()
    if "beluga" in hostname or "blg" in hostname:
        return True
    if "cedar" in hostname or "cdr" in hostname:
        return True
    if "gra" in hostname:
        return True
    return False


def success(save_dir: str):
    filename = ".success"
    from pathlib import Path
    Path(str(save_dir), filename).touch()
