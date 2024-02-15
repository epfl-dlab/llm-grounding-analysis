import logging
import os
from datetime import datetime
from logging import Logger
from typing import Any

import pytz

import inspect

from src.utils.io import save_json


def prepare_logger(output_dir: str) -> Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

    # Log to file
    fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def prepare_output_dir(base_dir: str = "./runs/") -> str:
    experiment_dir = os.path.join(
        base_dir, datetime.now(tz=pytz.timezone("Europe/Zurich")).strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


output_dir = prepare_output_dir()
logger = prepare_logger(output_dir)


def freeze_args(args: Any) -> None:
    # Retrieve caller filename
    caller_frame = inspect.stack()[1]
    caller_filename_full = caller_frame.filename
    caller_filename_only = os.path.splitext(os.path.basename(caller_filename_full))[0]

    # Save args to json file
    save_json(args.__dict__, os.path.join(output_dir, f"{caller_filename_only}_args"))


def get_output_dir() -> str:
    return output_dir


def get_logger() -> Logger:
    return logger
