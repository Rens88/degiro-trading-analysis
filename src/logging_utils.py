from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path


def setup_logger(log_path: str = "logs/app.log") -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("degiro_app")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def log(message: str, start_time: float, logger: logging.Logger | None = None) -> str:
    now = datetime.now().isoformat(timespec="seconds")
    elapsed = time.perf_counter() - start_time
    line = f"[{now}] [+{elapsed:.2f}s] {message}"
    print(line)
    if logger:
        logger.info(line)
    return line
