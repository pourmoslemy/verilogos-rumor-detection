from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("topolie")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)

    logfile = output_dir / "run.log"
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.info("Logger initialized. log_file=%s", logfile)
    return logger
