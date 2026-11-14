"""Shared helpers: logging, timing, reproducibility, time-window arithmetic."""

from __future__ import annotations

import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Log to stdout and (optionally) to a file. Safe to call more than once."""
    logger = logging.getLogger("renfc")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


log = logging.getLogger("renfc")


@contextmanager
def timed(label: str):
    """Context manager that logs the wall time of a block."""
    t0 = time.time()
    yield
    log.info("%s finished in %.1f s", label, time.time() - t0)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if present) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def to_timedelta(window: str) -> pd.Timedelta:
    """Parse strings such as ``'5min'``, ``'1h'``, ``'1D'`` into a Timedelta."""
    return pd.Timedelta(window)


def steps_in(duration_hours: float, window: str) -> int | None:
    """Number of whole ``window`` steps that make up ``duration_hours``.

    Returns ``None`` when the duration is not an integer multiple of the window
    (for example 1 h at a 45-min window), so callers can skip that combination
    instead of silently rounding it.
    """
    ratio = pd.Timedelta(hours=duration_hours) / to_timedelta(window)
    if abs(ratio - round(ratio)) > 1e-9:
        return None
    return int(round(ratio))


def window_sort_key(window: str) -> float:
    """Sort key so that windows order by duration, not alphabetically."""
    return to_timedelta(window).total_seconds()


def pretty_window(window: str) -> str:
    """Human label for a window string: ``'1h' -> '1 h'``, ``'1D' -> '1 day'``."""
    td = to_timedelta(window)
    secs = td.total_seconds()
    if secs % 86400 == 0:
        n = int(secs // 86400)
        return f"{n} day" + ("s" if n > 1 else "")
    if secs % 3600 == 0:
        return f"{int(secs // 3600)} h"
    return f"{int(secs // 60)} min"
