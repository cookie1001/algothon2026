"""
Price loading. Days down the rows, instruments across the columns.

`prices.txt` is already stored in that orientation (one row per day, one
column per instrument), so no transpose happens anywhere in this package.
The only transpose in the whole codebase is inside `getMyPosition`, where the
evaluator's (n_instruments, n_days) input is flipped once at the boundary.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PRICES_PATH = os.path.join(REPO_ROOT, "prices.txt")


def load_prices_by_day(path: str = DEFAULT_PRICES_PATH) -> Tuple[np.ndarray, List[str]]:
    """Returns ((n_days, n_instruments) float array, instrument names)."""
    frame = pd.read_csv(path, sep=r"\s+", header=0, index_col=None)
    return frame.to_numpy(dtype=float), list(frame.columns)


def load_prices_frame(path: str = DEFAULT_PRICES_PATH) -> pd.DataFrame:
    """Same data as a DataFrame, for exploratory work in notebooks."""
    return pd.read_csv(path, sep=r"\s+", header=0, index_col=None)


def to_evaluator_orientation(prices_by_day: np.ndarray) -> np.ndarray:
    """(n_days, n_inst) -> (n_inst, n_days), the shape eval.py passes in.

    Only needed when you want to call `getMyPosition` directly rather than
    through the backtester, which handles the boundary for you.
    """
    return prices_by_day.T
