"""JSON serialization helpers for API responses."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def clean_json_value(value):
    """Convert NumPy-heavy payloads into JSON-safe Python objects."""

    if isinstance(value, Mapping):
        return {key: clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


_clean_json_value = clean_json_value
