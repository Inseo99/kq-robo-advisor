from __future__ import annotations

import math

import numpy as np

from kq_tool.api.serialization import clean_json_value


def test_clean_json_value_converts_numpy_scalars_and_arrays() -> None:
    payload = {
        "int": np.int64(3),
        "float": np.float64(1.5),
        "bool": np.bool_(True),
        "array": np.array([1, 2, 3]),
    }

    cleaned = clean_json_value(payload)

    assert cleaned == {
        "int": 3,
        "float": 1.5,
        "bool": True,
        "array": [1, 2, 3],
    }


def test_clean_json_value_replaces_nan_with_none() -> None:
    cleaned = clean_json_value({"a": np.float64(np.nan), "b": float("nan")})

    assert cleaned == {"a": None, "b": None}
    assert not any(isinstance(value, float) and math.isnan(value) for value in cleaned.values())


def test_clean_json_value_handles_nested_tuples() -> None:
    assert clean_json_value({"x": (np.int64(1), np.float64(2.0))}) == {"x": [1, 2.0]}


def test_server_reuses_api_serialization_helper() -> None:
    import server

    assert server._kq_clean_json_value is clean_json_value
    assert server._clean({"x": np.float64(np.nan)}) == {"x": None}
