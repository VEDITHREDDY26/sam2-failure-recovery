from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.signals import SIGNAL_COLUMNS, read_signals, signals_from_masks, write_signals


def _frames() -> list[np.ndarray]:
    full = np.zeros((4, 6), dtype=bool)
    full[1:3, 2:5] = True
    return [full, np.zeros((4, 6), dtype=bool), full]


def _two_objects() -> pd.DataFrame:
    a = signals_from_masks("vid", 1, _frames(), [2.5, -4.0, 0.1 + 0.2])
    b = signals_from_masks("vid", 2, _frames(), [1.0 / 3.0, float("nan"), 7.0])
    return pd.concat([a, b], ignore_index=True)


def test_helper_values() -> None:
    df = signals_from_masks("vid", 1, _frames(), [2.0, -1.0, 2.0])
    assert tuple(df.columns) == SIGNAL_COLUMNS
    row = df.iloc[0]
    assert (row["area"], row["centre_x"], row["centre_y"], row["empty"]) == (6, 3.0, 1.5, False)
    second = df.iloc[1]
    assert second["empty"] and np.isnan(second["centre_x"]) and second["area"] == 0


def test_round_trip_bit_exact(tmp_path: Path) -> None:
    df = _two_objects()
    path = tmp_path / "s" / "vid.csv"
    write_signals(path, df)
    back = read_signals(path)
    pd.testing.assert_frame_equal(back, df, check_exact=True)
    assert path.read_text().splitlines()[0] == ",".join(SIGNAL_COLUMNS)
    assert ",1\n" in path.read_text()


def test_wrong_column_order_raises(tmp_path: Path) -> None:
    df = _two_objects()[list(reversed(SIGNAL_COLUMNS))]
    with pytest.raises(ValueError, match="columns"):
        write_signals(tmp_path / "x.csv", df)


def test_duplicate_key_raises(tmp_path: Path) -> None:
    df = _two_objects()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        write_signals(tmp_path / "x.csv", df)


def test_empty_area_mismatch_raises(tmp_path: Path) -> None:
    df = _two_objects()
    df.loc[0, "empty"] = True
    with pytest.raises(ValueError, match="empty"):
        write_signals(tmp_path / "x.csv", df)


def test_centre_nan_mismatch_raises(tmp_path: Path) -> None:
    df = _two_objects()
    df.loc[1, "centre_x"] = 1.0
    with pytest.raises(ValueError, match="centre"):
        write_signals(tmp_path / "x.csv", df)


def test_helper_length_mismatch() -> None:
    with pytest.raises(ValueError):
        signals_from_masks("vid", 1, _frames(), [1.0])
