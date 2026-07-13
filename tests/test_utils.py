from __future__ import annotations

import numpy as np
import pytest

from mylhd import anadata
from mylhd.labcom_retrieve import LHDData
from mylhd.utils import detect_data, map_onoff, wait_for_opendata


def test_map_onoff_maps_boolean_signal_with_deadtime() -> None:
    t_ref = np.array([0.0, 1.0, 2.0, 3.0])
    state_ref = np.array([False, False, True, False])
    t_target = np.array([0.5, 1.5, 2.0, 2.4, 3.0, 3.4])

    mapped = map_onoff(t_ref, state_ref, t_target, deadtime_rise=0.2, deadtime_fall=0.1)

    np.testing.assert_array_equal(mapped, np.array([False, False, False, True, True, False]))


def test_map_onoff_rejects_multidimensional_state() -> None:
    with pytest.raises(ValueError, match="must be 1D arrays"):
        map_onoff([0.0, 1.0], [[False, True]], [0.5])


def test_wait_for_opendata_retries_until_data_is_available(monkeypatch) -> None:
    monkeypatch.setattr("mylhd.utils.time.sleep", lambda _: None)
    attempts = {"count": 0}

    def fake_retrieve(*, diag: str, shotno: int, subno: int = 1) -> dict[str, int | str]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FileNotFoundError
        return {"diag": diag, "shotno": shotno, "subno": subno}

    result = wait_for_opendata("shotinfo", 12345, subno=2, retry_delay=1, retrieve_func=fake_retrieve)

    assert result == {"diag": "shotinfo", "shotno": 12345, "subno": 2}
    assert attempts["count"] == 3


def test_detect_data_finds_future_shot(monkeypatch) -> None:
    monkeypatch.setattr("mylhd.utils.time.sleep", lambda _: None)
    monkeypatch.setattr("mylhd.utils.time.time", lambda: 0.0)

    def fake_retrieve(*, diag: str, shotno: int, subno: int = 1) -> object:
        if shotno == 101:
            return object()
        raise FileNotFoundError

    assert detect_data("shotinfo", 100, retry_delay=1, retrieve_func=fake_retrieve, search_num=2) == 101


def test_anadata_reexports_shared_utility_functions() -> None:
    assert anadata.wait_for_opendata is wait_for_opendata
    assert anadata.detect_data is detect_data


def test_lhd_data_maps_int8_samples_to_asymmetric_voltage_range() -> None:
    raw = np.array([np.iinfo(np.int8).min, 0, np.iinfo(np.int8).max], dtype=np.int8)
    data = LHDData(
        data=raw,
        time=np.arange(raw.size),
        metadata={"RangeLow": "-2.0", "RangeHigh": "6.0", "ImageType": "Int8"},
    )

    expected = (raw.astype(float) - np.iinfo(np.int8).min) * (8.0 / 255.0) - 2.0
    np.testing.assert_allclose(data.val, expected)


def test_lhd_data_rejects_unknown_range_image_type() -> None:
    data = LHDData(
        data=np.array([0]),
        time=np.array([0.0]),
        metadata={"RangeLow": "-1.0", "RangeHigh": "1.0", "ImageType": "float32"},
    )

    with np.testing.assert_raises_regex(ValueError, "Unsupported ImageType"):
        data.get_val()
