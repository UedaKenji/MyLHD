from __future__ import annotations

import numpy as np

from mylhd.cts_utls import fb_viewer


def test_plot_mwscat_map_uses_old_48_channel_layout_without_mwscat2(monkeypatch) -> None:
    calls: list[tuple[str, np.ndarray]] = []

    class FakeRetriever:
        def retrieve_multiple_channels(self, *, diag_name, shot, channels):
            calls.append((diag_name, channels))
            return {}

    def stop_after_channel_retrieval(**kwargs):
        raise RuntimeError("stop test after channel retrieval")

    monkeypatch.setattr(fb_viewer, "LHDRetriever", FakeRetriever)
    monkeypatch.setattr(fb_viewer.anadata.KaisekiData, "retrieve_opendata", stop_after_channel_retrieval)

    assert fb_viewer.plot_mwscat_map(148159, is_print=False) is None
    assert len(calls) == 1
    assert calls[0][0] == "mwscat"
    np.testing.assert_array_equal(calls[0][1], np.arange(1, 49))
