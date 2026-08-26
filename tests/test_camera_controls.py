"""Tests for the camera control mapping.

`build_controls` imports libcamera, which exists only on the Pi, so every test
here skips on a dev machine. They still earn their place: the mapping is the
kind of code that is only ever exercised on hardware, and the `--ev` interaction
below cost a full capture session to find.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("libcamera", reason="libcamera is only present on the Pi")

from picam_yolo.server.cameras import build_controls  # noqa: E402


def test_defaults_keep_ae_in_charge():
    c = build_controls()
    assert c["AeEnable"] is True
    assert "ExposureTime" not in c


def test_short_mode_selects_the_short_ae_profile():
    from libcamera import controls

    assert build_controls(exposure_mode="short")["AeExposureMode"] is (
        controls.AeExposureModeEnum.Short
    )


def test_manual_exposure_disables_ae():
    """A pinned shutter with AE still running would be silently overridden."""
    c = build_controls(exposure_us=5000, gain=12)
    assert c["AeEnable"] is False
    assert c["ExposureTime"] == 5000
    assert c["AnalogueGain"] == 12.0
    # Manual is manual: no AE profile should linger in the dict.
    assert "AeExposureMode" not in c


def test_ev_is_only_set_when_asked_for():
    """Measured: --ev 0.7 pushed `short` back from 20ms to 33ms, because AE
    spends the extra brightness budget on a longer shutter. It must never appear
    unless explicitly requested."""
    assert "ExposureValue" not in build_controls(exposure_mode="short")
    assert build_controls(ev=0.7)["ExposureValue"] == 0.7


def test_rejects_unknown_modes():
    with pytest.raises(ValueError):
        build_controls(exposure_mode="fast")
    with pytest.raises(ValueError):
        build_controls(metering="clever")
