"""Phase 3 tests: analyze_mridangam() unification. Monkeypatches onset
detection/bpm estimation (same convention as the rest of the suite -- no
real audio files) so these exercise the routing/scoring logic itself."""

import os
import numpy as np
import pytest

import mridangam_analysis as ma

BPM = 80.0
BEAT = 60.0 / BPM


def _patch_onsets(monkeypatch, gaps):
    times = np.concatenate([[0.0], np.cumsum(gaps)])
    monkeypatch.setattr(ma.ta, "detect_onset_times", lambda fp: (times, times[-1]))
    monkeypatch.setattr(ma.ta, "estimate_bpm", lambda fp: BPM)


def test_script_provided_directly_uses_script_provided_mode(monkeypatch):
    _patch_onsets(monkeypatch, np.array([BEAT] * 7))
    result = ma.analyze_mridangam("fake.wav", script="1 1 1 1 1 1 1")
    assert result["mode"] == "script_provided"
    assert "validated" in result["reason"].lower()


def test_co_located_script_file_found_and_used(tmp_path, monkeypatch):
    audio = tmp_path / "recording.wav"
    audio.write_bytes(b"fake")
    script_file = tmp_path / "recording.txt"
    script_file.write_text("1 1 1 1 1 1 1")
    _patch_onsets(monkeypatch, np.array([BEAT] * 7))
    result = ma.analyze_mridangam(str(audio), script=None)
    assert result["mode"] == "script_found"
    assert "recording.txt" in result["reason"]


def test_no_script_anywhere_falls_back_to_inference(tmp_path, monkeypatch):
    audio = tmp_path / "solo.wav"
    audio.write_bytes(b"fake")
    pattern = [1.0, 0.5, 0.5, 1.0]
    gaps = np.array(pattern * 8) * BEAT
    _patch_onsets(monkeypatch, gaps)
    result = ma.analyze_mridangam(str(audio), script=None)
    assert result["mode"] == "inferred"
    assert result["cycle_length"] == 4
    assert result["pattern_confidence"] > 0.5
    assert "no script file found" in result["reason"].lower()
    assert "baseline" in result and result["baseline"] is not None


def test_no_script_and_no_inferable_pattern_still_resolves_to_an_answer(tmp_path, monkeypatch):
    # never asks a question -- always returns SOMETHING with a reason,
    # even when there's nothing confident to report
    audio = tmp_path / "random.wav"
    audio.write_bytes(b"fake")
    rng = np.random.default_rng(5)
    gaps = BEAT * (1 + rng.uniform(-0.9, 3.0, size=40))
    _patch_onsets(monkeypatch, gaps)
    result = ma.analyze_mridangam(str(audio), script=None)
    assert result["mode"] == "inferred"
    assert result["pattern_confidence"] == 0.0
    assert isinstance(result["reason"], str) and len(result["reason"]) > 0


def test_find_co_located_script_checks_both_extensions(tmp_path):
    audio = tmp_path / "take1.wav"
    audio.write_bytes(b"fake")
    assert ma._find_co_located_script(str(audio)) is None
    (tmp_path / "take1.script").write_text("1 2 3")
    assert ma._find_co_located_script(str(audio)) == str(tmp_path / "take1.script")
