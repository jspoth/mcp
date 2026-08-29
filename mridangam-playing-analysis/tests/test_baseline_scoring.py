"""Phase 1/2 tests: baseline construction and repetition scoring, synthetic
ground truth (gap durations built directly, no audio/librosa)."""

import numpy as np
import pytest

import baseline_scoring as bs

BPM = 80.0
BEAT = 60.0 / BPM


def gaps_from_pattern(pattern_beats, n_repeats, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_repeats):
        for frac in pattern_beats:
            g = BEAT * frac
            if noise:
                g *= 1 + rng.uniform(-noise, noise)
            out.append(g)
    return np.array(out)


def test_clean_pattern_baseline_matches_true_values_with_full_confidence():
    pattern = [1.0, 0.5, 0.5, 1.0]
    gaps = gaps_from_pattern(pattern, 6)
    result = bs.build_baseline(gaps, cycle_length=4)
    expected = [BEAT * f for f in pattern]
    assert np.allclose(result["baseline"], expected, atol=1e-6)
    assert result["baseline_confidence"] == 1.0
    assert result["num_repetitions"] == 6
    assert result["reason"]


def test_one_off_repetition_lowers_only_that_position_confidence():
    pattern = [1.0, 0.5, 0.5, 1.0]
    gaps = gaps_from_pattern(pattern, 6)
    gaps[2 * 4 + 1] *= 2.5  # corrupt position 1 in repetition index 2 only
    result = bs.build_baseline(gaps, cycle_length=4)
    assert result["position_confidence"][1] < 1.0
    assert result["position_confidence"][0] == 1.0
    assert result["position_confidence"][2] == 1.0
    assert result["baseline_confidence"] < 1.0


def test_not_enough_gaps_for_one_repetition_reports_cleanly():
    result = bs.build_baseline([0.5, 0.5], cycle_length=4)
    assert result["baseline"] is None
    assert result["baseline_confidence"] == 0.0
    assert "not enough" in result["reason"].lower()


def test_score_repetitions_clean_recording_has_no_bad_positions():
    pattern = [1.0, 0.5, 0.5, 1.0]
    gaps = gaps_from_pattern(pattern, 6)
    baseline = bs.build_baseline(gaps, 4)["baseline"]
    scored = bs.score_repetitions(gaps, 4, baseline)
    assert all(len(r["out_of_baseline_positions"]) == 0 for r in scored["repetitions"])
    assert scored["reason"]


def test_score_repetitions_flags_real_deviation_not_artifact():
    pattern = [1.0, 0.5, 0.5, 1.0]
    gaps = gaps_from_pattern(pattern, 6)
    baseline = bs.build_baseline(gaps, 4)["baseline"]
    # repetition 3, position 2: genuinely rushed (not an isolated
    # short/long spike relative to ITS OWN local neighbors, so
    # _flag_suspect_gaps shouldn't treat it as a detection artifact)
    gaps[3 * 4 + 2] *= 1.5
    scored = bs.score_repetitions(gaps, 4, baseline)
    rep3 = scored["repetitions"][3]
    assert 2 in rep3["out_of_baseline_positions"]
    assert rep3["deviations_pct"][2] is not None
    assert abs(rep3["deviations_pct"][2]) > 8.0


def test_score_repetitions_excludes_detection_artifacts_from_scoring():
    pattern = [1.0, 1.0, 1.0, 1.0]
    gaps = gaps_from_pattern(pattern, 6)
    baseline = bs.build_baseline(gaps, 4)["baseline"]
    # inject an isolated very short gap (classic duplicate-onset artifact
    # shape: much shorter than its local neighbors) by splitting one gap
    idx = 10
    gaps = np.insert(gaps, idx, gaps[idx] * 0.05)
    scored = bs.score_repetitions(gaps, 4, baseline)
    # the injected gap's repetition should have at least one suspect flag,
    # and that position should not count as a real "out_of_baseline" miss
    flat_suspect = [s for r in scored["repetitions"] for s in r["suspect"]]
    assert any(flat_suspect)
