"""
Synthetic ground-truth tests for the onsets -> suspect-gap-flagging ->
subdivision-inference -> timing-classification pipeline in
timing_analysis.py. No audio files or librosa onset detection involved —
onset_times are constructed directly so the tests exercise exactly the
pipeline's own logic, not librosa's.
"""

import numpy as np
import pytest

import timing_analysis as ta

BPM = 80.0
BEAT = 60.0 / BPM          # 0.75s: one stroke/beat
HALF_BEAT = BEAT / 2       # 0.375s: two strokes/beat
THIRD_BEAT = BEAT / 3      # 0.25s: three strokes/beat


def onsets_from_gaps(gaps, start=0.0):
    times = [start]
    for g in gaps:
        times.append(times[-1] + g)
    return np.array(times)


def reliable(intervals):
    return [iv for iv in intervals if not iv["suspect"]]


# ---------------------------------------------------------------------------
# Perfect / clean recordings
# ---------------------------------------------------------------------------

def test_perfect_tempo_one_stroke_per_beat():
    onsets = onsets_from_gaps([BEAT] * 12)
    intervals = ta.compute_tempo_intervals(onsets, BPM)
    assert all(iv["subdivision"] == 1 for iv in intervals)
    assert all(iv["in_tempo"] for iv in intervals)
    assert all(not iv["suspect"] for iv in intervals)
    assert all(abs(iv["deviation_pct"]) < 1e-6 for iv in intervals)


def test_perfect_tempo_two_strokes_per_beat():
    onsets = onsets_from_gaps([HALF_BEAT] * 12)
    intervals = ta.compute_tempo_intervals(onsets, BPM)
    # once enough data exists to infer subdivision 2, it's backfilled onto
    # the leading gaps too -- a clean recording shouldn't misreport its
    # first few gaps as "out of tempo" just because they came first.
    assert all(iv["subdivision"] == 2 for iv in intervals)
    assert all(iv["in_tempo"] for iv in intervals)
    assert all(not iv["suspect"] for iv in intervals)


def test_small_deviation_within_tolerance():
    rng = np.random.default_rng(0)
    gaps = BEAT * (1 + rng.uniform(-0.03, 0.03, size=20))
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM, tolerance_pct=8.0)
    assert all(not iv["suspect"] for iv in intervals)
    assert all(iv["in_tempo"] for iv in intervals)


def test_deviation_beyond_tolerance_is_flagged_out_of_tempo_not_suspect():
    # a consistent +15% rush is a real timing error, not a detection artifact
    gaps = [BEAT * 0.85] * 12
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM, tolerance_pct=8.0)
    assert all(not iv["suspect"] for iv in intervals)
    assert all(not iv["in_tempo"] for iv in intervals)


# ---------------------------------------------------------------------------
# Detection artifacts: duplicate / missed onsets
# ---------------------------------------------------------------------------

def test_one_duplicate_onset_flagged_short_and_excluded_from_stats():
    gaps = [BEAT] * 10
    onsets = onsets_from_gaps(gaps)
    # inject a spurious duplicate very close to onset index 5
    onsets = np.insert(onsets, 5, onsets[5] - 0.01)
    intervals = ta.compute_tempo_intervals(onsets, BPM)

    suspects = [iv for iv in intervals if iv["suspect"]]
    assert len(suspects) == 1
    assert suspects[0]["suspect_reason"] == "short"
    # the duplicate produces a wildly extreme deviation on its own row --
    # demonstrating why excluding it from aggregate stats actually matters,
    # not just that the exclusion filter is a no-op tautology
    assert abs(suspects[0]["deviation_pct"]) > 500
    # every gap that IS counted toward stats has a sane, real deviation
    assert all(abs(iv["deviation_pct"]) < 10 for iv in reliable(intervals))


def test_one_missed_onset_flagged_long_and_excluded_from_stats():
    gaps = [BEAT] * 5 + [2 * BEAT] + [BEAT] * 5
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM)

    suspects = [iv for iv in intervals if iv["suspect"]]
    assert len(suspects) == 1
    assert suspects[0]["suspect_reason"] == "long"
    assert suspects[0]["interval_sec"] == pytest.approx(2 * BEAT)
    # subdivision inference wasn't derailed by the missed onset
    assert all(iv["subdivision"] == 1 for iv in intervals)


def test_two_consecutive_missed_onsets_still_flagged():
    # missing 2 onsets in a row merges 3 beats into one gap (~3x)
    gaps = [BEAT] * 5 + [3 * BEAT] + [BEAT] * 5
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM)

    suspects = [iv for iv in intervals if iv["suspect"]]
    assert len(suspects) == 1
    assert suspects[0]["suspect_reason"] == "long"


def test_short_burst_of_missed_onsets_stays_flagged():
    # a burst up to ~local_window (4) long gaps in a row, flanked by clean
    # gaps on both sides, should still get flagged -- the local median has
    # enough clean neighbors on each side to not be pulled toward the slow
    # value.
    gaps = [HALF_BEAT] * 4 + [BEAT] * 4 + [HALF_BEAT] * 4
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM)

    burst = intervals[4:8]
    assert all(iv["suspect"] and iv["suspect_reason"] == "long" for iv in burst)


@pytest.mark.xfail(
    reason=(
        "KNOWN LIMITATION: _flag_suspect_gaps() only looks at the "
        "local_window=4 nearest neighbors on each side. Once a run of "
        "missed-onset-sized gaps is longer than roughly 2x that window, "
        "the local median re-centers on the slow gaps themselves and the "
        "interior of the run is no longer flagged suspect -- the pipeline "
        "instead (mis)reads it as a legitimate subdivision change. Fixing "
        "this needs either a longer/adaptive window or actual onset "
        "confidence scores (see review point #7), not just a threshold "
        "tweak, so it's left as a documented gap rather than patched "
        "blindly."
    ),
    strict=True,
)
def test_long_burst_of_missed_onsets_is_NOT_reliably_flagged():
    gaps = [HALF_BEAT] * 6 + [BEAT] * 10 + [HALF_BEAT] * 6
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM)

    burst = intervals[6:16]
    # this is what SHOULD happen; xfail documents that it currently doesn't
    assert all(iv["suspect"] and iv["suspect_reason"] == "long" for iv in burst)


# ---------------------------------------------------------------------------
# Subdivision changes must not be mistaken for onset errors
# ---------------------------------------------------------------------------

def _assert_at_most_one_boundary_error(intervals):
    """
    A centered window (see detect_subdivision_windowed's docstring) removes
    the multi-gap causal LAG a trailing window used to cause, but it can't
    remove a genuine TIE: the single gap sitting exactly at a symmetric
    transition's midpoint gets a window evenly split between old- and
    new-subdivision evidence, which is real ambiguity (a listener given
    only that local context couldn't resolve it either), not a bug. Assert
    the strong bound this actually supports: at most one misclassified gap
    total, not "eventually corrects itself within `window` gaps."
    """
    wrong = [iv for iv in intervals if not iv["in_tempo"]]
    assert len(wrong) <= 1, f"expected at most 1 boundary-tie error, got {len(wrong)}"


def test_subdivision_change_one_to_two_not_flagged_suspect():
    # every gap here is machine-perfect for its own subdivision -- a real,
    # error-free subdivision change must not cost the performer several
    # false "out of tempo" gaps just because the change happened partway
    # through (that was the bug: a trailing-only window lagged by up to
    # `window` gaps after every transition).
    gaps = [BEAT] * 6 + [HALF_BEAT] * 8
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM)

    assert all(not iv["suspect"] for iv in intervals)
    _assert_at_most_one_boundary_error(intervals)
    assert all(iv["subdivision"] == 1 for iv in intervals[:6])
    assert all(iv["subdivision"] == 2 for iv in intervals[-3:])


def test_subdivision_change_two_to_one_not_flagged_suspect():
    gaps = [HALF_BEAT] * 8 + [BEAT] * 6
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM)

    assert all(not iv["suspect"] for iv in intervals)
    _assert_at_most_one_boundary_error(intervals)
    assert all(iv["subdivision"] == 2 for iv in intervals[:8])
    assert all(iv["subdivision"] == 1 for iv in intervals[-3:])


def test_subdivision_change_one_to_three_not_flagged_suspect():
    # ratio 1/3 = 0.33 is BELOW min_gap_fraction (0.4) on its own, so this
    # is the sharpest transition case for a false "duplicate onset" flag
    gaps = [BEAT] * 6 + [THIRD_BEAT] * 9
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM)

    assert all(not iv["suspect"] for iv in intervals)
    _assert_at_most_one_boundary_error(intervals)
    assert all(iv["subdivision"] == 3 for iv in intervals[-3:])


# ---------------------------------------------------------------------------
# Sustained real timing errors must NOT be swallowed as detection artifacts
# ---------------------------------------------------------------------------

def test_sustained_half_tempo_passage_is_real_error_not_suspect():
    # a performer genuinely dragging to half tempo for a whole passage --
    # must be reported as a real, sustained timing error, not filtered out
    # as if it were a run of missed onsets.
    gaps = [BEAT] * 5 + [2 * BEAT] * 8
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM)

    sustained = intervals[6:]  # skip the single transition gap
    assert all(not iv["suspect"] for iv in sustained)
    assert all(not iv["in_tempo"] for iv in sustained)
    assert all(iv["deviation_pct"] == pytest.approx(-50.0, abs=1.0) for iv in sustained)


def test_gradual_acceleration_not_flagged_suspect():
    # smooth rushing (2% faster each gap) is a real trend, not noise --
    # neighboring gaps are always close in ratio so nothing should trip
    # the local-outlier suspect check.
    gaps = [BEAT * (0.98 ** i) for i in range(20)]
    onsets = onsets_from_gaps(gaps)
    intervals = ta.compute_tempo_intervals(onsets, BPM)

    assert all(not iv["suspect"] for iv in intervals)
    # later gaps should show a clearly increasing (rushing) deviation
    assert intervals[-1]["deviation_pct"] > intervals[5]["deviation_pct"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(target_bpm=0),
    dict(target_bpm=-10),
    dict(window=0),
    dict(window=-1),
    dict(min_gap_fraction=0),
    dict(min_gap_fraction=1.5),
    dict(max_gap_fraction=1.0),
    dict(max_gap_fraction=0.5),
])
def test_invalid_parameters_raise(kwargs):
    onsets = onsets_from_gaps([BEAT] * 5)
    base = dict(onset_times=onsets, target_bpm=BPM)
    base.update(kwargs)
    with pytest.raises(ValueError):
        ta.compute_tempo_intervals(**base)


def test_mismatched_subdivision_list_length_raises():
    onsets = onsets_from_gaps([BEAT] * 5)  # 5 gaps
    with pytest.raises(ValueError):
        ta.compute_tempo_intervals(onsets, BPM, subdivision=[1, 2])


def test_correct_length_subdivision_list_is_accepted():
    onsets = onsets_from_gaps([BEAT] * 5)  # 5 gaps
    intervals = ta.compute_tempo_intervals(onsets, BPM, subdivision=[1, 1, 2, 2, 1])
    assert [iv["subdivision"] for iv in intervals] == [1, 1, 2, 2, 1]


def test_ndarray_subdivision_does_not_crash():
    # np.array(...) == "auto" returns an array, not a bool -- evaluating
    # that in a plain `if` used to raise "truth value of an array is
    # ambiguous" before reaching the ndarray branch it was meant to hit
    onsets = onsets_from_gaps([BEAT] * 5)
    intervals = ta.compute_tempo_intervals(onsets, BPM, subdivision=np.array([1, 1, 2, 2, 1]))
    assert [iv["subdivision"] for iv in intervals] == [1, 1, 2, 2, 1]
    assert all(isinstance(iv["subdivision"], int) for iv in intervals)


def test_invalid_subdivision_string_raises():
    onsets = onsets_from_gaps([BEAT] * 5)
    with pytest.raises(ValueError):
        ta.compute_tempo_intervals(onsets, BPM, subdivision="atuo")


@pytest.mark.parametrize("bad_subdivision", [0, -2, [0, 1, 1, 1, 1], [1, 1, -1, 1, 1]])
def test_invalid_subdivision_values_raise(bad_subdivision):
    onsets = onsets_from_gaps([BEAT] * 5)
    with pytest.raises(ValueError):
        ta.compute_tempo_intervals(onsets, BPM, subdivision=bad_subdivision)
