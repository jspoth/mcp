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


from timing_analysis import (
    detect_subdivision_windowed,
    _flag_suspect_gaps,
)


def test_detect_subdivision_windowed_real_recording():
    """
    Regression test using a real 87.55s recording.

    Expected behavior:
      - target BPM ~= 99.384
      - beat interval ~= 0.60372s
      - local subdivision detection should identify multiple
        rhythmic regions rather than forcing one global subdivision
      - subdivision 4 should dominate the recording
      - subdivision 1 should appear in the slower/longer-gap regions
      - subdivision 2 should appear in several regions
      - isolated 3/5 results should not dominate
    """

    onset_times = np.array([
        0.78947846, 1.57895692, 1.97369615, 2.22911565, 2.85605442,
        2.92571429, 4.29569161, 4.38857143, 4.62077098, 5.57278912,
        5.66566893, 6.78022676, 7.91800454, 8.26630385, 8.42884354,
        8.52172336, 8.66104308, 8.80036281, 9.07900227, 9.26476190,
        9.40408163, 9.54340136, 9.65950113, 10.00780045, 10.14712018,
        10.30965986, 10.44897959, 10.56507937, 10.86693878, 11.00625850,
        11.12235828, 11.30811791, 11.47065760, 11.81895692, 11.91183673,
        12.12081633, 12.28335601, 12.39945578, 12.72453515, 12.88707483,
        13.16571429, 13.32825397, 13.88553288, 14.25705215, 14.86077098,
        15.46448980, 16.04498866, 16.37006803, 16.64870748, 16.97378685,
        17.55428571, 17.90258503, 18.04190476, 18.13478458, 18.29732426,
        18.43664399, 18.71528345, 18.90104308, 18.99392290, 19.15646259,
        19.29578231, 19.57442177, 19.71374150, 19.92272109, 20.17814059,
        20.48000000, 20.64253968, 20.78185941, 20.92117914, 21.06049887,
        21.40879819, 21.52489796, 21.71065760, 21.87319728, 21.98929705,
        22.15183673, 22.33759637, 22.47691610, 22.59301587, 22.80199546,
        22.89487528, 23.56825397, 23.87011338, 24.45061224, 25.10077098,
        25.68126984, 26.26176871, 26.47074830, 27.12090703, 27.42276644,
        27.60852608, 27.72462585, 27.88716553, 28.02648526, 28.30512472,
        28.46766440, 28.60698413, 28.72308390, 28.88562358, 29.23392290,
        29.37324263, 29.51256236, 29.67510204, 29.79120181, 30.11628118,
        30.20916100, 30.41814059, 30.53424036, 30.72000000, 31.02185941,
        31.13795918, 31.32371882, 31.43981859, 31.60235828, 31.74167800,
        31.92743764, 32.08997732, 32.27573696, 32.39183673, 32.50793651,
        33.11165533, 33.45995465, 34.04045351, 34.69061224, 35.29433107,
        35.50331066, 35.87482993, 36.17668934, 36.68752834, 37.03582766,
        37.19836735, 37.33768707, 37.47700680, 37.61632653, 37.91818594,
        38.05750567, 38.22004535, 38.33614512, 38.47546485, 38.84698413,
        38.98630385, 39.12562358, 39.28816327, 39.42748299, 39.56680272,
        39.72934240, 39.89188209, 40.00798186, 40.12408163, 40.30984127,
        40.96000000, 41.28507937, 41.86557823, 42.39963719, 42.74793651,
        43.04979592, 43.63029478, 43.86249433, 44.55909297, 44.88417234,
        45.00027211, 45.16281179, 45.27891156, 45.44145125, 45.72009070,
        45.85941043, 45.99873016, 46.16126984, 46.30058957, 46.62566893,
        46.76498866, 46.92752834, 47.06684807, 47.20616780, 47.36870748,
        47.50802721, 47.67056689, 47.76344671, 47.92598639, 48.08852608,
        48.71546485, 49.04054422, 49.62104308, 49.92290249, 50.22476190,
        50.82848073, 51.15356009, 51.43219955, 51.71083900, 52.29133787,
        52.68607710, 52.77895692, 52.96471655, 53.10403628, 53.22013605,
        53.54521542, 53.68453515, 53.84707483, 53.96317460, 54.10249433,
        54.40435374, 54.56689342, 54.72943311, 54.86875283, 55.03129252,
        55.19383220, 55.35637188, 55.49569161, 55.61179138, 55.77433107,
        55.93687075, 56.54058957, 56.81922902, 57.16752834, 57.42294785,
        58.00344671, 58.37496599, 58.65360544, 59.18766440, 59.51274376,
        60.04680272, 60.37188209, 60.53442177, 60.69696145, 60.85950113,
        60.95238095, 61.27746032, 61.39356009, 61.53287982, 61.69541950,
        61.85795918, 62.48489796, 62.80997732, 63.15827664, 63.36725624,
        64.04063492, 64.31927438, 64.62113379, 65.20163265, 65.48027211,
        66.08399093, 66.40907029, 66.54839002, 66.71092971, 66.98956916,
        67.26820862, 67.43074830, 67.59328798, 67.70938776, 67.84870748,
        68.54530612, 68.84716553, 69.17224490, 69.45088435, 70.03138322,
        70.35646259, 70.63510204, 71.19238095, 71.51746032, 72.09795918,
        72.44625850, 72.56235828, 72.67845805, 72.88743764, 72.98031746,
        73.16607710, 73.28217687, 73.39827664, 73.60725624, 73.74657596,
        73.88589569, 74.53605442, 74.81469388, 75.16299320, 75.41841270,
        76.02213152, 76.37043084, 76.60263039, 77.18312925, 77.48498866,
        78.08870748, 78.41378685, 78.78530612, 78.99428571, 79.57478458,
        79.92308390, 80.15528345, 80.80544218, 81.08408163, 81.71102041,
        82.01287982, 82.38439909, 82.61659864, 83.19709751, 83.52217687,
        83.77759637, 84.38131519, 84.68317460, 85.98349206, 87.19092971,
    ])

    target_bpm = 99.38401442307692

    # ------------------------------------------------------------------
    # Basic sanity checks on the fixture
    # ------------------------------------------------------------------

    assert len(onset_times) == 300
    assert np.all(np.diff(onset_times) > 0)

    beat_interval = 60.0 / target_bpm

    assert np.isclose(
        beat_interval,
        0.603718820861678,
        atol=1e-9,
    )

    # ------------------------------------------------------------------
    # Flag suspect gaps BEFORE subdivision inference
    # ------------------------------------------------------------------

    gaps = np.diff(onset_times)

    suspect, reasons = _flag_suspect_gaps(
        gaps,
        min_gap_fraction=0.4,
        max_gap_fraction=1.7,
        local_window=4,
    )

    assert len(suspect) == len(gaps)
    assert len(reasons) == len(gaps)

    # We expect some onset-detection artifacts in this real recording,
    # but certainly not every gap to be rejected.
    assert np.any(suspect)
    assert np.any(~suspect)

    reliable_mask = ~suspect

    # ------------------------------------------------------------------
    # Windowed subdivision detection
    # ------------------------------------------------------------------

    subdivisions = detect_subdivision_windowed(
        onset_times,
        target_bpm=target_bpm,
        window=6,
        candidates=range(1, 7),
        tolerance_pct=15.0,
        hysteresis_margin=0.05,
        min_confidence=0.5,
        reliable_mask=reliable_mask,
    )

    # One subdivision per gap.
    assert len(subdivisions) == len(onset_times) - 1

    # Every result must be one of the supported subdivisions.
    assert all(n in range(1, 7) for n in subdivisions)

    # ------------------------------------------------------------------
    # Regression expectations
    #
    # Based on the current real-recording result:
    #
    #   4 -> 162
    #   1 ->  75
    #   2 ->  43
    #   3 ->  13
    #   5 ->   6
    #
    # The exact boundaries may move slightly as onset detection changes,
    # so don't assert the complete sequence here.
    # ------------------------------------------------------------------

    counts = {
        n: subdivisions.count(n)
        for n in range(1, 7)
    }

    print("\nCURRENT SUBDIVISION COUNTS:")
    print(counts)

    print("\nCURRENT SUBDIVISION SEQUENCE:")
    print(subdivisions)

    # Subdivision 4 is clearly the dominant local subdivision.
    assert counts[4] > counts[1]
    assert counts[4] > counts[2]

    # There should be substantial evidence for the slower regions.
    assert counts[1] >= 30

    # And substantial evidence for subdivision 2.
    assert counts[2] >= 20

    # 3 and 5 should be secondary rather than dominating.
    assert counts[3] < counts[4]
    assert counts[5] < counts[4]

    # ------------------------------------------------------------------
    # Verify that the detector actually changes subdivision locally.
    # A global detector would return one value throughout.
    # ------------------------------------------------------------------

    unique_subdivisions = set(subdivisions)

    assert len(unique_subdivisions) >= 3

    # ------------------------------------------------------------------
    # Check representative regions from the observed output.
    #
    # These are intentionally broad rather than asserting every index.
    # ------------------------------------------------------------------

    # Early region transitions from 1 -> 4.
    assert subdivisions[0] == 1
    assert 4 in subdivisions[14:42]

    # Middle/later recording contains sustained subdivision-2 regions.
    assert 2 in subdivisions[150:195]
    assert 2 in subdivisions[220:250]

    # Final region transitions toward 3 and then back to 1.
    assert 3 in subdivisions[250:275]
    assert subdivisions[-1] == 1
