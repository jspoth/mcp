"""
Phase 0 tests: cycle_detection.detect_cycle() against known-ground-truth
subdivision sequences. Purely data-oriented -- no onset_times, no audio,
no timing_analysis.py dependency at all, matching the module's own
isolation.
"""

import numpy as np
import pytest

import cycle_detection as cd


def repeat(pattern, n_repeats):
    return list(pattern) * n_repeats


# ---------------------------------------------------------------------------
# The three coordinator-specified ground-truth cases
# ---------------------------------------------------------------------------

def test_seven_element_pattern_x5():
    seq = repeat([6, 6, 6, 2, 3, 4, 5], 5)
    result = cd.detect_cycle(seq)
    assert result["cycle_length"] == 7
    assert result["pattern_confidence"] > 0.7
    assert result["reason"]


def test_four_element_pattern_x10():
    seq = repeat([1, 2, 3, 4], 10)
    result = cd.detect_cycle(seq)
    assert result["cycle_length"] == 4
    assert result["pattern_confidence"] > 0.7
    assert result["reason"]


def test_eight_element_sequence_that_is_really_period_four_x5():
    # [1,1,2,3,1,1,2,3] x 5 is literally [1,1,2,3] x 10 -- the shorter
    # period is the correct, simplest explanation: a true period-4 signal
    # is trivially ALSO periodic at every multiple of 4 (8, 12, ...), since
    # two back-to-back copies of the period-4 pattern are indistinguishable
    # from "one repetition of an 8-long pattern." Autocorrelation will show
    # strong peaks at both lag=4 and lag=8; detect_cycle() must not report
    # the redundant lag=8 "explanation" when lag=4 already explains the
    # data equally well -- same "simplest explanation wins ties" principle
    # timing_analysis.py's own _best_subdivision() uses for subdivision
    # inference. Reporting 8 wouldn't be WRONG, exactly, but it would be a
    # needlessly complex answer to a simpler question, and would silently
    # hide the fact that [1,1,2,3] is the actual repeating unit.
    seq = repeat([1, 1, 2, 3, 1, 1, 2, 3], 5)
    result = cd.detect_cycle(seq)
    assert result["cycle_length"] == 4
    assert result["reason"]


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------

def test_return_shape_is_exactly_three_keys():
    result = cd.detect_cycle(repeat([2, 2, 2, 1, 1], 6))
    assert set(result.keys()) == {"cycle_length", "pattern_confidence", "reason"}


def test_reason_present_on_every_branch():
    success = cd.detect_cycle(repeat([2, 1, 1], 6))
    too_short = cd.detect_cycle([1, 2, 3])
    rng = np.random.default_rng(2)
    no_pattern = cd.detect_cycle(rng.integers(1, 5, size=40).tolist())

    for result in (success, too_short, no_pattern):
        assert isinstance(result["reason"], str) and len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# No real pattern / insufficient data
# ---------------------------------------------------------------------------

def test_random_non_repeating_sequence_reports_no_cycle():
    rng = np.random.default_rng(1)
    seq = rng.integers(1, 7, size=60).tolist()
    result = cd.detect_cycle(seq)
    assert result["cycle_length"] is None
    assert result["pattern_confidence"] == 0.0


def test_constant_sequence_reports_no_meaningful_cycle():
    # every value the same -- technically "periodic" at every lag
    # trivially, but not a meaningful pattern; autocorrelation of a
    # constant (mean-removed) signal is ~0 everywhere, so this must not
    # produce a false-positive cycle.
    seq = repeat([3], 30)
    result = cd.detect_cycle(seq)
    assert result["cycle_length"] is None


def test_too_short_sequence_reports_no_cycle_cleanly():
    result = cd.detect_cycle([1, 2])
    assert result["cycle_length"] is None
    assert result["pattern_confidence"] == 0.0
    assert "too few" in result["reason"].lower()


# ---------------------------------------------------------------------------
# Refinement: reject on insufficient CONSISTENT repetitions, not just a
# strong autocorrelation peak
# ---------------------------------------------------------------------------

def test_only_two_repetitions_rejected_even_if_perfectly_periodic():
    seq = repeat([2, 1, 1, 2, 1], 2)  # below MIN_CONSISTENT_REPETITIONS (3)
    result = cd.detect_cycle(seq, min_repetitions=3)
    assert result["cycle_length"] is None
    assert result["pattern_confidence"] == 0.0


def test_period_two_candidate_rejected_outright():
    # alternating values could superficially look "periodic" at lag 2, but
    # period-2 candidates are rejected by design (_MIN_CANDIDATE_LEN = 3) --
    # too easy to match by chance on low-cardinality integer data.
    seq = [1, 4] * 20
    result = cd.detect_cycle(seq)
    assert result["cycle_length"] != 2


def test_one_partially_off_repetition_among_many_lowers_confidence_but_still_detects():
    # realistic corruption: one repetition drifts (still in-range small
    # subdivision values, like a player briefly misjudging a phrase), not
    # replaced wholesale by values far outside the pattern's own range --
    # see test_severely_corrupted_repetition_is_a_known_limitation below
    # for why that harsher case is documented separately rather than
    # asserted as a guarantee.
    pattern = [2, 2, 1, 1]
    clean = cd.detect_cycle(repeat(pattern, 6))
    assert clean["cycle_length"] == 4

    seq = repeat(pattern, 6)
    bad_start = len(pattern) * 2
    seq[bad_start:bad_start + 4] = [3, 3, 2, 2]  # shifted by +1, still plausible subdivision values
    corrupted = cd.detect_cycle(seq)

    assert corrupted["cycle_length"] == 4
    assert corrupted["pattern_confidence"] < clean["pattern_confidence"]


def test_severely_corrupted_repetition_is_a_known_limitation():
    # KNOWN LIMITATION, documented rather than silently accepted: one
    # repetition out of six wholesale-replaced by values ~2.5x outside the
    # pattern's own range is a severe corruption (~40% of lag-4 pairwise
    # comparisons are contaminated). Autocorrelation-based candidate
    # detection genuinely cannot recover the true period here even with
    # outlier-robust clipping (_robust_clip) -- the periodicity signal at
    # the true period ends up negative/not a local max, so the candidate
    # never reaches the consistency-check stage that would otherwise catch
    # it. This asserts the ACTUAL (safe, non-crashing, honest) behavior --
    # reporting no confident cycle -- rather than a stronger guarantee the
    # algorithm can't currently make. See cycle_detection.py's
    # _MIN_PERIODICITY_STRENGTH comment for the calibration reasoning, and
    # the Phase 0 completion notes for what a fix would need (a
    # segmentation-search approach that doesn't route through a single
    # global autocorrelation pass, rather than further threshold tuning).
    pattern = [2, 2, 1, 1]
    seq = repeat(pattern, 6)
    bad_start = len(pattern) * 2
    for i in range(bad_start, bad_start + len(pattern)):
        seq[i] = 5
    result = cd.detect_cycle(seq)
    assert result["cycle_length"] is None
    assert result["pattern_confidence"] == 0.0
    assert result["reason"]


def test_confidence_reflects_both_periodicity_and_consistency():
    # a real pattern with injected per-position noise should score lower
    # than the perfectly clean version -- pins "confidence isn't just the
    # raw autocorrelation peak" rather than only checking it drops on ONE
    # of the two signals.
    pattern = [2, 1, 2, 1, 1]
    clean = cd.detect_cycle(repeat(pattern, 8))

    rng = np.random.default_rng(3)
    noisy_seq = repeat(pattern, 8)
    # jitter ~20% of positions by +/-1, simulating imperfect subdivision
    # detection rather than a perfectly clean synthetic signal
    idx = rng.choice(len(noisy_seq), size=len(noisy_seq) // 5, replace=False)
    for i in idx:
        noisy_seq[i] = max(1, noisy_seq[i] + rng.choice([-1, 1]))
    noisy = cd.detect_cycle(noisy_seq)

    assert noisy["cycle_length"] == clean["cycle_length"] == 5
    assert noisy["pattern_confidence"] < clean["pattern_confidence"]
