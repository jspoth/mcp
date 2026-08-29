"""
Phase 0 of the MCP v2 plan: cycle-boundary detection.

Purely data-oriented -- detect_cycle() takes a subdivision sequence (a
plain list/array of small positive integers, e.g. the output of
timing_analysis.py's detect_subdivision_windowed()) and knows nothing
about audio files, onset detection, MCP, or LLMs. Kept isolated on purpose:
easy to unit-test with hand-built sequences, and easy to swap out the
detection algorithm later without touching anything that calls it.

Does NOT interpret the sequence into a human-readable pattern (e.g.
"6 6 6 2 3 4 5") -- that's later work. This module answers exactly one
question: does this sequence repeat, and if so, with what period and how
confidently.
"""

import numpy as np

# Minimum number of consistent repetitions before a candidate cycle length
# is trusted at all -- 2 repetitions can't distinguish "this is the
# pattern" from "this happened to repeat once by chance."
MIN_CONSISTENT_REPETITIONS = 3

# A repetition must match the per-position median within this many units
# (subdivision values are small positive integers, so this is an absolute
# tolerance, not a relative one -- a relative tolerance breaks down at
# small integer values like 1 vs 2) to count as "consistent."
_REPETITION_MATCH_TOLERANCE = 0.5

# Candidate cycle lengths below this are rejected outright -- period-2
# candidates are dominated by false positives on short/noisy low-
# cardinality integer sequences (easy to match by chance when values only
# range 1-6) and rarely correspond to a meaningful rhythmic phrase anyway.
_MIN_CANDIDATE_LEN = 3
_MAX_CANDIDATE_LEN_FRACTION = 0.4  # candidate cycle length capped at 40% of
                                   # the sequence length, so at least ~2-3
                                   # repetitions are structurally possible

# A candidate's raw autocorrelation must clear this to even be considered
# -- filters out lags where the "local max" is really just noise wobbling
# around zero. Deliberately low: this is only a pre-filter to avoid
# wasting time validating pure-noise lags, NOT the real rejection
# mechanism -- that's the consistent-repetitions check below, which
# validates directly against the segmented data rather than a correlation
# heuristic. Calibrated against a real case: one repetition out of six
# fully replaced by out-of-pattern values (a severe, ~40%-of-comparisons-
# contaminated corruption) still leaves real periodicity signal at the
# true period, just weak (~0.10) -- a higher threshold here would reject
# that candidate before it ever reaches the consistency check that's
# actually meant to judge it.
_MIN_PERIODICITY_STRENGTH = 0.05


def _robust_clip(signal: np.ndarray, max_mad: float = 1.5) -> np.ndarray:
    """Clip values to within `max_mad` median-absolute-deviations of the
    median. Raw autocorrelation weights outlier MAGNITUDE quadratically
    (it's a sum of products), so one repetition that's uniformly off (e.g.
    a rushed passage briefly played with much higher subdivision values)
    can dominate and even invert the correlation profile for lags that
    every OTHER repetition agrees on -- confirmed live: an otherwise-clean
    period-4 pattern with one repetition replaced by out-of-range values
    made autocorrelation negative at the true period, hiding it entirely.
    Clipping bounds that one block's leverage while still preserving that
    it's different (just not unboundedly so)."""
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    if mad == 0:
        return signal
    bound = max_mad * mad
    return np.clip(signal, median - bound, median + bound)


def _autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation (lag 0 == 1.0) of a 1-D real signal,
    mean-removed so a constant signal doesn't produce spurious peaks, and
    outlier-clipped first (see _robust_clip) so one aberrant block can't
    dominate the whole profile."""
    signal = _robust_clip(signal)
    x = signal - signal.mean()
    if np.allclose(x, 0):
        return np.zeros(len(signal))
    full = np.correlate(x, x, mode="full")
    mid = len(full) // 2
    ac = full[mid:]
    return ac / ac[0]


def _segment(values: np.ndarray, period: int) -> np.ndarray:
    """Split `values` into complete chunks of length `period`, dropping any
    trailing partial chunk. Returns shape (n_reps, period)."""
    n_reps = len(values) // period
    return values[: n_reps * period].reshape(n_reps, period)


def _repetition_consistency(segments: np.ndarray) -> tuple:
    """
    Compare each repetition (row) against the per-position median across
    all repetitions. A repetition "matches" if at least 70% of its
    positions are within _REPETITION_MATCH_TOLERANCE (absolute) of that
    position's median.

    Returns (n_consistent: int, consistency_score: float in [0, 1]).
    consistency_score is the fraction of (repetition, position) pairs that
    matched -- finer-grained than n_consistent alone, since it also
    reflects HOW consistent the consistent-enough repetitions are.
    """
    close = np.abs(segments - np.median(segments, axis=0)) <= _REPETITION_MATCH_TOLERANCE
    consistency_score = float(close.mean())
    n_consistent = int((close.mean(axis=1) >= 0.7).sum())
    return n_consistent, consistency_score


def detect_cycle(subdivision_sequence, min_repetitions: int = MIN_CONSISTENT_REPETITIONS) -> dict:
    """
    Find a repeating cycle length in `subdivision_sequence`, without
    trusting a single autocorrelation peak in isolation.

    Algorithm (per the explicit refinement this builds on -- confidence
    must reflect BOTH periodicity and repetition consistency, not just the
    strongest autocorrelation peak):
      1. Autocorrelation over the sequence gives CANDIDATE periods (local
         maxima clearing a minimum strength), not an answer.
      2. Each candidate period is validated by actually segmenting the
         sequence at that period and checking the repetitions agree with
         each other (_repetition_consistency) -- a candidate with a
         locally-strong autocorrelation peak but inconsistent segments is
         rejected here even though step 1 would have accepted it.
      3. A candidate is rejected outright if it doesn't produce at least
         `min_repetitions` consistent repetitions.
      4. Among surviving candidates, pattern_confidence blends BOTH
         signals via geometric mean, so one weak signal can't be masked by
         the other being strong. Ties in confidence are broken toward the
         SHORTER period -- the same "simplest explanation wins" rule
         timing_analysis.py's own _best_subdivision() uses, since a
         period-N sequence trivially also "repeats" at every multiple of N
         (e.g. a true period of 4 also looks periodic at lag 8, 12, ...).

    Returns exactly:
      {"cycle_length": int or None, "pattern_confidence": float in [0, 1],
       "reason": str}
    `reason` is always present, on every branch (success, no-pattern, or
    input-too-short) -- explainability is required on every path.
    """
    seq = np.asarray(subdivision_sequence, dtype=float)

    if len(seq) < (_MIN_CANDIDATE_LEN * min_repetitions):
        return {
            "cycle_length": None,
            "pattern_confidence": 0.0,
            "reason": (
                f"Only {len(seq)} value(s) in the sequence -- too few to reliably "
                f"find a repeating pattern (need at least "
                f"{_MIN_CANDIDATE_LEN * min_repetitions})."
            ),
        }

    max_period = max(_MIN_CANDIDATE_LEN, int(len(seq) * _MAX_CANDIDATE_LEN_FRACTION))
    hi = min(max_period, len(seq) - 1)
    if hi < _MIN_CANDIDATE_LEN:
        return {
            "cycle_length": None,
            "pattern_confidence": 0.0,
            "reason": "Sequence too short relative to the minimum repetition count to test any candidate cycle length.",
        }

    ac = _autocorrelation(seq)

    candidates = []
    for lag in range(_MIN_CANDIDATE_LEN, hi + 1):
        if ac[lag] < _MIN_PERIODICITY_STRENGTH:
            continue
        is_local_max = (
            (lag == _MIN_CANDIDATE_LEN or ac[lag] >= ac[lag - 1])
            and (lag == hi or ac[lag] >= ac[lag + 1])
        )
        if is_local_max:
            candidates.append((lag, float(ac[lag])))
    # Evaluate shortest-period candidates first so a tie in confidence
    # naturally keeps the shorter period without needing a second pass.
    candidates.sort(key=lambda t: t[0])

    best = None
    for period, periodicity_strength in candidates:
        segments = _segment(seq, period)
        if len(segments) < min_repetitions:
            continue
        n_consistent, consistency_score = _repetition_consistency(segments)
        if n_consistent < min_repetitions:
            continue
        confidence = round(float(np.sqrt(max(periodicity_strength, 0.0) * consistency_score)), 3)
        if best is None or confidence > best[0]:
            best = (confidence, period, len(segments), n_consistent, periodicity_strength, consistency_score)

    if best is None:
        return {
            "cycle_length": None,
            "pattern_confidence": 0.0,
            "reason": (
                f"No candidate cycle length produced at least {min_repetitions} "
                f"mutually consistent repetitions -- this sequence doesn't show "
                f"a clear repeating pattern (or it's too short/noisy to confirm one)."
            ),
        }

    confidence, period, n_reps, n_consistent, periodicity_strength, consistency_score = best
    return {
        "cycle_length": period,
        "pattern_confidence": confidence,
        "reason": (
            f"Cycle length {period} detected: {n_reps} repetitions, "
            f"{n_consistent} of them consistent with each other "
            f"(periodicity {periodicity_strength:.2f}, consistency {consistency_score:.2f})."
        ),
    }
