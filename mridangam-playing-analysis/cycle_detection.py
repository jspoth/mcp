"""
Cycle and section detection for rhythmic subdivision sequences.

Phase 0 goals:
    1. Detect a repeating cycle within a section.
    2. Detect multiple rhythmic sections in one recording.
    3. Return confidence and human-readable reasons.

This module operates only on a subdivision sequence. It does not know
anything about audio, MCP, or LLMs.
"""

import numpy as np


MIN_CONSISTENT_REPETITIONS = 3

_REPETITION_MATCH_TOLERANCE = 0.5
_MIN_CANDIDATE_LEN = 3
_MAX_CANDIDATE_LEN_FRACTION = 0.4
_MIN_PERIODICITY_STRENGTH = 0.05

# Section scanning.
_SECTION_SCAN_REPETITIONS = 3
_SECTION_STEP = 1
_MIN_SECTION_LENGTH = 12

# Two adjacent windows are considered the same rhythmic section when their
# detected cycle lengths agree.
_MIN_STABLE_WINDOWS = 3


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _robust_clip(signal: np.ndarray, max_mad: float = 1.5) -> np.ndarray:
    """Limit the influence of extreme subdivision outliers."""
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))

    if mad == 0:
        return signal

    bound = max_mad * mad
    return np.clip(signal, median - bound, median + bound)


def _autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Return normalized, mean-removed autocorrelation."""
    signal = _robust_clip(signal)

    x = signal - signal.mean()

    if np.allclose(x, 0):
        return np.zeros(len(signal))

    full = np.correlate(x, x, mode="full")
    mid = len(full) // 2
    ac = full[mid:]

    if ac[0] == 0:
        return np.zeros(len(signal))

    return ac / ac[0]


def _segment(values: np.ndarray, period: int) -> np.ndarray:
    """Split values into complete repetitions of period."""
    n_reps = len(values) // period

    if n_reps == 0:
        return np.empty((0, period))

    return values[: n_reps * period].reshape(n_reps, period)


def _repetition_consistency(segments: np.ndarray) -> tuple:
    """
    Compare repetitions against the per-position median.

    Returns:
        (number_of_consistent_repetitions, consistency_score)
    """
    if len(segments) == 0:
        return 0, 0.0

    median = np.median(segments, axis=0)

    close = (
        np.abs(segments - median)
        <= _REPETITION_MATCH_TOLERANCE
    )

    consistency_score = float(close.mean())

    n_consistent = int(
        (close.mean(axis=1) >= 0.7).sum()
    )

    return n_consistent, consistency_score


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

def detect_cycle(
    subdivision_sequence,
    min_repetitions: int = MIN_CONSISTENT_REPETITIONS,
) -> dict:
    """
    Detect a repeating cycle in one rhythmic section.

    Returns:

        {
            "cycle_length": int | None,
            "pattern_confidence": float,
            "reason": str
        }
    """

    seq = np.asarray(
        subdivision_sequence,
        dtype=float,
    )

    if len(seq) < _MIN_CANDIDATE_LEN * min_repetitions:
        return {
            "cycle_length": None,
            "pattern_confidence": 0.0,
            "reason": (
                f"Only {len(seq)} values; too few to reliably "
                f"detect a repeating cycle."
            ),
        }

    max_period = max(
        _MIN_CANDIDATE_LEN,
        int(len(seq) * _MAX_CANDIDATE_LEN_FRACTION),
    )

    hi = min(
        max_period,
        len(seq) - 1,
    )

    if hi < _MIN_CANDIDATE_LEN:
        return {
            "cycle_length": None,
            "pattern_confidence": 0.0,
            "reason": "Sequence too short for cycle detection.",
        }

    ac = _autocorrelation(seq)

    candidates = []

    for lag in range(
        _MIN_CANDIDATE_LEN,
        hi + 1,
    ):
        strength = float(ac[lag])

        if strength < _MIN_PERIODICITY_STRENGTH:
            continue

        left = ac[lag - 1] if lag > 0 else strength
        right = ac[lag + 1] if lag < len(ac) - 1 else strength

        is_local_max = (
            strength >= left
            and strength >= right
        )

        if is_local_max:
            candidates.append(
                (lag, strength)
            )

    # Shorter periods are preferred when confidence is nearly tied.
    candidates.sort(key=lambda x: x[0])

    best = None

    for period, periodicity_strength in candidates:

        segments = _segment(seq, period)

        if len(segments) < min_repetitions:
            continue

        (
            n_consistent,
            consistency_score,
        ) = _repetition_consistency(segments)

        if n_consistent < min_repetitions:
            continue

        confidence = float(
            np.sqrt(
                max(periodicity_strength, 0.0)
                * consistency_score
            )
        )

        candidate = {
            "confidence": confidence,
            "period": period,
            "n_repetitions": len(segments),
            "n_consistent": n_consistent,
            "periodicity": periodicity_strength,
            "consistency": consistency_score,
        }

        if best is None:
            best = candidate
            continue

        if confidence > best["confidence"] + 0.02:
            best = candidate

    if best is None:
        return {
            "cycle_length": None,
            "pattern_confidence": 0.0,
            "reason": (
                f"No cycle produced at least "
                f"{min_repetitions} consistent repetitions."
            ),
        }

    confidence = best["confidence"]
    period = best["period"]

    return {
        "cycle_length": int(period),
        "pattern_confidence": round(
            confidence,
            3,
        ),
        "reason": (
            f"Cycle length {period} detected from "
            f"{best['n_consistent']}/{best['n_repetitions']} "
            f"consistent repetitions; "
            f"periodicity={best['periodicity']:.2f}, "
            f"consistency={best['consistency']:.2f}."
        ),
    }


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _candidate_cycle_at(
    sequence: np.ndarray,
    start: int,
    cycle_length: int,
) -> dict:
    """
    Test whether a cycle is stable starting at a particular position.

    The window contains enough repetitions to make the decision meaningful.
    """

    end = start + (
        cycle_length * _SECTION_SCAN_REPETITIONS
    )

    if end > len(sequence):
        return {
            "cycle_length": None,
            "confidence": 0.0,
        }

    result = detect_cycle(
        sequence[start:end],
        min_repetitions=_SECTION_SCAN_REPETITIONS,
    )

    return {
        "cycle_length": result["cycle_length"],
        "confidence": result["pattern_confidence"],
    }


def _find_local_cycle(
    sequence: np.ndarray,
    start: int,
) -> dict:
    """
    Detect the locally dominant cycle around start.

    We use a short local window rather than analysing the entire recording.
    """

    remaining = len(sequence) - start

    if remaining < (
        _MIN_SECTION_LENGTH
    ):
        return {
            "cycle_length": None,
            "confidence": 0.0,
        }

    max_period = max(
        _MIN_CANDIDATE_LEN,
        min(
            int(remaining / _SECTION_SCAN_REPETITIONS),
            int(remaining * 0.4),
        ),
    )

    window_end = min(
        len(sequence),
        start + max(
            _MIN_SECTION_LENGTH,
            max_period * _SECTION_SCAN_REPETITIONS,
        ),
    )

    window = sequence[start:window_end]

    result = detect_cycle(
        window,
        min_repetitions=_SECTION_SCAN_REPETITIONS,
    )

    return {
        "cycle_length": result["cycle_length"],
        "confidence": result["pattern_confidence"],
    }


def detect_sections(
    subdivision_sequence,
    min_section_length: int = 15,
    min_improvement: float = 0.10,
) -> list:
    """
    Recursively split a sequence when two independently detected cycles
    explain the data substantially better than one cycle for the whole region.

    Example:

        [7-cycle] [5-cycle] [7-cycle]

    becomes:

        section 1 -> cycle 7
        section 2 -> cycle 5
        section 3 -> cycle 7
    """

    seq = np.asarray(subdivision_sequence, dtype=float)
    n = len(seq)

    if n == 0:
        return []

    def split_region(start: int, end: int) -> list:
        length = end - start

        # Too short to split reliably.
        if length < min_section_length * 2:
            return [{
                "start": start,
                "end": end,
                "sequence": seq[start:end].tolist(),
            }]

        whole = detect_cycle(
            seq[start:end],
            min_repetitions=MIN_CONSISTENT_REPETITIONS,
        )

        whole_confidence = whole["pattern_confidence"]

        best_split = None

        # Try every possible boundary.
        for boundary in range(
            start + min_section_length,
            end - min_section_length + 1,
        ):
            left = detect_cycle(
                seq[start:boundary],
                min_repetitions=MIN_CONSISTENT_REPETITIONS,
            )

            right = detect_cycle(
                seq[boundary:end],
                min_repetitions=MIN_CONSISTENT_REPETITIONS,
            )

            if (
                left["cycle_length"] is None
                or right["cycle_length"] is None
            ):
                continue

            # Both sides should explain their regions better than
            # the unsplit region.
            split_score = (
                left["pattern_confidence"]
                + right["pattern_confidence"]
            ) / 2.0

            improvement = split_score - whole_confidence

            if improvement < min_improvement:
                continue

            candidate = {
                "boundary": boundary,
                "score": split_score,
                "improvement": improvement,
            }

            if (
                best_split is None
                or candidate["score"] > best_split["score"]
            ):
                best_split = candidate

        # No useful split → this is one section.
        if best_split is None:
            return [{
                "start": start,
                "end": end,
                "sequence": seq[start:end].tolist(),
            }]

        boundary = best_split["boundary"]

        # Recursively split both sides.
        left_sections = split_region(
            start,
            boundary,
        )

        right_sections = split_region(
            boundary,
            end,
        )

        return left_sections + right_sections

    return split_region(0, n)


# ---------------------------------------------------------------------------
# Public combined API
# ---------------------------------------------------------------------------

def detect_cycles_by_section(
    subdivision_sequence,
    min_repetitions: int = MIN_CONSISTENT_REPETITIONS,
) -> dict:
    """
    Detect rhythmic sections first, then independently detect a cycle
    within each section.
    """

    seq = np.asarray(
        subdivision_sequence,
        dtype=float,
    )

    sections = detect_sections(seq)

    results = []

    for index, section in enumerate(sections):

        cycle = detect_cycle(
            section["sequence"],
            min_repetitions=min_repetitions,
        )

        results.append(
            {
                "section": index,
                "start": section["start"],
                "end": section["end"],
                "cycle_length": cycle["cycle_length"],
                "pattern_confidence": cycle[
                    "pattern_confidence"
                ],
                "reason": cycle["reason"],
            }
        )

    return {
        "sections": results,
        "num_sections": len(results),
        "reason": (
            f"Analyzed {len(results)} rhythmic "
            f"section(s) independently."
        ),
    }