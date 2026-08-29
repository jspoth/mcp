"""
Phase 1/2 of the MCP v2 plan: baseline construction from repetitions, and
per-repetition scoring against that baseline. Pure over gap durations +
a confirmed cycle_length (from cycle_detection.detect_cycle) -- no audio,
onset, or MCP knowledge, same isolation convention as cycle_detection.py.

Reuses timing_analysis._flag_suspect_gaps() for the noise-vs-real-deviation
call at scoring time -- a position whose gap is flagged suspect by the
existing local-neighbor heuristic is a likely detection artifact, not
scored as a real timing miss; anything else that deviates from baseline is.
"""

import numpy as np

import timing_analysis as ta

_POSITION_MATCH_TOLERANCE = 0.25  # relative; gap durations are continuous, unlike subdivisions


def build_baseline(gaps, cycle_length: int) -> dict:
    """
    Segment `gaps` (raw onset-to-onset durations, seconds) into repetitions
    of `cycle_length`, and build the canonical per-position baseline as the
    per-position MEDIAN duration (robust to one bad repetition, unlike a
    mean). Returns:
      baseline: list[float] (length cycle_length) -- per-position median duration
      position_confidence: list[float in [0,1]] -- per-position agreement
          across repetitions (tight agreement = high; one wildly different
          repetition lowers this position's confidence rather than being
          silently averaged into the median unnoticed)
      baseline_confidence: float in [0,1] -- mean of position_confidence
      num_repetitions: int
      reason: str, always present
    """
    gaps = np.asarray(gaps, dtype=float)
    n_reps = len(gaps) // cycle_length
    if n_reps < 1:
        return {
            "baseline": None, "position_confidence": None, "baseline_confidence": 0.0,
            "num_repetitions": 0,
            "reason": f"Not enough gaps ({len(gaps)}) for even one repetition of cycle length {cycle_length}.",
        }
    segments = gaps[: n_reps * cycle_length].reshape(n_reps, cycle_length)
    baseline = np.median(segments, axis=0)

    tol = np.maximum(baseline * _POSITION_MATCH_TOLERANCE, 1e-6)
    close = np.abs(segments - baseline) <= tol  # (n_reps, cycle_length)
    position_confidence = close.mean(axis=0)  # fraction of reps matching, per position
    baseline_confidence = float(position_confidence.mean())

    return {
        "baseline": [round(float(v), 4) for v in baseline],
        "position_confidence": [round(float(v), 3) for v in position_confidence],
        "baseline_confidence": round(baseline_confidence, 3),
        "num_repetitions": n_reps,
        "reason": (
            f"Baseline built from {n_reps} repetitions of {cycle_length} positions; "
            f"average per-position agreement {baseline_confidence:.2f}"
            + (" (some positions vary more than others -- see position_confidence)"
               if position_confidence.min() < 0.7 else "")
            + "."
        ),
    }


def score_repetitions(gaps, cycle_length: int, baseline: list, tolerance_pct: float = 8.0) -> dict:
    """
    Score each repetition's positions against `baseline` (from
    build_baseline). For each gap: reuses timing_analysis._flag_suspect_gaps
    (local-neighbor based, independent of the baseline) to decide whether a
    bad match is a likely DETECTION ARTIFACT (excluded from scoring, same
    convention as compute_tempo_intervals) or a REAL deviation from the
    player's own baseline (scored).

    Returns:
      repetitions: list of dicts, one per repetition:
        {index, deviations_pct (per position, None where suspect),
         suspect (per position, bool), out_of_baseline_positions (list of
         position indices that deviated beyond tolerance_pct and were NOT
         suspect -- i.e. genuine misses, not noise)}
      reason: str
    """
    gaps = np.asarray(gaps, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    n_reps = len(gaps) // cycle_length
    suspect, _ = ta._flag_suspect_gaps(gaps)

    reps = []
    for r in range(n_reps):
        start = r * cycle_length
        rep_gaps = gaps[start:start + cycle_length]
        rep_suspect = suspect[start:start + cycle_length]
        deviations = []
        bad_positions = []
        for p in range(cycle_length):
            if rep_suspect[p]:
                deviations.append(None)  # artifact, not a real timing measurement
                continue
            dev_pct = (rep_gaps[p] - baseline[p]) / baseline[p] * 100 if baseline[p] > 0 else 0.0
            deviations.append(round(float(dev_pct), 2))
            if abs(dev_pct) > tolerance_pct:
                bad_positions.append(p)
        reps.append({
            "index": r,
            "deviations_pct": deviations,
            "suspect": [bool(s) for s in rep_suspect],
            "out_of_baseline_positions": bad_positions,
        })

    total_bad = sum(len(r["out_of_baseline_positions"]) for r in reps)
    return {
        "repetitions": reps,
        "reason": (
            f"Scored {n_reps} repetitions against the baseline: "
            f"{total_bad} position(s) across all repetitions deviated beyond "
            f"{tolerance_pct:.0f}% and weren't detection artifacts -- real timing misses."
        ),
    }
