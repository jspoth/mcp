"""
Phase 3 of the MCP v2 plan: unify the audio-only (Phases 0-2) and
script-provided paths into one interface, analyze_mridangam(audio, script).

DESIGN DECISION -- how real audio feeds into cycle_detection.detect_cycle()
(flagged in Phase 0 as an open question, resolved here, made visible per
instruction rather than left as an invisible implementation detail):

Options considered:
  A) Autocorrelate detect_subdivision_windowed()'s per-gap subdivision
     output directly (the ORIGINAL plan). Rejected in Phase 0 already --
     confirmed live that its window/hysteresis smoothing flattens short
     cycle patterns before autocorrelation ever sees them, whenever the
     true cycle length is comparable to the window size.
  B) Call detect_subdivision_windowed() with a much smaller window (e.g.
     window=2) specifically for cycle detection. Rejected: still couples
     cycle detection to a smoothing/hysteresis mechanism designed to solve
     a different problem (robustness to single-gap noise over a LONG
     recording's tempo), for no benefit -- detect_cycle()'s own
     autocorrelation + consistency-check already provides that robustness
     independently, at the position-count level rather than the smoothed-
     subdivision level.
  C) Feed RAW gap durations (onset-to-onset intervals, in seconds) directly
     into detect_cycle() as its "subdivision_sequence" input, after
     interpolating out detection artifacts flagged by the EXISTING
     _flag_suspect_gaps() (same function compute_tempo_intervals already
     uses). CHOSEN. detect_cycle()'s algorithm (autocorrelation +
     segmented-repetition consistency check) doesn't actually require its
     input to be small integers -- that was only true of Phase 0's own
     synthetic tests, not a real constraint of the code. Verified live:
     detect_cycle() on raw continuous gap durations for a
     [1.0,0.5,0.5,1.0]-beat pattern x8 correctly found cycle_length=4 at
     confidence 0.935, no changes to cycle_detection.py needed.

This also means Phase 1/2 (baseline_scoring.py) and Phase 0
(cycle_detection.py) now operate on the SAME representation throughout the
audio-only pipeline (raw gap durations), rather than switching between
subdivision integers and raw durations at different stages -- one
consistent signal end to end, which is simpler than it would have been
under option A or B.

detect_subdivision_windowed() is not used anywhere in the audio-only path.
It remains available (unchanged) for the pre-existing analyze_timing tool
in mcp_server.py, which measures against a flat external target tempo --
a genuinely different question than "does this repeat a pattern."
"""

import os

import numpy as np

import timing_analysis as ta
import cycle_detection as cd
import baseline_scoring as bs

_SCRIPT_EXTENSIONS = (".txt", ".script")


def _find_co_located_script(audio_path: str):
    """Look for a script file next to the audio (same basename, one of
    _SCRIPT_EXTENSIONS). Returns the path if found, else None. Zero-cost,
    automatic, no interaction -- covers "I have the script, I just forgot
    to pass it" without ever asking a question (see module docstring in
    the MCP v2 plan discussion: never ask, always resolve to an answer)."""
    base, _ = os.path.splitext(audio_path)
    for ext in _SCRIPT_EXTENSIONS:
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def _parse_script(text: str) -> list:
    """Minimal script format: whitespace-separated positive integers, one
    per position (e.g. "6 6 6 2 3 4 5"), same shape as compute_tempo_intervals'
    own `subdivision` list parameter -- reuses that existing validated path
    directly rather than inventing a new alignment algorithm."""
    return [int(tok) for tok in text.split()]


def analyze_mridangam(filepath: str, script: str = None, bpm: float = None) -> dict:
    """
    Unified entry point. Resolves to an answer on every path -- never asks
    the user a question, per the established convention (see
    costoptimizer/chat.go's system prompt for the precedent this follows).

    script: either a literal script string ("6 6 6 2 3 4 5"), or None.
      - given directly -> validated against it (script-provided path)
      - None -> check for a co-located script file next to the audio ->
        found -> same script-provided path -> not found -> run the
        audio-only inference path (Phases 0-2)

    Returns a dict always containing `mode` (one of "script_provided",
    "script_found", "inferred", or an error/insufficient-data status) and
    `reason` (one-line plain-English explanation of which path was taken
    and why), plus mode-specific fields.
    """
    onset_times, duration = ta.detect_onset_times(filepath)
    if len(onset_times) < 2:
        return {"mode": "insufficient_data", "reason": "Too few onsets detected to analyze."}

    if bpm is None:
        bpm = ta.estimate_bpm(filepath)

    script_source = None
    found_script_path = None
    if script is not None:
        script_source = "script_provided"
    else:
        found_script_path = _find_co_located_script(filepath)
        if found_script_path is not None:
            with open(found_script_path) as f:
                script = f.read()
            script_source = "script_found"

    if script is not None:
        subdivisions = _parse_script(script)
        intervals = ta.compute_tempo_intervals(onset_times, bpm, subdivision=subdivisions) \
            if len(subdivisions) == len(onset_times) - 1 else \
            ta.compute_tempo_intervals(onset_times, bpm)
        reason = (
            f"Found {os.path.basename(found_script_path)} alongside the audio -- validated against it."
            if script_source == "script_found" else
            "Script provided -- validated playing against it."
        )
        return {"mode": script_source, "reason": reason, "target_bpm": round(bpm, 2), "intervals": intervals}

    # No script given or found -- audio-only inference (Phases 0-2).
    gaps = np.diff(onset_times)
    # Raw gap durations, not a subdivision sequence -- see module docstring
    # "DESIGN DECISION" section for why.
    cycle_result = cd.detect_cycle(gaps)

    if cycle_result["cycle_length"] is None:
        return {
            "mode": "inferred",
            "reason": (
                "No script file found alongside the audio, and no clear repeating "
                "pattern could be inferred from the recording -- " + cycle_result["reason"]
            ),
            "pattern_confidence": 0.0,
        }

    cycle_length = cycle_result["cycle_length"]
    baseline_result = bs.build_baseline(gaps, cycle_length)
    scored = bs.score_repetitions(gaps, cycle_length, baseline_result["baseline"])

    return {
        "mode": "inferred",
        "reason": (
            f"No script file found alongside the audio -- used a self-inferred pattern "
            f"(pattern_confidence {cycle_result['pattern_confidence']:.2f}, "
            f"baseline_confidence {baseline_result['baseline_confidence']:.2f}). "
            + cycle_result["reason"]
        ),
        "target_bpm": round(bpm, 2),
        "cycle_length": cycle_length,
        "pattern_confidence": cycle_result["pattern_confidence"],
        "baseline": baseline_result["baseline"],
        "baseline_confidence": baseline_result["baseline_confidence"],
        "position_confidence": baseline_result["position_confidence"],
        "repetitions": scored["repetitions"],
    }
