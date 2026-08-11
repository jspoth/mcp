"""
Timing accuracy analysis — offline, no classifier/model involved.

Given a recording and a target tempo, detects when strokes were actually
played (onset detection), then for each pair of CONSECUTIVE strokes computes
the actual instantaneous tempo (60 / gap-in-seconds) and compares it directly
to the target tempo. There is no fixed beat grid anchored to the first
onset — that approach was tried earlier and abandoned because it compounds
drift over a long recording and stops being locally meaningful. This
consecutive-gap approach is the same one app.py's /analyze endpoint uses.

USAGE:
    python timing_analysis.py path/to/recording.m4a --bpm 80
    python timing_analysis.py path/to/recording.m4a            # auto-estimates tempo
"""

import argparse
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAMPLE_RATE = 22050


def detect_onset_times(filepath, sr=SAMPLE_RATE, min_onset_gap_sec=0.05):
    """Detect stroke onset times (seconds) across the whole recording."""
    y, sr = librosa.load(filepath, sr=sr)

    # Onset detection needs a little audio context before a transient to
    # register it, so a click sitting right at t=0 can be missed. Prepend a
    # brief silence pad and subtract it back out of the returned times so
    # callers still see times relative to the original, unpadded recording.
    pad_sec = 0.25
    pad_samples = int(pad_sec * sr)
    y_padded = np.concatenate([np.zeros(pad_samples, dtype=y.dtype), y])

    onset_frames = librosa.onset.onset_detect(y=y_padded, sr=sr, backtrack=True, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr) - pad_sec

    # backtrack finds the nearest preceding STRICT local minimum of onset
    # strength, and the padded region is flat zero, so it always stops
    # right at (or a few ms before) the pad boundary — it can't walk
    # arbitrarily far into the padding chasing a minimum that doesn't
    # exist there. Any resulting negative time is small quantization
    # error, not a detection artifact, and represents a real stroke the
    # padding was added specifically to catch — so clip it to 0 rather
    # than drop it.
    onset_times = np.clip(onset_times, 0, None)

    # Clipping near-zero negative times to 0 can pile up two onsets that
    # were genuinely a few ms apart onto the exact same instant. More
    # generally, onsets closer together than any real stroke interval are
    # almost always the same physical stroke crossing the onset threshold
    # twice (attack + decay), not two strokes. Merge them by keeping the
    # earlier one.
    if len(onset_times) > 1:
        deduped = [onset_times[0]]
        for t in onset_times[1:]:
            if t - deduped[-1] >= min_onset_gap_sec:
                deduped.append(t)
        onset_times = np.array(deduped)

    duration = len(y) / sr
    return onset_times, duration


def estimate_bpm(filepath, sr=SAMPLE_RATE):
    y, sr = librosa.load(filepath, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.asarray(tempo).item())


def _score_subdivisions(gaps, beat_interval, candidates, tolerance_pct):
    """Score each candidate subdivision n by the fraction of gaps landing
    within tolerance_pct of the expected gap (beat_interval / n). Shared by
    detect_subdivision() and detect_subdivision_windowed() so the two don't
    drift apart."""
    scores = {}
    for n in candidates:
        expected_gap = beat_interval / n
        rel_dev = np.abs(gaps - expected_gap) / expected_gap
        scores[n] = float(np.mean(rel_dev <= (tolerance_pct / 100)))
    return scores


def _best_subdivision(scores, min_confidence):
    """Pick the smallest n whose fit is close to the best-fitting candidate.
    If even the best candidate doesn't clear min_confidence, the gaps don't
    convincingly match any subdivision (e.g. a noisy or misdetected
    passage) — fall back to 1 rather than confidently reporting whichever
    n happened to be least-bad. Also guards an empty scores dict (e.g. an
    empty `candidates` iterable) rather than letting max() raise."""
    if not scores:
        return 1
    best_score = max(scores.values())
    if best_score < min_confidence:
        return 1
    close_enough = [n for n, s in scores.items() if s >= best_score - 0.10]
    return min(close_enough)


def _flag_suspect_gaps(gaps, min_gap_fraction=0.4, max_gap_fraction=1.7, local_window=4):
    """
    Flag gaps that look like onset-detection artifacts rather than real,
    if mistimed, strokes — deliberately done BEFORE any subdivision or
    target-tempo comparison, using only each gap's local neighborhood, so
    a bad onset can't skew which subdivision the surrounding passage gets
    scored as (which is what happened when this check ran after
    subdivision inference).

    For gap i, compares it to the median of nearby gaps (local_window on
    each side, whatever's available at the edges):
      - much shorter than its neighbors -> likely a duplicate/false onset
        (leftover noise, or one stroke's attack+decay both crossing the
        onset threshold)
      - much longer than its neighbors  -> likely a missed onset (a real
        stroke onset detection failed to catch), not genuinely slow
        playing — e.g. a 2x gap at 80bpm/2-per-beat is far more plausibly
        one missed stroke than a stroke played at exactly half tempo

    Returns (suspect: bool ndarray, reasons: list of "short"/"long"/None),
    both length len(gaps).
    """
    n = len(gaps)
    suspect = np.zeros(n, dtype=bool)
    reasons = [None] * n
    for i in range(n):
        neighbors = np.concatenate([
            gaps[max(0, i - local_window): i],
            gaps[i + 1: i + 1 + local_window],
        ])
        if len(neighbors) < 2:
            # a single neighbor's "median" is just that one value -- too
            # noisy a baseline to flag a gap against, so abstain instead
            # of comparing (mainly matters for very short recordings)
            continue
        local_median = np.median(neighbors)
        if local_median <= 0:
            continue
        ratio = gaps[i] / local_median
        if ratio < min_gap_fraction:
            suspect[i] = True
            reasons[i] = "short"
        elif ratio > max_gap_fraction:
            suspect[i] = True
            reasons[i] = "long"
    return suspect, reasons


def detect_subdivision(onset_times, target_bpm, candidates=range(1, 7), tolerance_pct=15.0,
                        min_confidence=0.5):
    """
    Infer how many strokes are being played per beat (e.g. 2 for a
    konnakol-style "ta-ka" subdivision) directly from the observed
    stroke-to-stroke gaps, instead of requiring the user to specify it.

    For each candidate subdivision n, checks what fraction of gaps land
    close to (target beat interval / n). Picks the smallest n whose fit is
    not meaningfully worse than the best-fitting candidate, so a passage
    played straight (n=1) isn't misread as a higher subdivision just
    because a larger n can trivially "explain" noisy gaps too. If no
    candidate clears min_confidence, returns 1 rather than guessing.
    """
    if len(onset_times) < 3:
        return 1

    gaps = np.diff(onset_times)
    beat_interval = 60.0 / target_bpm
    scores = _score_subdivisions(gaps, beat_interval, candidates, tolerance_pct)
    return _best_subdivision(scores, min_confidence)


def detect_subdivision_windowed(onset_times, target_bpm, window=6, candidates=range(1, 7),
                                 tolerance_pct=15.0, hysteresis_margin=0.15, min_confidence=0.5,
                                 reliable_mask=None):
    """
    Like detect_subdivision(), but LOCAL rather than global: returns one
    subdivision value per gap, inferred from nearby gaps rather than the
    whole recording. This handles pieces where the subdivision genuinely
    changes mid-recording (e.g. a phrase played straight, then a passage
    played with two strokes per beat) — a single global subdivision would
    misread whichever section doesn't match it.

    The window for gap i is CENTERED: the `window` reliable gaps nearest to
    i by index distance, from either side — not a trailing/causal window of
    only past gaps. This is offline analysis with the whole recording
    available upfront, so nothing requires a decision to lag behind a
    change: a trailing-only window would report a subdivision change up to
    `window` gaps late (real bug — even a machine-perfect recording that
    changes subdivision got several gaps immediately after the change
    reported as "out of tempo" against the old subdivision's expected
    rate). A centered window collapses that down to at most the single gap
    sitting exactly at a symmetric transition's midpoint, where the window
    is genuinely tied between old- and new-subdivision evidence — that one
    gap is real ambiguity, not something a bigger window or a cleverer
    tiebreak can resolve, since the local evidence alone can't tell which
    side it belongs to. Gaps near the very start of the recording naturally
    draw their window from whatever reliable gaps are nearby (mostly ahead
    of them), so no special backfill case is needed there either.

    reliable_mask: optional bool array (length len(onset_times) - 1, from
    _flag_suspect_gaps()) marking which gaps are trustworthy. Gaps marked
    False are excluded from every window — instead of counting toward
    it — so a leftover false onset or missed onset can't drag the window's
    fit toward the wrong subdivision. They still get a subdivision value
    assigned (whatever `current` is at that point), just don't influence
    the decision.

    Hysteresis: only switches away from the current subdivision if a
    candidate's fit beats it by at least hysteresis_margin (a fraction,
    e.g. 0.15 = 15 percentage points) AND clears min_confidence outright —
    otherwise ordinary human timing variance within one window would
    flicker between adjacent subdivisions. The very first decision (once
    enough reliable gaps exist to make one) is committed directly without
    the hysteresis check, since there's no real "current" yet to require a
    margin over.

    Returns a list of ints, one per gap (length len(onset_times) - 1).
    """
    gaps = np.diff(onset_times)
    n_gaps = len(gaps)
    if n_gaps == 0:
        return []
    if reliable_mask is None:
        reliable_mask = np.ones(n_gaps, dtype=bool)

    beat_interval = 60.0 / target_bpm
    min_window = min(4, window)
    reliable_idx = np.flatnonzero(reliable_mask)
    subdivisions = []
    current = 1
    initialized = False

    for i in range(n_gaps):
        if len(reliable_idx) < min_window:
            subdivisions.append(current)
            continue

        nearest = sorted(reliable_idx.tolist(), key=lambda idx: (abs(idx - i), idx))[:window]
        window_idx = sorted(nearest)

        window_gaps = gaps[window_idx]
        scores = _score_subdivisions(window_gaps, beat_interval, candidates, tolerance_pct)
        candidate_n = _best_subdivision(scores, min_confidence)
        # .get(..., 0.0) rather than scores[...]: _best_subdivision can
        # return 1 as a fallback when `scores` is empty (empty `candidates`
        # iterable), in which case 1 isn't necessarily a key in scores.
        candidate_score = scores.get(candidate_n, 0.0)
        current_score = scores.get(current, 0.0)

        if not initialized:
            current = candidate_n
            initialized = True
        elif (candidate_n != current
                and candidate_score >= min_confidence
                and candidate_score >= current_score + hysteresis_margin):
            current = candidate_n

        subdivisions.append(current)

    return subdivisions


def compute_tempo_intervals(onset_times, target_bpm, tolerance_pct=8.0, subdivision="auto",
                             window=6, min_gap_fraction=0.4, max_gap_fraction=1.7):
    """
    Pipeline: onsets -> gap inspection -> flag suspicious short/long gaps
    (local-neighborhood based, no target tempo involved yet) -> subdivision
    inference using only the reliable gaps -> per-gap timing classification
    against the now-clean subdivision.

    Earlier versions inferred subdivision first and only checked for
    suspiciously short gaps afterward, using the subdivision that check was
    supposed to help protect — a leftover false onset sitting inside a
    windowed-subdivision lookback could bias which subdivision a whole
    passage got scored as before it was ever flagged. Flagging suspects
    first (via _flag_suspect_gaps(), independent of subdivision/target
    tempo) and excluding them from the subdivision-inference window instead
    avoids that.

    For each pair of consecutive detected strokes, computes the actual
    instantaneous tempo (60 / gap-in-seconds) and compares it directly to
    the target tempo. This avoids anchoring to a fixed grid from the first
    onset, which compounds drift over a long recording.

    subdivision: "auto" (default) infers strokes-per-beat LOCALLY per gap
    via detect_subdivision_windowed() — handles konnakol-style subdivided
    playing, including a piece that changes subdivision partway through,
    without requiring manual input. Pass an int to fix one subdivision for
    the whole recording (or 1 to disable adjustment entirely), or a list of
    ints (one per gap, matching len(onset_times) - 1) to supply your own
    per-gap subdivisions directly. window is passed through to
    detect_subdivision_windowed() when subdivision="auto".

    min_gap_fraction / max_gap_fraction: a gap under min_gap_fraction (or
    over max_gap_fraction) times the median of its local neighbor gaps is
    flagged "suspect" instead of scored as a timing error — a much-shorter
    gap is more likely a leftover false-positive onset than a real stroke
    played implausibly fast, and a much-longer gap is more likely a missed
    onset than a real stroke played implausibly slow. Suspect gaps are
    still returned (with a "suspect_reason" of "short" or "long") but
    excluded from tempo statistics by print_report.

    Returns a list of dicts, one per gap between consecutive strokes:
    start, end, interval_sec, instantaneous_bpm, deviation_pct, in_tempo,
    subdivision (the value actually used for that gap's comparison),
    suspect, suspect_reason.
    """
    if target_bpm <= 0:
        raise ValueError("target_bpm must be greater than 0")
    if window < 1:
        raise ValueError("window must be >= 1")
    if not (0 < min_gap_fraction < 1):
        raise ValueError("min_gap_fraction must be between 0 and 1")
    if max_gap_fraction <= 1:
        raise ValueError("max_gap_fraction must be greater than 1")

    n_gaps = max(0, len(onset_times) - 1)
    gaps = np.diff(onset_times)
    pre_suspect, pre_reasons = _flag_suspect_gaps(gaps, min_gap_fraction, max_gap_fraction)

    # NOTE: isinstance(subdivision, str) must be checked before comparing
    # subdivision == "auto" — if subdivision is an ndarray/list, that
    # comparison returns an array (or bool per-element for lists in some
    # NumPy versions), and evaluating an array's truthiness in the `if`
    # raises "truth value of an array is ambiguous" instead of falling
    # through to the list/tuple/ndarray branch below as intended.
    if isinstance(subdivision, str):
        if subdivision != "auto":
            raise ValueError(f'subdivision string must be "auto", got {subdivision!r}')
        subdivisions = detect_subdivision_windowed(onset_times, target_bpm, window=window,
                                                     reliable_mask=~pre_suspect)
    elif isinstance(subdivision, (list, tuple, np.ndarray)):
        subdivisions = [int(s) for s in subdivision]
        if len(subdivisions) != n_gaps:
            raise ValueError(
                f"subdivision list must contain exactly {n_gaps} values "
                f"(one per gap), got {len(subdivisions)}"
            )
        if any(s < 1 for s in subdivisions):
            raise ValueError("subdivision values must all be >= 1")
    else:
        if not isinstance(subdivision, (int, np.integer)) or subdivision < 1:
            raise ValueError(
                'subdivision must be "auto", a positive integer, or a list of positive integers'
            )
        subdivisions = [int(subdivision)] * n_gaps

    intervals = []
    for i in range(len(onset_times) - 1):
        start, end = onset_times[i], onset_times[i + 1]
        interval_sec = end - start
        if interval_sec <= 0:
            continue
        gap_subdivision = subdivisions[i] if i < len(subdivisions) else 1
        effective_target_bpm = target_bpm * gap_subdivision
        instantaneous_bpm = 60.0 / interval_sec
        deviation_pct = (instantaneous_bpm - effective_target_bpm) / effective_target_bpm * 100
        suspect = bool(pre_suspect[i])
        intervals.append({
            "index": i,
            "start": float(start),
            "end": float(end),
            "interval_sec": float(interval_sec),
            "instantaneous_bpm": float(instantaneous_bpm),
            "deviation_pct": float(deviation_pct),
            "in_tempo": bool(abs(deviation_pct) <= tolerance_pct) and not suspect,
            "subdivision": gap_subdivision,
            "suspect": suspect,
            "suspect_reason": pre_reasons[i],
        })
    return intervals


def print_report(intervals, target_bpm):
    if not intervals:
        print("\nNo gaps between onsets to analyze.")
        return

    mixed_subdivision = len(set(iv["subdivision"] for iv in intervals)) > 1
    header = f"\n{'gap':<6} {'time range':<18} {'stroke rate':<12} {'deviation %':<13} {'status':<20}"
    if mixed_subdivision:
        header += " subdivision"
    print(header)
    for iv in intervals:
        if iv["suspect"]:
            status = ("SUSPECT (possible missed onset)" if iv["suspect_reason"] == "long"
                      else "SUSPECT (possible duplicate onset)")
        else:
            status = "in tempo" if iv["in_tempo"] else "OUT OF TEMPO"
        time_range = f"{iv['start']:5.2f}s - {iv['end']:5.2f}s"
        line = (
            f"{iv['index']:<6} {time_range:<18} {iv['instantaneous_bpm']:8.1f}     "
            f"{iv['deviation_pct']:+6.1f}%      {status:<20}"
        )
        if mixed_subdivision:
            line += f" {iv['subdivision']}"
        print(line)

    suspect_count = sum(1 for iv in intervals if iv["suspect"])
    scored = [iv for iv in intervals if not iv["suspect"]]
    subdivisions_used = sorted(set(iv["subdivision"] for iv in intervals))

    print(f"\nMusical tempo: {target_bpm:.1f} bpm")
    if subdivisions_used != [1]:
        if len(subdivisions_used) == 1:
            n = subdivisions_used[0]
            print(f"Detected subdivision: {n} strokes/beat "
                  f"(expected stroke rate {target_bpm * n:.1f}/min — the 'stroke rate' "
                  f"column above is strokes-per-minute, not the {target_bpm:.1f} bpm "
                  f"musical tempo itself)")
        else:
            print(f"Detected changing subdivision over the recording: {subdivisions_used} strokes/beat "
                  f"(see per-gap 'subdivision' below for where it changes)")
    print(f"Gaps detected: {len(intervals)}")
    if suspect_count:
        short_count = sum(1 for iv in intervals if iv["suspect_reason"] == "short")
        long_count = sum(1 for iv in intervals if iv["suspect_reason"] == "long")
        print(f"Reliable timing gaps: {len(scored)}")
        print(f"Detection uncertainty (not counted as timing errors below, but shown "
              f"above with their raw numbers — the audio, not the performance, is in "
              f"question here):")
        if short_count:
            print(f"  Possible duplicate onsets: {short_count}")
        if long_count:
            print(f"  Possible missed onsets: {long_count}")

    if not scored:
        print("No reliable gaps left to compute tempo statistics from.")
        return

    deviations = np.array([iv["deviation_pct"] for iv in scored])
    out_of_tempo = sum(1 for iv in scored if not iv["in_tempo"])
    mean_dev = deviations.mean()
    std_dev = deviations.std()
    bias = "rushing (faster than target)" if mean_dev > 0 else "dragging (slower than target)"

    print(f"Out of tempo: {out_of_tempo}/{len(scored)} reliable gaps")
    print(f"Average deviation from target: {abs(mean_dev):.1f}% {bias}" if mean_dev != 0
          else "Average deviation from target: 0.0%")
    print(f"Consistency (std dev of per-gap deviation %): {std_dev:.1f}%  (lower = steadier tempo)")


def plot_timing(intervals, target_bpm, out_path):
    """
    Single-panel line chart of the CURRENT model: instantaneous bpm for each
    gap between consecutive onsets, plotted against a flat horizontal target
    bpm line. Points/segments are colored by whether that gap was in or out
    of tempo (or flagged suspect). No fixed beat grid is involved.

    The x-axis is the gap's actual start time in the recording, not its
    index — gaps aren't evenly spaced in time (that's the whole point of
    measuring them), so an index axis visually compresses/stretches the
    timeline in a way that doesn't match the audio.
    """
    if not intervals:
        print("\nNo intervals to plot.")
        return

    times = [iv["start"] for iv in intervals]
    bpms = [iv["instantaneous_bpm"] for iv in intervals]

    def color_for(iv):
        if iv["suspect"]:
            return "tab:gray"
        return "tab:green" if iv["in_tempo"] else "tab:red"

    colors = [color_for(iv) for iv in intervals]
    subdivisions_used = sorted(set(iv["subdivision"] for iv in intervals))
    effective_targets = [target_bpm * iv["subdivision"] for iv in intervals]

    fig, ax = plt.subplots(figsize=(max(8, len(intervals) * 0.4), 5))

    if len(subdivisions_used) == 1:
        n = subdivisions_used[0]
        target_label = f"target ({target_bpm:.0f} bpm)" if n == 1 else \
            f"target ({target_bpm:.0f} bpm x {n} strokes/beat = {target_bpm * n:.0f})"
        ax.axhline(effective_targets[0], color="black", linewidth=1.5, linestyle="--", label=target_label)
    else:
        # subdivision changes mid-recording: draw the target as a step line
        # that follows the detected subdivision at each gap, instead of one
        # flat line that would be wrong for part of the recording
        ax.step(times, effective_targets, where="post", color="black", linewidth=1.5,
                 linestyle="--", label="target (adjusted for detected subdivision)")
    ax.plot(times, bpms, color="tab:blue", linewidth=1.2, zorder=2)
    ax.scatter(times, bpms, c=colors, s=50, zorder=3)

    ax.set_xlabel("time in recording (s)")
    ax.set_ylabel("stroke rate (strokes/min)")
    ax.set_title("Instantaneous tempo per gap vs target (green = in tempo, red = out of tempo, gray = suspect)")

    handles, labels = ax.get_legend_handles_labels()
    if any(iv["suspect"] for iv in intervals):
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:gray",
                               markersize=8, label="suspect (likely false or missed onset)"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved timing chart to {out_path}")


def analyze_timing(filepath, bpm=None, plot=True, window=6, min_gap_fraction=0.4, max_gap_fraction=1.7):
    onset_times, duration = detect_onset_times(filepath)
    if len(onset_times) == 0:
        print("No onsets detected.")
        return

    if bpm is None:
        bpm = estimate_bpm(filepath)
        print(f"Estimated tempo: {bpm:.1f} bpm")
        print("Note: this is auto-estimated and used as the target for every "
              "deviation number below — librosa's beat tracker is tuned for "
              "harmonic/percussive mixes and can misjudge solo percussion. "
              "If this looks off, re-run with --bpm to set it explicitly.")
        if not (30 <= bpm <= 300):
            print(f"WARNING: {bpm:.1f} bpm is outside a plausible tempo range for "
                  f"this kind of playing — treat the results below with real "
                  f"suspicion and pass --bpm explicitly instead.")

    intervals = compute_tempo_intervals(onset_times, bpm, window=window,
                                         min_gap_fraction=min_gap_fraction, max_gap_fraction=max_gap_fraction)
    print_report(intervals, bpm)

    if plot:
        song_name = os.path.splitext(os.path.basename(filepath))[0]
        out_path = f"{song_name}_timing.png"
        plot_timing(intervals, bpm, out_path)

    return intervals


def main():
    parser = argparse.ArgumentParser(description="Analyze stroke timing accuracy against a target tempo.")
    parser.add_argument("filepath", help="Path to the audio/video recording")
    parser.add_argument("--bpm", type=float, default=None, help="Target tempo (auto-estimated if omitted)")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating the visual chart")
    parser.add_argument("--window", type=int, default=6,
                         help="Trailing window size (in gaps) for local subdivision detection (default: 6)")
    parser.add_argument("--min-gap-fraction", type=float, default=0.4,
                         help="Gaps shorter than this fraction of their local neighbor gaps are "
                              "flagged suspect (likely false onset) instead of scored (default: 0.4)")
    parser.add_argument("--max-gap-fraction", type=float, default=1.7,
                         help="Gaps longer than this multiple of their local neighbor gaps are "
                              "flagged suspect (likely missed onset) instead of scored (default: 1.7)")
    args = parser.parse_args()

    analyze_timing(args.filepath, bpm=args.bpm, plot=not args.no_plot, window=args.window,
                    min_gap_fraction=args.min_gap_fraction, max_gap_fraction=args.max_gap_fraction)


if __name__ == "__main__":
    main()
