"""
MCP server exposing timing_analysis.py as tools for an LLM client — analyze
the timing/tempo accuracy of a percussion practice recording from chat.

Thin wrapper: all analysis logic lives in timing_analysis.py. Never returns
raw audio, only computed stats (and optionally a rendered chart). Payloads
are logged to stderr before returning (stdout/stdin are the MCP channel).

Install (inside a venv):
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Run directly (sanity check — sits waiting for stdio input, expected):
    python mcp_server.py

--------------------------------------------------------------------------
Claude Desktop config snippet (~/Library/Application Support/Claude/
claude_desktop_config.json on macOS; replace the paths with your own):

{
  "mcpServers": {
    "mridangam-playing-analysis": {
      "command": "/path/to/mridangam-playing-analysis/venv/bin/python",
      "args": ["/path/to/mridangam-playing-analysis/mcp_server.py"]
    }
  }
}

Or with Claude Code CLI:
    claude mcp add mridangam-playing-analysis -- \\
        /path/to/mridangam-playing-analysis/venv/bin/python \\
        /path/to/mridangam-playing-analysis/mcp_server.py
--------------------------------------------------------------------------
"""

import json
import sys
import uuid
from datetime import datetime, timezone

import numpy as np

from timing_analysis import (
    ANALYSIS_VERSION,
    compute_tempo_intervals,
    detect_onset_times,
    estimate_bpm,
    plot_timing,
)

# mcp==2.0.0 renamed FastMCP to MCPServer; fall back to the old name if needed.
try:
    from mcp.server.fastmcp import FastMCP as _MCPServerClass
except ImportError:
    from mcp.server.mcpserver import MCPServer as _MCPServerClass

try:
    from mcp.server.fastmcp import Image
except ImportError:
    from mcp.server.mcpserver import Image

mcp = _MCPServerClass("mridangam-playing-analysis")


def _log(event, level="info", **fields):
    """One-line JSON log to stderr — stdin/stdout are the MCP channel."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    print(json.dumps(entry, default=str), file=sys.stderr)


def _build_analysis(filepath: str, bpm: float | None = None) -> dict:
    """
    Core implementation, separate from the @mcp.tool wrapper so it can be
    called and tested directly. Returns a plain-data stats dict — never the
    raw waveform or audio bytes.
    """
    try:
        onset_times, duration = detect_onset_times(filepath)
    except ValueError as exc:
        return {"filepath": filepath, "analysis_status": "invalid_audio", "error": str(exc)}

    if len(onset_times) == 0:
        return {
            "filepath": filepath,
            "analysis_status": "insufficient_data",
            "error": "No onsets detected in this recording.",
            "duration_sec": float(duration),
        }

    MIN_ONSETS = 5
    if len(onset_times) < MIN_ONSETS:
        return {
            "filepath": filepath,
            "analysis_status": "insufficient_data",
            "duration_sec": float(duration),
            "onset_count": int(len(onset_times)),
            "error": f"Only {len(onset_times)} onset(s) detected — too few to say "
                     f"anything meaningful about timing. Try a longer recording.",
        }

    target_bpm = float(bpm) if bpm is not None else estimate_bpm(filepath)
    bpm_was_estimated = bpm is None
    # Mirrors the CLI's plausibility check — librosa's beat tracker can
    # misjudge solo percussion tempo.
    bpm_warning = None
    if bpm_was_estimated and not (30 <= target_bpm <= 300):
        bpm_warning = (
            f"Estimated tempo ({target_bpm:.1f} bpm) is outside a plausible "
            f"range for this kind of playing — treat the results below with "
            f"real suspicion and pass bpm explicitly instead."
        )

    if target_bpm <= 0:
        return {
            "filepath": filepath,
            "analysis_status": "invalid_input",
            "error": f"{'Auto-estimated' if bpm_was_estimated else 'Given'} tempo "
                     f"({target_bpm:.1f} bpm) isn't usable — pass a positive bpm explicitly.",
            "duration_sec": float(duration),
        }

    intervals = compute_tempo_intervals(onset_times, target_bpm)

    if not intervals:
        return {
            "filepath": filepath,
            "analysis_status": "insufficient_data",
            "target_bpm": target_bpm,
            "bpm_was_estimated": bpm_was_estimated,
            "duration_sec": float(duration),
            "onset_count": int(len(onset_times)),
            "error": "Only one onset detected — no gaps to analyze.",
        }

    # Suspect gaps (likely duplicate/missed onsets, not real mistiming) are
    # excluded from the stats — same rule as the CLI's print_report.
    reliable = [iv for iv in intervals if not iv["suspect"]]
    short_count = sum(1 for iv in intervals if iv.get("suspect_reason") == "short")
    long_count = sum(1 for iv in intervals if iv.get("suspect_reason") == "long")
    # Each gap's deviation is measured against target_bpm * its subdivision —
    # surface what subdivisions were used so the LLM doesn't misread a
    # subdivided stroke rate as a plain deviation from target_bpm.
    subdivisions_used = sorted(set(iv["subdivision"] for iv in intervals))

    if not reliable:
        return {
            "filepath": filepath,
            "analysis_status": "low_confidence",
            "target_bpm": round(target_bpm, 2),
            "bpm_was_estimated": bpm_was_estimated,
            "bpm_warning": bpm_warning,
            "duration_sec": round(float(duration), 2),
            "onset_count": int(len(onset_times)),
            "total_gaps": len(intervals),
            "subdivisions_used": subdivisions_used,
            "possible_duplicate_onsets": short_count,
            "possible_missed_onsets": long_count,
            "error": "No reliable gaps — every gap looks like a detection artifact "
                     "(duplicate/missed onset), not real timing data.",
            "gaps": intervals,
        }

    deviations = np.array([iv["deviation_pct"] for iv in reliable])
    in_tempo_count = sum(1 for iv in reliable if iv["in_tempo"])
    total = len(reliable)
    mean_dev = float(deviations.mean())
    std_dev = float(deviations.std())

    summary = {
        "filepath": filepath,
        "analysis_status": "success",
        "target_bpm": round(target_bpm, 2),
        "bpm_was_estimated": bpm_was_estimated,
        "bpm_warning": bpm_warning,
        "duration_sec": round(float(duration), 2),
        "onset_count": int(len(onset_times)),
        "total_gaps": len(intervals),
        "subdivisions_used": subdivisions_used,
        "reliable_gaps": total,
        "possible_duplicate_onsets": short_count,
        "possible_missed_onsets": long_count,
        "in_tempo_count": in_tempo_count,
        "out_of_tempo_count": total - in_tempo_count,
        "in_tempo_pct": round(100.0 * in_tempo_count / total, 1),
        "average_deviation_pct": round(mean_dev, 2),
        "average_abs_deviation_pct": round(float(np.abs(deviations).mean()), 2),
        "bias": (
            "rushing (faster than target)"
            if mean_dev > 0
            else "dragging (slower than target)"
            if mean_dev < 0
            else "none"
        ),
        "consistency_std_dev_pct": round(std_dev, 2),
        "gaps": intervals,
    }
    return summary


@mcp.tool()
def analyze_timing(filepath: str, bpm: float | None = None, include_chart: bool = False):
    """
    Analyze stroke timing accuracy in a mridangam (or any percussion)
    recording against a target tempo.

    Detects stroke onsets, estimates tempo if not given, and computes
    gap-by-gap deviation from the target tempo. Returns only the computed
    statistics (never the raw audio) — nothing is written to disk unless
    include_chart is set.

    Args:
        filepath: Path to a local audio/video file to analyze.
        bpm: Target tempo in beats per minute. If omitted, it is
            auto-estimated from the recording.
        include_chart: If True, also renders a chart PNG (from the computed
            stats, not the raw audio) and attaches it to the response. Off
            by default, since it's the one path that touches disk.
    """
    request_id = uuid.uuid4().hex[:8]
    summary = _build_analysis(filepath, bpm)
    summary["analysis_version"] = ANALYSIS_VERSION

    # Logged before returning — stdin/stdout are the MCP channel, so this is
    # the transparency mechanism instead of an interactive prompt.
    _log(
        "analyze_timing_result",
        level="warning" if "error" in summary else "info",
        request_id=request_id,
        filepath=filepath,
        include_chart=include_chart,
        result=summary,
    )

    if not include_chart or ("error" in summary and "gaps" not in summary):
        return summary

    # Written to a temp file only because Image() needs a path; removed
    # again right after it reads the bytes in, so nothing persists on disk.
    import os
    import tempfile

    song_name = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join(tempfile.gettempdir(), f"{song_name}_timing_mcp.png")
    try:
        plot_timing(summary["gaps"], summary["target_bpm"], out_path)
        image = Image(path=out_path)
        _log("chart_rendered", request_id=request_id, filepath=filepath)
        return [summary, image]
    except Exception as exc:  # chart generation is a nice-to-have, not required
        _log("chart_generation_skipped", level="warning", request_id=request_id,
             filepath=filepath, error=str(exc))
        return summary
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


if __name__ == "__main__":
    mcp.run()
