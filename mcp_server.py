"""
MCP server exposing timing_analysis.py as tools for an LLM client
(Claude Desktop, Claude Code, etc.) — analyze the timing/tempo accuracy of a
percussion practice recording directly from your chat client.

This is a thin wrapper: all analysis logic lives in timing_analysis.py
(detect_onset_times, estimate_bpm, compute_tempo_intervals, plot_timing).
This file only adapts those functions to the MCP tool-calling protocol.

Security / transparency design:
  - The tool takes a filepath and reads the audio locally via the existing
    functions (which use librosa). The raw audio bytes / waveform sample
    array are NEVER included in the tool's return value — only computed
    stats (and, optionally, a rendered chart image) go back to the LLM host.
  - stdin/stdout are reserved for the MCP JSON-RPC protocol channel, so this
    server never uses print()/input() on them. Since an MCP server can't do
    an interactive y/n confirmation prompt the way a CLI script can,
    transparency here comes from logging the exact payload being returned
    to stderr before it's sent, plus most MCP hosts showing the tool
    call/result inline in the chat.

Install (inside a venv):
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Run directly (for a quick sanity check — it will just sit waiting for
stdio protocol input, which is expected):
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

import numpy as np

from timing_analysis import (
    compute_tempo_intervals,
    detect_onset_times,
    estimate_bpm,
    plot_timing,
)

# This installed SDK version (mcp==2.0.0) renamed the old FastMCP class to
# MCPServer, but it's the same ergonomic decorator-based API. Fall back to
# FastMCP under its older name if that's what's installed instead.
try:
    from mcp.server.fastmcp import FastMCP as _MCPServerClass
except ImportError:
    from mcp.server.mcpserver import MCPServer as _MCPServerClass

try:
    from mcp.server.fastmcp import Image
except ImportError:
    from mcp.server.mcpserver import Image

mcp = _MCPServerClass("mridangam-playing-analysis")


def _build_analysis(filepath: str, bpm: float | None = None) -> dict:
    """
    Core implementation, kept separate from the @mcp.tool wrapper so it can
    be called directly (and verified) without going through the MCP
    protocol layer.

    Reuses the Phase-1 functions in timing_analysis.py directly rather than
    reimplementing any onset/tempo logic. Returns ONLY a plain-data stats
    dict — never the raw waveform, never audio bytes, never anything
    derived straight from librosa.load()'s sample array.
    """
    onset_times, duration = detect_onset_times(filepath)

    if len(onset_times) == 0:
        return {
            "filepath": filepath,
            "error": "No onsets detected in this recording.",
            "duration_sec": float(duration),
        }

    target_bpm = float(bpm) if bpm is not None else estimate_bpm(filepath)
    bpm_was_estimated = bpm is None

    if target_bpm <= 0:
        return {
            "filepath": filepath,
            "error": f"{'Auto-estimated' if bpm_was_estimated else 'Given'} tempo "
                     f"({target_bpm:.1f} bpm) isn't usable — pass a positive bpm explicitly.",
            "duration_sec": float(duration),
        }

    intervals = compute_tempo_intervals(onset_times, target_bpm)

    if not intervals:
        return {
            "filepath": filepath,
            "target_bpm": target_bpm,
            "bpm_was_estimated": bpm_was_estimated,
            "duration_sec": float(duration),
            "onset_count": int(len(onset_times)),
            "error": "Only one onset detected — no gaps to analyze.",
        }

    # Gaps flagged "suspect" (likely a duplicate or missed onset from
    # detection, not a real mistimed stroke) are excluded from the tempo
    # statistics — same rule timing_analysis.print_report applies. Without
    # this, a single missed onset (e.g. a 2x gap read as -50% deviation)
    # can dominate the average/std of an otherwise clean recording.
    reliable = [iv for iv in intervals if not iv["suspect"]]
    short_count = sum(1 for iv in intervals if iv.get("suspect_reason") == "short")
    long_count = sum(1 for iv in intervals if iv.get("suspect_reason") == "long")
    # Every "instantaneous_bpm"/"deviation_pct" in `gaps` is measured against
    # target_bpm * that gap's subdivision, not target_bpm directly (e.g. a
    # gap under subdivision 2 is compared to 2x target_bpm as the expected
    # stroke rate). Surface what subdivision(s) were actually used so a
    # caller (an LLM paraphrasing this to the user) doesn't present a
    # 160-bpm stroke rate as a deviation from an 80-bpm target without that
    # context.
    subdivisions_used = sorted(set(iv["subdivision"] for iv in intervals))

    if not reliable:
        return {
            "filepath": filepath,
            "target_bpm": round(target_bpm, 2),
            "bpm_was_estimated": bpm_was_estimated,
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
        "target_bpm": round(target_bpm, 2),
        "bpm_was_estimated": bpm_was_estimated,
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
def analyze_timing(filepath: str, bpm: float | None = None):
    """
    Analyze stroke timing accuracy in a mridangam (or any percussion)
    recording against a target tempo.

    Detects stroke onsets, estimates tempo if not given, and computes
    gap-by-gap deviation from the target tempo. Returns only the computed
    statistics (never the raw audio) plus an optional chart image.

    Args:
        filepath: Path to a local audio/video file to analyze.
        bpm: Target tempo in beats per minute. If omitted, it is
            auto-estimated from the recording.
    """
    summary = _build_analysis(filepath, bpm)

    # Transparency mechanism per music.md: stdin/stdout are the MCP
    # protocol channel, so an interactive confirmation prompt isn't
    # possible here. Instead, log the exact payload being returned to
    # stderr before returning it.
    print(
        f"[mcp_server] analyze_timing returning payload:\n"
        f"{json.dumps(summary, indent=2)}",
        file=sys.stderr,
    )

    if "error" in summary and "gaps" not in summary:
        return summary

    # Optional nice-to-have: also render and return the chart as an MCP
    # image content block. Chart is generated from the already-computed
    # `intervals`/stats, not from raw audio, and only its rendered PNG
    # bytes (not the waveform) are attached to the response.
    try:
        import os
        import tempfile

        song_name = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(tempfile.gettempdir(), f"{song_name}_timing_mcp.png")
        plot_timing(summary["gaps"], summary["target_bpm"], out_path)
        image = Image(path=out_path)
        print(f"[mcp_server] chart rendered to {out_path}", file=sys.stderr)
        return [summary, image]
    except Exception as exc:  # chart generation is a nice-to-have, not required
        print(f"[mcp_server] chart generation skipped: {exc}", file=sys.stderr)
        return summary


if __name__ == "__main__":
    mcp.run()
