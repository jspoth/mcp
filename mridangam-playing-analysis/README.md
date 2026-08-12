# mridangam-playing-analysis

An MCP server (and standalone CLI) for analyzing the **timing/tempo accuracy**
of a percussion practice recording — built around mridangam and konnakol
practice, but the underlying method (onset detection + tempo-gap analysis)
works for any single-line percussive recording.

It answers "which beats weren't at tempo," not "which stroke was played."
No classifier, no training data, no machine learning — just signal
processing (onset detection) and arithmetic.

## What it does

1. Detects likely stroke onsets in a recording (onset detection via
   [`librosa`](https://librosa.org)).
2. For each pair of consecutive strokes, computes the actual instantaneous
   tempo (60 / gap-in-seconds) and compares it to your target tempo.
3. Auto-detects subdivision (e.g. two strokes per beat, as in konnakol-style
   subdivided playing) directly from the gap pattern — no manual
   configuration needed, and it adapts if the subdivision changes partway
   through a recording.
4. Reports which specific gaps were out of tempo, by how much, and overall
   consistency stats — as text, and optionally a chart.

## Why no fixed grid?

An earlier version of this anchored a fixed beat grid to the first detected
onset and measured drift from it. That compounds error over a long
recording and stops being locally meaningful — being consistently 10ms
fast for five minutes looks like a large "deviation" by the end, even
though the actual playing was solid throughout. This tool instead compares
each gap only to its immediate neighbors, so feedback stays locally
accurate regardless of recording length.

## Two ways to use it

### As a standalone CLI (no AI/LLM involved)

```bash
python timing_analysis.py path/to/recording.wav --bpm 80
python timing_analysis.py path/to/recording.wav            # auto-estimates tempo
```

Prints a per-gap report and saves a chart PNG.

### As an MCP server (for use with Claude Desktop, Claude Code, or any
MCP-compatible client)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then register it with your MCP client — see the config snippet and setup
instructions in the header comment of `mcp_server.py`.

Once registered, just ask your LLM client something like *"analyze the
timing of my practice recording at `/path/to/recording.wav`, target tempo
80bpm"* — it calls the `analyze_timing` tool, which returns the stats (and
optionally a chart image) for the model to discuss with you.

## Privacy / security design

- The MCP tool takes a **filepath**, not audio data. It reads the file
  locally and only ever returns **computed statistics** (and, optionally, a
  separately-rendered chart image) — never the raw audio bytes, waveform,
  or anything derived directly from the sample array.
- The server never touches `stdin`/`stdout` with its own output — those are
  reserved for the MCP protocol channel. Instead, it logs the exact payload
  it's about to return to `stderr` before sending it, so what's being
  shared with the LLM is inspectable.
- Nothing is uploaded, stored, or sent anywhere beyond your own chosen LLM
  client and provider.

## Files

- `timing_analysis.py` — the core analysis engine (onset detection, tempo
  comparison, subdivision detection, reporting, charting). Usable standalone
  or imported by other tools.
- `mcp_server.py` — a thin MCP wrapper around `timing_analysis.py`, adding
  no new analysis logic of its own.
- `requirements.txt` — Python dependencies.

## License

MIT — see [LICENSE](LICENSE).
