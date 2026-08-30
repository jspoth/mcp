"""
Progress tracking across sessions: append one record per analyze_timing()
call (date/time, filename, and duplicate-upload detection) to a local
append-only log, and let that history be queried back per filename.

Deliberately a flat JSONL file rather than a database -- one process, no
concurrent writers to coordinate, and "read everything, filter in Python" is
plenty fast for how many practice sessions a single user logs.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

DEFAULT_LOG_PATH = os.path.join(os.path.dirname(__file__), "progress_log.jsonl")


def _file_hash(filepath, chunk_size=1 << 20):
    """SHA-256 of the file's bytes -- used to recognize the same recording
    being analyzed again (possibly re-uploaded under a different filename)
    as a re-run rather than a new practice session, and vice versa: a
    reused filename with different content is NOT treated as a duplicate."""
    digest = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_all(log_path):
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_progress(filepath, result, log_path=DEFAULT_LOG_PATH):
    """
    Append one record for this analysis run. `result` is analyze_timing()'s
    return dict; only the fields relevant to tracking progress over time are
    kept (not the full per-gap detail, which is already logged separately
    by mcp_server.py's _log() for a single run).

    If this exact file's content (by hash, not filename) already appears
    earlier in the log, `duplicate_of` is set to that earlier record's
    timestamp -- callers should surface this rather than silently counting
    a re-upload of the same recording as a new practice session.
    """
    file_hash = _file_hash(filepath)
    previous = next(
        (r for r in _read_all(log_path) if r.get("file_hash") == file_hash),
        None,
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": os.path.basename(filepath),
        "file_hash": file_hash,
        "analysis_status": result.get("analysis_status"),
        "target_bpm": result.get("target_bpm"),
        "in_tempo_pct": result.get("in_tempo_pct"),
        "average_deviation_pct": result.get("average_deviation_pct"),
        "consistency_std_dev_pct": result.get("consistency_std_dev_pct"),
        "duplicate_of": previous["timestamp"] if previous else None,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def get_progress_history(filename=None, log_path=DEFAULT_LOG_PATH):
    """
    Return recorded sessions, oldest first, optionally filtered to one
    filename (basename match, so callers don't need to pass the full path
    they originally analyzed). Returns [] if nothing has been logged yet.
    """
    records = _read_all(log_path)
    if filename is None:
        return records
    return [r for r in records if r["filename"] == os.path.basename(filename)]
