"""
progress_store.py: record_progress()/get_progress_history() are a flat
JSONL append + filtered read, exercised here against a tmp_path log file so
tests never touch the real progress_log.jsonl.

record_progress() hashes the file's actual bytes (to recognize a re-upload
of the same recording under a different filename as a duplicate, and a
reused filename with different content as NOT a duplicate) so tests write
real temp files rather than passing fake paths.
"""

from progress_store import record_progress, get_progress_history


def _audio_file(tmp_path, name, content=b"fake-audio-bytes"):
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


def test_record_progress_writes_relevant_fields(tmp_path):
    log_path = tmp_path / "progress_log.jsonl"
    audio = _audio_file(tmp_path, "part1_audio.m4a")
    result = {
        "analysis_status": "success",
        "target_bpm": 99.38,
        "in_tempo_pct": 82.5,
        "average_deviation_pct": -1.2,
        "consistency_std_dev_pct": 4.7,
    }

    record = record_progress(audio, result, log_path=str(log_path))

    assert record["filename"] == "part1_audio.m4a"
    assert record["analysis_status"] == "success"
    assert record["in_tempo_pct"] == 82.5
    assert record["duplicate_of"] is None
    assert "timestamp" in record
    assert "file_hash" in record


def test_get_progress_history_returns_empty_when_no_log(tmp_path):
    log_path = tmp_path / "does_not_exist.jsonl"
    assert get_progress_history(log_path=str(log_path)) == []


def test_get_progress_history_filters_by_filename(tmp_path):
    log_path = tmp_path / "progress_log.jsonl"

    audio1 = _audio_file(tmp_path, "part1_audio.m4a", b"content-a")
    audio2 = _audio_file(tmp_path, "part2_audio.m4a", b"content-b")

    record_progress(audio1, {"analysis_status": "success"}, log_path=str(log_path))
    record_progress(audio2, {"analysis_status": "success"}, log_path=str(log_path))
    record_progress(audio1, {"analysis_status": "low_confidence"}, log_path=str(log_path))

    all_records = get_progress_history(log_path=str(log_path))
    assert len(all_records) == 3

    part1_only = get_progress_history(filename="part1_audio.m4a", log_path=str(log_path))
    assert len(part1_only) == 2
    assert all(r["filename"] == "part1_audio.m4a" for r in part1_only)
    assert part1_only[1]["analysis_status"] == "low_confidence"


def test_get_progress_history_preserves_append_order(tmp_path):
    log_path = tmp_path / "progress_log.jsonl"
    audio = _audio_file(tmp_path, "a.m4a")

    record_progress(audio, {"analysis_status": "success", "consistency_std_dev_pct": 8.0},
                     log_path=str(log_path))
    record_progress(audio, {"analysis_status": "success", "consistency_std_dev_pct": 4.0},
                     log_path=str(log_path))

    records = get_progress_history(filename="a.m4a", log_path=str(log_path))
    assert [r["consistency_std_dev_pct"] for r in records] == [8.0, 4.0]


# ---------------------------------------------------------------------------
# Duplicate-upload detection: keyed on file CONTENT (hash), not filename --
# a re-upload under a new name is still a duplicate, and a reused filename
# with different content is NOT one.
# ---------------------------------------------------------------------------

def test_same_content_different_filename_is_flagged_duplicate(tmp_path):
    log_path = tmp_path / "progress_log.jsonl"
    original = _audio_file(tmp_path, "session1.m4a", b"identical-bytes")
    reupload = _audio_file(tmp_path, "session1_copy.m4a", b"identical-bytes")

    first = record_progress(original, {"analysis_status": "success"}, log_path=str(log_path))
    second = record_progress(reupload, {"analysis_status": "success"}, log_path=str(log_path))

    assert first["duplicate_of"] is None
    assert second["duplicate_of"] == first["timestamp"]


def test_same_filename_different_content_is_not_duplicate(tmp_path):
    log_path = tmp_path / "progress_log.jsonl"
    take1 = _audio_file(tmp_path, "practice.m4a", b"take-one-bytes")

    record_progress(take1, {"analysis_status": "success"}, log_path=str(log_path))

    # Overwrite the same filename with genuinely different content --
    # simulates a new recording saved over the same name.
    with open(take1, "wb") as f:
        f.write(b"take-two-different-bytes")

    second = record_progress(take1, {"analysis_status": "success"}, log_path=str(log_path))
    assert second["duplicate_of"] is None
