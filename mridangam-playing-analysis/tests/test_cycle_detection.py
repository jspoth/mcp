"""
Phase 0 tests: cycle_detection.detect_cycle() against known-ground-truth
subdivision sequences. Purely data-oriented -- no onset_times, no audio,
no timing_analysis.py dependency at all, matching the module's own
isolation.
"""

import numpy as np
import pytest

import numpy as np

from cycle_detection import (
    detect_cycle,
    detect_sections,
    detect_cycles_by_section,
)


def test_single_cycle():
    pattern = [6, 6, 6, 2, 3, 4, 5]
    sequence = pattern * 5

    result = detect_cycle(sequence)

    assert result["cycle_length"] == 7
    assert result["pattern_confidence"] > 0.5


def test_multiple_sections_detected():
    section1 = [6, 6, 6, 2, 3, 4, 5] * 5
    section2 = [1, 2, 3, 4, 5] * 5
    section3 = [6, 6, 6, 2, 3, 4, 5] * 5

    sequence = section1 + section2 + section3

    result = detect_cycles_by_section(sequence)

    print(result)

    cycles = [
        s["cycle_length"]
        for s in result["sections"]
    ]

    assert cycles == [7, 5, 7]

def test_detect_sections():
    section1 = [6, 6, 6, 2, 3, 4, 5] * 5
    section2 = [1, 2, 3, 4, 5] * 5
    section3 = [6, 6, 6, 2, 3, 4, 5] * 5

    sequence = section1 + section2 + section3

    sections = detect_sections(sequence)

    print("\nSECTIONS:")
    for s in sections:
        print(
            s["start"],
            s["end"],
            s["sequence"]
        )