"""Tests for drawing the verification subset: seeded, sorted, order independent, size checks
and the committed 60-video file.
"""

from pathlib import Path

import pytest

from experiments.verify import draw_subset

NAMES = [f"sav_{i:06d}" for i in range(155)]


def test_draw_is_seeded_sorted_and_independent_of_input_order() -> None:
    first = draw_subset.draw(NAMES)
    assert first == draw_subset.draw(list(reversed(NAMES)))
    assert first == sorted(first) and len(set(first)) == 60 and set(first) <= set(NAMES)
    assert first != draw_subset.draw(NAMES, seed=1)


def test_draw_refuses_bad_sizes_and_duplicates() -> None:
    with pytest.raises(ValueError):
        draw_subset.draw(NAMES, size=156)
    with pytest.raises(ValueError):
        draw_subset.draw(NAMES + NAMES[:1])


def test_committed_subset_is_60_distinct_names_of_the_val_list() -> None:
    subset = draw_subset.load_subset()
    assert len(subset) == 60 and subset == sorted(subset)
    assert all(name.startswith("sav_") for name in subset)


def test_main_writes_the_file(tmp_path: Path) -> None:
    names = tmp_path / "names.txt"
    names.write_text("\n".join(NAMES) + "\n")
    out = tmp_path / "subset.txt"
    draw_subset.main(["--names-file", str(names), "--out", str(out)])
    assert draw_subset.load_subset(out) == draw_subset.draw(NAMES)
