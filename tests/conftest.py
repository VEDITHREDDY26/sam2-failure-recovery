from pathlib import Path

import pytest

from tests.fixtures.synthetic_video import SyntheticVideo, make_synthetic_video


@pytest.fixture
def synthetic_video(tmp_path: Path) -> SyntheticVideo:
    return make_synthetic_video(tmp_path)
