"""Hardware string recorded next to the baseline timing so the FPS can be reported honestly."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import cv2

CPUINFO = Path("/proc/cpuinfo")


def cpu_model() -> str:
    if CPUINFO.is_file():
        for line in CPUINFO.read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def hardware_fields() -> dict[str, str | int]:
    return {
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count() or 0,
        "cv2_version": cv2.__version__,
        "cv2_threads": cv2.getNumThreads(),
    }
