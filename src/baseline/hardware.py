"""Hardware string recorded next to the baseline timing so the FPS can be reported honestly."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import cv2

CPUINFO = Path("/proc/cpuinfo")


def _sysctl_brand() -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _cpuinfo_model() -> str:
    try:
        if CPUINFO.is_file():
            for line in CPUINFO.read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def cpu_model() -> str:
    if sys.platform == "darwin":
        brand = _sysctl_brand()
        if brand:
            return brand
    return _cpuinfo_model() or platform.processor() or platform.machine()


def hardware_fields() -> dict[str, str | int]:
    return {
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count() or 0,
        "cv2_version": cv2.__version__,
        "cv2_threads": cv2.getNumThreads(),
    }
