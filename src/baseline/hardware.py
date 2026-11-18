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
    """Return the macOS CPU brand string from sysctl, or '' if it cannot be read."""
    try:
        # Never fail the run because the CPU name cannot be read.
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
    """Return the CPU model name from /proc/cpuinfo on Linux, or '' if unavailable."""
    try:
        if CPUINFO.is_file():
            for line in CPUINFO.read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def cpu_model() -> str:
    """Return a human-readable CPU model string for the machine running the baseline.

    Tries sysctl on macOS, then /proc/cpuinfo, then the platform module's processor and machine
    names, so the string is never empty on a normal system.
    """
    # macOS reports the CPU through sysctl; other systems fall back to /proc/cpuinfo, then platform.
    if sys.platform == "darwin":
        brand = _sysctl_brand()
        if brand:
            return brand
    return _cpuinfo_model() or platform.processor() or platform.machine()


def hardware_fields() -> dict[str, str | int]:
    """Return the hardware and OpenCV fields stored in each timing row.

    Returns:
        A dict with cpu_model, cpu_count (0 if unknown), cv2_version and cv2_threads (the number of
        threads OpenCV will use), so FPS numbers can be interpreted against the machine.
    """
    return {
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count() or 0,
        "cv2_version": cv2.__version__,
        "cv2_threads": cv2.getNumThreads(),
    }
