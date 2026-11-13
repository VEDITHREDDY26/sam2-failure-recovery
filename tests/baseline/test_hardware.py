"""CPU model lookup, independent of the machine running the tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.baseline import hardware


def _fake_run(stdout: str, code: int = 0):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, code, stdout=stdout, stderr="")

    return run


@pytest.fixture
def no_cpuinfo(monkeypatch, tmp_path):
    monkeypatch.setattr(hardware, "CPUINFO", tmp_path / "missing")
    monkeypatch.setattr(hardware.platform, "processor", lambda: "fallback-cpu")


def test_darwin_uses_sysctl_brand(monkeypatch, no_cpuinfo):
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.subprocess, "run", _fake_run("Apple M4\n"))
    assert hardware.cpu_model() == "Apple M4"


@pytest.mark.parametrize("failure", ["missing", "timeout", "nonzero", "empty"])
def test_darwin_sysctl_failure_falls_back(monkeypatch, no_cpuinfo, failure):
    def raise_missing(*a, **k):
        raise FileNotFoundError

    def raise_timeout(*a, **k):
        raise subprocess.TimeoutExpired("sysctl", 5)

    runs = {
        "missing": raise_missing,
        "timeout": raise_timeout,
        "nonzero": _fake_run("junk", 1),
        "empty": _fake_run("  \n"),
    }
    monkeypatch.setattr(hardware.sys, "platform", "darwin")
    monkeypatch.setattr(hardware.subprocess, "run", runs[failure])
    assert hardware.cpu_model() == "fallback-cpu"


def test_linux_reads_model_name(monkeypatch, tmp_path: Path):
    info = tmp_path / "cpuinfo"
    info.write_text("processor\t: 0\nmodel name\t: Intel Xeon @ 2.20GHz\nflags\t: x\n")
    monkeypatch.setattr(hardware, "CPUINFO", info)
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    assert hardware.cpu_model() == "Intel Xeon @ 2.20GHz"
