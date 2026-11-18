"""Check a verified run against the cached SAM 2 alone masks.

    python -m experiments.verify.pilot_check \
        --config configs/recovery/verified_rules_csrt_tiny.yaml \
        --prompts prompts/davis_val.json [--videos a b] [--out results/verify_pilot_check.csv]

Every video whose candidates were all rejected must equal the cached sam2_<model> label maps on
every frame (a rejection restarts SAM 2 and must leave the un-prompted output); every other
video must equal them on all frames before its first accepted re-prompt. Also checks that the
candidate log has one row per re-prompt or rejection event. Writes one verdict row per video;
exit code 1 if any video fails.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.e2.pilot_check import VideoCheck, check_video, read_events
from experiments.e2.run_online import model_of
from src.common.config import DATA_ROOT_ENV
from src.recovery.config import load_recovery_config
from src.recovery.online_runner import frame_paths
from src.recovery.verify import candidate_path
from src.sam2_runner.prompts import load_batch_prompts
from src.sam2_runner.run import frames_dir_for
from src.sam2_runner.smoke import load_data_config

CHECK_COLUMNS = (
    "video",
    "frames",
    "accepted",
    "rejected",
    "all_rejected",
    "compared_frames",
    "candidate_rows",
    "verdict",
    "first_diff",
)
DEFAULT_OUT = Path("results/verify_pilot_check.csv")


@dataclass(frozen=True)
class VerifiedCheck:
    """Verdict for one video of a verified run.

    Attributes:
        video: Video name.
        frames: Number of frames in the video.
        accepted: Accepted re-prompts (events with action reprompted).
        rejected: Candidates the rule rejected (events with action verify_rejected).
        compared: Number of frames compared with the SAM 2 alone masks.
        candidate_rows: Rows in the candidate log, or -1 when the log is missing.
        verdict: "identical" when everything matches, otherwise the reason of the failure.
        first_diff: Frame index of the first difference, or None.
    """

    video: str
    frames: int
    accepted: int
    rejected: int
    compared: int
    candidate_rows: int
    verdict: str
    first_diff: int | None

    @property
    def all_rejected(self) -> bool:
        """True when every candidate was rejected and none accepted."""
        return self.accepted == 0 and self.rejected > 0

    @property
    def ok(self) -> bool:
        """True when the video passed the check."""
        return self.verdict == "identical"

    def as_row(self) -> tuple[object, ...]:
        """The values as a row in the order of CHECK_COLUMNS."""
        diff = "" if self.first_diff is None else self.first_diff
        return (
            self.video,
            self.frames,
            self.accepted,
            self.rejected,
            self.all_rejected,
            self.compared,
            self.candidate_rows,
            self.verdict,
            diff,
        )


def verified_check(
    output_root: Path, method: str, reference: str, split: str, video: str, stems: Sequence[str]
) -> VerifiedCheck:
    """Check one video of a verified run against the SAM 2 alone masks.

    Args:
        output_root: Folder holding the saved masks and logs.
        method: The verified method.
        reference: SAM 2 alone method used as the reference.
        split: Test set name.
        video: Video name.
        stems: Frame file stems of the video.

    Returns:
        The verdict. A mask comparison that passes still fails with candidate_log_mismatch
        when the candidate log does not have one row per accepted or rejected candidate.
    """
    events = read_events(output_root, method, split, video)
    accepted = int((events["action"] == "reprompted").sum())
    rejected = int((events["action"] == "verify_rejected").sum())
    # Frame-by-frame comparison with SAM 2 alone from the E2 pilot check.
    base: VideoCheck = check_video(output_root, method, reference, split, video, stems)
    # The candidate log must have one row per accepted or rejected candidate.
    log = candidate_path(output_root, method, split, video)
    rows = len(pd.read_csv(log)) if log.is_file() else -1
    verdict = base.verdict
    if verdict == "identical" and rows != accepted + rejected:
        verdict = "candidate_log_mismatch"
    return VerifiedCheck(
        video, base.frames, accepted, rejected, base.compared, rows, verdict, base.first_diff
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point: check verified runs against SAM 2 alone.

    Reads the config, the prompts file, the saved masks, events and candidate logs (data root
    from the environment). Writes one verdict row per video to --out and prints one line per
    video.

    Args:
        argv: Command line arguments; None means sys.argv.

    Returns:
        0 if every video passed, 1 if any failed. A missing data root variable ends with the
        usage error code 2.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not os.environ.get(DATA_ROOT_ENV):
        parser.error(f"{DATA_ROOT_ENV} is not set")
    # The reference is the cached SAM 2 alone run of the same model.
    run_cfg = load_recovery_config(args.config)
    reference = f"sam2_{model_of(run_cfg)}"
    data = load_data_config()
    prompts = load_batch_prompts(args.prompts)
    names = list(args.videos) if args.videos else list(prompts.videos)
    output_root = data.data_root / data.output_root
    # Check each video; frame stems give the frame order.
    checks = [
        verified_check(
            output_root,
            run_cfg.method,
            reference,
            prompts.split,
            video,
            [p.stem for p in frame_paths(frames_dir_for(data, prompts.split, video))],
        )
        for video in names
    ]
    # Write the verdict table, print one line per video, then set the exit code.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([c.as_row() for c in checks], columns=list(CHECK_COLUMNS)).to_csv(
        args.out, index=False
    )
    for c in checks:
        kind = "all candidates rejected" if c.all_rejected else f"{c.accepted} accepted"
        where = "" if c.ok else f" at frame {c.first_diff}"
        print(f"{c.video}: {c.verdict}{where} ({c.compared} frames compared, {kind})")
    print(f"verdicts -> {args.out}")
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
