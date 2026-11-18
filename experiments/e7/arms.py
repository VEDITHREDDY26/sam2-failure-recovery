"""The four E7 arms: (rule detector | oracle flags) x (CSRT box | oracle box), Tiny."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Arm:
    """One E7 arm: which detector and which box source the run uses.

    Attributes:
        name: Short arm name used in tables.
        method: Method name of the run; contains "oracle" when ground truth is used.
        detector: "rules" or "oracle" (oracle flags).
        box_source: "csrt" or "oracle" (oracle box).
    """

    name: str
    method: str
    detector: str
    box_source: str

    @property
    def oracle(self) -> bool:
        """True if the arm reads ground truth, through its detector or its box source."""
        return "oracle" in (self.detector, self.box_source)


ARMS = (
    Arm("rules_csrt", "fix_rules_csrt_tiny", "rules", "csrt"),
    Arm("rules_oracle_box", "fix_rules_oracle_box_tiny", "rules", "oracle"),
    Arm("oracle_flags_csrt", "fix_oracle_flags_csrt_tiny", "oracle", "csrt"),
    Arm("oracle_flags_oracle_box", "fix_oracle_flags_oracle_box_tiny", "oracle", "oracle"),
)
ALONE = "sam2_tiny"
TEST_SETS = ("davis_val", "sav_test")
