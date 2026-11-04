"""Contract between the trigger and the model-based impact-analysis / repair layers.

The trigger only CALLS request_analysis(); deciding whether a repair is needed, choosing it,
validating it and applying it all happen on the other side. In the toy simulator (env.py) the
simulator implements this contract; later the DWAFP C++ layers (impact analysis, Stage I repair,
validation) would implement it through a thin adapter.
"""

from dataclasses import dataclass
from typing import Protocol

from .state import RuntimeSnapshot


@dataclass(frozen=True)
class AnalysisOutcome:
    """What the trigger learns from one analysis request."""

    threatened: bool  # the impact analysis found a deadline that will be missed
    repaired: bool    # a validated repair was applied (the task is then no longer monitored)


class AnalysisRepairInterface(Protocol):
    def request_analysis(self, snapshot: RuntimeSnapshot, task_id: str) -> AnalysisOutcome:
        """Runs impact analysis (and repair if needed) for one executing task."""
        ...
