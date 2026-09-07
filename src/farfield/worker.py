"""The toy keep-if-better campaign was removed. Product path is `farfield research`.

Kernel modules (graph oracle, predicates, ledger) still serve the mission.
This module exists so leftover experiment scripts fail with a named reason
instead of silently inventing a second scientist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CampaignError(RuntimeError):
    """The control-arm campaign is gone."""


def run_campaign(*_args: Any, **_kwargs: Any) -> None:
    raise CampaignError(
        "the control-arm campaign and Wave A–F ledger extras (dossier, F0"
        " install, Wave E audit) were removed; the product path is"
        " `farfield research`"
    )


def load_dossier(_project: Path | str) -> None:
    raise CampaignError(
        "Wave A dossiers were removed with the control arm; see mission/summary.md"
        " from `farfield research`"
    )
