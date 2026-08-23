"""K: this-campaign world model (hypotheses, evidence, open falsifiers).

The graph is a kernel corpus object and is not part of K.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .judge import Hypothesis
from .ledger import content_digest


@dataclass(frozen=True)
class WorldModel:
    hypotheses: tuple[Hypothesis, ...]
    evidence_ids: tuple[str, ...]
    killed: tuple[str, ...]
    focus: str
    open_falsifiers: tuple[dict[str, Any], ...]
    pre_judgment: bool
    spans: tuple[dict[str, Any], ...] = ()

    def digest(self) -> str:
        return content_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypotheses": [
                {
                    "id": hyp.id,
                    "statement": hyp.statement,
                    "zone": hyp.zone,
                    "cheapest_falsifier": hyp.cheapest_falsifier,
                    "falsifier_cost": hyp.falsifier_cost,
                    "is_core": hyp.is_core,
                }
                for hyp in self.hypotheses
            ],
            "evidence_ids": list(self.evidence_ids),
            "killed": list(self.killed),
            "focus": self.focus,
            "open_falsifiers": [dict(item) for item in self.open_falsifiers],
            "pre_judgment": self.pre_judgment,
            "spans": [dict(item) for item in self.spans],
        }

    @classmethod
    def from_judge(
        cls,
        hypotheses: Sequence[Hypothesis],
        judge: Any,
        evidence_ids: Sequence[str],
        open_falsifiers: Sequence[dict[str, Any]],
        spans: Sequence[Any] = (),
    ) -> "WorldModel":
        return cls(
            hypotheses=tuple(hypotheses),
            evidence_ids=tuple(evidence_ids),
            killed=tuple(sorted(judge.killed)),
            focus=str(judge.focus),
            open_falsifiers=tuple(dict(item) for item in open_falsifiers),
            pre_judgment=bool(judge.pre_judgment()),
            spans=tuple(
                {
                    "source_id": span.source_id,
                    "start": span.start,
                    "end": span.end,
                    "title": span.title,
                }
                for span in spans
            ),
        )
