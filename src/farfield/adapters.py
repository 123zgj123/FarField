"""Provider-neutral rendering helpers.

These helpers receive a contract already verified by ``FarfieldRuntime``. They do
not read mutable paths or call any external agent framework.
"""

from __future__ import annotations

from typing import Any


def render_bounded_agent_handoff(contract: dict[str, Any]) -> str:
    evidence = "\n".join(
        f"- [{item['id']}] {item['description']}"
        for item in contract["completion_clauses"]
    )
    constraints = "\n".join(f"- {item}" for item in contract["constraints"])
    budget = "\n".join(
        f"- {key}: {value}" for key, value in sorted(contract["budget"].items())
    )
    return (
        f"Mission: {contract['goal']}\n\n"
        f"Completion clauses:\n{evidence}\n\n"
        f"Constraints:\n{constraints or '- none'}\n\n"
        f"Budget:\n{budget}\n\n"
        "Authority boundary: optimize only the execution plan inside the Farfield "
        "project workspace. If evidence challenges the objective, return an "
        "OBJECTIVE_CHALLENGE with artifact references; do not redefine success.\n"
    )


# Backward-friendly name for users who choose Codex as one provider. Farfield has
# no dependency on Codex and does not execute /goal through this function.
render_codex_goal_handoff = render_bounded_agent_handoff
