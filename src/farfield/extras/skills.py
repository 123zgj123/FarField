"""Admitted research skills: SKILL.md for the model, plugin.py for the host.

Argus compiled runtime H = {Memory, Skills, Tools, Verifiers, Routing}
without training weights. FarField does the same *inside a mission*:
after each landing (including evidence) H is recompiled so the next
refine or generate is a better idea. Skills stay trusted plugins.

Causal force lives in `plugins.PluginHost`. A directory with only SKILL.md
is prompt colour. A trusted `plugin.py` next to it is a hook the host
calls. Distillation writes SKILL.md only — never plugin.py.

A skill is not evidence. It cannot kill, revive, rewrite expected_direction,
or climb the promotion ladder. A candidate whose body claims to override a
gate is refused, not installed.

Interchange:

- FarField + DeepSeek Harness: `.agents/skills/<name>/SKILL.md`
  (dsh-skill-filesystem rank 200)
- Durable catalog across missions: `var/skills/` (gitignored)
- Cursor skills stay in `.cursor/skills` and are not injected into research
  generation prompts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .domain import claim_covers_topic, folded_terms

STAGES = ("generate", "diagnose", "probe", "writeup", "intern")
BODY_CAP = 1200
SELECT_CAP = 4

# A skill that tells the model to skip the evaluation function is not a skill.
_FORBIDDEN = (
    "override the graph",
    "skip the probe",
    "bypass the ladder",
    "hard-code treatment",
    "hard-code control",
    "change expected_direction",
    "ignore the falsifier",
)


class SkillError(ValueError):
    """The file is not an admissible skill."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    source: str
    digest: str
    stages: tuple[str, ...]
    admitted: bool
    audience: str = "research"
    entry: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
            "digest": self.digest,
            "stages": list(self.stages),
            "admitted": self.admitted,
            "audience": self.audience,
            "entry": self.entry,
        }


def skill_digest(name: str, description: str, body: str) -> str:
    payload = json.dumps(
        {"name": name, "description": description, "body": body},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "skill")[:64]


def parse_skill_md(text: str, *, source: str = "") -> Skill:
    raw = text.strip()
    if not raw.startswith("---"):
        raise SkillError("SKILL.md must start with YAML frontmatter")
    rest = raw[3:]
    fence = rest.find("\n---")
    if fence < 0:
        raise SkillError("SKILL.md frontmatter is not closed")
    front, body = rest[:fence], rest[fence + 4 :].strip()
    meta: dict[str, str] = {}
    for line in front.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip().lower()] = value.strip().strip("\"'")
    name = _slug(meta.get("name") or Path(source).parent.name or "skill")
    description = (meta.get("description") or "").strip()
    if not description:
        raise SkillError(f"{name}: description is required")
    stage_raw = meta.get("stage") or meta.get("stages") or "generate"
    stages = tuple(
        item for item in (part.strip() for part in stage_raw.split(",")) if item in STAGES
    ) or ("generate",)
    admitted = meta.get("admitted", "true").lower() != "false"
    audience = (meta.get("audience") or meta.get("runtime") or "research").strip().lower()
    if audience not in {"research", "cursor", "all"}:
        audience = "research"
    entry = (meta.get("entry") or "").strip()
    lowered = body.lower()
    for banned in _FORBIDDEN:
        if banned in lowered:
            raise SkillError(f"{name}: skill body claims to {banned}")
    return Skill(
        name=name,
        description=description,
        body=body,
        source=source,
        digest=skill_digest(name, description, body),
        stages=stages,
        admitted=admitted,
        audience=audience,
        entry=entry,
    )


def render_skill_md(skill: Skill) -> str:
    admitted = "true" if skill.admitted else "false"
    stages = ", ".join(skill.stages)
    lines = [
        "---",
        f"name: {skill.name}",
        f"description: {skill.description}",
        f"stage: {stages}",
        f"admitted: {admitted}",
        f"audience: {skill.audience}",
    ]
    if skill.entry:
        lines.append(f"entry: {skill.entry}")
    lines.extend(["---", "", skill.body.rstrip(), ""])
    return "\n".join(lines)


def load_skill_file(path: Path) -> Skill:
    return parse_skill_md(path.read_text(encoding="utf-8"), source=str(path))


def load_skill_tree(root: Path) -> tuple[Skill, ...]:
    """Every SKILL.md under root, one level of skill directories."""
    root = Path(root)
    if not root.is_dir():
        return ()
    found: list[Skill] = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        try:
            found.append(load_skill_file(skill_md))
        except (SkillError, OSError):
            continue
    return tuple(found)


def default_skill_roots(
    repo: Path,
    workspace: Path | None = None,
    extra: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    """Same order dsh-skill-filesystem uses. Cursor skills are not research plugins."""
    repo = Path(repo)
    roots = [
        repo / ".dsh" / "skills",
        repo / ".agents" / "skills",
        repo / "experiments" / ".agents" / "skills",
    ]
    if workspace is not None:
        ws = Path(workspace)
        roots.extend(
            [
                ws / ".agents" / "skills",
                ws / ".dsh" / "skills",
            ]
        )
    roots.extend(Path(item) for item in extra)
    return tuple(roots)


def load_catalog(
    repo: Path,
    workspace: Path | None = None,
    extra: tuple[Path, ...] = (),
) -> tuple[Skill, ...]:
    """Deduplicate by digest; unadmitted files stay out of the prompt."""
    seen: set[str] = set()
    catalog: list[Skill] = []
    for root in default_skill_roots(repo, workspace, extra):
        for skill in load_skill_tree(root):
            if skill.digest in seen or not skill.admitted:
                continue
            if skill.audience == "cursor":
                continue
            seen.add(skill.digest)
            catalog.append(skill)
    return tuple(catalog)


def select_skills(
    catalog: Iterable[Skill],
    *,
    topic: str,
    stage: str,
    extra: str = "",
    limit: int = SELECT_CAP,
) -> tuple[Skill, ...]:
    """Skills whose description or body overlaps the topic, for this stage.

    No overlap still returns skills tagged for the stage whose description
    names a general research procedure (empty topic → nothing, so tests that
    do not pass a topic do not inject).
    """
    if not topic.strip():
        return ()
    wanted = folded_terms(topic) | folded_terms(extra)
    ranked: list[tuple[int, str, Skill]] = []
    for skill in catalog:
        if stage not in skill.stages:
            continue
        hay = folded_terms(skill.description) | folded_terms(skill.body[:BODY_CAP])
        score = len(wanted & hay)
        if score <= 0 and not _is_general(skill):
            continue
        ranked.append((-score if score else 0, skill.name, skill))
    ranked.sort()
    picked = []
    seen = set()
    for _, _, skill in ranked:
        if skill.name in seen:
            continue
        seen.add(skill.name)
        picked.append(skill)
        if len(picked) >= limit:
            break
    if not picked:
        picked = [
            skill
            for skill in catalog
            if stage in skill.stages and _is_general(skill)
        ][:limit]
    return tuple(picked)


def _is_general(skill: Skill) -> bool:
    blob = f"{skill.name} {skill.description}".lower()
    return any(
        marker in blob
        for marker in (
            "two-arm",
            "domain lock",
            "claim-domain",
            "fair probe",
            "farfield",
            "object-domain",
        )
    )


def skills_prompt_block(skills: Iterable[Skill]) -> str:
    """Empty when nothing matches, so cached prompts keep their digest."""
    items = list(skills)
    if not items:
        return ""
    lines = [
        "Admitted skills from earlier work or the installed catalog. PluginHost"
        " already runs their hooks; the text below is procedure, not a license"
        " to skip a hook, override a gate, rewrite probe arithmetic, or climb"
        " the promotion ladder:\n"
    ]
    for skill in items:
        excerpt = skill.body.strip().replace("\n", " ")
        if len(excerpt) > 400:
            excerpt = excerpt[:397] + "..."
        lines.append(f"- {skill.name}: {skill.description}")
        lines.append(f"  {excerpt}")
    return "\n".join(lines) + "\n\n"


def distill_skill(
    *,
    topic: str,
    seed_label: str,
    card: Any,
    diagnosis: Any,
    probe: dict[str, Any],
    brief: Any | None = None,
) -> Skill | None:
    """Mint a skill from a supporting run. Numbers come from the probe, not the model.

    Returns None when the evaluation function would not admit the result:
    non-support, topic drift, or missing arms. Distillation is not RSI and
    does not change weights.
    """
    if str(probe.get("verdict") or "") != "supports":
        return None
    if probe.get("treatment") is None or probe.get("control") is None:
        return None
    if not claim_covers_topic(card.claim, card.mechanism, topic, seed_label):
        return None
    pair = getattr(card, "pair", ("", ""))
    name = _slug(f"{pair[0]}-x-{pair[1]}")
    title = getattr(brief, "title", "") if brief is not None else ""
    idea = getattr(brief, "idea", "") if brief is not None else ""
    description = (
        f"Reusable procedure from a supporting two-arm probe of {pair[0]} ×"
        f" {pair[1]}. Use when researching {topic}."
    )[:1024]
    lines = [
        f"# {name}",
        "",
        "## When",
        f"Topic: {topic}",
        f"Claim: {card.claim}",
        f"Mechanism: {card.mechanism}",
    ]
    if title:
        lines.append(f"Title: {title}")
    lines.extend(
        [
            "",
            "## Procedure",
            str(getattr(diagnosis, "experiment", "") or ""),
            f"Treatment arm: {getattr(diagnosis, 'treatment_arm', '')}",
            f"Control arm: {getattr(diagnosis, 'control_arm', '')}",
            f"Measure: {probe.get('measure') or ''}",
            "",
            "## Observed (from the run; do not hard-code next time)",
            f"treatment={probe.get('treatment')} control={probe.get('control')}",
            f"expected_direction={getattr(diagnosis, 'expected_direction', '')}",
        ]
    )
    if idea:
        lines.extend(["", "## Next steps", idea])
    lines.extend(
        [
            "",
            "## Verifier",
            "Rerun a symmetric two-arm probe. Both arms call the same measure;"
            " only a mechanism flag differs. Do not paste the numbers above into"
            " experiment.py. Do not retitle the claim into a different field.",
            "",
        ]
    )
    body = "\n".join(lines)
    skill = Skill(
        name=name,
        description=description,
        body=body.strip() + "\n",
        source="distill",
        digest=skill_digest(name, description, body.strip() + "\n"),
        stages=("generate", "diagnose", "probe"),
        admitted=True,
        audience="research",
        entry="",
    )
    lowered = skill.body.lower()
    if any(banned in lowered for banned in _FORBIDDEN):
        return None
    return skill


def write_skill(dest_dir: Path, skill: Skill) -> Path:
    dest = Path(dest_dir) / skill.name
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "SKILL.md"
    path.write_text(render_skill_md(skill), encoding="utf-8")
    return path
