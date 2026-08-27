"""Cheap diagnosis before any experiment: what would change our mind?

The old probe asked "show the idea working" — an experiment the idea's
author picks is the experiment the idea wins. This stage asks the
scientific question instead: *what is the strongest competing explanation,
and what is the smallest experiment whose outcome separates the two?*

The model writes the diagnosis; the discipline is that the expected
outcome is **pre-registered as arithmetic**: a direction over the two
arms the probe must run (`treatment` vs `control`) and a margin. After
the probe runs, `judge_probe` computes supports / weakens / uninformative
without asking any model anything. A verdict the model cannot argue with
is the only kind that may move a hypothesis up the research state ladder.

The diagnosis reads only the hypothesis card — claim, mechanism,
prediction. It deliberately does not read the briefing: the briefing is
written *after* the evidence now (cheap diagnosis before expensive
writeup), and a later revision of the briefing therefore cannot touch a
registration that never depended on it.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from typing import Any

from ..models import BlockedRecord
from .evidence import MIN_HEAVY_REPLICATES
from .generate import GeneratedCard, GenerationRefused
from .llm import Completion
from .probeexp import COMPUTE_TIERS, DEFAULT_TIER

DIRECTIONS = ("treatment_lower", "treatment_higher")
DEFAULT_MARGIN = 0.05

SYSTEM = (
    "You are designing the cheapest experiment that discriminates between"
    " two explanations. Answer with one JSON object and nothing else."
)

TEMPLATE = """A research hypothesis survived the novelty gates. Before any experiment and before any write-up, name what would change our mind about it.

Researcher's topic: {topic}
Claim: {claim}
Proposed mechanism: {mechanism}
Prediction: {prediction}

Answer with JSON:
{{"alternative": "<=50 words: the strongest COMPETING explanation for the same predicted effect — not a restatement of the mechanism",
 "experiment": "<=60 words: the smallest two-arm experiment that separates mechanism from alternative; the treatment arm isolates the mechanism, the control arm removes exactly it",
 "treatment_arm": "<=25 words: what the treatment arm does",
 "control_arm": "<=25 words: what the control arm does (same in every other way)",
 "expected_direction": "treatment_lower" or "treatment_higher" — the metric relation IF the mechanism is real,
 "alternative_direction": "treatment_lower" or "treatment_higher" — the metric relation IF the alternative explains the effect instead,
 "expected_if_alternative": "<=30 words: what the metrics look like if the alternative explains the effect instead",
 "margin": <number>,
 "margin_reason": "<=25 words: why THIS margin — what size of separation would be indistinguishable from noise or constant factors in this experiment",
 "mechanism_flag": "<=20 words: the single boolean or config key that turns the claimed mechanism on; treatment and control may differ only here",
 "compute_tier": "host" (default, 600s stdlib) or "host-heavy" (3600s, numpy allowed) — the pre-registered budget the host reproduction needs; ask for host-heavy only when the experiment cannot be expressed at stdlib scale}}

`expected_direction` and `alternative_direction` are pre-registered: after the probe runs, the verdict is computed from the two numbers and these fields, with no further judgment. A design where mechanism and alternative predict the SAME direction has no discriminating power and will be rejected — find a regime where they predict opposite signs. `margin` is the minimum relative separation (0.01 to 0.5) that counts as a real difference; pick it for this experiment, do not copy a default. Both arms must share one measure and differ by a single mechanism flag; a treatment that always wins by construction is not a diagnosis. The measured quantity is a property of the CLAIM's object, not the distant mechanism's internal cost.
"""

FOLLOWUP = """

An earlier probe of this hypothesis failed to separate the arms (verdict: uninformative). That experiment was:
{experiment}

Do not repeat that design. Design a MORE discriminating experiment: a more extreme construction, a larger effect surface, or a regime where mechanism and alternative predict opposite signs — so the arms can actually separate this time. Do not use treatment or control numbers from any previous run; they are withheld so this redesign cannot chase a win. Iterate until the experiment can discriminate, not until the idea wins.
"""

SWITCH_MECHANISM = """

This hypothesis line is stagnant: two consecutive valid experiments were UNINFORMATIVE on the same mechanism. You MUST change one of: the intervention (mechanism flag), the metric, the scale, the competing explanation, or the world slice. Rewriting the previous experiment in new words is forbidden.
"""

DISCARDED = """

This mission already tried these designs and they failed to discriminate:
{experiments}

Do not repeat any of them. Do not look for treatment/control numbers — none are provided, because chasing a preferred sign is not a diagnosis.
"""

PROGRAM_CONTEXT = """

A committed research program from earlier missions constrains this diagnosis. It is not evidence and cannot climb:
{program}
Design the two arms to advance that program. Do not spray a new experiment that ignores it.
"""

WORLD_CONTEXT = """

A frozen experimental world will be bound for the cheap probe. Design the two arms against THIS instance, not an invented stand-in.
World: {world_id} — {title}
Schema: {schema}
Files in data/: {files}
How to load: {load_hint}
The treatment and control must read the same attested bytes and differ only by the mechanism flag. Do not subsample the instance just to make a preferred sign easier.
"""

WORLD_DYNAMICS = """

This world is forward-simulatable. The trusted runtime executes each lever on the evolving state until a stop (absorbing, plateau, or horizon) and measures every observable (relative change, start to stop):
{lever_lines}
Observables the runtime can compute on this world: {observables}

Add to the JSON:
"world_lever": the ONE lever above your mechanism acts through — the treatment arm's intervention must be an instance of it; write "none" if the claimed mechanism has no handle in this world,
"world_observable": the ONE observable above that the claim's predicted quantity is about (required unless world_lever is "none").
An honest "none" means the prediction has no handle on this object: skip the probe rather than invent data. A surrogate binding that pretends a handle exists is worse than no world. A lever the simulation shows inert cannot separate your arms — pick a responsive one or say "none".
"""

SCOUT_FEEDBACK = """

The runtime's forward simulation of the bound world (world facts, not experiment results — no treatment/control numbers exist in it):
{scout}
Use this dose-response reading to pick a lever and regime where the arms can actually separate.
"""

CONSTRUCTED_WORLD_CONTEXT = """

The bound world was constructed this iteration for the registered experiment. It is not a freeze and cannot corroborate.
World: {world_id} — {title}
Schema: {schema}
Files in data/: {files}
How to load: {load_hint}
Excerpt:
{excerpt}
Design the two arms against THIS instance. Both arms must read data/world.json and differ only by the mechanism flag.
"""

REQUIREMENT_CONTEXT = """

No attested freeze matches this claim (object_type={object_type}). Do not bind a substitute catalog fixture.
Do not invent a dataset and call it verification. Do not construct a GENERATED stand-in for this missing object. Record the requirement for a later host freeze. Design the two arms against the named object so a freeze can run them; both arms will read data/ of that freeze.
"""

LINEAGE_CONTEXT = """

Retrieved papers describe how this claim's object developed. Design the two arms against THAT object, not a toy that would make the idea win.
Named instance: {named_instance}
Development path: {lineage}
Why this is the claim's object: {why_this_object}
Papers: {cite_ids}
"""

PAPER_BASELINES = """

Papers retrieved this mission (not related-work colour). The control arm should instantiate one named method from this list on the bound world:
{papers}
JSON may also include:
"baseline_cite_id": "<one cite_id above, or empty if none fit>",
"baseline_method": "<=25 words: the paper's method the control implements"
If baseline_cite_id is set it must be one of the ids above. Do not invent a citation.
"""


class DiagnosisRefused(GenerationRefused):
    pass


def _refuse(attempted: str, unlock: str) -> DiagnosisRefused:
    return DiagnosisRefused(
        BlockedRecord(
            missing_capability="model_diagnosis_schema",
            attempted=attempted,
            unlock_condition=unlock,
        )
    )


@dataclass(frozen=True)
class Diagnosis:
    card_id: str
    alternative: str
    experiment: str
    treatment_arm: str
    control_arm: str
    expected_direction: str
    alternative_direction: str
    expected_if_alternative: str
    margin: float
    margin_reason: str
    baseline_cite_id: str = ""
    baseline_method: str = ""
    mechanism_flag: str = "mechanism_enabled"
    # Pre-registered compute budget for the host reproduction. Chosen
    # before any probe runs; a bigger budget is never a reaction to a
    # result. See probeexp.COMPUTE_TIERS.
    compute_tier: str = DEFAULT_TIER
    # The world-dynamics binding (dynworld). `world_lever` is the runtime
    # lever the mechanism acts through — "none" is the honest answer when
    # the claim has no handle in the bound world, and it unbinds the
    # world. `world_observable` is the runtime-computed quantity the
    # claim's prediction is about. Both are validated against the
    # bridge's vocabulary: the model may choose a handle; it may not
    # invent one.
    world_lever: str = ""
    world_observable: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "card_id": self.card_id,
            "alternative": self.alternative,
            "experiment": self.experiment,
            "treatment_arm": self.treatment_arm,
            "control_arm": self.control_arm,
            "expected_direction": self.expected_direction,
            "alternative_direction": self.alternative_direction,
            "expected_if_alternative": self.expected_if_alternative,
            "margin": self.margin,
            "margin_reason": self.margin_reason,
            "mechanism_flag": self.mechanism_flag,
            "compute_tier": self.compute_tier,
        }
        if self.baseline_cite_id:
            payload["baseline_cite_id"] = self.baseline_cite_id
            payload["baseline_method"] = self.baseline_method
        if self.world_lever:
            payload["world_lever"] = self.world_lever
            payload["world_observable"] = self.world_observable
        return payload


def _parse(completion: Completion) -> dict[str, Any]:
    text = completion.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _refuse(
            "parse a JSON diagnosis",
            f"the model answers with one JSON object; it answered with {text[:80]!r} ({exc})",
        ) from exc
    if not isinstance(payload, dict):
        raise _refuse(
            "read a JSON object for the diagnosis",
            f"the top-level value is an object, not {type(payload).__name__}",
        )
    return payload


def diagnosis_from_payload(card_id: str, payload: dict[str, Any]) -> Diagnosis:
    fields = {}
    for name in (
        "alternative",
        "experiment",
        "treatment_arm",
        "control_arm",
        "expected_if_alternative",
        "margin_reason",
    ):
        value = str(payload.get(name) or "").strip()
        if not value:
            raise _refuse(
                f"read a non-empty {name}",
                f"the diagnosis fills every field; {name} was empty"
                + (
                    ": say why this margin and not another"
                    if name == "margin_reason"
                    else ""
                ),
            )
        fields[name] = value
    direction = str(payload.get("expected_direction") or "").strip()
    if direction not in DIRECTIONS:
        raise _refuse(
            "read expected_direction",
            "expected_direction is treatment_lower or treatment_higher,"
            " pre-registered before the probe runs",
        )
    alternative_direction = str(payload.get("alternative_direction") or "").strip()
    if alternative_direction not in DIRECTIONS:
        raise _refuse(
            "read alternative_direction",
            "alternative_direction is treatment_lower or treatment_higher —"
            " what the metrics do IF the competing explanation is the real"
            " cause, pre-registered like expected_direction",
        )
    if alternative_direction == direction:
        # The heart of discriminative preregistration: an experiment where
        # both stories predict the same sign can only ever flatter the
        # mechanism ("having it beats not having it"). A live batch showed
        # 14/21 supports with inflated separations from exactly such designs.
        raise _refuse(
            "register a discriminating design",
            "mechanism and alternative predict the SAME direction, so this"
            " experiment cannot tell them apart; redesign it around a regime"
            " where the two explanations predict opposite signs",
        )
    raw_margin = payload.get("margin", DEFAULT_MARGIN)
    try:
        margin = float(raw_margin)
    except (TypeError, ValueError) as exc:
        raise _refuse(
            "read margin", "margin is a number between 0.01 and 0.5"
        ) from exc
    if not 0.01 <= margin <= 0.5:
        raise _refuse("read margin", "margin is a number between 0.01 and 0.5")
    from .plugins import refuse_diagnosis

    locked = refuse_diagnosis(
        treatment_arm=fields["treatment_arm"],
        control_arm=fields["control_arm"],
        experiment=fields["experiment"],
        alternative=fields["alternative"],
        expected_direction=direction,
        alternative_direction=alternative_direction,
        claim=str(payload.get("claim") or payload.get("_claim") or ""),
        mechanism=str(payload.get("mechanism") or payload.get("_mechanism") or ""),
        topic=str(payload.get("topic") or payload.get("_topic") or ""),
    )
    if locked:
        raise _refuse("write a two-arm diagnosis", locked)
    cite_id = str(payload.get("baseline_cite_id") or "").strip()
    method = str(payload.get("baseline_method") or "").strip()
    allowed = payload.get("_allowed_baseline_ids")
    if cite_id and isinstance(allowed, (set, frozenset, list, tuple)):
        if cite_id not in set(allowed):
            raise _refuse(
                "name a retrieved paper as the control baseline",
                "baseline_cite_id must be a paper retrieved this mission; "
                "do not invent a citation",
            )
    tier = str(payload.get("compute_tier") or "").strip().lower()
    if tier and tier not in COMPUTE_TIERS:
        raise _refuse(
            "read compute_tier",
            "compute_tier is one of "
            + ", ".join(sorted(COMPUTE_TIERS))
            + " — a pre-registered budget, not an invented one",
        )
    world_lever = str(payload.get("world_lever") or "").strip()
    world_observable = str(payload.get("world_observable") or "").strip()
    levers = payload.get("_world_levers")
    if levers:
        # The bound world declares runtime dynamics. The model may pick a
        # handle from the measured vocabulary or say "none"; a lever or
        # observable it invented would be a transition rule written after
        # seeing the claim — a larger invented world.
        lever_vocab = {str(item) for item in levers}
        obs_vocab = {str(item) for item in (payload.get("_world_observables") or [])}
        if not world_lever:
            world_lever = "none"
        if world_lever != "none" and world_lever not in lever_vocab:
            raise _refuse(
                "bind a runtime world lever",
                "world_lever must be one of "
                + ", ".join(sorted(lever_vocab))
                + " (or 'none' when the mechanism has no handle in this"
                " world); the runtime cannot pull a lever the bridge does"
                " not declare",
            )
        if world_lever != "none" and world_observable not in obs_vocab:
            raise _refuse(
                "bind a runtime world observable",
                "world_observable must be one of "
                + ", ".join(sorted(obs_vocab))
                + " — the runtime-computed quantity the claim's prediction"
                " is about, not an invented measurement",
            )
        if world_lever == "none":
            world_observable = ""
    return Diagnosis(
        card_id=card_id,
        expected_direction=direction,
        alternative_direction=alternative_direction,
        margin=margin,
        baseline_cite_id=cite_id,
        baseline_method=method,
        mechanism_flag=str(payload.get("mechanism_flag") or "mechanism_enabled").strip()
        or "mechanism_enabled",
        compute_tier=tier or DEFAULT_TIER,
        world_lever=world_lever,
        world_observable=world_observable,
        **fields,
    )


def write_diagnosis(
    client: Any,
    card: GeneratedCard,
    topic: str,
    *,
    prior_probe: dict[str, Any] | None = None,
    failed_experiments: list[str] | tuple[str, ...] | None = None,
    retries: int = 1,
    skills: str = "",
    world: Any = None,
    works: list[Any] | None = None,
    program: str | tuple[str, ...] = "",
    requirement: Any = None,
    dynamics: dict[str, Any] | None = None,
    world_path: Any = None,
) -> Diagnosis:
    """`prior_probe` is this hypothesis line's last probe from the research
    state, or an in-mission attempt that failed to discriminate. An
    uninformative earlier design is quoted back so the follow-up experiment
    must sharpen, not repeat — the spiral that keeps a stalled speculative
    line moving. `failed_experiments` are additional discarded designs from
    this mission; they are quoted as text, never as treatment/control
    numbers.

    A schema refusal — above all a non-discriminating design, where both
    explanations predict the same direction — is quoted back for `retries`
    more attempts. Only the final refusal propagates.
    """
    prompt = TEMPLATE.format(
        topic=topic,
        claim=card.claim,
        mechanism=card.mechanism,
        prediction=card.prediction,
    )
    if skills:
        prompt = skills + prompt
    if (
        prior_probe
        and prior_probe.get("verdict") == "uninformative"
        and str(prior_probe.get("experiment") or "").strip()
    ):
        prompt += FOLLOWUP.format(experiment=prior_probe["experiment"])
    if prior_probe and str(prior_probe.get("world_scout") or "").strip():
        prompt += SCOUT_FEEDBACK.format(scout=prior_probe["world_scout"])
    if prior_probe and prior_probe.get("must_switch_mechanism"):
        prompt += SWITCH_MECHANISM
    discarded: list[str] = []
    for text in list(failed_experiments or []) + list(
        (prior_probe or {}).get("discarded") or []
    ):
        item = str(text or "").strip()
        if not item or item in discarded:
            continue
        if prior_probe and item == str(prior_probe.get("experiment") or "").strip():
            continue
        discarded.append(item)
    if discarded:
        prompt += DISCARDED.format(
            experiments="\n".join(f"- {item}" for item in discarded[:8])
        )
    program_text = program
    if isinstance(program_text, (list, tuple)):
        program_text = "\n".join(f"- {line}" for line in program_text if str(line).strip())
    program_text = str(program_text or "").strip()
    if program_text:
        prompt += PROGRAM_CONTEXT.format(program=program_text)
    if world is not None:
        files = list(getattr(world, "files", ()) or [])
        role = str(getattr(world, "role", "") or "")
        excerpt = ""
        if role == "generated":
            try:
                from .genworld import load_world_payload, world_excerpt

                excerpt = world_excerpt(load_world_payload(world.root))
            except Exception:
                excerpt = ""
            prompt += CONSTRUCTED_WORLD_CONTEXT.format(
                world_id=getattr(world, "id", "world"),
                title=getattr(world, "title", ""),
                schema=getattr(world, "schema", "") or "symbolic_trace",
                files=", ".join(files) or "world.json",
                load_hint=getattr(world, "load_hint", "")
                or "json.loads(Path('data/world.json').read_text())",
                excerpt=excerpt or "(see data/world.json)",
            )
        else:
            prompt += WORLD_CONTEXT.format(
                world_id=getattr(world, "id", "world"),
                title=getattr(world, "title", ""),
                schema=getattr(world, "schema", "") or "attested files",
                files=", ".join(files) or "(see data/)",
                load_hint=getattr(world, "load_hint", "") or "read files under data/",
            )
    lever_vocab: tuple[str, ...] = ()
    obs_vocab: tuple[str, ...] = ()
    if world is not None and dynamics:
        from .dynworld import card_lines

        lever_vocab = tuple(sorted(dynamics.get("levers") or ()))
        obs_vocab = tuple(dynamics.get("observables") or ())
        if lever_vocab:
            prompt += WORLD_DYNAMICS.format(
                lever_lines=card_lines(dynamics),
                observables=", ".join(obs_vocab),
            )
    if world_path is not None:
        prompt += LINEAGE_CONTEXT.format(
            named_instance=getattr(world_path, "named_instance", "") or "",
            lineage=getattr(world_path, "lineage", "") or "",
            why_this_object=getattr(world_path, "why_this_object", "") or "",
            cite_ids=", ".join(
                str(item) for item in (getattr(world_path, "cite_ids", ()) or ())
            ),
        )
    if requirement is not None and world is None:
        object_type = ""
        if hasattr(requirement, "object_type"):
            object_type = str(requirement.object_type or "")
        elif isinstance(requirement, dict):
            object_type = str(requirement.get("object_type") or "")
        prompt += REQUIREMENT_CONTEXT.format(
            object_type=object_type or "formula",
        )
    allowed_ids: set[str] = set()
    paper_lines: list[str] = []
    for work in works or []:
        cite = ""
        title = ""
        abstract = ""
        if hasattr(work, "cite_id"):
            cite = str(work.cite_id() or "")
            title = str(getattr(work, "title", "") or "")
            abstract = str(getattr(work, "abstract", "") or "")
        elif isinstance(work, dict):
            cite = str(work.get("arxiv_id") or work.get("cite_id") or work.get("id") or "")
            title = str(work.get("title") or "")
            abstract = str(work.get("abstract") or "")
        if not cite:
            continue
        allowed_ids.add(cite)
        cue = " ".join(abstract.split())[:180]
        paper_lines.append(f"- [{cite}] {title}" + (f" — {cue}" if cue else ""))
    if paper_lines:
        prompt += PAPER_BASELINES.format(papers="\n".join(paper_lines[:12]))
    refusal: DiagnosisRefused | None = None
    for attempt in range(retries + 1):
        ask = prompt
        if refusal is not None:
            ask = (
                prompt
                + "\n\nYour previous diagnosis was rejected:"
                f" {refusal.record.unlock_condition}."
                " Write a new diagnosis that cannot be rejected for the"
                " same reason."
            )
        completion = client.complete(
            ask,
            purpose=f"diagnosis:{card.card_id}"
            + (f":retry{attempt}" if attempt else ""),
            system=SYSTEM,
        )
        completion.assert_usable()
        try:
            return diagnosis_from_payload(
                card.card_id,
                {
                    **_parse(completion),
                    "_allowed_baseline_ids": allowed_ids,
                    "_claim": card.claim,
                    "_mechanism": card.mechanism,
                    "_topic": topic,
                    "_world_levers": lever_vocab,
                    "_world_observables": obs_vocab,
                },
            )
        except DiagnosisRefused as exc:
            refusal = exc
    assert refusal is not None
    raise refusal


def judge_probe(
    diagnosis: Diagnosis, treatment: float, control: float
) -> dict[str, Any]:
    """Supports / weakens / uninformative, by arithmetic alone.

    The separation is relative to the control arm (absolute when the
    control is zero). Inside the margin, the probe failed to discriminate
    — that is a reading about the experiment, not about the idea.
    """
    base = abs(control)
    if base > 0:
        separation = (treatment - control) / base
    else:
        separation = treatment - control
    if abs(separation) < diagnosis.margin:
        verdict = "uninformative"
    else:
        lower = treatment < control
        expected_lower = diagnosis.expected_direction == "treatment_lower"
        verdict = "supports" if lower == expected_lower else "weakens"
    return {
        "verdict": verdict,
        "treatment": treatment,
        "control": control,
        "separation": round(separation, 6),
        "margin": diagnosis.margin,
        "expected_direction": diagnosis.expected_direction,
        "alternative": diagnosis.alternative,
    }


# Two-sided 95% Student-t critical values for df 1..30 — the standard
# table (identical to what scipy.stats.t.ppf(0.975, df) returns), kept
# inline because this runtime is dependency-free. Past df 30 the normal
# quantile is within half a percent and takes over.
_T_CRITICAL_95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def t_critical_95(df: int) -> float:
    """The two-sided 95% Student-t critical value for `df` degrees of freedom."""
    if df < 1:
        raise ValueError("a confidence interval needs at least 1 degree of freedom")
    if df <= len(_T_CRITICAL_95):
        return _T_CRITICAL_95[df - 1]
    return statistics.NormalDist().inv_cdf(0.975)


def _separation(treatment: float, control: float) -> float:
    base = abs(control)
    return (treatment - control) / base if base > 0 else treatment - control


def judge_replicated(
    diagnosis: Diagnosis, pairs: list[tuple[float, float]]
) -> dict[str, Any]:
    """The confirmation-tier verdict: arithmetic over executor-owned reruns.

    `pairs` is one `(treatment, control)` measurement per registered seed,
    produced by the trusted executor rerunning the *registered* script —
    never reported by the script itself. The model may generate
    computation; it may not generate the evidence that audits it.

    The verdict is `supports` only when the **entire** 95% confidence
    interval of the per-seed separations clears the pre-registered margin
    on the expected side; `weakens` when it clears the margin on the
    opposite side. Everything else is `uninformative` with the reason
    named — including the invariance case: if every seed produced the
    same numbers, the experiment did not consume the registered
    variation, and stability under variation cannot be distinguished
    from a script that ignores its seed.
    """
    margin = diagnosis.margin
    expected_lower = diagnosis.expected_direction == "treatment_lower"
    base: dict[str, Any] = {
        "replicated": False,
        "n": len(pairs),
        "margin": margin,
        "expected_direction": diagnosis.expected_direction,
        "alternative": diagnosis.alternative,
    }
    for pair in pairs:
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            for value in pair
        ):
            return {
                **base,
                "verdict": "uninformative",
                "reason": "a replicate arm is not a finite number",
            }
    count = len(pairs)
    if count < MIN_HEAVY_REPLICATES:
        return {
            **base,
            "verdict": "uninformative",
            "reason": (
                f"only {count} executor reruns completed; at least "
                f"{MIN_HEAVY_REPLICATES} are required for a confirmation verdict"
            ),
        }
    if len({(float(t), float(c)) for t, c in pairs}) == 1:
        return {
            **base,
            "n": count,
            "invariant": True,
            "verdict": "uninformative",
            "reason": (
                "every registered seed produced identical numbers; the "
                "experiment is invariant under the registered variation, "
                "which is indistinguishable from a script that ignores "
                "its seed — stability cannot be confirmed"
            ),
        }
    separations = [_separation(float(t), float(c)) for t, c in pairs]
    mean = statistics.fmean(separations)
    spread = statistics.stdev(separations)
    half = t_critical_95(count - 1) * spread / math.sqrt(count)
    ci_low, ci_high = mean - half, mean + half
    if expected_lower:
        supports, weakens = ci_high <= -margin, ci_low >= margin
    else:
        supports, weakens = ci_low >= margin, ci_high <= -margin
    verdict = "supports" if supports else "weakens" if weakens else "uninformative"
    result: dict[str, Any] = {
        **base,
        "replicated": True,
        "verdict": verdict,
        "mean_separation": round(mean, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
    }
    if verdict == "uninformative":
        result["reason"] = (
            f"the 95% interval of the per-seed separations "
            f"[{result['ci_low']}, {result['ci_high']}] does not clear the "
            f"pre-registered margin {margin} on one side"
        )
    return result
