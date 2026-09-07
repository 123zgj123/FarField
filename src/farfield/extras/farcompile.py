"""Compile a far-field noun onto a scouted world lever.

Generation still samples a distant *concept* from the jump graph. The
compile slots say how that noun becomes a declared runtime lever on
attested fields. Welding failure refuses the candidate; the jump retries
its remaining far labels. Exhausting the landing switches operator.
This is not a menu of five lever names replacing far-field search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .dynworld import RESPONSE_FLOOR, contrast_meaning, is_contrast
from .ideakind import (
    PROBE,
    QUESTION,
    observational_first,
    parse_idea_kind,
    prediction_tokens_for,
)
from .lifecycle import (
    ORIGIN_LOCAL,
    causal_overclaim_reason,
    handle_is_blacklisted,
    is_local_origin_label,
)
from .prior import content_tokens

COMPILE_CAPABILITY = "model_far_compile"


@dataclass(frozen=True)
class WorldMenu:
    """Mission-level scout: the prior every card of this freeze must read."""

    levers: tuple[str, ...]
    observables: tuple[str, ...]
    inert: frozenset[str]
    load_hint: str = ""
    attested_fields: tuple[str, ...] = ()
    room_to_move: bool = False
    world_id: str = ""
    blocked_handles: frozenset[str] = frozenset()

    def with_blocked(self, blocked: tuple[str, ...] | list[str]) -> "WorldMenu":
        extra = frozenset(str(item).strip() for item in blocked if str(item).strip())
        if not extra or extra <= self.blocked_handles:
            return self
        return WorldMenu(
            levers=self.levers,
            observables=self.observables,
            inert=self.inert,
            load_hint=self.load_hint,
            attested_fields=self.attested_fields,
            room_to_move=self.room_to_move,
            world_id=self.world_id,
            blocked_handles=self.blocked_handles | extra,
        )

    def handles(self) -> tuple[str, ...]:
        """Every lever/observable pair that is declared and not inert."""
        return tuple(
            f"{lever}/{observable}"
            for lever in self.levers
            for observable in self.observables
            if f"{lever}/{observable}" not in set(self.inert)
        )

    def unused_handles(
        self,
        used: tuple[str, ...] | list[str],
        *,
        reuse_scope: str = "spray",
        lineage_handles: tuple[str, ...] | list[str] = (),
    ) -> tuple[str, ...]:
        """Spray forbids used handles. Lineage deepening may reuse its own."""
        taken = {str(item) for item in used if str(item).strip()} | set(
            self.blocked_handles
        )
        if reuse_scope == "lineage":
            allowed = {str(item).strip() for item in lineage_handles if str(item).strip()}
            taken = {item for item in taken if item not in allowed}
            taken -= {item.split("/", 1)[0] for item in allowed}
        bare = {item for item in taken if "/" not in item}
        return tuple(
            handle
            for handle in self.handles()
            if handle not in taken and handle.split("/", 1)[0] not in bare
        )

    def unused_observational(self, used: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return tuple(
            handle
            for handle in self.unused_handles(used)
            if is_contrast(handle.split("/", 1)[0])
        )

    def unused_interventional(self, used: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        return tuple(
            handle
            for handle in self.unused_handles(used)
            if not is_contrast(handle.split("/", 1)[0])
        )

    def scorable_questions(
        self, used: tuple[str, ...] | list[str] = ()
    ) -> tuple[str, ...]:
        """Questions this freeze can score now. Not an idea and not a topic."""
        lines: list[str] = []
        for handle in self.unused_observational(used):
            lever, _, observable = handle.partition("/")
            meaning = contrast_meaning(lever) or lever
            lines.append(f"{handle} — {meaning}; measure {observable}")
        return tuple(lines)

    def prompt_block(self, used_levers: tuple[str, ...] | list[str] = ()) -> str:
        if not self.levers:
            return ""
        used = [str(item) for item in used_levers if str(item).strip()]
        fields = ", ".join(self.attested_fields) or self.load_hint or "(see load_hint)"
        observational = self.unused_observational(used)
        unused_levers = [
            lever
            for lever in self.levers
            if not any(u.split("/", 1)[0] == lever for u in used)
        ]
        acquire_handles = self.unused_interventional(used)
        constraint = ""
        if used or observational or acquire_handles:
            constraint = (
                "HARD CONSTRAINT by idea_kind:\n"
                f"- question: world_lever/world_observable MUST be one of "
                f"{', '.join(observational) or '(none left — do not invent a two-arm)'}. "
                "Do not invent an intervention.\n"
                f"- probe: weld a manipulable handle "
                f"({', '.join(acquire_handles) or '(none)'}). "
                "Fair two-arm, one freeze, one measure.\n"
                f"- acquire: name a missing capability as a harvest recipe "
                f"({', '.join(acquire_handles) or 'none unused'}); success is a "
                "new attested world, not a two-arm effect.\n"
                "- theory: lock a derivation to registered quantities; do not invent a two-arm.\n"
                + (
                    f"Unused levers: {', '.join(unused_levers)}. "
                    if unused_levers
                    else "Every lever is used; a used lever is allowed only with an unused observable. "
                )
                + "Spray may not repeat a used handle; the same lineage may reuse one.\n"
            )
            if observational:
                constraint += (
                    "This freeze can still score unused observational handles. "
                    "A question that welds dropout / replay_gate / "
                    "freeze_validators is a causal overclaim. "
                    "idea_kind=probe may use a manipulable lever; "
                    "idea_kind=acquire is the honest product when the lever "
                    "is not feelable on this freeze.\n"
                )
        if self.blocked_handles:
            constraint += (
                "Blocked attractor handles (failed on this freeze; do not compile): "
                f"{', '.join(sorted(self.blocked_handles))}.\n"
            )
        contrasts = [lever for lever in self.levers if is_contrast(lever)]
        contrast_note = ""
        if contrasts:
            contrast_note = (
                "Observational handles (`contrast:<stratum>`): treatment = the records "
                "the world already recorded in that stratum, control = the rest, matched "
                "on the world's natural strata (episode template, updates per episode); "
                "the mechanism flag selects the stratum, the measure is identical. On a "
                "replayed history these are the handles that ask about the world's own "
                "dynamics — an interventional lever there can only subsample records. "
                f"Available contrasts: {', '.join(contrasts)}.\n"
            )
            questions = self.scorable_questions(used)
            if questions:
                contrast_note += (
                    "Scorable questions on this freeze (compile onto one; not a "
                    "new topic):\n"
                    + "\n".join(f"- {item}" for item in questions)
                    + "\n"
                )
        return (
            "This mission's bound world already declared runtime dynamics. "
            "Compile the distant concept onto them; do not paste the far "
            "noun into the title and skip the handle.\n"
            f"Declared levers: {', '.join(self.levers)}\n"
            + contrast_note +
            f"Declared observables: {', '.join(self.observables)}\n"
            f"Inert lever/observable pairs (do not register): "
            f"{', '.join(sorted(self.inert)) or '(none)'}\n"
            f"Attested field paths: {fields}\n"
            "Compiled lever/observable handles already used this mission: "
            f"{', '.join(used) or '(none)'}\n"
            + constraint +
            "Prefer an unused declared lever over restating a used one "
            "as a new graph noun.\n"
            "JSON must also include:\n"
            '"idea_kind": "question | acquire | theory | probe",\n'
            '"world_lever": one declared lever,\n'
            '"world_observable": one declared observable that this lever '
            "actually moves,\n"
            '"far_maps_to_lever": "<=40 words: how the distant concept '
            'becomes that lever on attested fields — knapsack must say '
            'how it becomes mask_tools, not just put the word in the title"'
        )


def menu_from_card(
    card: dict[str, Any] | None,
    fixture: Any = None,
) -> WorldMenu:
    payload = card if isinstance(card, dict) else {}
    levers_map = payload.get("levers") if isinstance(payload.get("levers"), dict) else {}
    levers = tuple(
        str(name)
        for name in sorted(levers_map)
        if str(name).strip() and str(name) != "none"
    )
    observables = tuple(
        str(name) for name in (payload.get("observables") or ()) if str(name).strip()
    )
    inert = frozenset(
        str(item) for item in (payload.get("inert") or ()) if str(item).strip()
    )
    room = False
    for lever, response in levers_map.items():
        if not isinstance(response, dict):
            continue
        for delta in response.values():
            try:
                if abs(float(delta)) >= RESPONSE_FLOOR:
                    room = True
                    break
            except (TypeError, ValueError):
                continue
        if room:
            break
    load_hint = str(getattr(fixture, "load_hint", "") or "").strip()
    layout = str(getattr(fixture, "attested_layout", "") or "").strip()
    fields = _attested_fields(load_hint, layout, observables)
    return WorldMenu(
        levers=levers,
        observables=observables,
        inert=inert,
        load_hint=load_hint or layout,
        attested_fields=fields,
        room_to_move=room,
        world_id=str(payload.get("world_id") or getattr(fixture, "id", "") or ""),
    )


def _attested_fields(
    load_hint: str, layout: str, observables: tuple[str, ...]
) -> tuple[str, ...]:
    known = (
        "created_tools",
        "created_tool_count",
        "returncode",
        "action_chars",
        "output_chars",
        "label",
        "steps",
        "traces",
        "own_payoff",
        "own_choice",
    )
    blob = f"{load_hint} {layout}".lower()
    found = [name for name in known if name in blob]
    for obs in observables:
        if obs not in found:
            found.append(obs)
    return tuple(found)


def assert_compile(
    *,
    far_label: str,
    world_lever: str,
    world_observable: str,
    far_maps_to_lever: str,
    prediction: str,
    menu: WorldMenu,
    used_levers: tuple[str, ...] | list[str] = (),
    refuse: Callable[..., Any],
    idea_kind: str = PROBE,
    origin: str = "",
    reuse_scope: str = "spray",
    lineage_handles: tuple[str, ...] | list[str] = (),
    design_sig: str = "",
    world_id: str = "",
    failure_mode: str = "",
) -> tuple[str, str, str]:
    """Presentation lint: the mapping sentence names the far noun and lever.

    This is not scientific compile. Object identity, handle kind, and
    measurement grounding live in `claimspec.typecheck_claim`.
    """
    lever = str(world_lever or "").strip()
    observable = str(world_observable or "").strip()
    mapping = str(far_maps_to_lever or "").strip()
    if lever not in menu.levers:
        raise refuse(
            f"compile {far_label!r} onto a declared world lever",
            "world_lever must be one of "
            + ", ".join(menu.levers)
            + "; pasting the distant noun into the title is not a handle",
            capability=COMPILE_CAPABILITY,
            failed_far=far_label,
        )
    if observable not in menu.observables:
        raise refuse(
            f"compile {far_label!r} onto a declared observable",
            "world_observable must be one of " + ", ".join(menu.observables),
            capability=COMPILE_CAPABILITY,
            failed_far=far_label,
        )
    pair_key = f"{lever}/{observable}"
    kind = parse_idea_kind(idea_kind)
    overclaim = causal_overclaim_reason(
        kind, claim="", lever=lever, prediction=prediction
    )
    if overclaim:
        raise refuse(
            f"keep {far_label!r} as an observational question",
            overclaim,
            capability="model_causal_overclaim",
            failed_far=far_label,
        )
    if handle_is_blacklisted(
        menu.blocked_handles,
        pair_key,
        world_id=world_id or menu.world_id,
        failure_mode=failure_mode,
        design_sig=design_sig,
    ) or handle_is_blacklisted(
        menu.blocked_handles,
        lever,
        world_id=world_id or menu.world_id,
        failure_mode=failure_mode,
        design_sig=design_sig,
    ):
        raise refuse(
            f"compile {far_label!r} onto a handle that is not an attractor",
            f"{pair_key} already failed with an equivalent design on this "
            "world version; a materially different design may retry, a "
            "spray landing may not repeat the same handle",
            capability=COMPILE_CAPABILITY,
            failed_far=far_label,
            used_handle=True,
        )
    leftover = menu.unused_observational(used_levers)
    if leftover and not is_contrast(lever) and observational_first(kind):
        raise refuse(
            f"compile {far_label!r} onto a scorable observational handle",
            f"this freeze still has unused observational handles ({', '.join(leftover)}); "
            f"{lever} only subsamples a replayed history. Write idea_kind=question on "
            "a contrast, or idea_kind=acquire if the interesting mechanism needs a "
            "world this freeze cannot feel",
            capability=COMPILE_CAPABILITY,
            failed_far=far_label,
            used_handle=True,
        )
    unused = menu.unused_handles(
        used_levers,
        reuse_scope=reuse_scope,
        lineage_handles=lineage_handles,
    )
    taken = {str(item) for item in used_levers if str(item).strip()}
    if reuse_scope == "lineage":
        allowed = {str(item).strip() for item in lineage_handles if str(item).strip()}
        taken = {item for item in taken if item not in allowed}
    bare_used = {item for item in taken if "/" not in item}
    if reuse_scope != "lineage" and (lever in bare_used or pair_key in taken):
        raise refuse(
            f"compile {far_label!r} onto a handle this mission has not used",
            f"{lever}/{observable} already compiled an earlier spray landing; "
            "the next far concept must weld onto an unused handle"
            + (f" — one of: {', '.join(unused)}" if unused else " — none remain on this world"),
            capability=COMPILE_CAPABILITY,
            failed_far=far_label,
            used_handle=True,
        )
    if pair_key in menu.inert:
        raise refuse(
            f"compile {far_label!r} onto a lever that moves {observable}",
            f"{pair_key} is inert in this world's scout; pick a "
            "lever/observable pair the runtime actually moves",
            capability=COMPILE_CAPABILITY,
            failed_far=far_label,
        )
    if len(mapping.split()) < 8:
        raise refuse(
            f"say how {far_label!r} becomes {lever}",
            "far_maps_to_lever must be a sentence that turns the distant "
            "concept into the declared lever on attested fields",
            capability=COMPILE_CAPABILITY,
            failed_far=far_label,
        )
    maps_l = mapping.lower()
    if lever.lower() not in maps_l:
        raise refuse(
            f"name {lever} in far_maps_to_lever",
            f"the mapping sentence must say how {far_label} becomes {lever} "
            "(for example knapsack → mask_tools), not only reuse the noun "
            "in the claim title",
            capability=COMPILE_CAPABILITY,
            failed_far=far_label,
        )
    far_tokens = content_tokens(far_label)
    if (
        far_tokens
        and not (content_tokens(mapping) & far_tokens)
        and origin != ORIGIN_LOCAL
        and not is_local_origin_label(far_label)
    ):
        raise refuse(
            f"keep {far_label!r} in the mapping sentence",
            "far_maps_to_lever must still name the distant concept it is "
            "compiling; a lever name without the far noun is a local tweak",
            capability=COMPILE_CAPABILITY,
            failed_far=far_label,
        )
    measurable = {observable.lower(), *(field.lower() for field in menu.attested_fields)}
    pred_l = str(prediction or "").lower()
    tokens = prediction_tokens_for(kind, measurable)
    if not any(token in pred_l for token in tokens if token):
        raise refuse(
            f"predict a runtime observable or attested field for {far_label!r}",
            "prediction must name the compiled observable, an attested "
            "field path, or — for acquire — the object_properties the "
            "harvested world must have",
            capability=COMPILE_CAPABILITY,
            failed_far=far_label,
        )
    return lever, observable, mapping
