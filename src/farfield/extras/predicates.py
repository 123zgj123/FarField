"""Model-written executable predicates, gated before they may kill a card.

This is P4's mechanism, and the point where "AI builds AI" stops being a
slogan: the model's output here is not a hypothesis but a *judgment operator*
-- code that will decide other cards' survival. That is exactly why nothing
the model writes runs as-is. Three layers stand between a proposal and a
kill:

**The compiler is a whitelist, not a sandbox patch.** `compile_predicate`
parses the source and refuses any construct outside a small allowed set: one
function `check(card)`, expressions, `if`/`for`/`return`, comprehensions, and
calls into a fixed namespace (`re`, `math`, and a handful of builtins). No
imports, no `while`, no attribute whose name starts with an underscore, no
exception handling to hide behind. The whitelist is the security model, so a
reader can audit it in one screen; anything clever enough to be unlistable is
refused by construction.

**Execution is budgeted and fallible.** A predicate runs under a line-event
budget; an infinite `for` dies by budget, and any exception is the
predicate's failure rather than the card's. A crashing predicate never kills
-- `run` raises, and callers treat that as evidence against the predicate.

**Installation is earned on history.** `gate_report` runs a candidate over
the archived card bench and a set of known-bad cards. The four gates: it
must kill something (a predicate that passes everything is decoration), it
must spare something (one that kills everything is a shutdown switch), it
must catch the known-bad examples (its stated purpose, demonstrated), and
its verdict must survive a meaning-preserving paraphrase of every bench card
(a judge that kills "presents" but spares "demonstrates" is grading wording
luck, not quality — the first installed generation of model judges died of
exactly this). Gated candidates then pass through `select_marginal`: a
predicate is installed only
if it kills bench cards no already-installed predicate kills, so the
installed set carries no redundant judge. That is the same marginal-gain
discipline the check registry uses, applied to model-authored judges.

The two kernel reference predicates at the bottom are written in the same
restricted language and compiled by the same compiler. They are the
executable form of the P4 brief -- a prediction must name a measurable
quantity, a claim must contain a falsifiable assertion -- and they double as
proof that the language is expressive enough to say something worth saying.
"""

from __future__ import annotations

import ast
import builtins
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

__all__ = [
    "CARD_FIELDS",
    "INSTALLED_PREDICATES",
    "KERNEL_PREDICATES",
    "PARAPHRASES",
    "Predicate",
    "PredicateError",
    "PredicateRefused",
    "compile_predicate",
    "criteria_lines",
    "gate_report",
    "paraphrase",
    "run",
    "select_marginal",
]


class PredicateError(Exception):
    """A predicate failed at runtime: crashed, over budget, or non-boolean."""


class PredicateRefused(Exception):
    """A predicate source was refused at compile time, with the reason."""


# What a predicate may read. A plain dict, not the card object: the judgment
# is over the card's text and its recorded numbers, never over live graph
# handles or anything with methods worth escaping through.
CARD_FIELDS = (
    "claim",
    "mechanism",
    "prediction",
    "falsifier",
    "concept_a",
    "concept_b",
    "alienness",
    "plausibility_signal",
)

LINE_BUDGET = 50_000

_ALLOWED_NODES = frozenset(
    {
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Expr,
        ast.Return,
        ast.Pass,
        ast.If,
        ast.For,
        ast.Break,
        ast.Continue,
        ast.Assign,
        ast.AugAssign,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.Tuple,
        ast.List,
        ast.Set,
        ast.Dict,
        ast.Subscript,
        ast.Slice,
        ast.Attribute,
        ast.Call,
        ast.keyword,
        ast.IfExp,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.USub,
        ast.UAdd,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.ListComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.DictComp,
        ast.comprehension,
        ast.JoinedStr,
        ast.FormattedValue,
    }
)

_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    )
}


def _refuse(reason: str) -> PredicateRefused:
    return PredicateRefused(reason)


def _validate(tree: ast.Module) -> None:
    body = tree.body
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
        raise _refuse("the source must be exactly one function definition")
    fn = body[0]
    if fn.name != "check":
        raise _refuse(f"the function must be named check, not {fn.name!r}")
    args = fn.args
    if (
        [a.arg for a in args.args] != ["card"]
        or args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or args.defaults
    ):
        raise _refuse("the signature must be exactly check(card)")
    if fn.decorator_list:
        raise _refuse("decorators are not allowed")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES and not isinstance(node, ast.expr_context):
            raise _refuse(f"{type(node).__name__} is not in the allowed subset")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise _refuse(f"underscore attribute {node.attr!r} is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise _refuse(f"underscore name {node.id!r} is not allowed")
        if isinstance(node, ast.FunctionDef) and node is not fn:
            raise _refuse("nested functions are not allowed")


@dataclass(frozen=True)
class Predicate:
    """A compiled judgment operator. `check` returns True when the card passes."""

    name: str
    source: str
    provenance: str
    rationale: str = ""
    _fn: Callable[[dict[str, Any]], Any] = field(repr=False, compare=False, default=None)  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "provenance": self.provenance,
            "rationale": self.rationale,
        }


def compile_predicate(
    name: str, source: str, *, provenance: str, rationale: str = ""
) -> Predicate:
    """Whitelist-validate and compile one predicate, or refuse with the reason."""
    if not name or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise _refuse(f"predicate name {name!r} must be a lowercase identifier")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise _refuse(f"not parseable python: {error}") from error
    _validate(tree)
    namespace: dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS),
        "re": re,
        "math": math,
    }
    exec(compile(tree, f"<predicate:{name}>", "exec"), namespace)  # noqa: S102
    return Predicate(
        name=name,
        source=source,
        provenance=provenance,
        rationale=rationale,
        _fn=namespace["check"],
    )


def run(predicate: Predicate, card: dict[str, Any]) -> bool:
    """One verdict under the line budget. True passes, False kills.

    Every failure mode belongs to the predicate: an exception, a blown
    budget, or a non-boolean return all raise `PredicateError`, and the
    caller must treat that as the predicate's problem rather than the
    card's. A card may only die from an explicit False.
    """
    view = {key: card.get(key) for key in CARD_FIELDS}
    spent = 0

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        nonlocal spent
        if event == "line":
            spent += 1
            if spent > LINE_BUDGET:
                raise PredicateError(
                    f"{predicate.name} exceeded the {LINE_BUDGET}-line budget"
                )
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        verdict = predicate._fn(view)
    except PredicateError:
        raise
    except Exception as error:
        raise PredicateError(f"{predicate.name} crashed: {error!r}") from error
    finally:
        sys.settrace(previous)
    if not isinstance(verdict, bool):
        raise PredicateError(
            f"{predicate.name} returned {type(verdict).__name__}, not bool"
        )
    return verdict


# Meaning-preserving rewrites, applied to the card's free-text fields.
# Each right-hand side says the same thing as the left in this genre of
# prose; a verdict that flips under one of these substitutions was reading
# vocabulary, not content. The list is deliberately short and auditable —
# every pair must be defensible as a synonym to a human reader — and it is
# pinned here because the installation gate's meaning must replay.
#
# The phrase pairs at the top exist because the second installed generation
# found the same disease in a different organ: a judge robust to verb swaps
# still killed "a later paper will report" while sparing "later work will
# report". Markers of futurity are wording too.
PARAPHRASES: tuple[tuple[str, str], ...] = (
    ("later work", "later paper"),
    ("subsequent work", "subsequent paper"),
    ("future work", "future papers"),
    ("demonstrates", "presents"),
    ("demonstrate", "present"),
    ("shows", "reveals"),
    ("show", "reveal"),
    ("yields", "delivers"),
    ("yield", "deliver"),
    ("reports", "documents"),
    ("report", "document"),
    ("proves", "certifies"),
    ("prove", "certify"),
    ("enables", "lets"),
    ("enable", "let"),
    ("exhibits", "displays"),
    ("exhibit", "display"),
    ("constructs", "builds"),
    ("construct", "build"),
)

_PARAPHRASED_FIELDS = ("claim", "mechanism", "prediction")


def paraphrase(card: dict[str, Any]) -> dict[str, Any]:
    """The same card said with different verbs.

    Word-boundary, case-insensitive, deterministic. Only the free-text
    fields are rewritten; the pair, the falsifier and the recorded numbers
    are facts, not phrasing.
    """
    rewritten = dict(card)
    for field_name in _PARAPHRASED_FIELDS:
        text = str(card.get(field_name) or "")
        for old, new in PARAPHRASES:
            text = re.sub(rf"\b{old}\b", new, text, flags=re.IGNORECASE)
        rewritten[field_name] = text
    return rewritten


def criteria_lines(predicates: Sequence[Predicate]) -> tuple[str, ...]:
    """The installed judges' standards, one line each, for the generator.

    This is the feedback loop's text: a scientist knows the review criteria
    before submitting, and a generator that is judged by executable
    predicates deserves to read what they demand.
    """
    return tuple(
        f"{predicate.name}: {predicate.rationale}"
        if predicate.rationale
        else predicate.name
        for predicate in predicates
    )


def gate_report(
    predicate: Predicate,
    bench: Sequence[dict[str, Any]],
    known_bad: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """The four installation gates, measured rather than asserted.

    `bench` is the archived history: every card the system has generated,
    survivors and killed alike. `known_bad` is a curated set of cards that a
    working text predicate has no excuse to pass. Any runtime failure on any
    input closes all gates: a judge that crashes on history cannot be trusted
    with the future. The paraphrase gate reruns every card through the same
    predicate with its verbs synonym-swapped and demands the same verdict:
    judging wording luck is the one failure mode history alone cannot show.
    """
    kills: list[int] = []
    errors: list[str] = []
    flips: list[int] = []
    for index, card in enumerate(bench):
        try:
            verdict = run(predicate, card)
            if not verdict:
                kills.append(index)
            if run(predicate, paraphrase(card)) != verdict:
                flips.append(index)
        except PredicateError as error:
            errors.append(str(error))
    bad_caught = 0
    for card in known_bad:
        try:
            verdict = run(predicate, card)
            if not verdict:
                bad_caught += 1
            if run(predicate, paraphrase(card)) != verdict:
                flips.append(-1)
        except PredicateError as error:
            errors.append(str(error))
    gates = {
        "fires_on_history": bool(kills),
        "spares_some_history": len(kills) < len(bench),
        "catches_known_bad": bool(known_bad) and bad_caught == len(known_bad),
        "never_errors": not errors,
        "verdict_survives_paraphrase": not flips,
    }
    return {
        "predicate": predicate.name,
        "provenance": predicate.provenance,
        "bench_size": len(bench),
        "bench_kills": kills,
        "bench_kill_rate": len(kills) / len(bench) if bench else 0.0,
        "known_bad": len(known_bad),
        "known_bad_caught": bad_caught,
        "paraphrase_flips": len(flips),
        "errors": errors[:5],
        "gates": gates,
        "installable": all(gates.values()),
    }


def select_marginal(
    reports: Iterable[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Greedy marginal-gain selection over gated candidates.

    A predicate enters the installed set only if it kills at least one bench
    card nobody already installed kills. Redundant judges are refused even
    when individually sound, because every installed predicate is one more
    thing a future reader must audit, and a judge that never changes a
    verdict pays no rent. Deterministic: gain first, smaller kill set second
    (the sharper judge wins ties), name last.
    """
    candidates = [dict(report) for report in reports if report["installable"]]
    installed: list[str] = []
    log: list[dict[str, Any]] = []
    covered: set[int] = set()
    while candidates:
        best = None
        best_key = None
        for report in candidates:
            gain = len(set(report["bench_kills"]) - covered)
            key = (-gain, len(report["bench_kills"]), report["predicate"])
            if best_key is None or key < best_key:
                best, best_key = report, key
        gain = len(set(best["bench_kills"]) - covered)
        if gain == 0:
            for report in sorted(candidates, key=lambda r: r["predicate"]):
                log.append(
                    {
                        "predicate": report["predicate"],
                        "installed": False,
                        "reason": "no marginal kill: every bench card it kills"
                        " is already killed by the installed set",
                    }
                )
            break
        covered |= set(best["bench_kills"])
        installed.append(best["predicate"])
        log.append(
            {
                "predicate": best["predicate"],
                "installed": True,
                "marginal_kills": gain,
                "covered_after": len(covered),
            }
        )
        candidates.remove(best)
    return installed, log


# The kernel's own text predicates: the P4 brief, in executable form. They go
# through the same compiler and the same gates as any model proposal, so the
# experiment can report them refused if the history says they over- or
# under-kill.
_MEASURABLE_SOURCE = '''
def check(card):
    """The prediction must name a measurable quantity, not just a direction.

    A prediction that cannot say what number would move is not a prediction,
    it is a mood. Measurable here means: a digit, a percentage, an asymptotic
    form, or one of the quantities this corpus family actually measures.
    """
    text = str(card["prediction"] or "").lower()
    if re.search(r"\\d", text) or "%" in text:
        return True
    quantity = re.search(
        r"\\b(accuracy|error rate|error bound|running time|runtime|query time|"
        r"update time|space usage|approximation (ratio|factor)|competitive ratio|"
        r"regret|sample complexity|communication complexity|round complexity|"
        r"convergence rate|success probability|failure probability|perplexity|"
        r"f1|bleu|recall|precision|throughput|latency|treewidth|degree|diameter|"
        r"entropy|variance|distortion|stretch|congestion|depth|width)\\b",
        text,
    )
    asymptotic = re.search(r"\\b(log|polylog|poly|exp)\\b|o\\(|omega\\(|theta\\(", text)
    return bool(quantity or asymptotic)
'''

_FALSIFIABLE_SOURCE = '''
def check(card):
    """The claim must contain an assertion that some observation could refute.

    The tell is a committed relation: an explicit bound, a comparison, an
    impossibility, an equivalence. A claim built entirely from "may", "could
    help" and "is related to" survives every possible world, and a card that
    survives every world says nothing about this one.
    """
    text = str(card["claim"] or "").lower()
    hedged = re.search(
        r"\\b(may|might|could|can potentially|possibly|perhaps)\\b", text
    )
    committed = re.search(
        r"\\b(at most|at least|no more than|no less than|strictly"
        r" (smaller|larger|less|greater|better|worse)|faster than|slower than|"
        r"impossible|cannot|no (\\w+ ){0,2}(algorithm|structure|model|method)"
        r" can|never|always|if and only if|equivalent to|(lower|upper) bound|"
        r"tight|necessary and sufficient|"
        r"outperforms?|dominates?|reduces? to|collapses? to|implies)\\b",
        text,
    ) or re.search(r"\\d|%|o\\(|omega\\(|theta\\(", text)
    if not committed:
        return False
    if hedged and not re.search(r"\\d|%", text):
        return False
    return True
'''

KERNEL_PREDICATES = (
    compile_predicate(
        "prediction_names_a_measurable_quantity",
        _MEASURABLE_SOURCE,
        provenance="kernel",
        rationale="a prediction with no quantity cannot be wrong, so it cannot be right",
    ),
    compile_predicate(
        "claim_contains_a_falsifiable_assertion",
        _FALSIFIABLE_SOURCE,
        provenance="kernel",
        rationale="a claim hedged out of every possible refutation asserts nothing",
    ),
)

_VAGUE_MECHANISM_SOURCE = '''
def check(card):
    claim = str(card['claim'] or '')
    mech = str(card['mechanism'] or '')
    mwords = len(mech.split())
    if mwords < 15:
        return False
    che = claim.lower()
    if any(p in che for p in ['may be related', 'could potentially', 'interesting ways', 'worth investigating', 'promising connection']):
        return False
    m = mech.lower()
    if mwords < 30:
        causal_tokens = {'because', 'by', 'via', 'through', 'if', 'then', 'thus', 'therefore', 'hence', 'so'}
        if not any(t in causal_tokens for t in m.split()):
            return False
    return True
'''

_GROUNDED_CONCEPT_SOURCE = '''
def check(card):
    claim = str(card.get("claim") or "").strip().lower()
    a = str(card.get("concept_a") or "").strip().lower()
    b = str(card.get("concept_b") or "").strip().lower()
    if not claim or not a or not b:
        return False
    for concept in (a, b):
        found = False
        if concept in claim:
            found = True
        else:
            singular = concept[:-1] if concept.endswith("s") else concept
            plural = concept + "s"
            if singular in claim or plural in claim:
                found = True
            else:
                claim_parts = re.split(r"[^a-z0-9]+", claim)
                claim_roots = set()
                for token in claim_parts:
                    if len(token) >= 4:
                        claim_roots.add(token)
                        if token.endswith("s"):
                            claim_roots.add(token[:-1])
                        if token.endswith("es"):
                            claim_roots.add(token[:-2])
                        if token.endswith("ies"):
                            claim_roots.add(token[:-3] + "y")
                concept_parts = re.split(r"[^a-z0-9]+", concept)
                for token in concept_parts:
                    if len(token) >= 4:
                        cand = set()
                        cand.add(token)
                        if token.endswith("s"):
                            cand.add(token[:-1])
                        if token.endswith("es"):
                            cand.add(token[:-2])
                        if token.endswith("ies"):
                            cand.add(token[:-3] + "y")
                        for root in cand:
                            if root in claim_roots:
                                found = True
                                break
                    if found:
                        break
        if not found:
            return False
    return len(claim) - (len(a) + len(b)) >= 60
'''

INSTALLED_PREDICATES = KERNEL_PREDICATES + (
    compile_predicate(
        "kills_vague_or_underspecified_mechanism",
        _VAGUE_MECHANISM_SOURCE,
        provenance="model:74933ce7946a",
        rationale=(
            "Rejects mechanisms too short to state a causal path, "
            "hedged claims, and short mechanisms lacking causal connectors."
        ),
    ),
    compile_predicate(
        "grounded_concept_elaboration",
        _GROUNDED_CONCEPT_SOURCE,
        provenance="model:b77ff3a4ba0d",
        rationale=(
            "Rejects claims that only restate the two concept labels rather "
            "than grounding them in a substantive assertion; also requires "
            "both named concepts or clear substrings to appear."
        ),
    ),
)
