"""Compile retrieved papers into the neighbourhood research state.

Polaris keeps a research wiki: one paper, one page, shared across the
lab. FarField's survey used to die with the mission, so the next
`generate_card` re-searched and prior-check could not see a restatement
found last time. This module compiles attested `FreshWork` rows into
the digested state. No LLM. Only `verified` rows may be cited;
`unverified` / `verify_pending` stay out. Legacy rows that predate the
field remain citable. Citations in a packet remain this-mission only;
the wiki is memory for generation and prior support, not a license to
cite unread papers. Title and limitation phrases feed
`researchspace.recover_trajectories` so a distant mechanism is a step
on a recovered development line, not a node of the concept graph.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from .livefeed import FreshWork, as_work
from .prior import APPLY_PAD, claim_phrases, content_tokens, fold_plural

WIKI_CAP = 40
PROMPT_CAP = 8
ABSTRACT_CAP = 800
GAP_CAP = 8
GAP_OVERLAP = 2
LIT_FAR_CAP = 6
# Bibliographic and reporting words. A title like "A very fresh preprint"
# is not a mechanism. Method bigrams (`shadow replay`) survive because
# they still have a technical head after this pad is removed.
_GENERIC_FAR = frozenset(
    {
        "analysis",
        "article",
        "case",
        "fresh",
        "note",
        "notes",
        "paper",
        "papers",
        "preprint",
        "report",
        "review",
        "study",
        "studies",
        "survey",
        "very",
    }
)
_CITE_PREFIX = re.compile(r"^\[.*?\]\s*")
_GAP_PAD = frozenset(
    {
        "however",
        "limitation",
        "future",
        "remain",
        "unclear",
        "unsolved",
        "cannot",
        "address",
        "consider",
        "leave",
        "without",
        "open",
        "problem",
    }
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_LIMIT = (
    "however",
    "limitation",
    "we do not",
    "we cannot",
    "future work",
    "open problem",
    "remains unclear",
    "not address",
    "do not consider",
    "left as",
    "unsolved",
    "fails to",
    "without evaluating",
    "we leave",
    "research gap",
    "gap remains",
    "remains open",
    "remain open",
    "remains an open",
    "may not generalize",
    "may not generalise",
    "unfortunately",
    "impossible in practice",
    "does not expose",
    "not yet",
)


def limitation_sentences(text: str, *, cap: int = 3) -> list[str]:
    """Deterministic gaps: sentences that name a limitation, not an LLM summary."""
    parts = [piece.strip() for piece in _SENTENCE.split(text or "") if piece.strip()]
    hits: list[str] = []
    seen: set[str] = set()
    for sentence in parts:
        lower = sentence.lower()
        if not any(marker in lower for marker in _LIMIT):
            continue
        key = " ".join(sentence.split())
        if key in seen:
            continue
        seen.add(key)
        hits.append(key[:240])
        if len(hits) >= cap:
            break
    return hits


def compile_entry(
    work: FreshWork,
    *,
    pair: tuple[str, str] | list[str] | None = None,
    seen_at: str = "",
) -> dict[str, Any]:
    cite = str(work.cite_id() or "").strip()
    title = str(work.title or "").strip()
    abstract = str(work.abstract or "")[:ABSTRACT_CAP]
    return {
        "cite_id": cite,
        "title": title,
        "published": str(work.published or "")[:10],
        "source": str(work.source or "arxiv"),
        "abstract": abstract,
        "url": str(work.href() or ""),
        "venue": str(work.venue or ""),
        "first_seen": str(seen_at or "")[:10],
        "last_seen": str(seen_at or "")[:10],
        "pairs": [list(pair)[:2]] if pair and len(list(pair)) >= 2 else [],
        "limitations": limitation_sentences(abstract),
        "references": list(work.references)[:48],
        "verified": True,
        "verify_layer": "admitted",
    }


def merge_entry(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Same cite_id: keep identity, extend pairs, prefer the longer abstract."""
    pairs: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for row in list(old.get("pairs") or []) + list(new.get("pairs") or []):
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        key = (str(row[0]), str(row[1]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append([key[0], key[1]])
        if len(pairs) >= 6:
            break
    old_abs = str(old.get("abstract") or "")
    new_abs = str(new.get("abstract") or "")
    first = str(old.get("first_seen") or "") or str(new.get("first_seen") or "")
    last = str(new.get("last_seen") or "") or str(old.get("last_seen") or "")
    verified = _row_verified(old) or _row_verified(new)
    layer = ""
    if _row_verified(new):
        layer = str(new.get("verify_layer") or "")
    if not layer:
        layer = str(old.get("verify_layer") or "")
    return {
        "cite_id": str(old.get("cite_id") or new.get("cite_id") or ""),
        "title": str(new.get("title") or old.get("title") or ""),
        "published": str(old.get("published") or new.get("published") or "")[:10],
        "source": str(old.get("source") or new.get("source") or "arxiv"),
        "abstract": (new_abs if len(new_abs) >= len(old_abs) else old_abs)[:ABSTRACT_CAP],
        "url": str(new.get("url") or old.get("url") or ""),
        "venue": str(new.get("venue") or old.get("venue") or ""),
        "first_seen": first[:10],
        "last_seen": last[:10],
        "pairs": pairs,
        "limitations": _merge_limitations(old, new),
        "references": _merge_references(old, new),
        "verified": verified,
        "verify_layer": layer or ("admitted" if verified else ""),
    }


def _row_verified(row: dict[str, Any]) -> bool:
    """Legacy wiki rows (no field) stay citable. Explicit False does not."""
    return row.get("verified") is not False


def _merge_references(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(old.get("references") or []) + list(new.get("references") or []):
        key = " ".join(str(item or "").split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= 48:
            break
    return out


def _merge_limitations(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(old.get("limitations") or []) + list(new.get("limitations") or []):
        text = " ".join(str(item or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text[:240])
        if len(out) >= 4:
            break
    return out


def key_of(entry: dict[str, Any]) -> str:
    return str(entry.get("cite_id") or entry.get("title") or "").strip()


def upsert(
    wiki: list[dict[str, Any]], entries: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Insert or merge; evict least-recently seen past WIKI_CAP. Returns (wiki, added)."""
    by_key: dict[str, dict[str, Any]] = {}
    for row in wiki:
        if isinstance(row, dict) and key_of(row):
            by_key[key_of(row)] = dict(row)
    added = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = key_of(entry)
        if not key:
            continue
        if key in by_key:
            by_key[key] = merge_entry(by_key[key], entry)
        else:
            by_key[key] = dict(entry)
            added += 1
    ranked = sorted(
        by_key.values(),
        key=lambda row: str(row.get("last_seen") or row.get("first_seen") or ""),
        reverse=True,
    )
    return ranked[:WIKI_CAP], added


def as_works(wiki: Iterable[dict[str, Any]]) -> list[FreshWork]:
    works: list[FreshWork] = []
    for row in wiki:
        if not isinstance(row, dict):
            continue
        if row.get("verified") is False:
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        works.append(
            FreshWork(
                title=title,
                published=str(row.get("published") or ""),
                arxiv_id=str(row.get("cite_id") or "") if row.get("source") == "arxiv" else "",
                abstract=str(row.get("abstract") or ""),
                source=str(row.get("source") or "arxiv"),
                work_id="" if row.get("source") == "arxiv" else str(row.get("cite_id") or ""),
                url=str(row.get("url") or ""),
                venue=str(row.get("venue") or ""),
                references=tuple(
                    str(item).strip()
                    for item in (row.get("references") or ())
                    if str(item).strip()
                ),
            )
        )
    return works


def merge_works(
    live: Iterable[FreshWork], wiki: Iterable[FreshWork]
) -> list[FreshWork]:
    """Live survey wins on the same cite_id; wiki fills papers this mission did not pull."""
    by_id: dict[str, FreshWork] = {}
    for raw in wiki or []:
        work = as_work(raw)
        if work is None:
            continue
        key = work.cite_id() or work.title
        if key:
            by_id[key] = work
    for raw in live or []:
        work = as_work(raw)
        if work is None:
            continue
        key = work.cite_id() or work.title
        if key:
            by_id[key] = work
    return list(by_id.values())


def aimed_gap(
    claim: str,
    mechanism: str,
    gaps: Sequence[str],
    *,
    min_overlap: int = GAP_OVERLAP,
) -> str:
    """First limitation sentence the card aims at, or empty.

    Overlap is distinctive content words, not span support: hitting a gap
    is not a license to declare it solved. Prior-kill still closes a
    restatement. Empty when nothing distinctive overlaps.
    """
    body = (content_tokens(claim) | content_tokens(mechanism)) - APPLY_PAD - _GAP_PAD
    if not body:
        return ""
    for raw in gaps:
        text = _CITE_PREFIX.sub("", str(raw or "")).strip()
        if not text:
            continue
        overlap = body & (content_tokens(text) - APPLY_PAD - _GAP_PAD)
        if len(overlap) >= min_overlap:
            return text
    return ""


def _wiki_row(item: Any) -> dict[str, Any] | None:
    """Accept a wiki dict or a FreshWork. Unverified rows do not contribute."""
    if isinstance(item, FreshWork):
        return {
            "title": item.title,
            "abstract": item.abstract,
            "limitations": limitation_sentences(item.abstract),
            "references": list(item.references),
            "verified": True,
        }
    if not isinstance(item, dict) or item.get("verified") is False:
        return None
    return item


def _mechanism_token_set(phrase: str) -> set[str]:
    return {token for token in content_tokens(phrase) if len(token) >= 3}


def _generic_mechanism(phrase: str) -> bool:
    """True when every content word is bibliographic pad or APPLY glue."""
    tokens = _mechanism_token_set(phrase)
    if not tokens:
        return True
    return tokens <= (_GENERIC_FAR | APPLY_PAD)


def _phrase_menu(
    wiki: Iterable[Any],
    topic: str,
    *,
    source: str,
    cap: int,
) -> tuple[str, ...]:
    """Compile a far menu from attested paper text. `source` selects blobs."""
    from .domain import folded_terms, topic_object_phrases, topic_terms

    blocked = set(topic_object_phrases(topic))
    blocked |= {word for word in topic_terms(topic) if len(word) >= 3}
    seen: set[str] = set()
    phrases: list[str] = []
    take_title = source in {"all", "titles"}
    take_limits = source in {"all", "limitations"}
    for raw in wiki or []:
        row = _wiki_row(raw)
        if row is None:
            continue
        if topic.strip() and not _row_names_topic(row, topic):
            continue
        blobs: list[str] = []
        if take_title:
            blobs.append(str(row.get("title") or ""))
        if take_limits:
            blobs.extend(str(item) for item in (row.get("limitations") or []) if item)
        for blob in blobs:
            for gram in claim_phrases(blob):
                if " " not in gram or gram in blocked or gram in seen:
                    continue
                tokens = folded_terms(gram)
                if tokens <= blocked or _generic_mechanism(gram):
                    continue
                if _clause_fragment(gram):
                    continue
                seen.add(gram)
                phrases.append(gram)
                if len(phrases) >= cap:
                    return tuple(phrases)
    return tuple(phrases)


def mechanism_phrases(
    wiki: Iterable[Any],
    topic: str,
    *,
    cap: int = LIT_FAR_CAP,
) -> tuple[str, ...]:
    """Distant mechanism labels from verified papers, not the concept graph.

    Titles and limitation sentences are the only attested text, and only
    from papers that name the topic object. Topic object phrases are
    stripped so the far menu cannot rename the field to a paper's own
    object. A unigram, a bibliographic pad (`fresh preprint`), or a
    phrase whose every word is APPLY glue is dropped. A title bigram the
    abstract only ever capitalises is a system's name, not a mechanism,
    and is dropped too (`described_methods`). Empty wiki is an empty
    menu — this does not invent mechanisms.
    """
    seen: set[str] = set()
    phrases: list[str] = []
    for raw in wiki or []:
        row = _wiki_row(raw)
        if row is None:
            continue
        if topic.strip() and not _row_names_topic(row, topic):
            continue
        described = described_methods(
            str(row.get("title") or ""),
            str(row.get("abstract") or ""),
            topic,
            cap=cap,
        )
        gaps = _phrase_menu([row], topic, source="limitations", cap=cap)
        for gram in (*described, *gaps):
            if gram in seen:
                continue
            seen.add(gram)
            phrases.append(gram)
            if len(phrases) >= cap:
                return tuple(phrases)
    return tuple(phrases)


def title_phrases(
    wiki: Iterable[Any],
    topic: str,
    *,
    cap: int = LIT_FAR_CAP,
) -> tuple[str, ...]:
    """Method bigrams from verified titles. One agent-ml-style lens."""
    return _phrase_menu(wiki, topic, source="titles", cap=cap)


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_METHOD_SENTENCE = re.compile(
    r"\b(?:we|this (?:paper|work|article))\b[^.!?]{0,60}?\b"
    r"(?:propose|introduce|present|develop|build|design|train|mid-train|"
    r"pretrain|fine-tune|derive|formulate|implement|combine|use|leverage|"
    r"apply|learn|fit|compile|evaluate|perform|release)\b",
    re.IGNORECASE,
)
# The means clause of a method sentence: "… through counterexample-guided
# code repair", "… validates each change using coding benchmarks", "… by
# combining X with the generative priors of video models". This is where
# a paper says *how*, as opposed to the name it gives the system.
_MEANS = re.compile(
    r"\b(?:through|via|by|using|with|on top of|based on|"
    r"(?:mid-)?train\w*\s+\S+\s+on)\s+([^,.;:()]{3,80})",
    re.IGNORECASE,
)
# The object of a proposing verb, after an optional Name: "introduce the
# Darwin Gödel Machine (DGM), a self-improving system that …".
_PROPOSED = re.compile(
    r"\b(?:propose|introduce|present|develop|build|design)\b\s+"
    r"(?:(?:the|a|an)\s+)?(?:[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*\s*(?:\([^)]*\))?\s*[,:]\s*)?"
    r"(?:(?:the|a|an)\s+)?([^,.;:()]{3,80})",
)


def _fold_word(word: str) -> str:
    return fold_plural(word.lower())


# Words that make an n-gram a clause fragment rather than a mechanism
# (`system that`, `its own`, `before acting`, `large amount`), plus the
# verbs a paper uses to announce itself (`introduce code`). They are not
# in APPLY_PAD because prior-support recall must still see them.
_MENU_STOP = frozenset(
    {
        "able", "after", "again", "all", "already", "also", "always", "amount",
        "any", "are", "available", "been", "before", "being",
        "both", "can", "could", "demonstrate", "does", "each", "either",
        "examine", "first", "further", "had", "has", "have", "how", "its",
        "introduce", "itself", "large", "many", "may", "more", "most",
        "must", "not", "number", "only", "organize", "own", "present",
        "propose", "provide", "release", "second", "several", "show", "still",
        "study", "such", "than", "that", "their", "them", "themselves",
        "then", "there", "these", "they", "third", "those", "thereby",
        "various", "very", "was", "well", "were", "what", "when", "where",
        "whether", "which", "while", "whose", "will", "would", "yet",
    }
)


def _clause_fragment(gram: str) -> bool:
    return any(token in _MENU_STOP for token in str(gram or "").split())


def surface_form(phrase: str, text: str) -> str | None:
    """The lowercase words `text` actually wrote for a folded n-gram.

    Menus should read `iteratively modifies its own code`, not the
    folding key `iteratively modifie`. Returns None when no contiguous
    occurrence of the phrase is written in lowercase, which is exactly
    the case of a system Name (`Darwin Gödel Machine`, `ARC-AGI`,
    `Live-SWE-agent`) as opposed to a description (`validator dropout`).
    """
    tokens = [_fold_word(item) for item in str(phrase or "").split() if item]
    if not tokens:
        return None
    words = _WORD.findall(str(text or ""))
    span = len(tokens)
    for start in range(0, len(words) - span + 1):
        window = words[start : start + span]
        if [_fold_word(item) for item in window] != tokens:
            continue
        if all(item[:1].islower() for item in window):
            return " ".join(item.lower() for item in window)
    return None


def lowercase_attested(phrase: str, text: str) -> bool:
    """True when `text` writes the phrase as ordinary words, not as a name."""
    return surface_form(phrase, text) is not None


def method_sentences(text: str, *, cap: int = 3) -> list[str]:
    """Sentences where the paper says what it does (`we propose ...`)."""
    parts = [piece.strip() for piece in _SENTENCE.split(text or "") if piece.strip()]
    hits: list[str] = []
    for sentence in parts:
        if _METHOD_SENTENCE.search(sentence):
            hits.append(sentence)
            if len(hits) >= cap:
                break
    return hits


def described_methods(
    title: str,
    abstract: str,
    topic: str,
    *,
    cap: int = LIT_FAR_CAP,
) -> tuple[str, ...]:
    """Mechanisms a paper *describes*, not systems it *names*.

    The cwm-iclr2027 mission offered `darwin godel`, `arc agi`,
    `live swe`, `open weight` as trajectory mechanisms: title bigrams
    that are proper names or release settings, so the cards read "a
    Darwin Gödel harness" and the probe had no lever to compile.

    Order of authority, all under the same topic-object and pad filters
    as `title_phrases`, and all required to be written in lowercase by
    the abstract (`lowercase_attested`):

    1. the means clause of a `we propose/introduce ...` sentence
       (`through counterexample-guided code repair`);
    2. the object of the proposing verb after its Name is skipped
       (`a self-improving system that modifies its own code`);
    3. title bigrams the abstract also writes as ordinary words.

    A row without an abstract has only its title, and keeps title
    phrases that are not capitalised inside the title itself.
    """
    title = str(title or "")
    abstract = str(abstract or "")
    rows = [{"title": title, "abstract": abstract, "verified": True}]
    title_candidates = list(_phrase_menu(rows, topic, source="titles", cap=cap * 4))
    kept: list[str] = []
    if not abstract.strip():
        # No prose to check against: refuse only phrases the title itself
        # writes as a Name (mid-title capital after the first word).
        for gram in title_candidates:
            if _named_in_title(gram, title):
                continue
            kept.append(gram)
        return tuple(kept[:cap])

    keys: set[str] = set()

    def _admit(gram: str) -> bool:
        written = surface_form(gram, abstract)
        if written is None or gram in keys or written in kept:
            return False
        keys.add(gram)
        kept.append(written)
        return len(kept) >= cap

    def _take(spans: Iterable[str]) -> None:
        span_rows = [
            {"title": span, "abstract": abstract, "verified": True}
            for span in spans
            if span.strip()
        ]
        if not span_rows:
            return
        for gram in _span_phrases(span_rows, topic, cap=cap * 4):
            if _admit(gram):
                return

    sentences = method_sentences(abstract)
    _take(match.group(1) for sentence in sentences for match in _MEANS.finditer(sentence))
    if len(kept) < cap:
        _take(match.group(1) for sentence in sentences for match in _PROPOSED.finditer(sentence))
    if len(kept) < cap:
        for gram in title_candidates:
            if _admit(gram):
                break
    return tuple(kept[:cap])


def _span_phrases(
    rows: list[dict[str, Any]],
    topic: str,
    *,
    cap: int,
) -> tuple[str, ...]:
    """Longest-first phrases of short spans, then the title bigram menu.

    A means clause of two to four content words (`counterexample guided
    code repair`) is one mechanism, not three overlapping bigrams; it is
    offered whole first, then the usual bigram/trigram menu fills in.
    """
    from .domain import folded_terms, topic_object_phrases, topic_terms
    from .prior import ordered_content_runs

    blocked = set(topic_object_phrases(topic))
    blocked |= {word for word in topic_terms(topic) if len(word) >= 3}
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for run in ordered_content_runs(str(row.get("title") or "")):
            if not 2 <= len(run) <= 4:
                continue
            gram = " ".join(run)
            if gram in seen or gram in blocked:
                continue
            if folded_terms(gram) <= blocked or _generic_mechanism(gram):
                continue
            if _clause_fragment(gram):
                continue
            seen.add(gram)
            out.append(gram)
    for gram in _phrase_menu(rows, topic, source="titles", cap=cap):
        if gram not in seen:
            seen.add(gram)
            out.append(gram)
    return tuple(out[:cap])


# The object of a limitation sentence: "we do not support dynamic
# inserts", "does not expose a ground-truth latent state", "require
# costly offline training", "a research gap remains in how to induce
# executable code as a world model".
_GAP_OBJECT = re.compile(
    r"\b(?:do not|does not|did not|cannot|can not|could not|without|not|"
    r"lack(?:s|ing)?|fails? to|requires?|nor|no|"
    r"remains? (?:in )?how to|unclear how to|impossible to|"
    r"open (?:problem|question)(?: of| is)?)\s+([^,.;:()]{3,80})",
    re.IGNORECASE,
)
_GAP_STOP = frozenset(
    {
        "however", "unfortunately", "typically", "proving", "remain",
        "remains", "require", "requires", "required", "return", "returns",
        "expose", "exposes", "support", "supports", "evaluate", "evaluated",
        "consider", "considered", "address", "addressed", "test", "tested",
        "generalize", "generalise", "practice",
    }
)


def described_gaps(
    limitations: Iterable[str],
    topic: str,
    *,
    abstract: str = "",
    cap: int = LIT_FAR_CAP,
) -> tuple[str, ...]:
    """What a paper says it does *not* do, as the object of the sentence.

    `we do not support dynamic inserts` → `dynamic inserts`, never
    `support dynamic`; `unfortunately, proving that most changes are net
    beneficial is impossible in practice` → `net beneficial`, never
    `unfortunately proving`. Objects of the limitation marker come first,
    then the bigram menu of the sentence with clause words removed. All
    phrases are the lowercase words the sentence wrote.
    """
    sentences = [str(item).strip() for item in (limitations or ()) if str(item).strip()]
    if not sentences:
        return ()
    # The on-object check reads title+abstract; the paper already passed
    # it, so the paper's abstract stands in for a bare sentence.
    context = str(abstract or "") or " ".join(sentences)
    kept: list[str] = []
    keys: set[str] = set()

    def _admit(gram: str, sentence: str) -> bool:
        if any(token in _GAP_STOP for token in gram.split()):
            return False
        written = surface_form(gram, sentence)
        if written is None or gram in keys or written in kept:
            return False
        keys.add(gram)
        kept.append(written)
        return len(kept) >= cap

    for sentence in sentences:
        spans = [match.group(1) for match in _GAP_OBJECT.finditer(sentence)]
        rows = [{"title": span, "abstract": context, "verified": True} for span in spans]
        if rows:
            for gram in _span_phrases(rows, topic, cap=cap * 4):
                if _admit(gram, sentence):
                    return tuple(kept)
    for sentence in sentences:
        rows = [{"title": sentence, "abstract": context, "verified": True}]
        for gram in _phrase_menu(rows, topic, source="titles", cap=cap * 4):
            if _admit(gram, sentence):
                return tuple(kept)
    return tuple(kept[:cap])


def _named_in_title(phrase: str, title: str) -> bool:
    """A phrase the title writes with a mid-sentence capital is a Name."""
    tokens = [_fold_word(item) for item in str(phrase or "").split() if item]
    words = _WORD.findall(str(title or ""))
    span = len(tokens)
    for start in range(0, len(words) - span + 1):
        window = words[start : start + span]
        if [_fold_word(item) for item in window] != tokens:
            continue
        if start == 0:
            window = window[1:]
        if all(item[:1].islower() for item in window):
            return False
    return True


def limitation_phrases(
    wiki: Iterable[Any],
    topic: str,
    *,
    cap: int = LIT_FAR_CAP,
) -> tuple[str, ...]:
    """Failure-mode bigrams from limitation sentences. Another lens.

    ARIS feeds idea-creator from wiki Limitations; agent-ml's failure
    analysis searches SOTA breakdowns. This is that menu, compiled,
    not a second generate personality.
    """
    return _phrase_menu(wiki, topic, source="limitations", cap=cap)


def _row_names_topic(row: dict[str, Any], topic: str) -> bool:
    if not str(topic or "").strip():
        return True
    from .worldfields import paper_on_claim_object

    return paper_on_claim_object(
        str(row.get("title") or ""),
        str(row.get("abstract") or ""),
        topic,
        topic,
    )


def gap_lines(wiki: Iterable[dict[str, Any]], topic: str = "") -> tuple[str, ...]:
    """Limitation sentences from on-claim wiki rows, for agenda targets.

    Noise papers stay in the wiki for honesty; they do not become jump
    targets. Empty `topic` keeps the legacy unfiltered menu so cached
    prompt tests stay byte-identical.
    """
    lines: list[str] = []
    for row in wiki:
        if not isinstance(row, dict) or row.get("verified") is False:
            continue
        if not _row_names_topic(row, topic):
            continue
        cite = str(row.get("cite_id") or row.get("title") or "").strip()
        for item in row.get("limitations") or []:
            text = " ".join(str(item or "").split())
            if not text:
                continue
            head = f"[{cite}] {text}" if cite else text
            if head not in lines:
                lines.append(head)
            if len(lines) >= GAP_CAP:
                return tuple(lines)
    return tuple(lines)


def prompt_lines(wiki: Iterable[dict[str, Any]], topic: str = "") -> tuple[str, ...]:
    """Shown to the generator. Empty keeps cached prompts byte-identical."""
    rows = [
        row
        for row in wiki
        if isinstance(row, dict) and row.get("verified") is not False
        and _row_names_topic(row, topic)
    ]
    lines: list[str] = []
    for row in rows[:PROMPT_CAP]:
        cite = str(row.get("cite_id") or "").strip()
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        year = str(row.get("published") or "")[:4]
        snippet = " ".join(str(row.get("abstract") or "").split())[:160]
        head = f"[{cite}] {title}" if cite else title
        if year:
            head += f" ({year})"
        if snippet:
            head += f" — {snippet}"
        lines.append(head)
    for gap in gap_lines(rows, topic=topic)[:4]:
        lines.append(
            "Literature gap (not a license to declare it solved): " + gap
        )
    return tuple(lines)


def block(lines: tuple[str, ...]) -> str:
    if not lines:
        return ""
    return (
        "Literature wiki for this neighbourhood (papers earlier missions "
        "actually retrieved; not evidence and not a license to restate them "
        "as new). Limitation sentences are unsolved agenda, not results "
        "you may declare solved. A claim already supported by a span in "
        "one of these papers is closed_by_prior, not a discovery:\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n"
    )
