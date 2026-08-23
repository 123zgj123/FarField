"""Standard-library literature fetch. Tests must pass a fixture and never hit the network.

Offsets are byte offsets into the stored response bytes, and `verify_span`
re-reads those bytes. A span of `0 .. len(text)` over an agent-authored summary
attests nothing, because there is no slice that could fail to match.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from xml.etree import ElementTree

from farfield.models import AttestedSpan, new_id

ARXIV_ATOM = "http://www.w3.org/2005/Atom"
ARXIV_API = "https://export.arxiv.org/api/query"
MIN_CLAIM_CHARS = 20


class LiteratureError(RuntimeError):
    pass


def search_literature(
    seed: str,
    dest_dir: Path,
    *,
    fixture: Path | str | None = None,
    max_results: int = 3,
    timeout: float = 15.0,
) -> tuple[AttestedSpan, ...]:
    """Return one attested span per source, each bound to stored bytes."""
    query = seed.strip()
    if not query:
        raise LiteratureError("arxiv query (seed) must be non-empty")
    if max_results < 1:
        raise LiteratureError("max_results must be at least 1")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if fixture is not None:
        raw = Path(fixture).read_bytes()
    else:
        url = (
            f"{ARXIV_API}?search_query={urllib.parse.quote(query)}"
            f"&start=0&max_results={max_results}"
        )
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                raw = response.read()
        except Exception as exc:  # noqa: BLE001 — extras must not crash the kernel
            raise LiteratureError(f"arxiv fetch failed: {exc}") from exc
    return _spans_from_atom(raw, dest_dir, query, max_results)


def search_arxiv(
    seed: str,
    dest_dir: Path,
    *,
    fixture: Path | str | None = None,
    timeout: float = 15.0,
) -> AttestedSpan:
    """First attested span. Kept for callers that only cite one source."""
    return search_literature(
        seed, dest_dir, fixture=fixture, max_results=1, timeout=timeout
    )[0]


def verify_span(span: AttestedSpan) -> bool:
    """Re-read the stored bytes at the declared offsets (INV-1 promotion rule)."""
    path = Path(urllib.parse.urlparse(span.artifact_uri).path)
    if not path.is_file():
        return False
    blob = path.read_bytes()
    if span.end > len(blob):
        return False
    return blob[span.start : span.end] == span.text.encode("utf-8")


def _spans_from_atom(
    raw: bytes, dest_dir: Path, query: str, max_results: int
) -> tuple[AttestedSpan, ...]:
    try:
        root = ElementTree.fromstring(raw.decode("utf-8"))
    except (ElementTree.ParseError, UnicodeDecodeError) as exc:
        raise LiteratureError("arxiv response is not Atom XML") from exc
    entries = root.findall(f"{{{ARXIV_ATOM}}}entry")
    if not entries:
        raise LiteratureError("arxiv response contained no entry")
    artifact = dest_dir / f"{new_id('lit')}.xml"
    artifact.write_bytes(raw)
    artifact_uri = artifact.resolve().as_uri()
    # Provenance lives beside the response, not inside the bytes the offsets
    # address, so the offset target stays exactly what the source returned.
    artifact.with_suffix(".query.json").write_text(
        json.dumps(
            {
                "query": query,
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "response_bytes": len(raw),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    spans: list[AttestedSpan] = []
    seen: set[str] = set()
    for entry in entries:
        if len(spans) >= max_results:
            break
        summary = _child_text(entry, "summary")
        if not summary.strip():
            continue
        claim = _claim_sentence(summary)
        if claim is None:
            continue
        start = raw.find(claim.encode("utf-8"))
        if start < 0:
            # The parsed text is not a verbatim slice of the response (entities,
            # re-encoding). Fail closed rather than attest an offset we cannot bind.
            continue
        source_id = _arxiv_source_id(_child_text(entry, "id"))
        if source_id in seen:
            continue
        seen.add(source_id)
        span = AttestedSpan(
            source_id=source_id,
            start=start,
            end=start + len(claim.encode("utf-8")),
            text=claim,
            artifact_uri=artifact_uri,
            title=re.sub(r"\s+", " ", _child_text(entry, "title")).strip(),
        )
        span.validate()
        if not verify_span(span):
            raise LiteratureError(f"span for {source_id} does not match stored bytes")
        spans.append(span)
    if not spans:
        raise LiteratureError(
            "no arxiv entry yielded a sentence that is a verbatim slice of the response"
        )
    return tuple(spans)


def _claim_sentence(summary: str) -> str | None:
    """First single-line sentence, so the offsets point at a real slice."""
    for match in re.finditer(r"[^.\n]+\.", summary):
        candidate = match.group(0).strip()
        if len(candidate) >= MIN_CLAIM_CHARS:
            return candidate
    return None


def _child_text(entry: ElementTree.Element, tag: str) -> str:
    node = entry.find(f"{{{ARXIV_ATOM}}}{tag}")
    if node is None or not (node.text or "").strip():
        return ""
    return node.text or ""


def _arxiv_source_id(raw_id: str) -> str:
    match = re.search(r"arxiv\.org/abs/([^/\s]+)", raw_id)
    if match:
        return f"arxiv:{match.group(1)}"
    return f"arxiv:{raw_id.rsplit('/', 1)[-1]}"
