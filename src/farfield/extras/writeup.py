"""Compile a short research note from fields the mission already attested.

The note is not a new generation. Title, gap, idea, papers, and any probe
number are copied into Markdown and into a self-contained article class
LaTeX file. Citations may only name papers retrieved this mission — the
same tooth `read_first` already has. If `pdflatex` is on PATH the TeX is
compiled; if it is not, the `.tex` is still a document that *can* compile.

The executable finish line is the sibling protocol in `packet.py`, not this
note. The note is the briefing; the protocol is what to run next week.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .brief import ResearchBrief
from .generate import GeneratedCard
from .livefeed import FreshWork


def _tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        # pdflatex under inputenc dies on these; live model text uses them
        # freely (the first live mission failed on µ and ≤).
        "µ": r"\(\mu\)",
        "μ": r"\(\mu\)",
        "≤": r"\(\le\)",
        "≥": r"\(\ge\)",
        "≠": r"\(\ne\)",
        "≈": r"\(\approx\)",
        "×": r"\(\times\)",
        "·": r"\(\cdot\)",
        "→": r"\(\rightarrow\)",
        "←": r"\(\leftarrow\)",
        "∈": r"\(\in\)",
        "Δ": r"\(\Delta\)",
        "δ": r"\(\delta\)",
        "ε": r"\(\varepsilon\)",
        "α": r"\(\alpha\)",
        "β": r"\(\beta\)",
        "λ": r"\(\lambda\)",
        "σ": r"\(\sigma\)",
        "Θ": r"\(\Theta\)",
        "√": r"\(\sqrt{}\)",
        "∞": r"\(\infty\)",
        "⌈": r"\(\lceil\)",
        "⌉": r"\(\rceil\)",
        "⌊": r"\(\lfloor\)",
        "⌋": r"\(\rfloor\)",
        "∑": r"\(\sum\)",
        "∏": r"\(\prod\)",
        "∪": r"\(\cup\)",
        "∩": r"\(\cap\)",
        "⊆": r"\(\subseteq\)",
        "⊂": r"\(\subset\)",
        "∀": r"\(\forall\)",
        "∃": r"\(\exists\)",
        "±": r"\(\pm\)",
        "≫": r"\(\gg\)",
        "≪": r"\(\ll\)",
        "∝": r"\(\propto\)",
        "γ": r"\(\gamma\)",
        "τ": r"\(\tau\)",
        "ρ": r"\(\rho\)",
        "π": r"\(\pi\)",
        "η": r"\(\eta\)",
        "κ": r"\(\kappa\)",
        "ω": r"\(\omega\)",
        "φ": r"\(\varphi\)",
        "Ω": r"\(\Omega\)",
        "–": "--",
        "—": "---",
        "“": "``",
        "”": "''",
        "‘": "`",
        "’": "'",
        "°": r"\(^\circ\)",
    }

    def one(ch: str) -> str:
        mapped = replacements.get(ch)
        if mapped is not None:
            return mapped
        # pdflatex under inputenc dies on any unmapped non-ASCII glyph.
        # Losing one character to a visible placeholder is strictly better
        # than losing the whole PDF (a live note failed on U+2308).
        if ord(ch) > 126:
            return "?"
        return ch

    return "".join(one(ch) for ch in text)


def _cite_key(work: FreshWork, index: int) -> str:
    raw = re.sub(r"[^A-Za-z0-9]+", "", work.cite_id() or f"p{index}")
    return raw or f"p{index}"


@dataclass(frozen=True)
class ResearchNote:
    markdown: str
    latex: str
    pdf: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "markdown": self.markdown,
            "latex": self.latex,
            "pdf": self.pdf,
        }


def render_note(
    topic: str,
    card: GeneratedCard,
    brief: ResearchBrief,
    works: list[FreshWork],
    *,
    probe: dict[str, Any] | None = None,
    prior: dict[str, Any] | None = None,
) -> ResearchNote:
    papers = works or []
    bib_md = []
    for work in papers:
        cite = work.cite_id()
        venue = f", {work.venue}" if work.venue else ""
        bib_md.append(f"- [{cite}] {work.title} ({work.published}{venue})")
    probe_md = ""
    if probe and probe.get("status") == "ran":
        verdict = probe.get("verdict") or "no verdict computed"
        probe_md = (
            f"A pre-registered two-arm probe ran ({probe.get('measure') or 'measure'}):"
            f" treatment = {probe.get('treatment')}, control = {probe.get('control')}."
            f" Pre-registered direction: {probe.get('expected_direction') or 'n/a'};"
            f" verdict by arithmetic: **{verdict}**."
        )
        if probe.get("alternative"):
            probe_md += f" Competing explanation tested against: {probe['alternative']}"
        probe_md += (
            " The arms run on a constructed dataset: this is evidence about the"
            " mechanism's coherence, not the full experiment."
        )
    elif probe:
        probe_md = (
            f"The probe did not produce its two arms ({probe.get('status')}:"
            f" {probe.get('error') or 'no detail'})."
        )
    prior_md = ""
    if prior:
        prior_md = (
            f"Closest supporting span: {prior.get('title')}"
            f" [{prior.get('cite_id')}], support {prior.get('support', prior.get('coverage'))}"
            + (f" — {prior['span']}" if prior.get("span") else "")
            + "."
        )

    markdown = "\n".join(
        [
            f"# {brief.title}",
            "",
            f"*Topic.* {topic}",
            "",
            f"*Combination.* {card.pair[0]} × {card.pair[1]}",
            "",
            "## Claim",
            "",
            card.claim,
            "",
            "## Why it might hold",
            "",
            card.mechanism,
            "",
            "## Gap",
            "",
            brief.gap,
            "",
            "## Idea",
            "",
            brief.idea,
            "",
            "## Approach",
            "",
            brief.approach,
            "",
            "## Baseline",
            "",
            brief.baseline,
            "",
            "## First steps",
            "",
            *[f"{i}. {step}" for i, step in enumerate(brief.first_steps, start=1)],
            "",
            "## Probe",
            "",
            probe_md or "No probe was run.",
            "",
            "## Risks",
            "",
            brief.risks,
            "",
            "## Related work retrieved this mission",
            "",
            *(bib_md or ["- (none retrieved)"]),
            "",
            *(["## Prior warning", "", prior_md, ""] if prior_md else []),
            f"*Prediction.* {card.prediction}",
            "",
        ]
    )

    bibitems = []
    for index, work in enumerate(papers, start=1):
        key = _cite_key(work, index)
        venue = f" {_tex_escape(work.venue)}." if work.venue else ""
        bibitems.append(
            f"\\bibitem{{{key}}} {_tex_escape(work.title)}."
            f" {_tex_escape(work.cite_id())} ({_tex_escape(work.published)}).{venue}"
        )
    related = (
        "The papers below are the ones retrieved for this idea; no other"
        " citations are permitted in this note.\n\n"
        + "\n\n".join(
            f"{_tex_escape(work.title)} [{_tex_escape(work.cite_id())}]."
            for work in papers
        )
        if papers
        else "No live papers were retrieved; the gap is written from the hypothesis alone."
    )
    probe_tex = _tex_escape(probe_md or "No probe was run.")
    prior_tex = (
        "\\section{Prior warning}\n\n" + _tex_escape(prior_md) + "\n\n"
        if prior_md
        else ""
    )
    steps = "\n".join(
        f"\\item {_tex_escape(step)}" for step in brief.first_steps
    )
    latex = "\n".join(
        [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{url}",
            rf"\title{{{_tex_escape(brief.title)}}}",
            rf"\author{{Farfield research note}}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
            r"\begin{abstract}",
            _tex_escape(brief.idea),
            r"\end{abstract}",
            r"\section{Topic and combination}",
            _tex_escape(topic),
            "",
            rf"{_tex_escape(card.pair[0])} $\times$ {_tex_escape(card.pair[1])}.",
            r"\section{Claim}",
            _tex_escape(card.claim),
            r"\section{Mechanism}",
            _tex_escape(card.mechanism),
            r"\section{Gap}",
            _tex_escape(brief.gap),
            r"\section{Approach}",
            _tex_escape(brief.approach),
            r"\section{Baseline}",
            _tex_escape(brief.baseline),
            r"\section{First steps}",
            r"\begin{enumerate}",
            steps,
            r"\end{enumerate}",
            r"\section{Related work}",
            related,
            r"\section{Probe}",
            probe_tex,
            prior_tex.rstrip(),
            r"\section{Risks}",
            _tex_escape(brief.risks),
            r"\section{Prediction}",
            _tex_escape(card.prediction),
            r"\begin{thebibliography}{99}",
            *bibitems,
            r"\end{thebibliography}",
            r"\end{document}",
            "",
        ]
    )
    return ResearchNote(markdown=markdown, latex=latex, pdf=None)


def compile_tex(latex: str, dest_dir: Path) -> str | None:
    """Run pdflatex twice if present. Return the pdf path or None."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tex = dest_dir / "note.tex"
    tex.write_text(latex, encoding="utf-8")
    engine = shutil.which("pdflatex")
    if not engine:
        return None
    for _ in range(2):
        try:
            completed = subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=str(dest_dir),
                timeout=40,
                capture_output=True,
                text=True,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode != 0:
            return None
    pdf = dest_dir / "note.pdf"
    return str(pdf.resolve()) if pdf.is_file() else None
