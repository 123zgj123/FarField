---
name: research-lit
description: Retrieve live papers from arXiv, Semantic Scholar, and OpenAlex, then admit only rows that pass verify_papers (arXiv id_list / CrossRef DOI / Scholar title overlap). Use when a mission needs today's literature, a briefing citation, or the neighbourhood Research Wiki.
stage: intern
admitted: true
audience: research
entry: plugin.py
---

# Research literature

FarField does not invent a bibliography. Host-side retrieve is multi-source (`CompositeFeed`: arXiv + Semantic Scholar + OpenAlex). Admission is `verify_papers`:

1. Named arXiv id → Export API `id_list`.
2. Named DOI → CrossRef `/works/{doi}`.
3. Otherwise Semantic Scholar title search; token overlap ≥ 0.6.

Statuses:

- `verified` — may enter the wiki, a briefing, `read_first`, or `baseline_cite_id`.
- `unverified` — the index answered and this work is not there. Do not cite.
- `verify_pending` — 429 / 5xx / timeout. Not a hallucination and not a citation.

Query the topic's object phrases, not the distant concept and not a claim written after the fact. OpenAlex gets those phrases **quoted** (`quoted_query`), not a bag of words. `survey_around` with a topic must not query the far concept alone. A retrieved row is on-claim only when it carries a topic bigram or two distinctive unigrams (`paper_on_claim_object`) — sharing `world` or `model` alone is noise. Retrieval retries 429/5xx once with backoff before `verify_pending`. Limitation sentences on verified abstracts become agenda targets, not solved claims. Method bigrams from verified titles (`title_phrases`) and limitation sentences (`limitation_phrases`) compile literature far menus so a code-world-model topic can weld `shadow replay` instead of the nearest cs.DS noun. When the graph does not cover the topic object, those menus occupy the jump cap — cosine graph nouns are not a fallback. Bibliographic pads (`fresh preprint`) are dropped. The probe sandbox stays offline.

Interns invoke `{"action":"skill","name":"research-lit","args":{"topic":"...","state_store":"...","anchor":"..."}}`. Canned rows use `works` plus `skip_verify` only in tests.
