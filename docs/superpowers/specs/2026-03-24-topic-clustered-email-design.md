# Topic-Clustered Daily Email Design

## Summary
Add a topic clustering step to the existing daily paper pipeline so the email is organized into 3–5 dynamic topic groups instead of a single flat ranked list. Each group should have a short label and a one-sentence summary, allowing the user to quickly understand the main topics and research problems represented in the day’s new papers.

## Goals
- Group the selected daily papers into dynamic topic clusters.
- Show 3–5 groups in the email when the paper set is large enough.
- Preserve the current reranking behavior and per-paper TL;DR generation.
- Make the email faster to scan by showing group headings and group summaries before paper-level details.
- Keep email delivery reliable even if clustering fails.

## Non-goals
- Changing retrieval logic for arXiv, bioRxiv, or medRxiv.
- Changing the reranking algorithm.
- Adding new output formats beyond the email.
- Persisting cluster results to files or external systems.
- Introducing fixed taxonomy buckets; clustering should remain dynamic.

## Current State
Today the pipeline in `src/zotero_arxiv_daily/executor.py` retrieves papers, reranks them, generates per-paper TL;DRs and affiliations, and then renders a flat HTML email through `src/zotero_arxiv_daily/construct_email.py`. The email presents papers one by one with title, authors, relevance score, affiliations, and TL;DR.

## Proposed Architecture
Keep the existing flow up through reranking and per-paper enrichment, then insert one new topic clustering stage before email rendering.

Proposed pipeline:
1. Fetch papers from configured sources.
2. Rerank papers against the Zotero corpus.
3. Truncate to the configured maximum number of papers.
4. Generate per-paper TL;DR and affiliations.
5. Cluster the selected papers into 3–5 dynamic topic groups.
6. Render the email by topic group.
7. Send the email.

The clustering stage operates only on the already-selected papers that are about to appear in the email. This keeps cost bounded and avoids changing retrieval or scoring semantics.

## Component Design
Add a new focused module, tentatively `src/zotero_arxiv_daily/topic_clusterer.py`.

Primary responsibilities:
- Build compact clustering input from selected papers.
- Ask the LLM to partition papers into 3–5 topic clusters.
- Parse and validate the LLM response.
- Return grouped paper data for email rendering.

Suggested data model:
- `PaperGroup`
  - `label: str`
  - `summary: str | None`
  - `papers: list[Paper]`

This grouping model should remain separate from the existing `Paper` dataclass so clustering concerns do not leak into the core paper representation.

## Clustering Input and Output
### Input to clustering
For each selected paper, the clusterer should build a compact record containing:
- a stable paper index
- title
- TL;DR if available
- otherwise abstract

The clusterer should prefer TL;DR because it is shorter and already optimized for summarization, but it must fall back to abstract when TL;DR is unavailable.

### Expected LLM output
The LLM should return a structured result containing:
- 3–5 topic groups when feasible
- a short, human-readable label for each group
- a one-sentence description of what the grouped papers are about
- the member paper indices for each group

The returned labels should describe the topic or research problem, not generic phrases like “Group 1” or “Interesting papers”.

## Validation Rules
The clustering result should be accepted only if all of the following hold:
- Every selected paper appears exactly once.
- No paper index is duplicated across groups.
- No group is empty.
- The group count is between 3 and 5 when the number of papers makes that feasible.
- Group labels are non-empty strings.

If the number of selected papers is too small to support meaningful 3–5-way grouping, the clusterer may return fewer groups, including a single group.

## Fallback Behavior
Clustering must be non-fatal.

Fallback rules:
- If there are too few papers to cluster meaningfully, return a single default group.
- If the clustering LLM call fails, log a warning and return a single default group.
- If the LLM returns malformed or invalid structure, retry once with stricter instructions.
- If validation still fails after the retry, discard the grouping and return a single default group.

Default fallback group behavior:
- Label: a neutral section heading such as `Relevant papers today`
- Summary: optional; may be omitted in fallback mode
- Papers: all selected papers in their existing relevance order

This ensures the daily email still arrives even when clustering is unavailable.

## Executor Integration
The executor should continue to own the pipeline orchestration.

Integration point:
- After TL;DR and affiliation generation finishes for the selected papers.
- Before `render_email(...)` is called.

Executor behavior changes:
- Call the topic clusterer with the final reranked paper list.
- Pass grouped results to email rendering.
- Preserve the current send/no-send logic for empty paper sets.

The clustering stage should not alter paper scores or paper ordering semantics beyond partitioning the papers into sections.

## Email Rendering Design
Update email rendering so the HTML is organized by topic groups instead of a single continuous paper list.

For each group, render:
1. Group heading
2. Group summary sentence, if present
3. Existing per-paper cards for the papers in that group

Within each group, preserve the existing relevance order from the reranker. This keeps the current ranking signal while making the email easier to scan by topic first.

The existing per-paper card layout should remain unchanged unless minor adjustments are needed to visually separate groups.

## Prompting Strategy
Use one additional LLM pass after per-paper TL;DR generation.

Prompt characteristics:
- Input should include only the compact paper records needed for clustering.
- Instructions should emphasize topic/problem-based grouping rather than source categories.
- Instructions should request 3–5 groups when feasible.
- Output should be structured enough to parse deterministically.
- Labels and summaries should be concise and email-friendly.

This is intended to be a moderate-cost addition rather than a multi-stage clustering workflow.

## Error Handling
- Failures in clustering must not prevent email delivery.
- Invalid cluster output should be detected explicitly rather than partially trusted.
- Partial or ambiguous groupings should be discarded unless they can be corrected trivially and safely.
- Logging should clearly distinguish between clustering success, retry, and fallback.

## Testing Strategy
Add focused tests for the new behavior.

### Clusterer tests
Create a new test module for clustering behavior, for example `tests/test_topic_clusterer.py`, covering:
- successful parsing of valid structured cluster output
- rejection of malformed output
- rejection of duplicated or missing paper indices
- fallback behavior after invalid output
- behavior when paper count is too small for 3–5 groups

### Email tests
Extend `tests/test_email.py` to cover:
- rendering grouped sections with headings
- rendering group summaries
- preserving existing paper cards beneath each group
- fallback rendering with a single default group

### Executor tests
Add or extend executor-level tests to verify:
- grouped rendering path when clustering succeeds
- fallback rendering path when clustering fails
- unchanged behavior when no papers are found

## Constraints and Decisions
- Clustering is dynamic, not based on a fixed taxonomy.
- Output is email-only.
- The solution may add moderate LLM cost, including cluster labels and summaries.
- The normal target is 3–5 groups.
- Retrieval, reranking, and recommendation scoring remain unchanged.

## Open Questions Resolved
- Grouping basis: dynamic topic clusters.
- Stability preference: dynamic day-by-day grouping.
- Output surface: email only.
- Cost tolerance: moderate extra LLM cost.
- Typical cluster count: 3–5 groups.

## Recommended Implementation Direction
Implement a single LLM-based topic clustering pass over the final reranked papers using title + TL;DR (or abstract fallback), validate the output strictly, and render the email in grouped sections with a short summary for each topic. If clustering fails for any reason, send the email as a single fallback section rather than failing the daily run.
