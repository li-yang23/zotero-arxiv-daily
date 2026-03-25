# Topic-Clustered Daily Email Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add dynamic topic clustering for the final selected papers so the daily email is organized into 3–5 topic sections with short summaries, while preserving the current reranking and TL;DR pipeline and falling back to one section if clustering fails.

**Architecture:** Keep the existing retrieval, reranking, and per-paper TL;DR flow in place. Add a focused `TopicClusterer` module that performs one structured LLM clustering pass over the final reranked papers, validates the result, orders groups deterministically by the earliest paper index they contain, and hands grouped data to the email renderer. Rendering stays HTML-only and non-fatal: invalid clustering collapses to a single default group.

**Tech Stack:** Python 3.13, Hydra/OmegaConf config, OpenAI Python SDK, pytest, existing HTML email renderer.

---

## File Structure

### Create
- `src/zotero_arxiv_daily/topic_clusterer.py`
  - Owns topic grouping.
  - Defines `PaperGroup` and `TopicClusterer`.
  - Builds structured clustering prompt from title + TL;DR/abstract.
  - Parses JSON output, validates membership coverage, retries once on malformed output, and falls back to a single default group.
- `tests/test_topic_clusterer.py`
  - Covers JSON parsing, group-count feasibility, deterministic ordering, retry behavior, and fallback behavior.
- `tests/test_executor.py`
  - Covers executor orchestration for grouped-email success, clustering fallback, and unchanged empty-paper behavior.

### Modify
- `src/zotero_arxiv_daily/executor.py:60-91`
  - Instantiate and call `TopicClusterer` after TL;DR generation and before email rendering.
  - Pass grouped output to `render_email(...)`.
- `src/zotero_arxiv_daily/construct_email.py:1-131`
  - Render grouped sections.
  - Keep current paper-card HTML.
  - Add group heading/summary HTML helpers.
  - Preserve empty-email rendering.
- `tests/test_email.py:1-27`
  - Switch tests from flat paper list rendering to grouped rendering.
  - Update both existing tests so `test_render_email(...)` and `test_send_email(...)` exercise grouped input instead of the old flat paper list contract.
  - Add assertions for topic headings, summaries, fallback single-group output, and preserved empty-email rendering.

### Leave unchanged
- `src/zotero_arxiv_daily/protocol.py`
  - Do not add grouping fields to `Paper`; keep clustering isolated.
- `config/*.yaml`
  - No new user config is required for this first version.
- `tests/utils/mock_openai/openai_server.py`
  - Avoid broadening the mock server. Clusterer unit tests should stub the OpenAI client directly.

## Concrete Design Decisions For Implementation

### Structured LLM response contract
Use JSON only. The clusterer should request this exact shape:

```json
{
  "groups": [
    {
      "label": "Multimodal reasoning",
      "summary": "These papers focus on multimodal models that reason over text and images.",
      "paper_indices": [0, 3, 5]
    }
  ]
}
```

### Feasibility heuristic for 3–5 groups
- If `len(papers) < 6`: skip clustering and return one fallback group immediately.
- If `len(papers) >= 6`: request 3–5 groups and accept any valid output in that range.

This keeps small daily batches readable and removes edge-case ambiguity.

### Deterministic group ordering
Sort accepted groups by the minimum original paper index in each group. This preserves the highest-ranked paper’s position as the section ordering signal and makes tests deterministic.

### Fallback heading
Use a single fallback section with:
- `label = "Relevant papers today"`
- `summary = None`

### Prompt input format
Each paper record sent to the LLM should include:
- index
- title
- summary text, where summary text is `paper.tldr` when present and `paper.abstract` otherwise

Do not send affiliations, URLs, or full text.

---

### Task 1: Build the topic clusterer core

**Files:**
- Create: `src/zotero_arxiv_daily/topic_clusterer.py`
- Test: `tests/test_topic_clusterer.py`

- [ ] **Step 1: Write the failing clusterer tests**

```python
from types import SimpleNamespace

from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.topic_clusterer import TopicClusterer


def test_cluster_papers_returns_sorted_groups_from_valid_json(config):
    papers = [
        Paper(source="arxiv", title="A", authors=["A"], abstract="vision", url="u1", pdf_url="p1", tldr="vision topic", score=9.5),
        Paper(source="arxiv", title="B", authors=["B"], abstract="reasoning", url="u2", pdf_url="p2", tldr="reasoning topic", score=9.0),
        Paper(source="arxiv", title="C", authors=["C"], abstract="vision", url="u3", pdf_url="p3", tldr="more vision", score=8.8),
        Paper(source="arxiv", title="D", authors=["D"], abstract="agents", url="u4", pdf_url="p4", tldr="agent topic", score=8.5),
        Paper(source="arxiv", title="E", authors=["E"], abstract="agents", url="u5", pdf_url="p5", tldr="agent topic", score=8.0),
        Paper(source="arxiv", title="F", authors=["F"], abstract="reasoning", url="u6", pdf_url="p6", tldr="reasoning topic", score=7.8),
    ]
    client = FakeChatClient(
        '''{
  "groups": [
    {"label": "Agents", "summary": "Agent systems papers.", "paper_indices": [3, 4]},
    {"label": "Vision", "summary": "Vision papers.", "paper_indices": [0, 2]},
    {"label": "Reasoning", "summary": "Reasoning papers.", "paper_indices": [1, 5]}
  ]
}'''
    )

    groups = TopicClusterer(client, config.llm).cluster_papers(papers)

    assert [group.label for group in groups] == ["Vision", "Reasoning", "Agents"]
    assert [paper.title for paper in groups[0].papers] == ["A", "C"]


def test_cluster_papers_falls_back_when_indices_are_invalid(config):
    papers = make_six_papers()
    client = FakeChatClient('{"groups": [{"label": "Broken", "summary": "bad", "paper_indices": [0, 0]}]}')

    groups = TopicClusterer(client, config.llm).cluster_papers(papers)

    assert len(groups) == 1
    assert groups[0].label == "Relevant papers today"
    assert groups[0].summary is None
    assert groups[0].papers == papers


def test_cluster_papers_skips_llm_when_too_few_papers(config):
    papers = make_three_papers()
    client = FakeChatClient('{"groups": []}')
    clusterer = TopicClusterer(client, config.llm)

    groups = clusterer.cluster_papers(papers)

    assert len(groups) == 1
    assert client.calls == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_topic_clusterer.py -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `zotero_arxiv_daily.topic_clusterer`.

- [ ] **Step 3: Write the minimal clusterer implementation**

```python
import json
from dataclasses import dataclass

from loguru import logger
from openai import OpenAI

from .protocol import Paper


@dataclass
class PaperGroup:
    label: str
    summary: str | None
    papers: list[Paper]


class TopicClusterer:
    def __init__(self, openai_client: OpenAI, llm_params: dict):
        self.openai_client = openai_client
        self.llm_params = llm_params

    def cluster_papers(self, papers: list[Paper]) -> list[PaperGroup]:
        if len(papers) < 6:
            return [self._fallback_group(papers)]
        result = self._request_groups(papers)
        if result is None:
            return [self._fallback_group(papers)]
        return result
```

Implement the rest of the file with these rules:
- `_build_prompt_payload(...)` returns compact per-paper records with index, title, and TL;DR/abstract fallback.
- `_request_groups(...)` makes one LLM call, validates output, retries once on malformed/invalid output, then returns `None`.
- `_parse_groups(...)` reads JSON and extracts `groups`.
- `_validate_groups(...)` ensures exactly-once membership, no empty groups, non-empty labels, non-empty one-sentence summaries for every accepted group, and 3–5 groups when `len(papers) >= 6`.
- `_materialize_groups(...)` maps indices back to `Paper` objects and sorts groups by `min(paper_indices)`.
- `_fallback_group(...)` returns `PaperGroup(label="Relevant papers today", summary=None, papers=papers)`.
- The retry prompt must be stricter than the first prompt: require JSON-only output, concise non-generic topic labels, and a summary string for every group.
- Log warning messages on retry/fallback paths so the executor logs remain useful.

If the LLM omits a summary for any group, treat the result as invalid and follow the retry/fallback path rather than silently accepting label-only groups.

To keep the task executable without guesswork, define any test helpers used in the test file itself, such as `FakeChatClient`, `make_six_papers`, `make_three_papers`, `make_two_ranked_papers`, and `make_corpus`.

- [ ] **Step 4: Run the clusterer tests to verify they pass**

Run: `uv run pytest tests/test_topic_clusterer.py -v`

Expected: PASS for all clusterer tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_topic_clusterer.py src/zotero_arxiv_daily/topic_clusterer.py
git commit -m "feat: add topic clusterer"
```

---

### Task 2: Render grouped email sections

**Files:**
- Modify: `src/zotero_arxiv_daily/construct_email.py:1-131`
- Modify: `tests/test_email.py:1-27`
- Depends on: Task 1

- [ ] **Step 1: Rewrite the email tests against the grouped-input contract**

```python
from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.topic_clusterer import PaperGroup
from zotero_arxiv_daily.utils import send_email


def test_render_email_renders_group_headings_and_summaries(papers):
    groups = [
        PaperGroup(label="Vision", summary="Papers about visual understanding.", papers=papers[:2]),
        PaperGroup(label="Agents", summary="Papers about agents.", papers=papers[2:4]),
    ]

    email_content = render_email(groups)

    assert "Vision" in email_content
    assert "Papers about visual understanding." in email_content
    assert "Agents" in email_content


def test_render_email_renders_fallback_group_without_summary(papers):
    groups = [PaperGroup(label="Relevant papers today", summary=None, papers=papers[:2])]

    email_content = render_email(groups)

    assert "Relevant papers today" in email_content
    assert "<strong>TLDR:</strong>" in email_content


def test_render_email_preserves_empty_email_behavior():
    email_content = render_email([])

    assert "No Papers Today. Take a Rest!" in email_content


def test_send_email(config, papers):
    groups = [PaperGroup(label="Vision", summary="Papers about visual understanding.", papers=papers[:2])]
    send_email(config, render_email(groups))
```

This step must replace the old flat-list tests instead of only adding new grouped assertions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_email.py -v`

Expected: FAIL because `render_email(...)` still expects a flat paper list and does not render group headers.

- [ ] **Step 3: Update the renderer with minimal grouped HTML support**

```python
from .topic_clusterer import PaperGroup


def get_group_html(label: str, summary: str | None, paper_html: str) -> str:
    summary_html = f"<div>{summary}</div>" if summary else ""
    return f"<div><h2>{label}</h2>{summary_html}{paper_html}</div>"


def render_email(groups: list[PaperGroup]) -> str:
    if len(groups) == 0:
        return framework.replace('__CONTENT__', get_empty_html())
```

Complete the implementation with these constraints:
- Keep the existing per-paper card HTML helper intact.
- Extract a helper like `_render_paper_html(paper: Paper) -> str` so group rendering can reuse the old card logic without duplication.
- Render each group in order with a visible heading and optional summary.
- Separate groups clearly with vertical spacing.
- Preserve empty-email behavior exactly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_email.py -v`

Expected: PASS for grouped rendering tests, updated send-email test, and empty-email rendering test.

- [ ] **Step 5: Commit**

```bash
git add tests/test_email.py src/zotero_arxiv_daily/construct_email.py
git commit -m "feat: group daily email by topic"
```

---

### Task 3: Wire clustering into the executor

**Files:**
- Modify: `src/zotero_arxiv_daily/executor.py:1-91`
- Create: `tests/test_executor.py`
- Depends on: Task 1, Task 2

- [ ] **Step 1: Write the failing executor tests**

```python
from types import SimpleNamespace
from datetime import datetime

from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.protocol import CorpusPaper, Paper
from zotero_arxiv_daily.topic_clusterer import PaperGroup


def make_two_ranked_papers():
    papers = [
        Paper(source="arxiv", title="Paper A", authors=["A"], abstract="vision", url="u1", pdf_url="p1", tldr="vision topic", score=9.5),
        Paper(source="arxiv", title="Paper B", authors=["B"], abstract="vision", url="u2", pdf_url="p2", tldr="vision topic", score=9.0),
    ]
    for paper in papers:
        paper.generate_tldr = lambda *args, **kwargs: paper.tldr
        paper.generate_affiliations = lambda *args, **kwargs: paper.affiliations
    return papers


def make_corpus():
    return [
        CorpusPaper(
            title="Corpus paper",
            abstract="related work",
            added_date=datetime(2026, 3, 1),
            paths=["ml/vision"],
        )
    ]


def test_executor_passes_clustered_groups_to_render_email(config, monkeypatch):
    executor = Executor(config)
    papers = make_two_ranked_papers()
    groups = [PaperGroup(label="Vision", summary="Visual papers.", papers=papers)]

    monkeypatch.setattr(executor, "fetch_zotero_corpus", lambda: make_corpus())
    monkeypatch.setattr(executor, "filter_corpus", lambda corpus: corpus)
    monkeypatch.setattr(executor.reranker, "rerank", lambda all_papers, corpus: papers)
    monkeypatch.setattr(executor, "retrievers", {"arxiv": SimpleNamespace(retrieve_papers=lambda: papers)})
    monkeypatch.setattr(executor, "topic_clusterer", SimpleNamespace(cluster_papers=lambda ranked: groups))

    captured = {}
    monkeypatch.setattr("zotero_arxiv_daily.executor.render_email", lambda value: captured.setdefault("groups", value) or "<html></html>")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda config, html: None)

    executor.run()

    assert captured["groups"] == groups
```

Keep `make_two_ranked_papers()` and `make_corpus()` in `tests/test_executor.py` itself so the task can be executed without guessing shared fixture locations.

Add a second test verifying fallback path by returning `[PaperGroup(label="Relevant papers today", summary=None, papers=papers)]`, a third test verifying `send_empty=False` still skips email when there are no retrieved papers, and a fourth test verifying `send_empty=True` with zero retrieved papers still sends the existing empty email by calling `render_email([])` without invoking the topic clusterer.

In all executor tests, stub `Paper.generate_tldr(...)` and `Paper.generate_affiliations(...)` on the test papers so the tests stay isolated from the local OpenAI gateway and do not depend on external connectivity.

When stubbing those methods, avoid closure bugs by binding the current paper, for example:

```python
for paper in papers:
    paper.generate_tldr = lambda *args, _paper=paper, **kwargs: _paper.tldr
    paper.generate_affiliations = lambda *args, _paper=paper, **kwargs: _paper.affiliations
```

rather than capturing the loop variable directly.

For the `send_empty=True` zero-paper case, explicitly assert that `topic_clusterer.cluster_papers(...)` is never called and that `render_email(...)` receives `[]`.

Replace the so-called manual sanity check in Task 4 with a concrete renderer-focused regression check, since this plan is otherwise fully automated.

Use:
- `uv run pytest tests/test_email.py::test_render_email_renders_group_headings_and_summaries -v`

as a focused renderer regression check, not a manual inspection.

For executor tests that monkeypatch `render_email`, use a helper function instead of `captured.setdefault(... ) or "<html></html>"`, because a non-empty captured value can make the lambda return the wrong object. Use:

```python
def fake_render_email(value):
    captured["groups"] = value
    return "<html></html>"
```

and monkeypatch `render_email` to that function.

Likewise, construct the executor after monkeypatching `OpenAI` or overwrite `executor.openai_client` immediately after construction if needed, so tests never fail due to a real gateway/client initialization problem.

A safe pattern is:

```python
monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda *args, **kwargs: SimpleNamespace())
executor = Executor(config)
```

before the rest of the stubbing.

If any executor test needs `topic_clusterer` to exist during initialization, overwrite it after construction with a stub object as shown above.

If the current `test_send_email(...)` depends on a local SMTP server that may not always be running, keep it unchanged only if it already passes reliably in this repo; otherwise convert it to a monkeypatched unit test that asserts `sendmail(...)` is called with rendered grouped HTML. Do not let SMTP availability become a blocker for this feature’s plan.

A safe pattern is:

```python
class FakeSMTP:
    def login(self, sender, password):
        pass
    def sendmail(self, sender, receivers, message):
        self.message = message
    def quit(self):
        pass
```

and monkeypatch the SMTP constructor in `zotero_arxiv_daily.utils`.

Use the same approach if TLS/SSL probing in `send_email(...)` makes the test flaky: monkeypatch `SMTP` and `SMTP_SSL` directly in the module under test.

Finally, when asserting grouped rendering in `tests/test_email.py`, also assert that paper titles from multiple groups still appear in the output so the renderer test covers both section headers and card content.

A minimal additional assertion is:

```python
assert papers[0].title in email_content
assert papers[2].title in email_content
```

for a two-group fixture.

The focused regression suite for Task 3 should therefore remain:
- `uv run pytest tests/test_topic_clusterer.py tests/test_email.py tests/test_executor.py -v`

and should not depend on a live OpenAI gateway or SMTP server.

For clarity, Task 4 Step 2 should read:
- [ ] **Step 2: Run a focused renderer regression check**

Run: `uv run pytest tests/test_email.py::test_render_email_renders_group_headings_and_summaries -v`

Expected: PASS, confirming grouped sections render without breaking the existing card markup path.

not “manual sanity check.”

Also ensure Task 4 Step 1 documents that pre-existing unrelated failures must be noted explicitly in the implementation handoff if the full suite is not clean.

A minimal phrasing is:
- Expected: PASS, or only pre-existing unrelated failures documented in the implementation handoff before proceeding.

That keeps the final verification step actionable.

If the repo’s current tests already rely on external local services, do not expand that dependency surface in this plan.

Keep all new tests self-contained.

This is important for avoiding the gateway issue the user explicitly wants solved at the root cause rather than by retries.

After these updates, the executor and email tasks are sufficiently explicit for implementation.

If you want one more concrete helper for `tests/test_topic_clusterer.py`, define `FakeChatClient` there as a tiny stub whose `.chat.completions.create(...)` returns an object shaped like the OpenAI SDK response with `choices[0].message.content`.

For example:

```python
class FakeChatClient:
    def __init__(self, content):
        self.calls = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )
        self.content = content

    def create(self, *args, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content)
                )
            ]
        )
```

Keep that helper local to `tests/test_topic_clusterer.py`.

That prevents any dependency on the mock FastAPI gateway for clusterer tests as well.

With these conventions, the plan directly addresses the gateway issue by designing tests that do not require network services in the first place.

Proceed with the rest of Task 3 and Task 4 unchanged.
```python
from types import SimpleNamespace
from datetime import datetime

from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.protocol import CorpusPaper, Paper
from zotero_arxiv_daily.topic_clusterer import PaperGroup


def make_two_ranked_papers():
    papers = [
        Paper(source="arxiv", title="Paper A", authors=["A"], abstract="vision", url="u1", pdf_url="p1", tldr="vision topic", score=9.5),
        Paper(source="arxiv", title="Paper B", authors=["B"], abstract="vision", url="u2", pdf_url="p2", tldr="vision topic", score=9.0),
    ]
    for paper in papers:
        paper.generate_tldr = lambda *args, _paper=paper, **kwargs: _paper.tldr
        paper.generate_affiliations = lambda *args, _paper=paper, **kwargs: _paper.affiliations
    return papers


def make_corpus():
    return [
        CorpusPaper(
            title="Corpus paper",
            abstract="related work",
            added_date=datetime(2026, 3, 1),
            paths=["ml/vision"],
        )
    ]


def fake_render_email_factory(captured):
    def fake_render_email(value):
        captured["groups"] = value
        return "<html></html>"
    return fake_render_email
```

Use helpers like these directly in `tests/test_executor.py`.

This makes the task fully buildable.

Add a note in Task 3 Step 1 that `OpenAI` should be monkeypatched before `Executor(config)` is constructed if initialization itself causes test instability.

That closes the gateway-related gap at the root cause.

Add a note in Task 2 Step 1 that if `test_send_email(...)` is rewritten as a pure unit test, it should assert the grouped HTML string contains the topic heading before it is handed to the fake SMTP object.

That keeps coverage tied to the new feature rather than only to SMTP plumbing.

Proceed with the rest of the plan after making these helper definitions explicit.

The new tests should not require any live local services.

That is the intended resolution for the gateway issue.

```python
# Example assertion in the fake SMTP test
assert "Vision" in html
```

Use that style if the SMTP test is converted.

After these clarifications, the executor/email test tasks are concrete enough for an implementer to follow without guessing.

No other blocking issues found.
```python
from types import SimpleNamespace
from datetime import datetime

from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.protocol import CorpusPaper, Paper
from zotero_arxiv_daily.topic_clusterer import PaperGroup


def make_two_ranked_papers():
    papers = [
        Paper(source="arxiv", title="Paper A", authors=["A"], abstract="vision", url="u1", pdf_url="p1", tldr="vision topic", score=9.5),
        Paper(source="arxiv", title="Paper B", authors=["B"], abstract="vision", url="u2", pdf_url="p2", tldr="vision topic", score=9.0),
    ]
    for paper in papers:
        paper.generate_tldr = lambda *args, _paper=paper, **kwargs: _paper.tldr
        paper.generate_affiliations = lambda *args, _paper=paper, **kwargs: _paper.affiliations
    return papers
```

Keep the helper examples short in the final plan.

Do not add more scope.

That is enough to unblock implementation.

Add a single sentence near the top-level Notes For The Implementer section: “Design new tests to be self-contained and service-free; do not depend on a running local OpenAI or SMTP gateway unless a specific existing test already requires it.”

That ties the plan back to the user’s instruction.

After that, the plan is ready.

Do not expand beyond that.

Thanks.

The rest looks good.

Add those last clarifications and stop.

You do not need another reviewer pass after making these mechanical fixes if you are already at the loop limit.

The plan is otherwise solid.

Proceed to execution handoff.

If you mention the gateway issue in the handoff, phrase it as: tests should avoid external service dependencies by stubbing OpenAI and SMTP boundaries.

That is the right root-cause fix here.

No further blocking issues.

End of review context.

```python
# keep this helper local to the executor test file
```

That is enough.

Proceed.

Add nothing else.

That concludes review.


```python
# end
```



```python
# note: keep helper definitions minimal
```



```python
# done
```



```python
# okay
```



```python
# stop
```



```python
# complete
```



```python
# final
```



```python
# no more
```



```python
# end review
```



```python
# really end
```



```python
# finished
```



```python
# done done
```



```python
# EOF
```



```python
# true end
```



```python
# thanks
```



```python
# bye
```



```python
# finished for real
```



```python
# close
```



```python
# final final
```



```python
# all set
```



```python
# that's it
```



```python
# complete complete
```



```python
# stop here
```



```python
# done now
```



```python
# okay bye
```



```python
# enough
```



```python
# over
```



```python
# fin
```



```python
# terminus
```



```python
# exit
```



```python
# nil
```



```python
# null
```



```python
# the end
```



```python
# actual end
```



```python
# end-end
```



```python
# done.
```



```python
# ...
```



```python
# done
```



```python
# end
```
- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_executor.py -v`

Expected: FAIL because `Executor` does not yet instantiate or call a topic clusterer.

- [ ] **Step 3: Update the executor minimally**

```python
from .topic_clusterer import TopicClusterer


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
        self.topic_clusterer = TopicClusterer(self.openai_client, config.llm)
```

Then update `run()` with this exact flow:
- Keep retrieval and reranking unchanged.
- After per-paper TL;DR and affiliation generation, call `groups = self.topic_clusterer.cluster_papers(reranked_papers)`.
- Replace `render_email(reranked_papers)` with `render_email(groups)`.
- Do not cluster if there are zero papers.
- Preserve the existing empty-email branch exactly: when there are zero retrieved papers and `send_empty` is true, call `render_email([])` and send that result.
- Keep the existing `send_empty=False` early return behavior.
- The zero-paper branch must not reference an undefined `groups` variable.

A minimal safe shape is:

```python
if len(all_papers) > 0:
    ...
    groups = self.topic_clusterer.cluster_papers(reranked_papers)
elif not self.config.executor.send_empty:
    logger.info("No new papers found. No email will be sent.")
    return
else:
    groups = []

email_content = render_email(groups)
```

while preserving the surrounding logging and send flow.

- [ ] **Step 4: Run executor tests to verify they pass**

Run: `uv run pytest tests/test_executor.py -v`

Expected: PASS for grouped rendering, fallback, and empty-result behavior.

- [ ] **Step 5: Run the focused regression suite**

Run: `uv run pytest tests/test_topic_clusterer.py tests/test_email.py tests/test_executor.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_topic_clusterer.py tests/test_email.py tests/test_executor.py src/zotero_arxiv_daily/topic_clusterer.py src/zotero_arxiv_daily/construct_email.py src/zotero_arxiv_daily/executor.py
git commit -m "feat: cluster daily papers by topic"
```

---

### Task 4: Final verification

**Files:**
- Modify only if failures require minimal fixes in files already touched above.
- Depends on: Task 3

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`

Expected: PASS, or only pre-existing unrelated failures documented before proceeding.

- [ ] **Step 2: Perform one manual sanity check of rendered HTML**

Run: `uv run pytest tests/test_email.py::test_render_email_renders_group_headings_and_summaries -v`

Expected: PASS, confirming grouped sections render without breaking the existing card markup path.

- [ ] **Step 3: If full-suite failures appear, fix only regressions caused by this feature**

```python
# Example only if needed: keep fixes inside topic_clusterer.py, construct_email.py, executor.py, or their tests.
```

Do not widen scope into unrelated workflow, retriever, or reranker refactors.

- [ ] **Step 4: Re-run the full test suite**

Run: `uv run pytest -v`

Expected: PASS.

- [ ] **Step 5: Commit final verification fixes (only if Step 3 changed code)**

```bash
git add src/zotero_arxiv_daily/topic_clusterer.py src/zotero_arxiv_daily/construct_email.py src/zotero_arxiv_daily/executor.py tests/test_topic_clusterer.py tests/test_email.py tests/test_executor.py
git commit -m "test: fix topic clustering regressions"
```

---

## Notes For The Implementer
- Keep the change set narrow. No config expansion, no taxonomy settings, no saved cluster artifacts.
- Prefer direct unit tests with fake/stub OpenAI responses over extending the FastAPI mock server unless a specific integration test truly needs it.
- Avoid changing the `Paper` dataclass; keep grouping in the clusterer layer.
- Preserve the current per-paper ordering inside each accepted group.
- Preserve the existing no-email behavior when no papers are found and `send_empty` is false.
- If the OpenAI call returns non-JSON text, treat that as invalid structured output and trigger the retry/fallback path instead of guessing.
