from zotero_arxiv_daily.paper_resolver import extract_page_metadata, normalize_title, title_similarity


def test_title_normalization_handles_case_punctuation_and_unicode_width():
    assert normalize_title("Prεεmpt:  Sanitizing Sensitive Prompts!") == normalize_title(
        "PRΕΕMPT — sanitizing sensitive prompts"
    )


def test_title_similarity_requires_close_title_match():
    assert title_similarity("A Paper: About LLM Security", "A Paper About LLM Security") == 1.0
    assert title_similarity("A Paper About LLM Security", "Unrelated Database Research") < 0.5


def test_extract_page_metadata_reads_scholarly_meta_and_relative_pdf():
    html = """
    <html><head>
      <meta name="citation_title" content="Paper One">
      <meta name="citation_author" content="Author A">
      <meta name="citation_author" content="Author B">
      <meta name="citation_abstract" content="This is a sufficiently detailed abstract about the paper and its main contribution to security research. It contains enough text for a summary.">
      <meta name="citation_pdf_url" content="/papers/one.pdf">
    </head><body></body></html>
    """

    metadata = extract_page_metadata(html, "https://conference.example/paper/one", "Paper One")

    assert metadata.title == "Paper One"
    assert metadata.authors == ["Author A", "Author B"]
    assert metadata.abstract.startswith("This is a sufficiently detailed abstract")
    assert metadata.pdf_url == "https://conference.example/papers/one.pdf"


def test_extract_page_metadata_falls_back_to_visible_conference_page_blocks():
    html = """
    <html><body>
      <h1>Attention is All You Need to Defend</h1>
      <p>Alice Example (University A), Bob Example (University B)</p>
      <p>This paper studies indirect prompt injection attacks in deployed language-model agents and introduces a detector that operates on attention features.</p>
      <p>Experiments across several models show that the method improves attack detection while preserving utility in benign agent tasks.</p>
      <a href="/paper.pdf">Paper</a>
    </body></html>
    """

    metadata = extract_page_metadata(
        html,
        "https://conference.example/paper/page",
        "Attention is All You Need to Defend",
    )

    assert metadata.authors == ["Alice Example (University A), Bob Example (University B)"]
    assert metadata.abstract.startswith("This paper studies indirect prompt injection")
    assert metadata.pdf_url == "https://conference.example/paper.pdf"


def test_extract_page_metadata_matches_collapsible_conference_author_block():
    html = """
    <div class="list-group-item">
      <b><a data-toggle="collapse" href="#collapse-38">LLM Unlearning Should Be Form-Independent</a></b>
      <div class="collapse authorlist" id="collapse-38">
        Xiaotian Ye<sup>1,2</sup>, Mengqi Zhang<sup>3</sup>, Shu Wu<sup>1</sup><br>
        <sup>1</sup>: University A, <sup>2</sup>: University B, <sup>3</sup>: University C
      </div>
    </div>
    """

    metadata = extract_page_metadata(
        html,
        "https://sp.example/accepted-papers.html#collapse-38",
        "LLM Unlearning Should Be Form-Independent",
    )

    assert metadata.authors == ["Xiaotian Ye, Mengqi Zhang, Shu Wu"]
