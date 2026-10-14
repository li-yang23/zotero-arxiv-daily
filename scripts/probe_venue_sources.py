"""Save public source responses for diagnosing upstream markup/API changes."""

import json
from pathlib import Path

import httpx


SOURCES = {
    "jmlr": "https://www.jmlr.org/papers/v27/",
    "iclr": "https://iclr.cc/virtual/2026/papers.html",
    "icml": "https://icml.cc/static/virtual/data/icml-2026-orals-posters.json",
    "neurips": "https://neurips.cc/virtual/2026/papers.html",
    "ndss": "https://www.ndss-symposium.org/ndss2026/accepted-papers/",
    "usenix": "https://www.usenix.org/conference/usenixsecurity26/technical-sessions",
    "sp": "https://www.ieee-security.org/TC/SP2026/accepted-papers.html",
    "ccs": "https://www.sigsac.org/ccs/CCS2026/program/accepted-papers.html",
    "tifs": "https://api.crossref.org/journals/1556-6013/works?filter=from-index-date:2026-07-16,from-pub-date:2026-01-01&rows=2",
}


if __name__ == "__main__":
    output = Path("outputs/venue-source-probe")
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for name, url in SOURCES.items():
            try:
                response = client.get(url)
                (output / f"{name}.txt").write_text(response.text, encoding="utf-8")
                result[name] = {"status": response.status_code, "url": str(response.url), "bytes": len(response.content)}
            except httpx.HTTPError as exc:
                result[name] = {"error": str(exc)}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
