import sys
from types import SimpleNamespace

import numpy as np

from zotero_arxiv_daily.reranker.local import LocalReranker


def test_local_reranker(config, monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model, trust_remote_code):
            self.model = model
            self.trust_remote_code = trust_remote_code

        def encode(self, values, **_kwargs):
            embedding_map = {
                "hello": [1.0, 0.0],
                "world": [0.0, 1.0],
                "ping": [1.0, 1.0],
            }
            return np.array([embedding_map[value] for value in values])

        def similarity(self, s1_feature, s2_feature):
            s1_normalized = s1_feature / np.linalg.norm(s1_feature, axis=1, keepdims=True)
            s2_normalized = s2_feature / np.linalg.norm(s2_feature, axis=1, keepdims=True)
            return SimpleNamespace(numpy=lambda: np.dot(s1_normalized, s2_normalized.T))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)
    assert np.allclose(score, np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]]))
