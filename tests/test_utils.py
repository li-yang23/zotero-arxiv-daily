import os
from copy import deepcopy

from zotero_arxiv_daily import utils as utils_module
from zotero_arxiv_daily.utils import _filter_mupdf_stderr, fetch_api_balance


def test_filter_mupdf_stderr_suppresses_known_icc_profile_noise(capfd):
    with _filter_mupdf_stderr():
        os.write(2, b"MuPDF error: format error: cmsOpenProfileFromMem failed\n")

    captured = capfd.readouterr()
    assert "cmsOpenProfileFromMem failed" not in captured.err


def test_filter_mupdf_stderr_preserves_unrelated_stderr(capfd):
    with _filter_mupdf_stderr():
        os.write(2, b"MuPDF error: format error: another issue\n")

    captured = capfd.readouterr()
    assert "MuPDF error: format error: another issue" in captured.err


def test_fetch_api_balance_uses_configured_json_path(config, monkeypatch):
    test_config = deepcopy(config)
    test_config.email.api_balance.enabled = True
    test_config.email.api_balance.endpoint = "https://api.example.com/balance"
    test_config.email.api_balance.api_key = "sk-test"
    test_config.email.api_balance.json_path = "data.balance"
    test_config.email.api_balance.currency = "USD"
    observed = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"data": {"balance": 12.34}}'

    def fake_urlopen(request, timeout):
        observed["url"] = request.full_url
        observed["authorization"] = request.headers["Authorization"]
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(utils_module, "urlopen", fake_urlopen)

    assert fetch_api_balance(test_config) == "12.34 USD"
    assert observed == {
        "url": "https://api.example.com/balance",
        "authorization": "Bearer sk-test",
        "timeout": 20,
    }


def test_fetch_api_balance_returns_none_when_disabled(config, monkeypatch):
    test_config = deepcopy(config)
    test_config.email.api_balance.enabled = False
    monkeypatch.setattr(
        utils_module,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not call balance endpoint")),
    )

    assert fetch_api_balance(test_config) is None


def test_fetch_api_balance_treats_false_string_as_disabled(config, monkeypatch):
    test_config = deepcopy(config)
    test_config.email.api_balance.enabled = "false"
    monkeypatch.setattr(
        utils_module,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not call balance endpoint")),
    )

    assert fetch_api_balance(test_config) is None
