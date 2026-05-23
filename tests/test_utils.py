import os

from zotero_arxiv_daily.utils import _filter_mupdf_stderr


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
