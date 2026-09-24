import imaplib
from email.message import EmailMessage

import pytest

from copy import deepcopy

from zotero_arxiv_daily.inbound_email import (
    EmailRequestProcessor,
    extract_markdown_attachment,
    parse_inbound_request,
)
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.request_models import RequestReport
from zotero_arxiv_daily.topic_clusterer import PaperGroup


def make_message(
    *,
    sender: str = "researcher@example.com",
    subject: str = "[论文摘要] Security papers",
    attachment: bytes | None = b"# NDSS 2026\nPaper One\n",
) -> bytes:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = "paperbot@example.com"
    message["Subject"] = subject
    message["Message-ID"] = "<request-1@example.com>"
    message.set_content("Please see the attached list.")
    if attachment is not None:
        message.add_attachment(
            attachment,
            maintype="text",
            subtype="markdown",
            filename="论文列表.md",
        )
    return message.as_bytes()


def test_parse_inbound_request_extracts_allowed_markdown_request():
    request = parse_inbound_request(
        make_message(),
        allowed_senders={"researcher@example.com"},
        subject_prefix="[论文摘要]",
        max_attachment_bytes=1024,
    )

    assert request is not None
    assert request.sender == "researcher@example.com"
    assert request.subject == "[论文摘要] Security papers"
    assert request.message_id == "<request-1@example.com>"
    assert "Paper One" in request.markdown


def test_parse_inbound_request_ignores_sender_outside_allowlist():
    request = parse_inbound_request(
        make_message(sender="attacker@example.com"),
        allowed_senders={"researcher@example.com"},
        subject_prefix="[论文摘要]",
        max_attachment_bytes=1024,
    )

    assert request is None


def test_parse_inbound_request_ignores_unrelated_subject():
    request = parse_inbound_request(
        make_message(subject="Hello"),
        allowed_senders={"researcher@example.com"},
        subject_prefix="[论文摘要]",
        max_attachment_bytes=1024,
    )

    assert request is None


def test_extract_markdown_attachment_requires_one_attachment():
    message = EmailMessage()
    message.set_content("No attachment")

    with pytest.raises(ValueError, match=r"没有找到 \.md 附件"):
        extract_markdown_attachment(message)


def test_extract_markdown_attachment_enforces_size_limit():
    raw_message = make_message(attachment=b"x" * 20)
    parsed = EmailMessage()
    from email import policy
    from email.parser import BytesParser

    parsed = BytesParser(policy=policy.default).parsebytes(raw_message)
    with pytest.raises(ValueError, match="超过 10 字节限制"):
        extract_markdown_attachment(parsed, max_bytes=10)


def test_email_request_processor_reads_replies_and_marks_message_seen(config):
    raw_message = make_message(sender=config.email.receiver)

    class FakeIMAP:
        def __init__(self, *_args, **_kwargs):
            self.logged_in = None
            self.stored = []
            self.logged_out = False

        def login(self, username, password):
            self.logged_in = (username, password)
            return "OK", []

        def select(self, mailbox):
            assert mailbox == "INBOX"
            return "OK", [b"1"]

        def uid(self, command, *args):
            if command == "search":
                return "OK", [b"1"]
            if command == "fetch":
                return "OK", [(b"1 (RFC822)", raw_message)]
            if command == "store":
                self.stored.append(args)
                return "OK", []
            raise AssertionError(command)

        def logout(self):
            self.logged_out = True

    paper = Paper(
        source="test",
        title="Paper One",
        authors=["Author"],
        abstract="Abstract",
        url="https://example.com/one",
        tldr="Summary",
    )

    class FakeExecutor:
        def __init__(self, _config):
            self.closed = False

        def process(self, groups):
            assert groups[0].label == "NDSS 2026"
            return RequestReport(
                requested_count=1,
                groups=[PaperGroup(label="NDSS 2026", summary=None, papers=[paper])],
            )

        def close(self):
            self.closed = True

    test_config = deepcopy(config)
    test_config.email_requests.imap_server = "imap.example.com"
    sent = []
    fake_imaps = []

    def make_fake_imap(*_args, **_kwargs):
        client = FakeIMAP()
        fake_imaps.append(client)
        return client

    processor = EmailRequestProcessor(
        test_config,
        imap_factory=make_fake_imap,
        executor_factory=FakeExecutor,
        send_email_func=lambda *args, **kwargs: sent.append((args, kwargs)),
    )

    assert processor.run_once() == 1
    assert len(fake_imaps) == 2
    assert all(
        client.logged_in == (test_config.email.sender, test_config.email.sender_password)
        for client in fake_imaps
    )
    assert fake_imaps[0].stored == []
    assert fake_imaps[1].stored == [(b"1", "+FLAGS", "(\\Seen)")]
    assert all(client.logged_out for client in fake_imaps)
    assert len(sent) == 1
    assert sent[0][1]["receiver"] == test_config.email.receiver
    assert sent[0][1]["subject"] == "Re: [论文摘要] Security papers"
    assert sent[0][1]["in_reply_to"] == "<request-1@example.com>"


def test_email_request_processor_retries_seen_flag_with_another_fresh_connection(config):
    raw_message = make_message(sender=config.email.receiver)
    clients = []

    class FakeIMAP:
        def __init__(self, *_args, **_kwargs):
            self.index = len(clients)
            self.stored = []
            self.logged_out = False
            clients.append(self)

        def login(self, _username, _password):
            return "OK", []

        def select(self, _mailbox):
            return "OK", [b"1"]

        def uid(self, command, *args):
            if command == "search":
                return "OK", [b"1"]
            if command == "fetch":
                return "OK", [(b"1 (RFC822)", raw_message)]
            if command == "store":
                if self.index == 1:
                    raise imaplib.IMAP4.abort("socket error: connection reset")
                self.stored.append(args)
                return "OK", []
            raise AssertionError(command)

        def logout(self):
            self.logged_out = True

    class FakeExecutor:
        def __init__(self, _config):
            pass

        def process(self, _groups):
            return RequestReport(requested_count=1)

        def close(self):
            pass

    test_config = deepcopy(config)
    test_config.email_requests.imap_server = "imap.example.com"
    test_config.email_requests.imap_mark_retries = 2
    test_config.email_requests.imap_retry_delay_seconds = 0
    sent = []
    sleeps = []
    processor = EmailRequestProcessor(
        test_config,
        imap_factory=FakeIMAP,
        executor_factory=FakeExecutor,
        send_email_func=lambda *args, **kwargs: sent.append((args, kwargs)),
        sleep_func=lambda seconds: sleeps.append(seconds),
    )

    assert processor.run_once() == 1
    assert len(sent) == 1
    assert len(clients) == 3
    assert clients[0].stored == []
    assert clients[1].stored == []
    assert clients[2].stored == [(b"1", "+FLAGS", "(\\Seen)")]
    assert all(client.logged_out for client in clients)
    assert sleeps == [0.0]
