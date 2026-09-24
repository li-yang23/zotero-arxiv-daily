import imaplib
import re
import ssl
import time
from dataclasses import dataclass
from email import policy
from email.message import Message
from email.parser import BytesParser
from email.utils import parseaddr
from typing import Callable

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from .request_email import render_request_email, render_request_error
from .request_executor import RequestExecutor
from .request_parser import parse_paper_list
from .utils import send_email


@dataclass(frozen=True)
class InboundRequest:
    sender: str
    subject: str
    message_id: str | None
    markdown: str


def extract_markdown_attachment(message: Message, max_bytes: int = 262_144) -> str:
    candidates: list[tuple[str | None, bytes, str | None]] = []
    for part in message.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        content_type = part.get_content_type().casefold()
        if not ((filename and filename.casefold().endswith(".md")) or content_type == "text/markdown"):
            continue
        payload = part.get_payload(decode=True)
        if payload is None:
            raw_payload = part.get_payload()
            payload = raw_payload.encode(part.get_content_charset() or "utf-8") if isinstance(raw_payload, str) else b""
        candidates.append((filename, payload, part.get_content_charset()))

    if not candidates:
        raise ValueError("没有找到 .md 附件")
    if len(candidates) > 1:
        raise ValueError("一封请求邮件只能包含一个 .md 附件")

    filename, payload, declared_charset = candidates[0]
    if len(payload) > max_bytes:
        raise ValueError(f"Markdown 附件超过 {max_bytes} 字节限制")

    encodings = [declared_charset, "utf-8-sig", "utf-8"]
    for encoding in encodings:
        if not encoding:
            continue
        try:
            markdown = payload.decode(encoding)
            break
        except (LookupError, UnicodeDecodeError):
            continue
    else:
        raise ValueError(f"无法按 UTF-8 解码附件 {filename or ''}".strip())

    if "\x00" in markdown:
        raise ValueError("Markdown 附件包含无效的 NUL 字符")
    return markdown


def parse_inbound_request(
    raw_message: bytes,
    *,
    allowed_senders: set[str],
    subject_prefix: str,
    max_attachment_bytes: int,
) -> InboundRequest | None:
    message = BytesParser(policy=policy.default).parsebytes(raw_message)
    sender = parseaddr(str(message.get("From", "")))[1].casefold()
    if not sender or sender not in allowed_senders:
        return None

    auto_submitted = str(message.get("Auto-Submitted", "no")).strip().casefold()
    precedence = str(message.get("Precedence", "")).strip().casefold()
    if auto_submitted not in {"", "no"} or precedence in {"bulk", "junk", "list"}:
        return None

    subject = str(message.get("Subject", "")).strip()
    normalized_subject = re.sub(r"^(?:re|fw|fwd):\s*", "", subject, flags=re.IGNORECASE)
    if not normalized_subject.casefold().startswith(subject_prefix.casefold()):
        return None

    markdown = extract_markdown_attachment(message, max_bytes=max_attachment_bytes)
    message_id = str(message.get("Message-ID", "")).strip() or None
    return InboundRequest(
        sender=sender,
        subject=subject,
        message_id=message_id,
        markdown=markdown,
    )


class EmailRequestProcessor:
    def __init__(
        self,
        config: DictConfig,
        *,
        imap_factory: Callable[..., imaplib.IMAP4_SSL] = imaplib.IMAP4_SSL,
        executor_factory: Callable[[DictConfig], RequestExecutor] = RequestExecutor,
        send_email_func=send_email,
        sleep_func: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.imap_factory = imap_factory
        self.executor_factory = executor_factory
        self.send_email_func = send_email_func
        self.sleep_func = sleep_func

        self.imap_server = str(OmegaConf.select(config, "email_requests.imap_server", default="") or "")
        self.imap_port = int(OmegaConf.select(config, "email_requests.imap_port", default=993))
        self.imap_username = str(
            OmegaConf.select(config, "email_requests.username", default=None)
            or OmegaConf.select(config, "email.sender", default="")
            or ""
        )
        self.imap_password = str(
            OmegaConf.select(config, "email_requests.password", default=None)
            or OmegaConf.select(config, "email.sender_password", default="")
            or ""
        )
        allowed_sender_value = str(
            OmegaConf.select(config, "email_requests.allowed_senders", default=None)
            or OmegaConf.select(config, "email.receiver", default="")
            or ""
        )
        self.allowed_senders = {
            sender.strip().casefold()
            for sender in re.split(r"[,;]", allowed_sender_value)
            if sender.strip()
        }
        self.subject_prefix = str(
            OmegaConf.select(config, "email_requests.subject_prefix", default="[论文摘要]")
        )
        self.mailbox = str(OmegaConf.select(config, "email_requests.mailbox", default="INBOX"))
        self.max_attachment_bytes = int(
            OmegaConf.select(config, "email_requests.max_attachment_bytes", default=262_144)
        )
        self.max_papers = int(OmegaConf.select(config, "email_requests.max_papers", default=100))
        self.max_messages_per_run = int(
            OmegaConf.select(config, "email_requests.max_messages_per_run", default=1)
        )
        self.imap_mark_retries = max(
            1,
            int(OmegaConf.select(config, "email_requests.imap_mark_retries", default=3)),
        )
        self.imap_retry_delay_seconds = max(
            0.0,
            float(OmegaConf.select(config, "email_requests.imap_retry_delay_seconds", default=2)),
        )

    def run_once(self) -> int:
        self._validate_config()
        client = self._connect_imap()
        processed = 0
        try:
            status, search_data = client.uid("search", None, "UNSEEN")
            if status != "OK":
                raise RuntimeError("Cannot search unread IMAP messages")
            message_uids = search_data[0].split() if search_data and search_data[0] else []
            logger.info(f"Found {len(message_uids)} unread email(s) in {self.mailbox}")

            for uid in message_uids:
                if processed >= self.max_messages_per_run:
                    break
                raw_message = self._fetch_message(client, uid)
                try:
                    request = parse_inbound_request(
                        raw_message,
                        allowed_senders=self.allowed_senders,
                        subject_prefix=self.subject_prefix,
                        max_attachment_bytes=self.max_attachment_bytes,
                    )
                except ValueError as exc:
                    parsed_message = BytesParser(policy=policy.default).parsebytes(raw_message)
                    sender = parseaddr(str(parsed_message.get("From", "")))[1].casefold()
                    subject = str(parsed_message.get("Subject", "")).strip()
                    if sender not in self.allowed_senders or not subject.casefold().startswith(self.subject_prefix.casefold()):
                        continue
                    request = InboundRequest(
                        sender=sender,
                        subject=subject,
                        message_id=str(parsed_message.get("Message-ID", "")).strip() or None,
                        markdown="",
                    )
                    html = render_request_error(str(exc), self.config.llm.language)
                    self._reply(request, html)
                    self._mark_seen_reliably(uid)
                    processed += 1
                    continue

                if request is None:
                    continue

                try:
                    groups = parse_paper_list(request.markdown, max_papers=self.max_papers)
                    executor = self.executor_factory(self.config)
                    try:
                        report = executor.process(groups)
                    finally:
                        executor.close()
                    html = render_request_email(report, self.config.llm.language)
                except Exception as exc:
                    logger.exception(f"Failed to process paper-list email from {request.sender}")
                    html = render_request_error(str(exc), self.config.llm.language)

                self._reply(request, html)
                self._mark_seen_reliably(uid)
                processed += 1
        finally:
            self._logout_quietly(client)
        logger.info(f"Processed {processed} paper-list request email(s)")
        return processed

    def _validate_config(self) -> None:
        missing = []
        if not self.imap_server:
            missing.append("email_requests.imap_server / IMAP_SERVER")
        if not self.imap_username:
            missing.append("email_requests.username or email.sender")
        if not self.imap_password:
            missing.append("email_requests.password or email.sender_password")
        if not self.allowed_senders:
            missing.append("email_requests.allowed_senders or email.receiver")
        if missing:
            raise ValueError("Missing email-request configuration: " + ", ".join(missing))

    def _fetch_message(self, client: imaplib.IMAP4_SSL, uid: bytes) -> bytes:
        # BODY.PEEK[] retrieves the complete message without setting \Seen.
        # The request is marked only after its reply has been accepted by SMTP.
        status, data = client.uid("fetch", uid, "(BODY.PEEK[])")
        if status != "OK":
            raise RuntimeError(f"Cannot fetch IMAP message UID {uid.decode(errors='replace')}")
        for item in data:
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], bytes):
                return item[1]
        raise RuntimeError(f"IMAP message UID {uid.decode(errors='replace')} had no message payload")

    def _connect_imap(self) -> imaplib.IMAP4_SSL:
        client = self.imap_factory(
            self.imap_server,
            self.imap_port,
            ssl_context=ssl.create_default_context(),
            timeout=30,
        )
        try:
            client.login(self.imap_username, self.imap_password)
            status, _ = client.select(self.mailbox)
            if status != "OK":
                raise RuntimeError(f"Cannot select IMAP mailbox {self.mailbox}")
            return client
        except Exception:
            self._logout_quietly(client)
            raise

    @staticmethod
    def _logout_quietly(client: imaplib.IMAP4_SSL) -> None:
        try:
            client.logout()
        except Exception:
            pass

    def _reply(self, request: InboundRequest, html: str) -> None:
        subject = request.subject if request.subject.casefold().startswith("re:") else f"Re: {request.subject}"
        self.send_email_func(
            self.config,
            html,
            receiver=request.sender,
            subject=subject,
            in_reply_to=request.message_id,
            references=request.message_id,
        )
        logger.info("SMTP accepted paper-summary reply")

    def _mark_seen(self, client: imaplib.IMAP4_SSL, uid: bytes) -> None:
        status, _ = client.uid("store", uid, "+FLAGS", "(\\Seen)")
        if status != "OK":
            raise RuntimeError(f"Cannot mark IMAP message UID {uid.decode(errors='replace')} as seen")

    def _mark_seen_reliably(self, uid: bytes) -> None:
        last_error: Exception | None = None
        for attempt in range(1, self.imap_mark_retries + 1):
            client = None
            try:
                # Paper processing can take hours. Never reuse the long-idle fetch
                # connection for the final state update.
                client = self._connect_imap()
                self._mark_seen(client, uid)
                logger.info(
                    f"Marked paper-list request UID {uid.decode(errors='replace')} as seen"
                )
                return
            except Exception as exc:
                last_error = exc
                if attempt >= self.imap_mark_retries:
                    break
                logger.warning(
                    "Failed to mark paper-list request as seen "
                    f"(attempt {attempt}/{self.imap_mark_retries}); retrying: {exc}"
                )
                self.sleep_func(self.imap_retry_delay_seconds)
            finally:
                if client is not None:
                    self._logout_quietly(client)
        raise RuntimeError(
            f"Cannot mark IMAP message UID {uid.decode(errors='replace')} as seen "
            f"after {self.imap_mark_retries} attempt(s): {last_error}"
        ) from last_error
