import tarfile
import re
import glob
import smtplib
import os
import sys
import tempfile
import json
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from contextlib import contextmanager
from loguru import logger
import datetime
from omegaconf import DictConfig, OmegaConf
from urllib.request import Request, urlopen

_PYMUPDF_LAYOUT_ACTIVATED = False
_IGNORED_MUPDF_STDERR_MESSAGES = (
    "MuPDF error: format error: cmsOpenProfileFromMem failed",
)


def _load_pymupdf4llm():
    global _PYMUPDF_LAYOUT_ACTIVATED
    if not _PYMUPDF_LAYOUT_ACTIVATED:
        import pymupdf.layout
        pymupdf.layout.activate()
        _PYMUPDF_LAYOUT_ACTIVATED = True

    import pymupdf4llm
    return pymupdf4llm


def _remove_ignored_mupdf_stderr(stderr_output: str) -> str:
    kept_lines = [
        line
        for line in stderr_output.splitlines(keepends=True)
        if not any(ignored in line for ignored in _IGNORED_MUPDF_STDERR_MESSAGES)
    ]
    return "".join(kept_lines)


@contextmanager
def _filter_mupdf_stderr():
    """Suppress known MuPDF ICC profile noise while preserving other stderr."""
    sys.stderr.flush()
    original_stderr_fd = os.dup(2)
    try:
        with tempfile.TemporaryFile(mode="w+b") as captured_stderr:
            os.dup2(captured_stderr.fileno(), 2)
            try:
                yield
            finally:
                sys.stderr.flush()
                os.dup2(original_stderr_fd, 2)
                captured_stderr.seek(0)
                filtered_output = _remove_ignored_mupdf_stderr(
                    captured_stderr.read().decode(errors="replace")
                )
                if filtered_output:
                    os.write(original_stderr_fd, filtered_output.encode())
    finally:
        os.close(original_stderr_fd)


def extract_tex_code_from_tar(file_path:str, paper_id:str) -> dict[str,str]:
    try:
        tar = tarfile.open(file_path)
    except tarfile.ReadError:
        logger.debug(f"Failed to find main tex file of {paper_id}: Not a tar file.")
        return None
 
    tex_files = [f for f in tar.getnames() if f.endswith('.tex')]
    if len(tex_files) == 0:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file.")
        tar.close()
        return None
    
    bbl_file = [f for f in tar.getnames() if f.endswith('.bbl')]
    match len(bbl_file) :
        case 0:
            if len(tex_files) > 1:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple tex files while no bbl file.")
                main_tex = None
            else:
                main_tex = tex_files[0]
        case 1:
            main_name = bbl_file[0].replace('.bbl','')
            main_tex = f"{main_name}.tex"
            if main_tex not in tex_files:
                logger.debug(f"Cannot find main tex file of {paper_id} from bbl: The bbl file does not match any tex file.")
                main_tex = None
        case _:
            logger.debug(f"Cannot find main tex file of {paper_id} from bbl: There are multiple bbl files.")
            main_tex = None

    if main_tex is None:
        logger.debug(f"Trying to choose tex file containing the document block as main tex file of {paper_id}")
    #read all tex files
    file_contents = {}
    for t in tex_files:
        f = tar.extractfile(t)
        content = f.read().decode('utf-8',errors='ignore')
        #remove comments
        content = re.sub(r'%.*\n', '\n', content)
        content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
        #remove redundant \n
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\\\\', '', content)
        #remove consecutive spaces
        content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
        if main_tex is None and re.search(r'\\begin\{document\}', content) and not any(w in t for w in ['example', 'sample']):
            main_tex = t
            logger.debug(f"Choose {t} as main tex file of {paper_id}")
        file_contents[t] = content
    
    if main_tex is not None:
        main_source:str = file_contents[main_tex]
        #find and replace all included sub-files
        include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
        for f in include_files:
            if not f.endswith('.tex'):
                file_name = f + '.tex'
            else:
                file_name = f
            main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
        file_contents["all"] = main_source
    else:
        logger.debug(f"Failed to find main tex file of {paper_id}: No tex file containing the document block.")
        file_contents["all"] = None
        
    tar.close()
    return file_contents

def extract_markdown_from_pdf(file_path:str) -> str:
    pymupdf4llm = _load_pymupdf4llm()
    with _filter_mupdf_stderr():
        return pymupdf4llm.to_markdown(file_path,use_ocr=False,header=False,footer=False,ignore_code=True)

def glob_match(path:str, pattern:str) -> bool:
    re_pattern = glob.translate(pattern,recursive=True)
    return re.match(re_pattern, path) is not None

def send_email(config:DictConfig, html:str):
    sender = config.email.sender
    receiver = config.email.receiver
    password = config.email.sender_password
    smtp_server = config.email.smtp_server
    smtp_port = config.email.smtp_port
    def _format_addr(s):
        name, addr = parseaddr(s)
        return formataddr((Header(name, 'utf-8').encode(), addr))

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_addr('Github Action <%s>' % sender)
    msg['To'] = _format_addr('You <%s>' % receiver)
    today = datetime.datetime.now().strftime('%Y/%m/%d')
    msg['Subject'] = Header(f'Daily arXiv {today}', 'utf-8').encode()

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
    except Exception as e:
        logger.debug(f"Failed to use TLS. {e}\nTry to use SSL.")
        try:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        except Exception as e:
            logger.debug(f"Failed to use SSL. {e}\nTry to use plain text.")
            server = smtplib.SMTP(smtp_server, smtp_port)

    server.login(sender, password)
    server.sendmail(sender, [receiver], msg.as_string())
    server.quit()


def fetch_api_balance(config: DictConfig) -> str | None:
    if not _config_bool(OmegaConf.select(config, "email.api_balance.enabled", default=False)):
        return None
    endpoint = OmegaConf.select(config, "email.api_balance.endpoint", default=None)
    api_key = OmegaConf.select(config, "email.api_balance.api_key", default=None)
    if not endpoint:
        logger.warning("API balance footer is enabled but email.api_balance.endpoint is not configured")
        return None

    headers = {"Accept": "application/json"}
    header_name = str(OmegaConf.select(config, "email.api_balance.header_name", default="Authorization") or "")
    header_prefix = str(OmegaConf.select(config, "email.api_balance.header_prefix", default="Bearer ") or "")
    if header_name and api_key:
        headers[header_name] = f"{header_prefix}{api_key}"

    try:
        request = Request(str(endpoint), headers=headers)
        with urlopen(request, timeout=20) as response:
            raw_body = response.read().decode("utf-8")
    except Exception as exc:
        logger.warning(f"Failed to fetch API balance: {exc}")
        return None

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        balance_text = raw_body.strip()
    else:
        json_path = OmegaConf.select(config, "email.api_balance.json_path", default=None)
        balance = _select_json_path(payload, str(json_path)) if json_path else _guess_balance_value(payload)
        balance_text = str(balance).strip() if balance is not None else ""

    if not balance_text:
        logger.warning("API balance response did not contain a usable balance value")
        return None
    currency = str(OmegaConf.select(config, "email.api_balance.currency", default="") or "").strip()
    return f"{balance_text} {currency}".strip()


def _select_json_path(payload, json_path: str):
    current = payload
    for part in json_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _config_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _guess_balance_value(payload):
    if not isinstance(payload, dict):
        return payload
    for key in ("balance", "remaining_balance", "available_balance", "credit", "credits", "quota", "remaining_quota"):
        if key in payload:
            return payload[key]
    for value in payload.values():
        if isinstance(value, dict):
            guessed = _guess_balance_value(value)
            if guessed is not None:
                return guessed
    return None
