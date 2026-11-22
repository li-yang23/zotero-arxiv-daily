"""Delivery state is durable in Git, not in an expiring Actions cache."""

import base64
import json
import hashlib
from pathlib import Path
from time import sleep, time
from urllib.parse import quote

import httpx
from loguru import logger


def retry_delay(error, attempt):
    """Retry transient failures, never an ordinary permissions rejection."""
    if not isinstance(error, httpx.HTTPStatusError):
        return 2 ** attempt
    response = error.response
    headers = response.headers
    message = response.text.lower()
    limited = response.status_code == 429 or (response.status_code == 403 and (
        headers.get("x-ratelimit-remaining") == "0" or "retry-after" in headers
        or "rate limit" in message or "abuse detection" in message
    ))
    if limited:
        delay = 60 * 2 ** attempt
        try:
            if "retry-after" in headers:
                delay = max(delay, float(headers["retry-after"]))
            if headers.get("x-ratelimit-remaining") == "0" and "x-ratelimit-reset" in headers:
                delay = max(delay, float(headers["x-ratelimit-reset"]) - time() + 1)
        except ValueError:
            pass
        # Do not retry early when the server asks for a longer wait.
        return delay if delay <= 300 else None
    if response.status_code in {409, 500, 502, 503, 504}:
        return 2 ** attempt
    return None


def empty_state():
    return {"version": 1, "venues": {}, "papers": {}, "deliveries": {}}


def validate_state(state):
    if state.get("version") != 1:
        raise ValueError("Unsupported venue monitor state version")
    for key in ("venues", "papers", "deliveries"):
        if not isinstance(state.get(key), dict):
            raise ValueError(f"Invalid venue monitor state: {key}")
    return state


class FileStateStore:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        return validate_state(json.loads(self.path.read_text())) if self.path.exists() else empty_state()

    def save(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


class GitHubStateStore:
    BRANCH = "codex/venue-monitor-state"
    FILE = "venue-monitor-state.json"

    def __init__(self, client, repository):
        self.client = client
        self.base = f"https://api.github.com/repos/{repository}"
        self.sha = None

    def request(self, method, path, **kwargs):
        for attempt in range(3):
            try:
                response = self.client.request(method, self.base + path, **kwargs)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    # Keep GitHub's explanation, not credentials or uploaded state.
                    try:
                        message = str(response.json().get("message", ""))[:500]
                    except (ValueError, AttributeError):
                        message = response.reason_phrase
                    details = {key: response.headers[key] for key in (
                        "x-github-request-id", "retry-after", "x-ratelimit-remaining", "x-ratelimit-reset"
                    ) if key in response.headers}
                    raise httpx.HTTPStatusError(
                        f"GitHub {method} {path}: HTTP {response.status_code}: {message}; {details}",
                        request=exc.request, response=response,
                    ) from exc
                return response.json()
            except httpx.HTTPError as exc:
                delay = retry_delay(exc, attempt)
                # Writes reconcile their SHA in save() before trying again.
                if method != "GET" or delay is None or attempt == 2:
                    raise
                logger.warning(f"GitHub read failed; retrying in {delay}s: {exc}")
                sleep(delay)

    def load(self):
        branch = quote(self.BRANCH, safe="")
        try:
            self.request("GET", f"/git/ref/heads/{branch}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            repo = self.request("GET", "")
            ref = self.request("GET", "/git/ref/heads/" + quote(repo["default_branch"], safe=""))
            self.request("POST", "/git/refs", json={
                "ref": f"refs/heads/{self.BRANCH}", "sha": ref["object"]["sha"],
            })
            state = empty_state()
            self.save(state)
            return state
        # An existing branch with missing/corrupt state must NOT silently reset delivery history.
        payload = self.request("GET", f"/contents/{self.FILE}", params={"ref": self.BRANCH})
        self.sha = payload["sha"]
        if payload.get("encoding") != "base64":
            payload = self.request("GET", f"/git/blobs/{self.sha}")
        return validate_state(json.loads(base64.b64decode(payload["content"])))

    def save(self, state):
        content = json.dumps(state, ensure_ascii=False).encode()
        body = {
            "message": "Update daily venue monitor delivery state",
            "branch": self.BRANCH,
            "content": base64.b64encode(content).decode(),
        }
        if self.sha:
            body["sha"] = self.sha
        desired_sha = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if desired_sha == self.sha:
            return  # An unchanged checkpoint needs no new GitHub write/commit.
        for attempt in range(3):
            try:
                payload = self.request("PUT", f"/contents/{self.FILE}", json=body)
                self.sha = payload["content"]["sha"]
                return
            except httpx.HTTPError as exc:
                delay = retry_delay(exc, attempt)
                if delay is None:
                    raise
                # Respect rate limits before even reading back the saved state.
                logger.warning(f"GitHub state write failed; checking its result after {delay}s: {exc}")
                sleep(delay)
                current = self.request("GET", f"/contents/{self.FILE}", params={"ref": self.BRANCH})
                if current["sha"] == desired_sha:
                    self.sha = desired_sha  # The ambiguous write actually committed exactly our content.
                    return
                if current["sha"] != self.sha or attempt == 2:
                    raise  # Never overwrite another writer's delivery history.
