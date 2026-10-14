"""Delivery state is durable in Git, not in an expiring Actions cache."""

import base64
import json
import hashlib
from pathlib import Path
from time import sleep
from urllib.parse import quote

import httpx


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
        response = self.client.request(method, self.base + path, **kwargs)
        response.raise_for_status()
        return response.json()

    def load(self):
        branch = quote(self.BRANCH, safe="")
        response = self.client.get(f"{self.base}/git/ref/heads/{branch}")
        if response.status_code == 404:
            repo = self.request("GET", "")
            ref = self.request("GET", "/git/ref/heads/" + quote(repo["default_branch"], safe=""))
            self.request("POST", "/git/refs", json={
                "ref": f"refs/heads/{self.BRANCH}", "sha": ref["object"]["sha"],
            })
            state = empty_state()
            self.save(state)
            return state
        response.raise_for_status()
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
        for attempt in range(3):
            try:
                payload = self.request("PUT", f"/contents/{self.FILE}", json=body)
                self.sha = payload["content"]["sha"]
                return
            except httpx.HTTPError as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in {409, 500, 502, 503, 504}:
                    raise
                current = self.request("GET", f"/contents/{self.FILE}", params={"ref": self.BRANCH})
                if current["sha"] == desired_sha:
                    self.sha = desired_sha  # The ambiguous write actually committed exactly our content.
                    return
                if current["sha"] != self.sha or attempt == 2:
                    raise  # Never overwrite another writer's delivery history.
                sleep(2 ** attempt)
