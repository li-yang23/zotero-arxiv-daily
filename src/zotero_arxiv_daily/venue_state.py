"""Delivery state is durable in Git, not in an expiring Actions cache."""

import base64
import json
from pathlib import Path
from urllib.parse import quote


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
        body = {
            "message": "Update daily venue monitor delivery state",
            "branch": self.BRANCH,
            "content": base64.b64encode(json.dumps(state, ensure_ascii=False).encode()).decode(),
        }
        if self.sha:
            body["sha"] = self.sha
        # No blind retry on an ambiguous write or SHA conflict, especially before SMTP.
        payload = self.request("PUT", f"/contents/{self.FILE}", json=body)
        self.sha = payload["content"]["sha"]
