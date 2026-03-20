"""
SessionStore — persists chat history to disk as JSON files.

Layout:
  sessions/
    {session_id}.json   ← one file per session
      {
        "session_id": "...",
        "title": "First user message (truncated)",
        "created_at": "ISO timestamp",
        "updated_at": "ISO timestamp",
        "messages": [
          {"role": "user"|"assistant", "content": "...", "ts": "ISO"}
        ]
      }
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings

settings = get_settings()


def _sessions_dir() -> Path:
    p = Path(settings.sessions_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _session_path(session_id: str) -> Path:
    return _sessions_dir() / f"{session_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_session(session_id: str) -> dict:
    path = _session_path(session_id)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "session_id": session_id,
        "title": "New chat",
        "created_at": _now(),
        "updated_at": _now(),
        "messages": [],
    }


def save_session(session_id: str, data: dict) -> None:
    data["updated_at"] = _now()
    path = _session_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_message(session_id: str, role: str, content: str) -> None:
    data = load_session(session_id)
    msg = {"role": role, "content": content, "ts": _now()}
    data["messages"].append(msg)
    # Set title from first user message
    if role == "user" and data["title"] == "New chat":
        data["title"] = content[:60] + ("…" if len(content) > 60 else "")
    save_session(session_id, data)


def list_sessions() -> list[dict]:
    """Return all sessions sorted by updated_at desc (for sidebar)."""
    sessions_dir = _sessions_dir()
    sessions = []
    for f in sessions_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            sessions.append({
                "session_id": data["session_id"],
                "title": data.get("title", "Chat"),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            continue
    return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)


def delete_session(session_id: str) -> None:
    path = _session_path(session_id)
    if path.exists():
        os.remove(path)


def get_messages(session_id: str) -> list[dict]:
    return load_session(session_id).get("messages", [])
