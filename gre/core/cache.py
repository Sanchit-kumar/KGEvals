"""Content-addressed cache (SQLite) for LLM calls and stage outputs.

Key = hash(adapter name + version/source hash + params + input hash + model + prompt version),
so changing only the resolve stage reuses cached extraction outputs.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _default(o: Any):
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json", by_alias=True)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=_default, ensure_ascii=False, separators=(",", ":"))


def content_hash(*parts: Any, n: int = 16) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(stable_json(p).encode())
        h.update(b"\x00")
    return h.hexdigest()[:n]


def source_hash(obj: Any) -> str:
    """Hash of the callable's whole module source, so editing user code (including module-level
    constants and helpers the function uses) invalidates its cached outputs."""
    try:
        mod = inspect.getmodule(obj)
        src = inspect.getsource(mod) if mod is not None else inspect.getsource(obj)
        return content_hash(getattr(obj, "__qualname__", ""), src, n=12)
    except (OSError, TypeError):
        try:
            return content_hash(inspect.getsource(obj), n=12)
        except (OSError, TypeError):
            return getattr(obj, "__qualname__", repr(obj))


class Cache:
    def __init__(self, path: str | Path | None, enabled: bool = True):
        self.enabled = enabled and path is not None
        self._lock = threading.Lock()
        if self.enabled:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(path), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("CREATE TABLE IF NOT EXISTS kv (ns TEXT, k TEXT, v TEXT, PRIMARY KEY(ns, k))")
            self._db.commit()

    def get(self, ns: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE ns=? AND k=?", (ns, key)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, ns: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)", (ns, key, stable_json(value)))
            self._db.commit()

    def close(self) -> None:
        if self.enabled:
            self._db.close()
