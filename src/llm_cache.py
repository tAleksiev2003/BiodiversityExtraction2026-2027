"""Disk cache for LLM answers, so no prompt is ever paid for twice.

The free SAIA tier allows only 400 requests per day (SPEC.md, section 5). Every answer is
stored under a hash of everything that determines it (model, full prompt, temperature).
A stopped run (Ctrl + C) therefore continues where it ended: all prompts that were
already answered come from the disk.

Only the answer and a few metadata are stored, not the prompt itself: prompts of the
full-text pipeline contain whole documents, and the run folder already keeps them.

Several threads use one cache at the same time (parallel requests). Reading needs no
lock, because a cache file only appears by an atomic rename, complete or not at all.
Writing is locked, and every writer uses its own temporary file name, so two threads
never write into the same temporary file.
"""
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}

CACHE_FILE_SUFFIX = ".json"


def cache_key(model: str, messages: list[Message], temperature: float) -> str:
    """Hash of model, full prompt and temperature.

    `sort_keys` makes the JSON text (and with it the hash) independent of the order in
    which the dict keys were written.
    """
    request = {"model": model, "messages": messages, "temperature": temperature}
    request_text = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(request_text.encode("utf-8")).hexdigest()


class ResponseCache:
    """One JSON file per answer: `<cache_dir>/<key>.json`."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._write_lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}{CACHE_FILE_SUFFIX}"

    def get(self, key: str) -> str | None:
        """The cached answer text, or None if this prompt was never answered."""
        path = self._path(key)
        if not path.exists():
            return None
        entry: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return entry["content"]

    def put(self, key: str, model: str, temperature: float, content: str) -> None:
        """Store an answer.

        The file is written under a temporary name and then renamed. A rename is atomic,
        so Ctrl + C during the write never leaves a half-written cache file behind.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "model": model,
            "temperature": temperature,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "content": content,
        }
        final_path = self._path(key)
        # Process and thread id in the name: two writers never share a temporary file.
        temporary_name = f"{final_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        temporary_path = final_path.with_name(temporary_name)
        with self._write_lock:
            temporary_path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary_path, final_path)
