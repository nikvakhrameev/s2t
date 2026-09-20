"""Append-only journal of transcriptions (JSON Lines).

Dictation only shows the final text; the journal keeps what Whisper heard, what
the LLM made of it and which guardrails fired, so STT mistakes, alias candidates
and guardrail thresholds can be worked out from real use:

    tail -n 5 history.jsonl | jq '{raw_text, text, cleanup_rejections}'
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING

from .config import Config

if TYPE_CHECKING:
    from .pipeline import Result


class History:
    """Not thread-safe; Engine appends from its single worker thread."""

    def __init__(self, config: Config) -> None:
        self.config = config.history
        self.path = config.resolve(config.history.path)
        self._default_language = config.stt.language
        self._models = {"stt": config.stt.model, "cleanup": config.cleanup.model}

    def append(
        self, result: Result, origin: str, language: str | None = None, file: str | None = None
    ) -> None:
        if not self.config.enabled or origin not in self.config.origins:
            return
        record = {
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "origin": origin,
            **({"file": file} if file else {}),
            "requested_language": language or self._default_language,
            **result.to_dict(),
            "models": self._models,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 0600: the journal holds everything that was dictated.
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as error:  # a broken journal must never break dictation
            print(f"[history] cannot write {self.path}: {error}", file=sys.stderr, flush=True)
