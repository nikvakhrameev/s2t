"""Glossary of fixed terms, shared by the STT prompt, alias replacement and the LLM.

glossary.yaml format:

    terms:
      - Kubernetes                      # canonical spelling only
      - term: PostgreSQL                # canonical + how it tends to be misheard
        aliases: [постгрес, постгря]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Term:
    term: str
    aliases: tuple[str, ...] = ()


@dataclass
class Glossary:
    terms: list[Term] = field(default_factory=list)
    _alias_re: re.Pattern[str] | None = field(default=None, repr=False)
    _alias_map: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._alias_map = {
            alias.lower(): t.term for t in self.terms for alias in t.aliases if alias.strip()
        }
        if self._alias_map:
            # Longest first so "клод код" wins over "клод".
            ordered = sorted(self._alias_map, key=len, reverse=True)
            body = "|".join(re.escape(a) for a in ordered)
            self._alias_re = re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)

    @classmethod
    def from_file(cls, path: Path) -> "Glossary":
        if not path.is_file():
            return cls()
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        terms: list[Term] = []
        for item in data.get("terms") or []:
            if isinstance(item, str):
                terms.append(Term(item.strip()))
            elif isinstance(item, dict) and item.get("term"):
                aliases = tuple(str(a).strip() for a in item.get("aliases") or [])
                terms.append(Term(str(item["term"]).strip(), aliases))
            else:
                raise ValueError(f"Bad glossary entry in {path}: {item!r}")
        return cls(terms)

    def __bool__(self) -> bool:
        return bool(self.terms)

    def canonical(self, limit: int | None = None) -> list[str]:
        names = [t.term for t in self.terms]
        return names[:limit] if limit else names

    def stt_prompt(self, max_chars: int) -> str | None:
        """Whisper biases towards spellings seen in its prompt (~224 tokens max)."""
        picked: list[str] = []
        size = 0
        for name in self.canonical():
            if size + len(name) + 2 > max_chars:
                break
            picked.append(name)
            size += len(name) + 2
        return ", ".join(picked) + "." if picked else None

    def apply_aliases(self, text: str) -> str:
        """Deterministic replacement of known mis-hearings with the canonical term."""
        if not self._alias_re:
            return text
        return self._alias_re.sub(lambda m: self._alias_map[m.group(0).lower()], text)


class GlossaryStore:
    """Reloads the glossary when the file changes on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime: float | None = None
        self._glossary = Glossary()
        self.version = 0

    def get(self) -> Glossary:
        mtime = self.path.stat().st_mtime if self.path.is_file() else None
        if mtime != self._mtime:
            self._glossary = Glossary.from_file(self.path)
            self._mtime = mtime
            self.version += 1
        return self._glossary
