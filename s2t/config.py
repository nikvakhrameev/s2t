"""Configuration: dataclasses with defaults, overridable from a YAML file."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_SEARCH_PATHS = (
    Path.cwd() / "config.yaml",
    Path.home() / ".config" / "s2t" / "config.yaml",
    PROJECT_ROOT / "config.yaml",
)


@dataclass
class VadConfig:
    enabled: bool = True
    model_path: str = "models/silero_vad.onnx"
    threshold: float = 0.5
    # Speech shorter than this is treated as noise (clicks, key presses).
    min_speech_ms: int = 150
    # Silence shorter than this never splits a speech segment.
    min_silence_ms: int = 250
    # Audio kept before the first / after the last speech segment.
    edge_pad_ms: int = 200
    # Internal pauses longer than this are shortened to exactly this length.
    max_pause_ms: int = 500


@dataclass
class SttConfig:
    model: str = "mlx-community/whisper-large-v3-turbo"
    # "auto" = detect per request. A hotkey / API call may override it.
    language: str = "auto"
    # "auto" picks the most probable language among these only.
    auto_languages: tuple[str, ...] = ("ru", "en")
    # Temperature fallback ladder: retried only when a decode looks degenerate.
    temperatures: tuple[float, ...] = (0.0, 0.2, 0.4)
    no_speech_threshold: float = 0.6
    compression_ratio_threshold: float = 2.4
    logprob_threshold: float = -1.0
    # Glossary terms are fed to Whisper as a prompt; cap its size (chars).
    max_prompt_chars: int = 600
    # Dropped when they make up a whole segment (classic Whisper hallucinations).
    hallucination_phrases: tuple[str, ...] = (
        "субтитры сделал",
        "субтитры создавал",
        "субтитры подготовил",
        "редактор субтитров",
        "корректор а.",
        "продолжение следует",
        "спасибо за просмотр",
        "подписывайтесь на канал",
        "thanks for watching",
        "thank you for watching",
        "subtitles by",
        "amara.org",
    )


@dataclass
class JevConfig:
    # Semantic check of every cleaned chunk against its raw text by TypeSafe's Jev
    # (see jev.py). Jev is a cloud API: with this on, the chunk texts leave the
    # machine.
    enabled: bool = False
    # The API key: $TYPESAFE_API_KEY, else the contents of this file. Keep it out
    # of config.yaml, which is under git.
    api_key_file: str = "~/.config/s2t/typesafe_api_key"
    # both: a chunk must pass the word-level heuristics and then Jev.
    # only: Jev replaces the heuristics (just the free "empty output" check stays).
    mode: str = "both"
    # An alias; pin the versioned id (jev-1.13.0) once the thresholds are tuned.
    model: str = "jev-latest"
    # Questions asked about each chunk (ids from jev.QUESTIONS; one request, answered
    # in parallel) -> the highest tolerated risk, 0..1. Above it the chunk is rejected.
    # The values are a starting point, not calibrated yet: scripts/eval_jev.py.
    questions: dict[str, float] = field(
        default_factory=lambda: {
            "content_added": 0.5,
            "content_dropped": 0.5,
            "polarity_flipped": 0.5,
            "details_changed": 0.5,
            "roles_swapped": 0.5,
            "responded": 0.5,
        }
    )
    # Wall-clock deadline for one verdict, counted from the moment it is requested.
    timeout_s: float = 1.5
    # Jev unreachable / timed out / over quota:
    # heuristics = the heuristics alone decide, reject = keep the raw chunk.
    on_error: str = "heuristics"


@dataclass
class CleanupConfig:
    enabled: bool = True
    model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    # Texts shorter than this skip the LLM entirely.
    min_words: int = 3
    # Long texts are cleaned in chunks of about this many characters.
    # Small chunks keep a 4-bit model stable and make guardrail fallbacks local.
    chunk_chars: int = 220
    # Guardrails (see cleanup.py): invented words and negation flips are never
    # accepted; besides fillers/stutters, at most this share of the transcript's
    # words may disappear. On violation the raw transcript is returned.
    max_dropped_ratio: float = 0.1
    # Share of the transcript's negations (не/not/...) that may disappear. 0.5:
    # losing 1 of 2 is fine (hesitation repeats), losing the only one is rejected.
    # 0 = strict. Gaining a negation is never accepted.
    max_negation_loss: float = 0.5
    max_length_ratio: float = 1.3
    # How many glossary terms to show the LLM.
    max_glossary_terms: int = 150
    # An output word counts as a respelling of a transcript word ("конекшенов" ->
    # "коннекшенов") at this difflib.get_close_matches ratio or above.
    spelling_similarity: float = 0.75
    # Same, after Cyrillic->Latin transliteration ("питоне" -> "Python").
    translit_similarity: float = 0.6
    # Words shorter than this are not checked by the invented/dropped-word guardrails.
    min_word_chars: int = 3
    # Words the cleaner may delete freely (single words of multi-word fillers too).
    fillers: tuple[str, ...] = (
        "ну", "вот", "это", "как", "бы", "типа", "в", "общем", "короче", "значит",
        "так", "самое", "то", "есть", "слушай", "смотри",
        "э", "ээ", "эээ", "эм", "мм", "ммм", "а", "и", "да", "же",
        "um", "uh", "er", "hmm", "like", "you", "know", "i", "mean", "so", "well",
        "actually", "basically", "kind", "sort", "of", "right", "okay", "ok",
    )
    # Negations that carry meaning; see max_negation_loss.
    negations: tuple[str, ...] = ("не", "ни", "нет", "not", "no", "never")
    jev: JevConfig = field(default_factory=JevConfig)


@dataclass
class HotkeyBinding:
    key: str = "alt_r"  # pynput key name: alt_r, cmd_r, ctrl_r, f13, ...
    language: str = "auto"  # auto | ru | en | ...


@dataclass
class OverlayConfig:
    # Pill above all windows while recording: pulsing dot + elapsed time.
    enabled: bool = True
    # top | bottom | top_left | top_right | bottom_left | bottom_right,
    # on the screen under the mouse pointer (menu bar and Dock excluded).
    position: str = "top"
    margin: int = 12  # points from the screen edge
    scale: float = 1.0  # 1.0 = 30 pt high


@dataclass
class DictationConfig:
    enabled: bool = True
    # Push-to-talk: hold the key to record, release to transcribe and paste.
    hotkeys: list[HotkeyBinding] = field(
        default_factory=lambda: [HotkeyBinding("alt_r", "auto")]
    )
    min_record_ms: int = 300
    max_record_s: int = 300
    paste: bool = True  # False = only copy to the clipboard
    restore_clipboard: bool = True
    sounds: bool = True
    # True = zero start latency, but the macOS mic indicator stays on.
    keep_mic_open: bool = False
    input_device: str | int | None = None
    overlay: OverlayConfig = field(default_factory=OverlayConfig)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class HistoryConfig:
    # Journal of every transcription: raw vs. cleaned text, timings, guardrail hits.
    enabled: bool = True
    # JSON Lines, one record per request. Holds everything that was dictated.
    path: str = "history.jsonl"
    # Which requests are journaled: dictation (hotkey) | api (HTTP) | cli (--local).
    origins: tuple[str, ...] = ("dictation", "api", "cli")


@dataclass
class Config:
    glossary_path: str = "glossary.yaml"
    vad: VadConfig = field(default_factory=VadConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    dictation: DictationConfig = field(default_factory=DictationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)

    def resolve(self, path: str) -> Path:
        return resolve_path(path)


def resolve_path(path: str) -> Path:
    """Relative paths in the config are relative to the project root."""
    p = Path(os.path.expanduser(path))
    return p if p.is_absolute() else PROJECT_ROOT / p


def _merge(obj: Any, data: dict[str, Any], where: str = "") -> Any:
    """Apply a dict of overrides on top of a dataclass instance."""
    fields = {f.name: f for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in fields:
            raise ValueError(f"Unknown config key: {where}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value, f"{where}{key}.")
        elif key == "hotkeys":
            setattr(obj, key, [HotkeyBinding(**item) for item in value])
        elif isinstance(current, tuple) and isinstance(value, list):
            # PyYAML parses bare `no`/`yes`/etc. as booleans, which would silently
            # drop a word like "no" from cleanup.negations - catch it here.
            if all(isinstance(v, str) for v in current):
                bad = [v for v in value if not isinstance(v, str)]
                if bad:
                    raise ValueError(
                        f"{where}{key}: entry {bad[0]!r} is not a string - quote it "
                        f'in YAML (e.g. "no")'
                    )
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)
    return obj


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    config = Config()
    candidates = [Path(path)] if path else list(CONFIG_SEARCH_PATHS)
    for candidate in candidates:
        if candidate.is_file():
            data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            return _merge(config, data)
    if path:
        raise FileNotFoundError(path)
    return config
