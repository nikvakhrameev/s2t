"""The full cycle: audio -> VAD -> STT -> alias replacement -> LLM cleanup."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import audio as audio_io
from .cleanup import LlmCleaner
from .config import Config
from .glossary import GlossaryStore
from .history import History
from .stt import WhisperMlx
from .vad import SileroVad


@dataclass
class Result:
    text: str = ""
    raw_text: str = ""
    language: str = ""
    audio_s: float = 0.0
    speech_s: float = 0.0  # what was actually sent to STT after VAD
    no_speech: bool = False
    cleanup_used: bool = False
    cleanup_rejected_chunks: int = 0  # units that fell back to the raw transcript
    # Every guardrail hit (cleanup.Rejection as a dict), incl. chunks rescued by a retry.
    cleanup_rejections: list[dict[str, Any]] = field(default_factory=list)
    # Every Jev request (jev.Verdict as a dict), accepted chunks too: threshold tuning.
    jev_checks: list[dict[str, Any]] = field(default_factory=list)
    hallucinations_dropped: int = 0
    timings_ms: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Pipeline:
    """Not thread-safe; use Engine, which pins all MLX work to one thread."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.glossary = GlossaryStore(config.resolve(config.glossary_path))
        self.vad = SileroVad(config.vad, config.resolve(config.vad.model_path))
        self.stt = WhisperMlx(config.stt)
        self.cleaner = LlmCleaner(config.cleanup)

    def load(self) -> None:
        self.stt.load()
        if self.config.cleanup.enabled:
            self.cleaner.load()
            self.cleaner.warm_up(self.glossary.get())

    def run(
        self,
        source: str | Path | bytes | np.ndarray,
        language: str | None = None,
        cleanup: bool | None = None,
    ) -> Result:
        cfg = self.config
        result = Result()
        timings = result.timings_ms
        started = clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[name] = round((now - clock) * 1000)
            clock = now

        use_llm = cfg.cleanup.enabled if cleanup is None else (cleanup and cfg.cleanup.enabled)
        if use_llm and self.cleaner.judge is not None:
            self.cleaner.judge.warm()  # the TLS handshake overlaps decoding and STT

        if isinstance(source, np.ndarray):
            samples = np.ascontiguousarray(source, dtype=np.float32)
        elif isinstance(source, bytes):
            samples = audio_io.decode_bytes(source)
        else:
            samples = audio_io.decode_file(source)
        result.audio_s = round(len(samples) / audio_io.SAMPLE_RATE, 2)
        lap("decode")

        if cfg.vad.enabled:
            vad = self.vad.process(samples)
            samples = vad.audio
            lap("vad")
            if not vad.has_speech:
                result.no_speech = True
                timings["total"] = round((time.perf_counter() - started) * 1000)
                return result
        result.speech_s = round(len(samples) / audio_io.SAMPLE_RATE, 2)

        glossary = self.glossary.get()
        stt = self.stt.transcribe(
            samples,
            language=language or cfg.stt.language,
            prompt=glossary.stt_prompt(cfg.stt.max_prompt_chars),
        )
        result.language = stt.language
        result.hallucinations_dropped = stt.dropped_segments
        result.raw_text = stt.text
        lap("stt")

        text = glossary.apply_aliases(stt.text)
        if text and use_llm:
            cleaned = self.cleaner.clean(text, glossary)
            text = cleaned.text
            result.cleanup_used = cleaned.used_llm
            result.cleanup_rejected_chunks = cleaned.rejected_chunks
            result.cleanup_rejections = [asdict(r) for r in cleaned.rejections]
            result.jev_checks = [asdict(v) for v in cleaned.jev_checks]
            lap("cleanup")
            if cleaned.jev_checks:  # the part of "cleanup" spent waiting for Jev's verdicts
                timings["jev"] = cleaned.jev_wait_ms
        result.text = text
        result.no_speech = not text
        timings["total"] = round((time.perf_counter() - started) * 1000)
        return result


class Engine:
    """Owns the models on a single worker thread.

    MLX streams are per-thread, so loading and every inference call must happen
    on the same thread. The HTTP server and the hotkey listener both submit here;
    requests are naturally serialized. Each result is appended to the history
    journal on that thread too; `origin` (dictation | api | cli) and `file` only
    label the journal record.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.history = History(config)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s2t-mlx")
        self._pipeline: Pipeline | None = None

    def _load(self) -> None:
        self._pipeline = Pipeline(self.config)
        self._pipeline.load()

    def load(self) -> None:
        self._executor.submit(self._load).result()

    def _run(self, source, language, cleanup, origin, file) -> Result:
        result = self._pipeline.run(source, language, cleanup)
        self.history.append(result, origin, language, file)
        return result

    def submit(
        self,
        source,
        language: str | None = None,
        cleanup: bool | None = None,
        origin: str = "api",
        file: str | None = None,
    ):
        return self._executor.submit(self._run, source, language, cleanup, origin, file)

    def run(
        self,
        source,
        language: str | None = None,
        cleanup: bool | None = None,
        origin: str = "api",
        file: str | None = None,
    ) -> Result:
        return self.submit(source, language, cleanup, origin, file).result()

    def _unload(self) -> None:
        import mlx.core as mx

        self._pipeline = None
        mx.clear_cache()

    def close(self) -> None:
        # Free the models on their own thread and join it: tearing MLX down from
        # the interpreter's exit path aborts with "recursive_mutex lock failed".
        self._executor.submit(self._unload)
        self._executor.shutdown(wait=True)
