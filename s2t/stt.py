"""Speech-to-text backend: Whisper on MLX.

The backend is deliberately small (load / transcribe) so language-specific
models (GigaAM for ru, Parakeet for en, ...) can be plugged in later and bound
to their own hotkeys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .config import SttConfig


@dataclass
class SttResult:
    text: str
    language: str
    dropped_segments: int = 0


def _norm(text: str) -> str:
    return re.sub(r"[^\w\s.]+", "", text.lower()).strip()


class WhisperMlx:
    def __init__(self, config: SttConfig) -> None:
        self.config = config
        self._model = None

    def load(self) -> None:
        import mlx.core as mx
        from mlx_whisper.transcribe import ModelHolder

        # Same holder transcribe() uses, so the weights are loaded exactly once.
        self._model = ModelHolder.get_model(self.config.model, mx.float16)
        # Warm-up compiles the Metal kernels; the first real request stays fast.
        self.transcribe(np.zeros(16_000, dtype=np.float32), language="en")

    def detect_language(self, audio: np.ndarray, allowed: tuple[str, ...]) -> str:
        """Language id restricted to `allowed`.

        Unrestricted Whisper detection sometimes picks uk/bg/be for short
        Russian clips full of English terms; choosing among the languages the
        user actually speaks removes that failure mode.
        """
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

        mel = log_mel_spectrogram(audio, n_mels=self._model.dims.n_mels, padding=N_SAMPLES)
        segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = self._model.detect_language(segment)
        return max(allowed, key=lambda lang: probs.get(lang, 0.0))

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "auto",
        prompt: str | None = None,
    ) -> SttResult:
        import mlx_whisper

        cfg = self.config
        if language == "auto":
            language = self.detect_language(audio, cfg.auto_languages)
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=cfg.model,
            language=language,
            initial_prompt=prompt,
            temperature=tuple(cfg.temperatures),
            no_speech_threshold=cfg.no_speech_threshold,
            compression_ratio_threshold=cfg.compression_ratio_threshold,
            logprob_threshold=cfg.logprob_threshold,
            # Conditioning on previous windows is what turns one hallucination
            # into a repetition loop on long audio.
            condition_on_previous_text=False,
            verbose=None,
        )

        kept: list[str] = []
        dropped = 0
        for segment in result.get("segments", []):
            text = segment["text"].strip()
            if not text:
                continue
            if self._is_hallucination(text, prompt):
                dropped += 1
                continue
            kept.append(text)
        return SttResult(" ".join(kept).strip(), result.get("language", language), dropped)

    def _is_hallucination(self, text: str, prompt: str | None) -> bool:
        norm = _norm(text)
        if any(norm.startswith(phrase) for phrase in self.config.hallucination_phrases):
            return True
        # On unclear audio Whisper may simply recite its own prompt back.
        return bool(prompt) and len(norm) > 20 and norm.rstrip(".") == _norm(prompt).rstrip(".")
