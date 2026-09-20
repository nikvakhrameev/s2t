"""Silero VAD on raw onnxruntime: trim edge silence and shorten internal pauses.

Whisper hallucinates mostly on non-speech audio ("Субтитры сделал ...",
"Thanks for watching"), so silence never reaches it: leading/trailing silence
is cut, long pauses are shortened, and audio without speech skips STT entirely.
Pauses are shortened (not removed) and filled with the original room tone, so
sentence boundaries stay audible and there are no digital-silence artifacts.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .audio import SAMPLE_RATE
from .config import VadConfig

MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
WINDOW = 512  # samples per VAD step at 16 kHz (32 ms)
CONTEXT = 64  # samples of the previous window prepended to each step


@dataclass
class VadResult:
    audio: np.ndarray
    segments: list[tuple[int, int]]  # speech regions in the ORIGINAL audio (samples)
    original_s: float
    kept_s: float

    @property
    def has_speech(self) -> bool:
        return bool(self.segments)


def ensure_model(path: Path) -> Path:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.rename(path)
    return path


class SileroVad:
    def __init__(self, config: VadConfig, model_path: Path) -> None:
        self.config = config
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(ensure_model(model_path)),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def speech_probs(self, audio: np.ndarray) -> np.ndarray:
        """Speech probability for every 32 ms window."""
        n_windows = -(-len(audio) // WINDOW)
        padded = np.zeros(n_windows * WINDOW, dtype=np.float32)
        padded[: len(audio)] = audio
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT), dtype=np.float32)
        rate = np.array(SAMPLE_RATE, dtype=np.int64)
        probs = np.empty(n_windows, dtype=np.float32)
        for i in range(n_windows):
            chunk = padded[i * WINDOW : (i + 1) * WINDOW][None, :]
            frame = np.concatenate([context, chunk], axis=1)
            out, state = self.session.run(None, {"input": frame, "state": state, "sr": rate})
            probs[i] = out[0, 0]
            context = frame[:, -CONTEXT:]
        return probs

    def speech_segments(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Speech regions as (start, end) sample offsets, with hysteresis."""
        cfg = self.config
        probs = self.speech_probs(audio)
        on, off = cfg.threshold, max(cfg.threshold - 0.15, 0.01)
        min_silence = cfg.min_silence_ms * SAMPLE_RATE // 1000
        min_speech = cfg.min_speech_ms * SAMPLE_RATE // 1000

        segments: list[tuple[int, int]] = []
        start: int | None = None
        silence_from: int | None = None
        for i, p in enumerate(probs):
            pos = i * WINDOW
            if p >= on:
                silence_from = None
                if start is None:
                    start = pos
            elif p < off and start is not None:
                if silence_from is None:
                    silence_from = pos
                if pos + WINDOW - silence_from >= min_silence:
                    segments.append((start, silence_from))
                    start, silence_from = None, None
        if start is not None:
            segments.append((start, silence_from if silence_from is not None else len(audio)))
        return [(s, min(e, len(audio))) for s, e in segments if e - s >= min_speech]

    def process(self, audio: np.ndarray) -> VadResult:
        cfg = self.config
        segments = self.speech_segments(audio)
        original_s = len(audio) / SAMPLE_RATE
        if not segments:
            return VadResult(np.zeros(0, dtype=np.float32), [], original_s, 0.0)

        edge_pad = cfg.edge_pad_ms * SAMPLE_RATE // 1000
        half_pause = cfg.max_pause_ms * SAMPLE_RATE // 2000
        # Keep each speech region plus half of the allowed pause on both sides:
        # pauses <= max_pause_ms survive untouched, longer ones shrink to max_pause_ms.
        keep: list[list[int]] = []
        for index, (start, end) in enumerate(segments):
            left = edge_pad if index == 0 else half_pause
            right = edge_pad if index == len(segments) - 1 else half_pause
            region = [max(0, start - left), min(len(audio), end + right)]
            if keep and region[0] <= keep[-1][1]:
                keep[-1][1] = max(keep[-1][1], region[1])
            else:
                keep.append(region)

        trimmed = np.concatenate([audio[s:e] for s, e in keep])
        return VadResult(trimmed, segments, original_s, len(trimmed) / SAMPLE_RATE)
