"""Decode any audio/video container to 16 kHz mono float32 using ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000


class AudioDecodeError(RuntimeError):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioDecodeError("ffmpeg not found in PATH (brew install ffmpeg)")
    return exe


def _run(args: list[str], stdin: bytes | None) -> np.ndarray:
    proc = subprocess.run(
        [_ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error", *args,
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        input=stdin,
        capture_output=True,
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", "replace").strip()
        raise AudioDecodeError(f"ffmpeg failed: {message or proc.returncode}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise AudioDecodeError("No audio stream could be decoded from the input")
    return samples


def decode_file(path: str | Path) -> np.ndarray:
    """Any file ffmpeg understands (mp3, m4a, ogg/opus, webm, mp4, wav, ...)."""
    path = Path(path)
    if not path.is_file():
        raise AudioDecodeError(f"No such file: {path}")
    return _run(["-i", str(path)], None)


def decode_bytes(data: bytes) -> np.ndarray:
    """Same as decode_file, for an in-memory upload (format is auto-detected).

    Goes through a temp file, not a pipe: MP4/M4A/MOV keep their index at the
    end of the file and ffmpeg cannot seek in a pipe, so it would decode nothing.
    """
    if not data:
        raise AudioDecodeError("Empty audio payload")
    with tempfile.NamedTemporaryFile(prefix="s2t-", suffix=".bin") as handle:
        handle.write(data)
        handle.flush()
        return _run(["-i", handle.name], None)


def resample_pcm(samples: np.ndarray, rate: int) -> np.ndarray:
    """Mono float32 PCM at an arbitrary rate -> 16 kHz."""
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if rate == SAMPLE_RATE:
        return samples
    return _run(["-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "pipe:0"],
                samples.tobytes())
