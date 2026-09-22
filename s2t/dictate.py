"""Push-to-talk dictation: hold a hotkey to record, release to transcribe + paste.

macOS permissions needed by the app that runs this (Terminal / iTerm / ...):
Microphone, Input Monitoring (global hotkey) and Accessibility (Cmd+V paste).
"""

from __future__ import annotations

import subprocess
import threading
import time

import numpy as np
import sounddevice as sd
from pynput import keyboard

from .audio import SAMPLE_RATE, resample_pcm
from .config import DictationConfig
from .overlay import Overlay
from .pipeline import Engine

SOUND_START = "/System/Library/Sounds/Tink.aiff"
SOUND_DONE = "/System/Library/Sounds/Pop.aiff"
SOUND_EMPTY = "/System/Library/Sounds/Basso.aiff"
KEY_V = keyboard.KeyCode.from_vk(9)  # kVK_ANSI_V


def _log(message: str) -> None:
    print(f"[dictate] {message}", flush=True)


def resolve_key(name: str):
    """'alt_r' / 'f13' -> pynput Key, single characters -> KeyCode."""
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(f"Unknown hotkey: {name!r} (use pynput key names, e.g. alt_r, cmd_r, f13)")


class Recorder:
    """Microphone capture.

    Opening a CoreAudio input takes ~300 ms. By default the stream is opened on
    key press (the start sound plays once it is live). With keep_open=True the
    stream stays open, so recording starts instantly - at the price of the macOS
    microphone indicator being on all the time.
    """

    def __init__(self, device: str | int | None, keep_open: bool = False) -> None:
        self.device = device
        self.keep_open = keep_open
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._rate = SAMPLE_RATE
        self._recording = False

    def open(self) -> None:
        if self._stream is not None:
            return
        try:
            self._stream = self._create(SAMPLE_RATE)
        except sd.PortAudioError:
            # Device refuses 16 kHz: record at its native rate, resample later.
            native = int(sd.query_devices(self.device, "input")["default_samplerate"])
            self._stream = self._create(native)
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _create(self, rate: int) -> sd.InputStream:
        self._rate = rate
        return sd.InputStream(
            samplerate=rate, channels=1, dtype="float32", device=self.device,
            callback=self._on_audio,
        )

    def _on_audio(self, data, frames, timestamp, status) -> None:
        if self._recording:
            self._chunks.append(data[:, 0].copy())

    def start(self) -> None:
        self._chunks = []
        self.open()
        self._recording = True

    def stop(self) -> np.ndarray:
        self._recording = False
        if not self.keep_open:
            self.close()
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return resample_pcm(np.concatenate(self._chunks), self._rate)


class Dictation:
    def __init__(self, config: DictationConfig, engine: Engine) -> None:
        self.config = config
        self.engine = engine
        self.bindings = {resolve_key(b.key): b.language for b in config.hotkeys}
        self.recorder = Recorder(config.input_device, config.keep_mic_open)
        self.overlay = Overlay(config.overlay)
        self._active = None  # the key currently held for recording
        self._started_at = 0.0
        self._lock = threading.Lock()
        self._typer = keyboard.Controller()
        self._listener: keyboard.Listener | None = None

    # -- hotkey handling -----------------------------------------------------

    def start(self) -> None:
        if self.config.keep_mic_open:
            self.recorder.open()
        self.overlay.start()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
        self.recorder.close()
        self.overlay.close()

    def _on_press(self, key) -> None:
        with self._lock:
            if self._active is not None or key not in self.bindings:
                return
            self._active = key
            self._started_at = time.monotonic()
        try:
            self.recorder.start()
        except Exception as error:  # noqa: BLE001 - must never kill the listener
            _log(f"cannot open microphone: {error}")
            with self._lock:
                self._active = None
            return
        # Both cues mean "the microphone is live now", so the timer shows what was really recorded.
        self.overlay.show()
        self._play(SOUND_START)

    def _on_release(self, key) -> None:
        with self._lock:
            if key != self._active:
                return
            self._active = None
            held_ms = (time.monotonic() - self._started_at) * 1000
        self.overlay.hide()
        samples = self.recorder.stop()
        if held_ms < self.config.min_record_ms:
            return
        language = self.bindings[key]
        # Never block the listener thread: macOS disables slow event taps.
        threading.Thread(target=self._process, args=(samples, language), daemon=True).start()

    # -- transcription + output ----------------------------------------------

    def _process(self, samples: np.ndarray, language: str) -> None:
        try:
            result = self.engine.run(
                samples[: self.config.max_record_s * SAMPLE_RATE], language, origin="dictation"
            )
        except Exception as error:  # noqa: BLE001
            _log(f"failed: {error}")
            self._play(SOUND_EMPTY)
            return
        if not result.text:
            _log(f"no speech ({result.audio_s}s of audio)")
            self._play(SOUND_EMPTY)
            return
        _log(f"{result.language} {result.timings_ms} -> {result.text}")
        self._output(result.text)
        self._play(SOUND_DONE)

    def _output(self, text: str) -> None:
        previous = _clipboard_get() if self.config.restore_clipboard else None
        _clipboard_set(text)
        if not self.config.paste:
            return
        time.sleep(0.05)
        with self._typer.pressed(keyboard.Key.cmd):
            # Physical key code, not the character: tap("v") cannot be resolved
            # while a non-Latin (e.g. Russian) keyboard layout is active.
            self._typer.tap(KEY_V)
        if previous is not None:
            time.sleep(0.4)  # let the target app read the clipboard first
            _clipboard_set(previous)

    def _play(self, sound: str) -> None:
        if self.config.sounds:
            subprocess.Popen(["afplay", sound], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# pbcopy/pbpaste pick the encoding from the locale; without it Cyrillic is mangled.
_UTF8_ENV = {"LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8", "PATH": "/usr/bin:/bin"}


def _clipboard_get() -> str | None:
    try:
        out = subprocess.run(["pbpaste"], capture_output=True, timeout=2, env=_UTF8_ENV)
        return out.stdout.decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def _clipboard_set(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), timeout=2, check=False, env=_UTF8_ENV)
