# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Local speech-to-text dictation for macOS / Apple Silicon (dev machine: MacBook M4 Pro, 24 GB). Russian + English. Everything runs on-device. Target: **2–3 s for the full cycle** on a typical 5–15 s dictation.

Pipeline: `any audio → ffmpeg (16 kHz mono f32) → Silero VAD → Whisper large-v3-turbo (MLX) → glossary alias replacement → LLM cleanup (MLX) + guardrails → text`.

Further reading (not auto-loaded — open when relevant):
- `docs/DECISIONS.md` — why things are the way they are; **read before touching guardrails, chunking, models or the glossary layers**.
- `docs/ROADMAP.md` — agreed-but-unbuilt work, research findings on alternative models, current status.
- `README.md` — user-facing docs (Russian), requirements→solution table, measured timings.

## Working with the user

- The user writes in Russian; reply in Russian. Code and code comments are English; `README.md`, `config.yaml`, `glossary.yaml` comments are Russian.
- Do the requested step, report, and **stop**. Do not chain into the next roadmap item unasked — the user steers actively and has interrupted runs that went further than asked (including an unrequested lint run via `uvx`).
- Design decisions are the user's. Give one recommendation with the trade-off, then implement what they choose without re-arguing (example: the negation tolerance in `docs/DECISIONS.md`).
- They prefer **tolerances/thresholds exposed in `config.yaml` over rigid exact-match rules** ("слишком хрупкие правила"). Every guardrail knob should be configurable.
- They asked to use subagents for research/web lookups to keep the main context small.
- The repo lives in the user's private GitHub repo `nikvakhrameev/s2t` (`origin`, branch `main`). Commit and push only when asked. `.idea/` is the user's IDE folder — leave it alone.

## Commands

```bash
uv sync                                        # Python is pinned to 3.12 (MLX wheels); system python3 is 3.14 - always go through uv
uv run pytest -q                               # fast: no MLX models are loaded (~0.5 s)
uv run pytest -q tests/test_core.py::test_negation_guard_understands_hesitations   # single test
scripts/make_test_audio.sh                     # regenerate tests/audio/* via macOS `say` (gitignored; VAD/decoder tests skip without it)

uv run s2t serve [--no-hotkeys]                # warm models + HTTP API (127.0.0.1:8765) + push-to-talk hotkeys
uv run s2t transcribe -v FILE...               # uses a running server if one answers, else loads models locally (~7 s)
uv run s2t transcribe --local -v FILE...       # force in-process; -v prints raw_text, timings, guardrail counters
uv run python scripts/eval_cleanup.py -v MODEL [MODEL...]   # compare cleanup LLMs: guardrail pass-rate + latency on fixed cases
curl -F file=@a.m4a -F language=ru http://127.0.0.1:8765/transcribe
```

No linter is configured. Models download on first use into `~/.cache/huggingface` (Whisper ~1.6 GB, LLM ~2.3 GB); the VAD ONNX (2 MB) downloads into `models/`.

## Architecture (what you cannot see from one file)

- **`pipeline.Engine` pins all MLX work to one worker thread.** MLX streams are per-thread: models must be loaded *and* called on the same thread. The FastAPI handlers and the hotkey listener both submit jobs to `Engine` (a 1-worker executor, so requests are serialized); never call `Pipeline` / MLX from another thread. `Engine` also appends every result to the history journal on that thread (`history.py`, so no locking; `origin` = `dictation | api | cli` labels the record, `history.origins` filters). `Engine.close()` unloads models on that thread and joins it — exiting without it aborts with `libc++abi ... recursive_mutex lock failed`. `cmd_serve` does this in a `finally`.
- **`Pipeline.run`** accepts a path, raw bytes (HTTP upload) or a 16 kHz float32 numpy array (microphone), and returns `Result` with per-stage `timings_ms`. If VAD finds no speech, STT is **never called** — the main anti-hallucination measure (on 6 s of noise Whisper outputs "Thank you." without VAD).
- **VAD (`vad.py`)** is Silero v5 on raw `onnxruntime`: 512-sample windows + 64-sample context, recurrent `state`. Edges are trimmed to `edge_pad_ms`; internal pauses longer than `max_pause_ms` are shortened to exactly that, filled with the original room tone (not digital zeros).
- **STT (`stt.py`)** reuses `mlx_whisper.transcribe.ModelHolder` so weights load once. `language: auto` is *restricted* to `stt.auto_languages` (`[ru, en]`) via our own `detect_language` — unrestricted detection picks uk/bg on short Russian clips with English terms. `condition_on_previous_text=False`; known hallucination phrases and prompt-echo segments are dropped. The backend is intentionally tiny (`load` / `transcribe`) so language-specific models can be added later and bound to their own hotkeys.
- **Glossary (`glossary.py`, `glossary.yaml`) works in three layers**: (1) canonical terms become Whisper's `initial_prompt` (capped at `stt.max_prompt_chars`, so file order = priority); (2) optional `aliases` are replaced deterministically by regex before the LLM; (3) canonical terms go into the LLM system prompt. It is filled **by hand**; `GlossaryStore` hot-reloads on mtime change and the LLM prefix KV cache (keyed by the term tuple) rebuilds itself.
- **Cleanup (`cleanup.py`)**: system prompt + few-shot turns are prefilled once into a KV cache; each request `deepcopy`s it and generates only the suffix (greedy, token budget ∝ input). The transcript is wrapped in `<transcript>` tags and treated as data. Text is split into small chunks (`chunk_chars: 220`).
- **Guardrails are deterministic code, not an LLM** (`LlmCleaner._accept`). A chunk's cleanup is rejected if the output: is empty / too long (`max_length_ratio`); contains *invented words* (every output word must match a transcript word exactly, by close spelling ≥0.75, or via Cyrillic→Latin transliteration ≥0.6, e.g. "питоне"→"Python"); contains mixed-script tokens; **gains** a negation, or loses more than `max_negation_loss` of them (default 0.5: losing the only "не" is rejected, 1 of 2 is tolerated; counts are taken after stripping fillers and collapsing repeated words/phrases); or drops more than `max_dropped_ratio` non-filler words. A rejected chunk is retried sentence-by-sentence; what still fails is returned as the raw transcript (`Result.cleanup_rejected_chunks`). `LlmCleaner._violation` names the guardrail that fired (`_accept` is a bool wrapper); every hit, including chunks rescued by the retry (`fallback: false`), lands in `Result.cleanup_rejections` and thus in the journal. A false reject is always safe — Whisper's text already has punctuation.
- **Dictation (`dictate.py`)**: pynput global listener, push-to-talk (hold = record, release = transcribe + paste via clipboard and Cmd+V, clipboard restored). The listener callback must never block (macOS disables slow event taps) — processing runs in a separate thread.
- **Recording indicator (`overlay.py`)**: an always-on-top pill (pulsing dot + `m:ss` timer) shown while the mic is live. It runs in a **helper process** (`python -m s2t.overlay`, AppKit needs a main thread with a run loop and ours belongs to uvicorn), driven over stdin (`show` / `hide`, EOF = exit). It must never take focus: accessory app, non-activating borderless `NSPanel`, mouse events ignored. `Dictation.stop()` closes it. See D10 in `docs/DECISIONS.md`.

## Contracts

- `POST /transcribe` (multipart): `file` (any container ffmpeg reads), optional `language` (`auto|ru|en|…`), optional `cleanup` (bool). Response = `pipeline.Result`: `text`, `raw_text`, `language`, `audio_s`, `speech_s`, `no_speech`, `cleanup_used`, `cleanup_rejected_chunks`, `cleanup_rejections` (list of `{reason, words, raw, llm, fallback}`), `hallucinations_dropped`, `timings_ms`. Undecodable input → HTTP 400. `GET /health`.
- `history.jsonl` (repo root, gitignored, mode 0600, contains everything dictated): one JSON object per request = `ts`, `origin`, optional `file`, `requested_language`, all `Result` fields, `models`. Write errors are logged to stderr and never fail the request. When testing with real models, point `history.path` at the scratchpad via a temp `--config` so the user's journal is not polluted.
- Config lookup order: `--config` → `./config.yaml` → `~/.config/s2t/config.yaml` → repo `config.yaml`. Unknown keys raise `ValueError` on purpose (typos must not be silent). `Config.resolve()` resolves relative paths against the repo root, not the cwd. When adding a config field: dataclass default in `config.py` + commented line in `config.yaml` + README.

## Hard-won gotchas (do not regress)

- **MP4/M4A/MOV cannot be decoded from a pipe** (index at end of file, ffmpeg can't seek) — `audio.decode_bytes` goes through a temp file, and zero decoded samples is an error. Regression test exists.
- **The 4-bit Qwen3-4B cleanup model corrupts text on long mixed Cyrillic/Latin chunks**: observed "Grafана", "Graf и Grafana", and replacing the unknown word "Варкере" with the glossary term "Redis". Consequences: chunks are small; glossary terms get **no free pass** in the invented-word check; keep these cases in tests. It also obeyed a prompt-injection ("reply banana") — the dropped-words guard is what catches that. The prefix KV cache was ruled out as the cause (cached and uncached outputs are identical).
- Paste uses the **physical key code** (`KeyCode.from_vk(9)`), not `tap("v")`, which fails under a Russian keyboard layout. `pbcopy`/`pbpaste` get an explicit UTF-8 locale or Cyrillic is mangled.
- Opening the microphone takes ~300 ms; the start sound plays *after* the stream is live. `dictation.keep_mic_open: true` removes the delay at the cost of a permanent mic indicator.
- Output that may be redirected to a file must use `flush=True` (`dictate._log`), otherwise logs vanish on SIGTERM.
- `torch` is installed only transitively (via `mlx-whisper`); our code never imports it — keep it that way (startup time, memory).
- Hugging Face downloads buffer before hitting disk, so `du` on the cache shows no progress for minutes; judge progress by network counters, not file size.

## Latency facts (M4 Pro, warm models)

- 14 s of speech: ~1.5–2.0 s total (STT ~0.9 s, LLM ~0.5–0.8 s, decode+VAD ~0.1 s). 54 s recording: ~5.3 s, LLM 3.3 s dominates. Model load ~7 s.
- Whisper always processes a 30 s window, so STT time is flat (~0.8–0.9 s) for anything under 30 s. VAD does **not** speed up short clips — it exists to prevent hallucinations and to cut windows on long audio.
- LLM time is generation-bound (~60 tok/s for 4B 4-bit); the ~590-token prompt prefix is free thanks to the KV cache. A higher-precision quant will slow generation roughly in proportion to its size.
- Always check `timings_ms` after touching the hot path.

## Testing hotkeys / dictation safely

Never paste into the user's focused window during tests. Pattern used so far: build the config with `dictation.paste = False` and `sounds = False`, simulate the hold with `pynput.keyboard.Controller().press/release(Key.alt_r)` (a lone Right Option types nothing), use a stub engine or feed `Dictation._process()` a decoded test file, and save/restore the clipboard (`pbpaste` → … → `pbcopy`). The terminal app already has Microphone and Input Monitoring permissions; real Cmd+V pasting (needs Accessibility) has **not** been verified by an agent. All test audio so far is synthetic (`say`); live-speech behaviour is unverified.

The terminal has **no Screen Recording permission**: `screencapture` returns only the wallpaper. To check the overlay, list windows by the helper's PID with `CGWindowListCopyWindowInfo` (layer and bounds need no permission), and to see it, run `overlay._serve` in-process with a faked `sys.stdin` and snapshot its own window via `CGWindowListCreateImage(..., windowNumber)`. Before simulating a hotkey, make sure the user's own `s2t serve` is not running (it would record and paste for real).
