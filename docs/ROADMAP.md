# Roadmap and status

Last updated: 2026-09-21 (history journal added).

## Status

- MVP works end to end: CLI, HTTP API, push-to-talk dictation. 17 unit tests pass (`uv run pytest -q`).
- Verified by an agent: decoding of ogg/opus, m4a, mp3, aiff, mp4 video; VAD trimming; RU/EN recognition with glossary terms; cleanup not answering questions / not obeying instructions; guardrails on real model glitches; hotkey → record → pipeline → clipboard (simulated key press); clean shutdown.
- **Not verified**: real Cmd+V paste into a focused app (needs Accessibility for the terminal); any live speech — all audio so far is macOS `say` TTS.
- Code is in the private GitHub repo `nikvakhrameev/s2t` (initial commit 2026-09-21). No linter configured.
- Leftovers: the research subagent left ~3 GB of downloaded models in the first session's scratchpad under `/private/tmp/claude-501/…/scratchpad` (safe to delete). Whisper turbo and Qwen3-4B-4bit are in `~/.cache/huggingface`.

## Next (agreed with the user, not built)

### 1. Semantic-divergence check with TypeSafe's **Jev** model
- The user asked for "быструю проверку через JEV модель на то, насколько по смыслу расходятся оригинал и получившийся текст".
- **Jev is a TypeSafe "System One" model** — use the `typesafe:typesafe-ai` skill (it appeared in the skill list late in the first session; invoke it first and read its live docs). An earlier guess that "JEV" meant an *embedding* model was wrong; nothing was built on it.
- Open questions to settle before coding:
  - Does Jev run locally or is it a cloud API? The project's premise is on-device processing — if transcripts would leave the machine, confirm with the user first.
  - Latency budget: the whole cycle must stay within 2–3 s; the check runs per chunk.
- Intended shape: an additional, **configurable** guard in `LlmCleaner._accept` (threshold in `config.yaml`, off-switch), complementing the word-level guards, which stay. It should target their blind spots: word reordering / role swaps, word-form changes, subtle paraphrase, double-negation flips now tolerated by `max_negation_loss`. Surface the score in `Result` so the threshold can be tuned from real dictations.
- Calibrate before choosing a threshold: score known-good pairs (`scripts/eval_cleanup.py` cases) against known-bad ones ("Graf и Grafana", "Варкере"→"Redis", "не сработает"→"сработает", "banana").

### 2. Compare cleanup models
`uv run python scripts/eval_cleanup.py -v <models>`. Baseline `Qwen3-4B-Instruct-2507-4bit`: 13/14 accepted (the one reject is the prompt-injection case — expected), mean 453 ms, max 647 ms.
Candidates (repos verified to exist by the research subagent; quality claims untested):
- `mlx-community/Qwen3-4B-Instruct-2507-6bit` (3.3 GB), `-8bit` (4.3 GB), `-4bit-DWQ-2510` (2.3 GB) — cheapest likely fix for the 4-bit token damage; expect slower generation with bigger quants.
- `mlx-community/Qwen3.5-4B-OptiQ-4bit` / `Qwen3.5-4B-MLX-8bit` — thinking is ON by default → `apply_chat_template(..., enable_thinking=False)`; hybrid cache is not trimmable (our `deepcopy` approach is fine).
- `mlx-community/gemma-4-E4B-it-qat-4bit` (6.8 GB) / `gemma-4-e4b-it-4bit` — thinking off by default; RotatingKVCache (window 512) → trimming fails silently, `deepcopy` is fine.
- The eval set lacks long multi-sentence chunks, which is where the glitches appeared — add some before trusting a comparison.

### 3. Glossary ergonomics
- ~~`history.jsonl`~~ — **done 2026-09-21** (`s2t/history.py`, `history.*` in config, README «Журнал»): raw vs. cleaned text, language, timings, every guardrail hit with its reason. Not included (not asked): saving the audio, rotation, a `s2t history` viewer. The journal is the data source for the remaining items here and for tuning in §4.
- `s2t glossary add "Term" -a alias …` / `s2t glossary list`.
- One-off script that mines term candidates from the user's repos/docs **for review** (never auto-add).
- Later: "fix last dictation" hotkey → alias candidates from the diff.

### 4. Verification on real use
- Live-speech testing by the user; tune `vad.*`, `max_pause_ms`, fillers list (`cleanup.FILLERS`) and few-shot examples from real transcripts.
- Verify paste with a Russian keyboard layout active.

## Later / ideas
- launchd autostart for `s2t serve`.
- Per-language STT backends on separate hotkeys. Research notes: `Qwen3-ASR-1.7B` (mlx-audio) supports `system_prompt`/hotwords, vendor WER only marginally better than Whisper large-v3; `ai-sage/GigaAM-Multilingual` beats Whisper on Russian but is weak on English, torch-only, no biasing; `parakeet-tdt-0.6b-v3` is fast, RU+EN, no biasing.
- Long dictations: cleanup dominates (3.3 s for a 54 s recording) — options are streaming paste per chunk or a faster cleanup model.
- "Thank you." is not in the hallucination phrase list on purpose (it is a legitimate dictation); with VAD on it does not occur.
