# Decisions

Why the project is built the way it is. Newest last. Each entry: decision → reason → what would change it.
Read this before changing guardrails, chunking, models or the glossary layers.

## D1. Whisper large-v3-turbo on MLX as the single RU+EN model

- The user asked to start with one multilingual model; two per-language hotkeys/models are a later option (`stt.py` is kept tiny so backends can be swapped; `dictation.hotkeys` already carries a per-key `language`).
- Decisive feature: `initial_prompt` gives **vocabulary biasing**, which the glossary requirement needs. Parakeet-TDT v3 (fast, RU+EN) and GigaAM (best Russian) have no biasing.
- Measured: ~0.9 s for up to 30 s of audio on M4 Pro.
- Would change it: a model with hotword support and clearly better RU quality (candidate: `Qwen3-ASR-1.7B` via mlx-audio — see ROADMAP).

## D2. Silence handling: shorten pauses, do not delete them; skip STT when there is no speech

- Whisper hallucinates on non-speech ("Thank you.", "Субтитры сделал…"). Verified: 6 s of noise → "Thank you." without VAD, empty with VAD.
- Pauses above `max_pause_ms` are shortened to that length and filled with the **original room tone**: sentence boundaries stay audible (punctuation quality) and there are no digital-zero discontinuities.
- Second layer kept on purpose: `condition_on_previous_text=False`, Whisper's own thresholds, and a phrase filter for known hallucinations / prompt echo.

## D3. `language: auto` is restricted to `[ru, en]`

- Unrestricted Whisper language ID sometimes picks uk/bg/be on short Russian clips full of English terms. We run detection ourselves and take the argmax over `stt.auto_languages` only.

## D4. Glossary = three layers, aliases optional, filled by hand

- Asked by the user: "does code-side replacement make sense, or just put the glossary in the LLM prompt?" Answer adopted: all three, each for a different job.
  1. Whisper prompt — fixes the error at the source; most valuable; ~224-token cap → file order is priority.
  2. Glossary in the LLM system prompt — handles inflected forms ("в докере" → "в Docker").
  3. Regex `aliases` — deterministic, zero hallucination risk; only for what 1–2 keep getting wrong (names, internal project names).
- LLM-only was rejected because a small model **misuses** the glossary (see D6).
- No automatic glossary growth: a wrong alias is a permanent deterministic error, and a bloated glossary dilutes the Whisper prompt. Tooling to make manual upkeep easy is on the roadmap.

## D5. Cleanup LLM: Qwen3-4B-Instruct-2507 4-bit, prefix KV cache, small chunks

- ~0.5–0.8 s per typical dictation; the ~590-token system prompt + few-shot prefix is prefilled once and `deepcopy`'d per request (deepcopy rather than trimming so that models with rotating/hybrid caches keep working).
- `chunk_chars` went 700 → 350 → **220** because the 4-bit model glitches on long mixed-script chunks (D6). Small chunks also make guardrail fallbacks local.
- Texts under `min_words` skip the LLM.

## D6. Guardrails are deterministic code; glossary terms get no free pass

- An LLM judge would double latency and share the cleaner's blind spots. Word-level checks cost ~1 ms and are predictable.
- Observed failures of the 4-bit model that shaped the rules: "Graf и Grafana" (inserted fragment), "Grafана" (mixed scripts), "в Варкере" → "в **Redis**" (unknown word replaced by a glossary term), obeying "ignore previous instructions… reply banana".
- Hence: every output word must trace back to a transcript word (exact / close spelling / transliteration); glossary words are **not** whitelisted; mixed-script tokens are rejected; only fillers and repeats may be dropped (plus `max_dropped_ratio`).
- An earlier word-level `similarity` ratio was removed: it scored "пул реквест"→"pull request" at 0.4 (false reject) yet let a dropped sentence through.
- Rejected chunk → retry per sentence → raw transcript. False rejects are safe by construction.
- Known blind spots (accepted): word reordering, word-form changes ("деплоим"→"деплоили"), words shorter than 3 letters, one dropped meaningful word within tolerance.

## D7. Negations: tolerance instead of exact equality (user decision)

- Problem raised by the user: a speaker who hesitates repeats negations ("я не… э-э… не уверен"), so exact equality of negation counts falsely rejects valid cleanups.
- Claude's recommendation was strict equality after normalization (strip fillers, collapse repeated words/phrases), arguing that a tolerance lets a real flip through when a chunk has two negations ("не могу не согласиться" → "могу не согласиться").
- **The user chose a tolerance: "слишком хрупкие правила".** Implemented as `cleanup.max_negation_loss` (default 0.5) *on top of* the normalization: gaining a negation is never accepted; losing the only one is always rejected; losing 1 of 2 passes. `0` restores strict mode. The double-negation trade-off is known and accepted; it is pinned in `test_negation_guard_understands_hesitations`.
- General lesson: prefer configurable tolerances over exact-match rules in this project.

## D8. One process, one MLX thread

- HTTP API and hotkeys live in the same `s2t serve` process so models are loaded once; all MLX work is serialized on `Engine`'s single worker thread (MLX streams are per-thread). `s2t transcribe` talks to a running server when there is one.

## D9. Push-to-talk UX details

- Hold-to-record on a lone modifier (Right Option) — types nothing, works in any app.
- Start cue plays after the mic is live (~300 ms open latency); `keep_mic_open` trades the macOS mic indicator for zero latency.
- Paste = clipboard + Cmd+V by **physical key code** (layout-independent), then the clipboard is restored.

## D10. Recording indicator = a helper process, not a window in `s2t serve`

- The user asked for an always-on-top element showing that recording is on and for how long. It is a pill with a pulsing dot and an `m:ss` timer (`s2t/overlay.py`), shown from "mic is live" (same moment as the start cue) until key release, so the timer equals the recorded length.
- AppKit windows need the main thread of their process plus a run loop, and the main thread of `s2t serve` runs uvicorn. Moving uvicorn off the main thread would touch signal handling and the MLX shutdown order, so the pill is a separate `python -m s2t.overlay` process, spawned once at start (cold start ~0.25 s, warm show ~15 ms) and driven over stdin (`show` / `hide`; EOF = exit, so it cannot outlive the server; a dead helper is respawned on the next command). It ignores SIGINT: Ctrl+C reaches the whole process group and the parent shuts it down.
- It must never take focus (the paste goes to the user's window): accessory activation policy, borderless non-activating `NSPanel`, `ignoresMouseEvents`, `orderFrontRegardless`. Level `NSScreenSaverWindowLevel` + `FullScreenAuxiliary` puts it above full-screen apps.
- PyObjC (`pyobjc-framework-Cocoa` / `-Quartz`) was already installed via `pynput`; it is now a declared dependency. `dictation.overlay.{enabled,position,margin,scale}` are config knobs.

## D11. Semantic guardrail: TypeSafe's Jev, optional, on top of the heuristics (user request)

- Asked by the user: check the cleaned text against the raw one with Jev; by default **in addition** to the heuristics, with a config switch to move the guardrail to Jev **entirely** (`cleanup.jev.mode: both | only`).
- D6 rejected an LLM judge (latency, shared blind spots). Jev is a different trade: it generates nothing, answers typed questions with probabilities in ~100 ms server-side, and its blind spots are not the cleaner's. It targets what D6 lists as accepted blind spots: role/order swaps, double-negation flips inside `max_negation_loss`, numbers shorter than 3 characters, one dropped meaningful word. Verified with the heuristics: 7 of the bad pairs in `scripts/eval_jev.py` pass them.
- **Jev is a cloud API** (no on-device option, apparently US-hosted, zero retention is enterprise-only). It breaks the "everything on-device" premise, so it is **off by default** and the only networked code lives in `s2t/jev.py`. Audio never leaves; chunk texts do.
- Raw `httpx` instead of `typesafe-sdk`: the contract is one POST, `httpx` is already a dependency, the SDK (0.7.0) broke compatibility twice in a week, pulls 5 packages and retries timeouts by default — the opposite of what a latency-critical call needs. No retries here at all.
- Questions follow the TypeSafe guidance: several narrow "bad = true" Nouls in **one** request (parallel: extra questions cost tokens, not time), gated by `max` — one confident red flag rejects, nothing is averaged. A single "is the meaning preserved?" question is what their docs warn against; it is in the catalog (`meaning_changed`, plus a Score `fidelity` and a Choice `edit_kind`) for comparison, not in the default set. Every question has its own threshold in `config.yaml` (D7's lesson). In-prompt examples deliberately differ from the eval pairs.
- Latency: measured from the dev machine — 812 ms cold, ~235 ms warm round trip (network only), keep-alive survives 20 s idle. Hence: the connection is warmed when `Pipeline.run` starts (handshake overlaps STT), requests run on helper threads while the next chunk is generated, and the deadline (`timeout_s`) is wall-clock from submission. A chunk whose words did not change (punctuation/case only) is not sent.
- `mode: only` keeps the free `empty` check and drops the rest. Known cost: Jev's docs say state content can steer its answers (prompt injection) and that double negatives and non-English text are weaker spots — which is why `both` is the default. `on_error: heuristics` makes the heuristics the fallback even in `only` mode.
- **Not calibrated yet**: the 0.5 thresholds are the docs' neutral default, no authenticated call had been made when this was written. Would change it: eval results showing a question does not separate good from bad pairs on Russian → reword or drop it; false rejects in the journal (`jev_checks`) → raise that question's threshold.
