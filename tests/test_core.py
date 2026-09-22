"""Fast tests for the model-free logic (no MLX, no downloads except the 2 MB VAD)."""

from pathlib import Path

import numpy as np
import pytest

from s2t.audio import SAMPLE_RATE, AudioDecodeError, decode_bytes, resample_pcm
from s2t.cleanup import LlmCleaner, dropped_words, split_chunks
from s2t.config import CleanupConfig, Config, VadConfig, load_config
from s2t.glossary import Glossary, GlossaryStore, Term
from s2t.vad import SileroVad


def test_glossary_aliases_are_case_insensitive_and_word_bounded():
    glossary = Glossary([Term("Kubernetes", ("кубернетес",)), Term("Claude Code", ("клод код",))])
    text = "Задеплоили в Кубернетес, а клод код помог. Некубернетесный текст."
    assert glossary.apply_aliases(text) == (
        "Задеплоили в Kubernetes, а Claude Code помог. Некубернетесный текст."
    )


def test_glossary_stt_prompt_respects_limit():
    glossary = Glossary([Term("A" * 10), Term("B" * 10), Term("C" * 10)])
    assert glossary.stt_prompt(25) == f"{'A' * 10}, {'B' * 10}."
    assert Glossary().stt_prompt(100) is None


def test_glossary_store_reloads_on_change(tmp_path: Path):
    path = tmp_path / "g.yaml"
    path.write_text("terms: [Redis]", encoding="utf-8")
    store = GlossaryStore(path)
    assert store.get().canonical() == ["Redis"]
    path.write_text("terms: [Redis, {term: Kafka, aliases: [кафка]}]", encoding="utf-8")
    import os

    os.utime(path, (1, 1))  # force a different mtime
    assert store.get().canonical() == ["Redis", "Kafka"]
    assert store.version == 2


def test_split_chunks_keeps_all_text():
    text = " ".join(f"Предложение номер {i}." for i in range(40))
    chunks = split_chunks(text, 200)
    assert len(chunks) > 1
    assert " ".join(chunks) == text
    assert all(len(c) <= 220 for c in chunks)


def test_guardrail_accepts_cleanup_and_rejects_rewrites():
    cleaner = LlmCleaner(CleanupConfig())
    raw = "ну в общем мы вчера э-э задеплоили сервис и он как бы упал"
    assert cleaner._accept(raw, "Мы вчера задеплоили сервис, и он упал.")
    assert not cleaner._accept(raw, "")
    assert not cleaner._accept(raw, "Конечно! Вот список причин, почему сервисы падают после деплоя.")
    assert not cleaner._accept("what is the capital of france", "The capital of France is Paris. " * 3)
    # dropping content (not fillers) is a meaning change
    long_raw = "сначала обновим базу данных потом перезапустим сервер и после этого проверим логи"
    assert dropped_words(long_raw, "Сначала обновим базу данных.") != []
    assert not cleaner._accept(long_raw, "Сначала обновим базу данных.")
    assert dropped_words("ёжик, ну, типа, пришёл", "Ежик пришел.") == []


def test_postprocess_strips_tags_and_quotes():
    assert LlmCleaner._postprocess("<transcript>«Привет, мир.»</transcript>") == "Привет, мир."
    assert LlmCleaner._postprocess("<think>hmm</think>Hello.") == "Hello."


def test_config_overrides_and_unknown_keys(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "vad: {max_pause_ms: 300}\n"
        "stt: {auto_languages: [ru]}\n"
        "dictation: {hotkeys: [{key: f13, language: en}]}\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.vad.max_pause_ms == 300
    assert config.stt.auto_languages == ("ru",)
    assert config.dictation.hotkeys[0].key == "f13"
    path.write_text("vad: {nope: 1}", encoding="utf-8")
    with pytest.raises(ValueError, match="vad.nope"):
        load_config(path)


def test_decode_rejects_garbage_and_resamples():
    with pytest.raises(AudioDecodeError):
        decode_bytes(b"definitely not audio")
    tone = np.sin(np.linspace(0, 440 * 2 * np.pi, 48_000, dtype=np.float32))
    assert abs(len(resample_pcm(tone, 48_000)) - SAMPLE_RATE) < 100


@pytest.mark.skipif(not Path("tests/audio/ru.ogg").is_file(), reason="needs generated sample audio")
def test_vad_trims_edges_and_shortens_pauses():
    from s2t.audio import decode_file

    config = Config()
    vad = SileroVad(VadConfig(), config.resolve(config.vad.model_path))
    audio = decode_file("tests/audio/ru.ogg")
    result = vad.process(audio)
    assert len(result.segments) == 3
    # 3 s of edge noise on both sides and 4 s + 3 s pauses are gone.
    assert result.kept_s < result.original_s - 9
    # Speech itself is fully preserved, plus paddings.
    speech = sum(e - s for s, e in result.segments) / SAMPLE_RATE
    assert result.kept_s >= speech

    silence = decode_file("tests/audio/silence.mp3")
    assert not vad.process(silence).has_speech


@pytest.mark.skipif(not Path("tests/audio/en.m4a").is_file(), reason="needs generated sample audio")
def test_decode_bytes_handles_mp4_family():
    """MP4/M4A cannot be decoded from a pipe (index at the end of the file)."""
    from s2t.audio import decode_file

    path = Path("tests/audio/en.m4a")
    assert abs(len(decode_bytes(path.read_bytes())) - len(decode_file(path))) < SAMPLE_RATE // 10


def test_guardrail_rejects_invented_words_and_negation_flips():
    cleaner = LlmCleaner(CleanupConfig())
    raw = "все это мы деплоим а мониторинг настроен в Grafana и написан на питоне в докере"
    assert cleaner._accept(raw, "Всё это мы деплоим, а мониторинг настроен в Grafana и написан на Python в Docker.")
    # real glitches seen from a 4-bit model
    assert not cleaner._accept(raw, raw.replace("в Grafana", "в Graf и Grafana"))
    assert not cleaner._accept(raw, raw.replace("Grafana", "Grafана"))
    assert not cleaner._accept("я думаю что дело в утечке в варкере", "Я думаю, что дело в утечке в Redis.")
    assert not cleaner._accept("я думаю это не сработает", "Я думаю, это сработает.")
    assert cleaner._accept("я не не знаю как это починить", "Я не знаю, как это починить.")
    # respelling of a misheard word is fine
    assert cleaner._accept("не хватило конекшенов в пуле базы", "Не хватило коннекшенов в пуле базы.")
    assert cleaner._accept("открыл пул реквест в гитхабе", "Открыл pull request в GitHub.")


def test_negation_guard_understands_hesitations():
    from s2t.cleanup import negation_count

    cleaner = LlmCleaner(CleanupConfig())
    # repeats separated by fillers / pauses are one negation
    assert negation_count("я не, э-э, не уверен") == negation_count("Я не уверен.") == 1
    assert cleaner._accept("я не э-э не уверен что это хорошая идея", "Я не уверен, что это хорошая идея.")
    # restarted phrase
    assert cleaner._accept(
        "я не думаю ну я не думаю что это сработает на проде", "Я не думаю, что это сработает на проде."
    )
    assert cleaner._accept("i do not um i do not think this works", "I do not think this works.")
    # hesitation the normalization cannot collapse (different words in between)
    assert cleaner._accept("я не хочу то есть не могу прийти завтра", "Я не могу прийти завтра.")
    # a real double negation counts as two; the default tolerance (0.5) lets one of
    # two go - an accepted trade-off - while strict mode rejects it
    assert negation_count("не могу не согласиться") == 2
    assert cleaner._accept("я не могу не согласиться с этим", "Я могу не согласиться с этим.")
    strict = LlmCleaner(CleanupConfig(max_negation_loss=0.0))
    assert not strict._accept("я не могу не согласиться с этим", "Я могу не согласиться с этим.")
    # losing the only negation or gaining one is rejected at any tolerance
    lenient = LlmCleaner(CleanupConfig(max_negation_loss=0.9))
    assert not lenient._accept("я думаю это не сработает на проде", "Я думаю, это сработает на проде.")
    assert not lenient._accept("мы решили это выкатывать сегодня", "Мы решили это не выкатывать сегодня.")
    assert not cleaner._accept("я думаю это не сработает на проде", "Я думаю, это сработает на проде.")
    assert not cleaner._accept("мы решили это выкатывать сегодня", "Мы решили это не выкатывать сегодня.")


def test_violation_names_the_guardrail():
    cleaner = LlmCleaner(CleanupConfig())
    raw = "я думаю что дело в утечке в варкере"
    assert cleaner._violation(raw, "Я думаю, что дело в утечке в варкере.") is None
    assert cleaner._violation(raw, "") == ("empty", [])
    assert cleaner._violation(raw, "Я думаю, что дело в утечке в Redis.") == ("invented_words", ["redis"])
    assert cleaner._violation(raw, "Я думаю, что дело в утечке. " * 3)[0] == "too_long"
    assert cleaner._violation("я думаю это не сработает", "Я думаю, это сработает.")[0] == "negation_lost"
    assert cleaner._violation("мы это выкатываем", "Мы это не выкатываем.")[0] == "negation_gained"
    reason, words = cleaner._violation(
        "сначала обновим базу данных потом перезапустим сервер", "Сначала обновим базу данных."
    )
    assert reason == "dropped_words" and "сервер" in words


def test_guardrail_similarity_and_length_knobs_are_configurable():
    from s2t.cleanup import dropped_words, negation_count

    # translit_similarity: default (0.6) accepts "воркере" -> "Docker" as a
    # respelling; a stricter knob rejects it as an invented word.
    raw = "я думаю что дело в утечке в воркере"
    cleaned = "Я думаю, что дело в утечке в Docker."
    assert LlmCleaner(CleanupConfig())._violation(raw, cleaned) is None
    assert LlmCleaner(CleanupConfig(translit_similarity=0.75))._violation(raw, cleaned) == (
        "invented_words", ["docker"]
    )

    # spelling_similarity: "сервера" -> "серваки" sits just below the default
    # cutoff (invented), a looser cutoff accepts it as a respelling.
    raw = "мы перезапустили сервера ночью"
    cleaned = "Мы перезапустили серваки ночью."
    assert LlmCleaner(CleanupConfig())._violation(raw, cleaned) == ("invented_words", ["серваки"])
    assert LlmCleaner(CleanupConfig(spelling_similarity=0.7))._violation(raw, cleaned) is None

    # min_word_chars: short invented words are ignored by default, caught once
    # the minimum is lowered.
    raw, cleaned = "мы это ок сделаем", "Мы это ой сделаем."
    assert LlmCleaner(CleanupConfig())._violation(raw, cleaned) is None
    assert LlmCleaner(CleanupConfig(min_word_chars=1))._violation(raw, cleaned) == (
        "invented_words", ["ой"]
    )

    # fillers: an added filler is dropped for free instead of tripping dropped_words.
    raw, cleaned = "сначала например обновим базу", "Сначала обновим базу."
    assert dropped_words(raw, cleaned) == ["например"]
    extra_filler = CleanupConfig(fillers=CleanupConfig().fillers + ("например",))
    assert dropped_words(raw, cleaned, extra_filler) == []

    # negations: an added negation is counted; configured entries are normalized
    # (case, "ё") the same way transcript text is.
    assert negation_count("это нельзя делать") == 0
    extra_negation = CleanupConfig(negations=CleanupConfig().negations + ("нельзя",))
    assert negation_count("это нельзя делать", extra_negation) == 1
    assert negation_count("нельзя", CleanupConfig(negations=("НЕЛЬЗЯ",))) == 1
    assert dropped_words("это ещё видно", "Это видно.", CleanupConfig(fillers=("Ещё",))) == []


def test_cleanup_config_word_lists_load_and_reject_unquoted_yaml_booleans(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "cleanup: {fillers: [ну, ладно], negations: [не, нельзя], spelling_similarity: 0.8}\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.cleanup.fillers == ("ну", "ладно")
    assert config.cleanup.negations == ("не", "нельзя")
    assert config.cleanup.spelling_similarity == 0.8

    # Unquoted `no` is parsed by PyYAML as the boolean False, not the word "no".
    path.write_text("cleanup: {negations: [не, no]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cleanup.negations"):
        load_config(path)


def test_clean_records_every_guardrail_hit(monkeypatch):
    cleaner = LlmCleaner(CleanupConfig())
    monkeypatch.setattr(cleaner, "_ensure_prefix", lambda terms: None)
    replies = {
        # the whole chunk is rejected, then retried sentence by sentence
        "мы это деплоим. дело в утечке в варкере.": "Мы это деплоим. Дело в утечке в Redis.",
        "мы это деплоим.": "Мы это деплоим.",
        "дело в утечке в варкере.": "Дело в утечке в Redis.",
    }
    monkeypatch.setattr(cleaner, "_generate", lambda terms, text: replies[text])
    result = cleaner.clean("мы это деплоим. дело в утечке в варкере.", Glossary([]))
    assert result.text == "Мы это деплоим. дело в утечке в варкере."
    assert result.rejected_chunks == 1
    assert [(r.reason, r.words, r.fallback) for r in result.rejections] == [
        ("invented_words", ["redis"], False),
        ("invented_words", ["redis"], True),
    ]
    assert result.rejections[1].raw == "дело в утечке в варкере."
    assert result.rejections[1].llm == "Дело в утечке в Redis."


def test_engine_journals_results(tmp_path: Path):
    import json

    from s2t.pipeline import Engine, Result

    config = Config()
    config.history.path = str(tmp_path / "log" / "history.jsonl")
    config.history.origins = ("dictation", "api")

    class StubPipeline:
        def run(self, source, language, cleanup):
            return Result(text="Привет, мир.", raw_text="ну привет мир", language="ru",
                          timings_ms={"total": 5})

    engine = Engine(config)
    engine._pipeline = StubPipeline()
    try:
        engine.run(b"", "ru", origin="dictation")
        engine.run(b"", origin="api", file="voice.m4a")
        engine.run(b"", origin="cli", file="skipped.wav")  # not in history.origins
    finally:
        engine._executor.shutdown(wait=True)

    path = Path(config.history.path)
    first, second = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert "ну привет мир" in path.read_text(encoding="utf-8")  # readable, not \u-escaped
    assert path.stat().st_mode & 0o777 == 0o600
    assert (first["origin"], first["requested_language"], first["raw_text"]) == ("dictation", "ru", "ну привет мир")
    assert first["text"] == "Привет, мир." and first["timings_ms"] == {"total": 5}
    assert first["cleanup_rejections"] == [] and "file" not in first and first["ts"]
    assert (second["origin"], second["file"], second["requested_language"]) == ("api", "voice.m4a", "auto")
    assert second["models"]["stt"] == config.stt.model

    config.history.enabled = False
    path.unlink()
    engine.history.append(Result(text="x"), "dictation")
    assert not path.exists()


def test_overlay_config_format_and_placement(tmp_path: Path):
    from s2t.config import OverlayConfig
    from s2t.overlay import Overlay, format_elapsed, pill_origin

    path = tmp_path / "config.yaml"
    path.write_text("dictation: {overlay: {position: bottom_right, scale: 1.5}}", encoding="utf-8")
    overlay_config = load_config(path).dictation.overlay
    assert (overlay_config.enabled, overlay_config.position, overlay_config.scale) == (True, "bottom_right", 1.5)
    path.write_text("dictation: {overlay: {nope: 1}}", encoding="utf-8")
    with pytest.raises(ValueError, match="dictation.overlay.nope"):
        load_config(path)
    with pytest.raises(ValueError, match="middle"):
        Overlay(OverlayConfig(position="middle"))

    assert [format_elapsed(s) for s in (0, 9.99, 61.5, 600)] == ["0:00", "0:09", "1:01", "10:00"]
    area, size = (100, 50, 1000, 800), (80, 30)  # AppKit: y grows upwards
    assert pill_origin("top", area, size, 12) == (560, 808)
    assert pill_origin("bottom", area, size, 12) == (560, 62)
    assert pill_origin("top_left", area, size, 12) == (112, 808)
    assert pill_origin("bottom_right", area, size, 12) == (1008, 62)

    disabled = Overlay(OverlayConfig(enabled=False))  # no helper process is ever spawned
    disabled.start(), disabled.show(), disabled.hide(), disabled.close()
    assert disabled._process is None


def test_dictation_shows_overlay_only_while_the_mic_is_live():
    from s2t.config import DictationConfig
    from s2t.dictate import Dictation, resolve_key

    events: list[str] = []

    class StubRecorder:
        def start(self):
            events.append("mic on")

        def stop(self):
            events.append("mic off")
            return np.zeros(0, dtype=np.float32)

    class DeadRecorder(StubRecorder):
        def start(self):
            raise OSError("no microphone")

    class StubOverlay:
        def show(self):
            events.append("show")

        def hide(self):
            events.append("hide")

    # min_record_ms: the key is "released too early", so the engine is never reached.
    dictation = Dictation(DictationConfig(sounds=False, paste=False, min_record_ms=60_000), engine=None)
    dictation.recorder, dictation.overlay = StubRecorder(), StubOverlay()
    key = resolve_key("alt_r")
    dictation._on_press(key)
    dictation._on_press(key)  # key auto-repeat while held
    dictation._on_release(key)
    assert events == ["mic on", "show", "hide", "mic off"]

    events.clear()
    dictation.recorder = DeadRecorder()
    dictation._on_press(key)
    dictation._on_release(key)
    assert events == []


def _jev_cleaner(monkeypatch, replies: dict[str, str], risks: dict[str, float] | None = None,
                 error: str | None = None, **jev):
    """LlmCleaner with a canned LLM and a canned Jev: `risks` maps an LLM output to
    the risk Jev reports for it (the single question is polarity_flipped, limit 0.5)."""
    from s2t.config import JevConfig
    from s2t.jev import Verdict

    config = CleanupConfig(jev=JevConfig(enabled=True, questions={"polarity_flipped": 0.5}, **jev))
    cleaner = LlmCleaner(config)
    monkeypatch.setattr(cleaner, "_ensure_prefix", lambda terms: None)
    monkeypatch.setattr(cleaner, "_generate", lambda terms, text: replies[text])
    asked = []

    class Pending:
        def __init__(self, raw, llm):
            risk = (risks or {}).get(llm, 0.0)
            self.verdict = Verdict(raw, llm, error=error) if error else Verdict(
                raw, llm, {"polarity_flipped": risk}, ["polarity_flipped"] if risk > 0.5 else [], ms=1
            )

        def result(self):
            return self.verdict

    def submit(raw, llm):
        asked.append(llm)
        return Pending(raw, llm)

    monkeypatch.setattr("s2t.cleanup.PendingVerdict", Pending)
    monkeypatch.setattr(cleaner.judge, "submit", submit)
    return cleaner, asked


def test_jev_runs_on_top_of_the_heuristics(monkeypatch):
    # a double-negation flip: within the heuristics' tolerance, caught by Jev
    raw = "мы это деплоим сегодня. я не могу не согласиться с этим."
    replies = {
        raw: "Мы это деплоим сегодня. Я могу не согласиться с этим.",
        "мы это деплоим сегодня.": "Мы это деплоим сегодня.",
        "я не могу не согласиться с этим.": "Я могу не согласиться с этим.",
    }
    cleaner, asked = _jev_cleaner(
        monkeypatch, replies,
        risks={replies[raw]: 0.9, "Я могу не согласиться с этим.": 0.8},
    )
    result = cleaner.clean(raw, Glossary([]))
    assert result.text == "Мы это деплоим сегодня. я не могу не согласиться с этим."
    assert [(r.reason, r.words, r.fallback) for r in result.rejections] == [
        ("jev", ["polarity_flipped"], False),
        ("jev", ["polarity_flipped"], True),
    ]
    # the sentence whose words did not change is not worth a request
    assert asked == [replies[raw], "Я могу не согласиться с этим."]
    assert [v.risks["polarity_flipped"] for v in result.jev_checks] == [0.9, 0.8]

    # what the heuristics reject never reaches Jev
    cleaner, asked = _jev_cleaner(monkeypatch, {"дело в утечке в варкере": "Дело в утечке в Redis."})
    result = cleaner.clean("дело в утечке в варкере", Glossary([]))
    assert result.rejections[0].reason == "invented_words" and asked == []


def test_jev_can_replace_the_heuristics(monkeypatch):
    replies = {"открыл пиар в гитхабе и жду ревью": "Открыл pull request в GitHub и жду ревью."}
    raw, llm = next(iter(replies.items()))
    assert not LlmCleaner(CleanupConfig())._accept(raw, llm)  # "pull", "request" are invented words
    cleaner, asked = _jev_cleaner(monkeypatch, replies, mode="only")
    assert cleaner.clean(raw, Glossary([])).text == llm and asked == [llm]
    # an empty output is still not worth a request
    cleaner, asked = _jev_cleaner(monkeypatch, {raw: ""}, mode="only")
    result = cleaner.clean(raw, Glossary([]))
    assert (result.text, result.rejections[0].reason, asked) == (raw, "empty", [])


def test_jev_failure_falls_back_as_configured(monkeypatch):
    good = {"ну мы это выкатываем сегодня": "Мы это выкатываем сегодня."}
    bad = {"дело в утечке в варкере": "Дело в утечке в Redis."}
    for mode, replies, on_error, reason in [
        ("both", good, "heuristics", None),  # the heuristics had passed
        ("both", good, "reject", "jev_error"),
        ("only", good, "heuristics", None),
        ("only", bad, "heuristics", "invented_words"),  # the heuristics take over
        ("only", good, "reject", "jev_error"),
    ]:
        cleaner, _ = _jev_cleaner(monkeypatch, replies, error="no answer in 1.5 s", mode=mode, on_error=on_error)
        result = cleaner.clean(next(iter(replies)), Glossary([]))
        assert [r.reason for r in result.rejections] == ([reason] if reason else []), (mode, on_error)
        assert result.jev_checks[0].error


def test_jev_judge_request_and_thresholds(monkeypatch):
    import json

    import httpx

    from s2t.config import JevConfig
    from s2t.jev import QUESTIONS, JevJudge

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": []})
        body = json.loads(request.content)
        seen.append(body)
        if body["state"]["cleaned_text"] == "overloaded":
            return httpx.Response(529, text="try later")
        return httpx.Response(200, json={
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 700, "output_tokens": 3},
            "answers": {
                "polarity_flipped": {"type": "noul", "noul": 0.91},
                "details_changed": {"type": "noul", "noul": 0.02},
                "fidelity": {"type": "score", "score": 1.5, "confidence": 0.4,
                             "probabilities": {"0": 0.1, "1": 0.4, "2": 0.4, "3": 0.1}},
                "edit_kind": {"type": "choice", "choice": "faithful", "confidence": 0.5,
                              "probabilities": {"faithful": 0.7, "content_dropped": 0.1, "content_added": 0.0,
                                                "meaning_changed": 0.2, "responded": 0.0}},
            },
        })

    judge = JevJudge(JevConfig(enabled=True, questions={
        "polarity_flipped": 0.5, "details_changed": 0.5, "fidelity": 0.4, "edit_kind": 0.4}))
    judge.load(transport=httpx.MockTransport(handler))
    verdict = judge.check("это не сработает", "Это сработает.")
    assert seen[0]["model"] == "jev-1.13.0"
    assert seen[0]["state"]["raw_transcript"] == "это не сработает"
    assert seen[0]["questions"]["fidelity"] == QUESTIONS["fidelity"]
    assert verdict.risks == {"polarity_flipped": 0.91, "details_changed": 0.02, "fidelity": 0.5, "edit_kind": 0.3}
    assert verdict.failed == ["polarity_flipped", "fidelity"]
    assert (verdict.model, verdict.tokens, verdict.error) == ("jev-1.13.0", 700, None)

    failed = judge.check("x", "overloaded")
    assert failed.error.startswith("HTTP 529") and failed.risks == {} and failed.failed == []


def test_jev_config_is_validated(tmp_path: Path, monkeypatch):
    from s2t.config import JevConfig
    from s2t.jev import JevJudge

    path = tmp_path / "config.yaml"
    path.write_text("cleanup: {jev: {enabled: true, mode: only, questions: {fidelity: 0.25}}}", encoding="utf-8")
    jev = load_config(path).cleanup.jev
    assert (jev.mode, jev.questions, jev.on_error) == ("only", {"fidelity": 0.25}, "heuristics")
    assert LlmCleaner(CleanupConfig()).judge is None  # off by default: nothing leaves the machine

    for broken in ({"mode": "jev"}, {"on_error": "accept"}, {"questions": {"polarity": 0.5}}, {"questions": {}}):
        with pytest.raises(ValueError, match=f"cleanup.jev.{next(iter(broken))}"):
            JevJudge(JevConfig(enabled=True, **broken))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    jev.api_key_file = str(tmp_path / "key")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        JevJudge(jev).load()


def test_jev_key_file_and_deadline(tmp_path: Path, monkeypatch):
    import time

    import httpx

    from s2t.config import JevConfig
    from s2t.jev import JevJudge

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    (tmp_path / "key").write_text("file-key\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer file-key"
        if request.url.path == "/v1/systemone":
            time.sleep(0.3)
        return httpx.Response(200, json={"answers": {"responded": {"type": "noul", "noul": 0.0}}})

    judge = JevJudge(JevConfig(enabled=True, api_key_file=str(tmp_path / "key"),
                               questions={"responded": 0.5}, timeout_s=0.05))
    judge.load(transport=httpx.MockTransport(handler))
    started = time.perf_counter()
    verdict = judge.check("raw", "llm")
    assert time.perf_counter() - started < 0.2  # the deadline is wall-clock
    assert verdict.error == "no answer in 0.05 s" and verdict.failed == []

    # a rejected key stops the start-up instead of failing every dictation
    with pytest.raises(RuntimeError, match="rejected"):
        judge.load(transport=httpx.MockTransport(lambda request: httpx.Response(401)))
