import json
import wave
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def stub_decoder(monkeypatch, request):
    if not request.node.name.startswith("test_ffmpeg_"):
        monkeypatch.setattr("ml.transcribe._decode_audio", lambda path, ffmpeg: str(path))


def make_wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 1600)


def test_transcribes_wav_to_schema_compatible_json(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")

    class Model:
        def transcribe(self, path, **kwargs):
            return iter([SimpleNamespace(start=0.0, end=0.09, text=" Привет ")]), (
                SimpleNamespace(language="ru", duration=0.1)
            )

    output = tmp_path / "transcripts"
    report = transcribe_batch(tmp_path, output, model_factory=lambda: Model())
    assert report[0]["status"] == "ok"
    assert json.loads((output / "1.json").read_text(encoding="utf-8")) == {
        "clip_id": 1,
        "model": "faster-whisper large-v3",
        "language": "ru",
        "duration": 0.1,
        "segments": [{"start": 0.0, "end": 0.09, "text": "Привет"}],
    }


def test_valid_rounded_end_remains_within_duration_and_chunks(tmp_path):
    from ml.chunking import chunk_transcript
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")

    class Model:
        def transcribe(self, audio, **kwargs):
            return iter([SimpleNamespace(start=0, end=0.106, text="ok")]), (
                SimpleNamespace(language="ru", duration=0.106)
            )

    out = tmp_path / "transcripts"
    first = transcribe_batch(tmp_path, out, model_factory=Model)[0]
    transcript = json.loads((out / "1.json").read_text(encoding="utf-8"))
    assert first["anomalies"] == []
    assert transcript["segments"][0]["end"] == 0.11
    assert transcript["duration"] == 0.11
    assert first["repair_metadata"] == {"original_duration": 0.106, "duration": 0.11}
    assert chunk_transcript(transcript)["chunks"][0]["text"] == "ok"
    assert transcribe_batch(tmp_path, out, model_factory=lambda: None)[0]["anomalies"] == []


def test_retry_without_vad_and_report_no_speech(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")
    make_wav(tmp_path / "audio" / "2.wav")
    calls = []

    class Model:
        def transcribe(self, path, **kwargs):
            calls.append((path, kwargs))
            items = (
                []
                if path.endswith("2.wav") or kwargs["vad_filter"]
                else [SimpleNamespace(start=0, end=0.09, text="Голос")]
            )
            return iter(items), SimpleNamespace(language="ru", duration=0.1)

    report = transcribe_batch(tmp_path, tmp_path / "transcripts", model_factory=Model)
    assert [item["status"] for item in report] == ["ok", "no_speech"]
    assert [item[1]["vad_filter"] for item in calls] == [True, False, True, False]
    assert calls[0][1]["vad_parameters"]["threshold"] < 0.5
    assert (
        json.loads((tmp_path / "transcripts" / "2.json").read_text(encoding="utf-8"))["segments"]
        == []
    )


def test_missing_and_corrupt_audio_are_errors_and_batch_continues(tmp_path):
    from ml.transcribe import transcribe_batch

    (tmp_path / "clips").mkdir()
    for i in (1, 2, 3):
        (tmp_path / "clips" / f"{i}.json").write_text("{}")
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "2.wav").write_bytes(b"bad")
    make_wav(tmp_path / "audio" / "3.wav")

    class Model:
        def transcribe(self, path, **kwargs):
            return iter([SimpleNamespace(start=0, end=0.1, text="ok")]), (
                SimpleNamespace(language="en", duration=0.1)
            )

    report = transcribe_batch(
        tmp_path, tmp_path / "transcripts", model_factory=Model, include_clips=True
    )
    assert [item["status"] for item in report] == ["missing_audio", "error", "ok"]
    assert not (tmp_path / "transcripts" / "1.json").exists()
    assert not (tmp_path / "transcripts" / "2.json").exists()
    assert (tmp_path / "audio" / "2.wav").read_bytes() == b"bad"


def test_existing_results_are_only_replaced_with_explicit_overwrite(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")
    out = tmp_path / "transcripts"
    out.mkdir()
    (out / "1.json").write_text("original")

    class Model:
        def transcribe(self, path, **kwargs):
            return iter([SimpleNamespace(start=0, end=0.1, text="ok")]), (
                SimpleNamespace(language="en", duration=0.1)
            )

    report = transcribe_batch(tmp_path, out, model_factory=Model)
    assert report[0]["status"] == "error"
    assert (out / "1.json").read_text() == "original"
    report = transcribe_batch(tmp_path, out, model_factory=Model, overwrite=True)
    assert report[0]["status"] == "ok"
    assert json.loads((out / "1.json").read_text(encoding="utf-8"))["clip_id"] == 1


def test_model_and_generator_errors_do_not_become_no_speech(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")
    make_wav(tmp_path / "audio" / "2.wav")

    class Model:
        def transcribe(self, path, **kwargs):
            def segments():
                if path.endswith("1.wav"):
                    raise ValueError("decode failed")
                yield SimpleNamespace(start=0, end=0.1, text="ok")

            return segments(), SimpleNamespace(language="en", duration=0.1)

    report = transcribe_batch(tmp_path, tmp_path / "transcripts", model_factory=Model)
    assert [item["status"] for item in report] == ["error", "ok"]
    assert "decode failed" in report[0]["error"]


def test_invalid_filenames_and_unsafe_output_are_reported(tmp_path):
    import pytest

    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "0.wav")
    make_wav(tmp_path / "audio" / "01.wav")
    report = transcribe_batch(tmp_path, tmp_path / "transcripts", model_factory=lambda: None)
    assert [item["status"] for item in report] == ["error", "error"]
    with pytest.raises(ValueError):
        transcribe_batch(tmp_path, tmp_path / "clips", model_factory=lambda: None, overwrite=True)


def test_cli_outputs_json_report_and_nonzero_for_missing_audio(tmp_path, capsys):
    from ml.transcribe import main

    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "1.json").write_text("{}")
    code = main(
        ["--input", str(tmp_path), "--out", str(tmp_path / "transcripts"), "--include-clips"]
    )
    assert code == 1
    assert json.loads(capsys.readouterr().out)[0]["status"] == "missing_audio"


def test_truncated_wav_is_an_error_even_when_model_returns_empty(tmp_path):
    from ml.transcribe import transcribe_batch

    audio = tmp_path / "audio" / "1.wav"
    make_wav(audio)
    audio.write_bytes(audio.read_bytes()[:-100])
    report = transcribe_batch(tmp_path, tmp_path / "transcripts", model_factory=lambda: None)
    assert report[0]["status"] == "error"
    assert not (tmp_path / "transcripts" / "1.json").exists()


def test_model_initialization_failure_is_reported_for_each_valid_input(tmp_path):
    from ml.transcribe import transcribe_batch

    for i in (1, 2):
        make_wav(tmp_path / "audio" / f"{i}.wav")
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("missing model")

    result = transcribe_batch(tmp_path, tmp_path / "transcripts", model_factory=broken)
    assert [item["status"] for item in result] == ["error", "error"]
    assert all("missing model" in item["error"] for item in result)
    assert len(calls) == 1


def test_cli_selects_small_cpu_and_writes_report(tmp_path, monkeypatch, capsys):
    import sys

    from ml.transcribe import main

    make_wav(tmp_path / "audio" / "1.wav")
    settings = []

    class Model:
        def __init__(self, name, **kwargs):
            settings.append((name, kwargs))

        def transcribe(self, path, **kwargs):
            return iter([SimpleNamespace(start=0, end=0.1, text="ok")]), (
                SimpleNamespace(language="en", duration=0.1)
            )

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Model))
    assert (
        main(
            [
                "--input",
                str(tmp_path),
                "--out",
                str(tmp_path / "transcripts"),
                "--model",
                "small",
                "--device",
                "cpu",
                "--compute-type",
                "int8",
            ]
        )
        == 0
    )
    assert settings == [("small", {"device": "cpu", "compute_type": "int8"})]
    assert json.loads(capsys.readouterr().out)[0]["status"] == "ok"
    assert (
        json.loads((tmp_path / "transcripts" / "1.json").read_text(encoding="utf-8"))["model"]
        == "faster-whisper small"
    )


def test_output_must_not_be_aliased_to_input_audio(tmp_path):
    import pytest

    from ml.transcribe import transcribe_batch

    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "audio").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    make_wav(outside / "1.wav")
    with pytest.raises(ValueError):
        transcribe_batch(root, outside, model_factory=lambda: None, overwrite=True)


def test_timestamp_anomalies_are_reported_without_changing_transcript(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")

    class Model:
        def transcribe(self, path, **kwargs):
            return iter(
                [
                    SimpleNamespace(start=0.08, end=0.2, text="Поздно"),
                    SimpleNamespace(start=0.02, end=0.05, text="Раньше"),
                ]
            ), SimpleNamespace(language="ru", duration=0.1)

    output = tmp_path / "transcripts"
    result = transcribe_batch(tmp_path, output, model_factory=Model)
    assert result[0]["status"] == "ok"
    assert result[0]["review_required"] is True
    assert {a["type"] for a in result[0]["anomalies"]} == {"end_after_duration", "out_of_order"}
    data = json.loads((output / "1.json").read_text(encoding="utf-8"))
    assert data["segments"] == [
        {"start": 0.08, "end": 0.2, "text": "Поздно"},
        {"start": 0.02, "end": 0.05, "text": "Раньше"},
    ]


def test_no_speech_marks_review_required(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")

    class Model:
        def transcribe(self, path, **kwargs):
            return iter([]), SimpleNamespace(language="ru", duration=0.1)

    result = transcribe_batch(tmp_path, tmp_path / "transcripts", model_factory=Model)
    assert result[0]["status"] == "no_speech"
    assert result[0]["review_required"] is True


def test_language_is_forwarded_for_russian_and_auto(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")
    calls = []

    class Model:
        def transcribe(self, path, **kwargs):
            calls.append(kwargs["language"])
            return iter([SimpleNamespace(start=0, end=0.1, text="ok")]), (
                SimpleNamespace(language="ru", duration=0.1)
            )

    transcribe_batch(tmp_path, tmp_path / "transcripts", model_factory=Model, language="ru")
    transcribe_batch(
        tmp_path, tmp_path / "transcripts", model_factory=Model, language="auto", overwrite=True
    )
    assert calls == ["ru", None]


def test_ffmpeg_decode_uses_float32_audio_not_a_filename(tmp_path, monkeypatch):
    import sys

    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")
    captured = []
    samples = object()

    class Array:
        def copy(self):
            return samples

    def frombuffer(data, dtype):
        assert data == b"\x00\x00\x80\x3f" and dtype == "<f4"
        return Array()

    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(frombuffer=frombuffer))

    def run(cmd, **kwargs):
        captured.append(cmd)
        return SimpleNamespace(returncode=0, stdout=b"\x00\x00\x80\x3f", stderr=b"")

    monkeypatch.setattr("ml.transcribe.subprocess.run", run)

    class Model:
        def transcribe(self, audio, **kwargs):
            assert audio is samples
            return iter([SimpleNamespace(start=0, end=0.1, text="ok")]), (
                SimpleNamespace(language="ru", duration=0.1)
            )

    result = transcribe_batch(tmp_path, tmp_path / "transcripts", model_factory=Model)
    assert result[0]["status"] == "ok"
    assert captured[0][0] == "ffmpeg"
    assert "f32le" in captured[0] and "16000" in captured[0] and "-y" not in captured[0]
    assert captured[0][-1] == "pipe:1"


def test_ffmpeg_decode_failure_is_reported_and_input_preserved(tmp_path, monkeypatch):
    from ml.transcribe import transcribe_batch

    audio = tmp_path / "audio" / "1.wav"
    make_wav(audio)
    original = audio.read_bytes()
    monkeypatch.setattr(
        "ml.transcribe.subprocess.run",
        lambda *a, **kw: SimpleNamespace(returncode=1, stdout=b"", stderr=b"decoder failed"),
    )
    result = transcribe_batch(tmp_path, tmp_path / "transcripts", model_factory=lambda: None)
    assert result[0]["status"] == "error"
    assert "decoder failed" in result[0]["error"]
    assert audio.read_bytes() == original
    assert not (tmp_path / "transcripts" / "1.json").exists()


def test_reversed_timestamps_are_reported_without_losing_data(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")

    class Model:
        def transcribe(self, audio, **kwargs):
            return iter([SimpleNamespace(start=0.08, end=0.03, text="Ошибка")]), (
                SimpleNamespace(language="ru", duration=0.1)
            )

    out = tmp_path / "transcripts"
    report = transcribe_batch(tmp_path, out, model_factory=Model)
    assert report[0]["status"] == "ok"
    assert report[0]["anomalies"][0]["type"] == "end_before_start"
    assert json.loads((out / "1.json").read_text(encoding="utf-8"))["segments"][0]["end"] == 0.03


def test_nonempty_model_output_also_requires_review(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")

    class Model:
        def transcribe(self, audio, **kwargs):
            return iter([SimpleNamespace(start=0, end=0.1, text="ok")]), (
                SimpleNamespace(language="ru", duration=0.1)
            )

    report = transcribe_batch(tmp_path, tmp_path / "transcripts", model_factory=Model)
    assert report[0]["status"] == "ok"
    assert report[0]["review_required"] is True


def test_rounding_keeps_raw_anomalies_and_retry_recovers_visible_ones(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")

    class Model:
        def transcribe(self, audio, **kwargs):
            return iter(
                [
                    SimpleNamespace(start=0.104, end=0.103, text="a"),
                    SimpleNamespace(start=0.02, end=0.203, text="b"),
                ]
            ), SimpleNamespace(language="ru", duration=0.2)

    out = tmp_path / "transcripts"
    first = transcribe_batch(tmp_path, out, model_factory=Model)[0]
    assert {a["type"] for a in first["anomalies"]} == {
        "end_before_start",
        "out_of_order",
        "end_after_duration",
    }
    data = json.loads((out / "1.json").read_text(encoding="utf-8"))
    assert data["segments"] == [
        {"start": 0.1, "end": 0.1, "text": "a"},
        {"start": 0.02, "end": 0.2, "text": "b"},
    ]
    second = transcribe_batch(tmp_path, out, model_factory=lambda: None)[0]
    assert second["status"] == "skipped"
    assert second["result_status"] == "ok"
    assert second["no_speech"] is False and second["review_required"] is True
    assert second["path"] == str(out / "1.json")
    assert [a["type"] for a in second["anomalies"]] == ["out_of_order"]


def test_skipped_no_speech_is_recovered(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")
    out = tmp_path / "transcripts"
    out.mkdir()
    (out / "1.json").write_text(
        json.dumps(
            {"clip_id": 1, "model": "model", "language": "ru", "duration": 1, "segments": []}
        ),
        encoding="utf-8",
    )
    result = transcribe_batch(tmp_path, out, model_factory=lambda: None)[0]
    assert result["status"] == "skipped"
    assert result["result_status"] == "no_speech" and result["no_speech"] is True
    assert result["review_required"] is True and result["anomalies"] == []


@pytest.mark.parametrize(
    "patch",
    [
        {"clip_id": 2},
        {"clip_id": True},
        {"model": 3},
        {"language": ""},
        {"duration": -1},
        {"duration": float("nan")},
        {"duration": True},
        {"segments": {}},
        {"segments": [{"start": -1, "end": 1, "text": "a"}]},
        {"segments": [{"start": 0, "end": float("inf"), "text": "a"}]},
        {"segments": [{"start": 0, "end": 1, "text": " "}]},
        {"segments": [{"start": 0, "end": 1, "text": 5}]},
        {"segments": [{"start": 0, "end": 1, "text": "a", "extra": 1}]},
        {"extra": 1},
    ],
)
def test_invalid_existing_transcript_is_error_and_not_overwritten(tmp_path, patch):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")
    out = tmp_path / "transcripts"
    out.mkdir()
    dest = out / "1.json"
    payload = {"clip_id": 1, "model": "m", "language": "ru", "duration": 1, "segments": []}
    payload.update(patch)
    original = json.dumps(payload)
    dest.write_text(original, encoding="utf-8")
    result = transcribe_batch(tmp_path, out, model_factory=lambda: None)[0]
    assert result["status"] == "error"
    assert dest.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "invalid", ["language", "missing_input", "missing_audio", "missing_clips", "empty", "output"]
)
def test_invalid_input_never_creates_report_or_output(tmp_path, invalid):
    from ml.transcribe import transcribe_batch

    root = tmp_path / "input"
    root.mkdir()
    out = tmp_path / "out"
    report = tmp_path / "report.jsonl"
    kwargs = {}
    if invalid not in ("missing_input", "missing_audio", "missing_clips", "empty"):
        make_wav(root / "audio" / "1.wav")
    if invalid == "language":
        kwargs["language"] = "xx"
    if invalid == "missing_input":
        root = tmp_path / "nonexistent"
    if invalid == "missing_clips":
        kwargs["include_clips"] = True
    if invalid == "empty":
        (root / "audio").mkdir()
    if invalid == "output":
        out = root / "audio" / "out"
    with pytest.raises(ValueError):
        transcribe_batch(root, out, model_factory=lambda: None, report_path=report, **kwargs)
    assert not report.exists()
    assert not out.exists()


def test_report_is_incremental_and_survives_interruption(tmp_path):
    from ml.transcribe import transcribe_batch

    root = tmp_path / "input"
    for i in (1, 2):
        make_wav(root / "audio" / f"{i}.wav")
    report_path = tmp_path / "report.jsonl"

    class Model:
        def transcribe(self, path, **kwargs):
            if path.endswith("2.wav"):
                raise KeyboardInterrupt()
            return iter([]), SimpleNamespace(language="ru", duration=0.1)

    with pytest.raises(KeyboardInterrupt):
        transcribe_batch(root, root / "transcripts", model_factory=Model, report_path=report_path)
    records = [json.loads(line) for line in report_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1 and records[0]["status"] == "no_speech"
    assert records[0]["review_required"] is True and records[0]["no_speech"] is True


def test_report_rejects_existing_and_input_paths(tmp_path):
    from ml.transcribe import transcribe_batch

    root = tmp_path / "input"
    audio = root / "audio" / "1.wav"
    make_wav(audio)
    original = audio.read_bytes()
    existing = tmp_path / "existing.jsonl"
    existing.write_text("keep", encoding="utf-8")
    for path in (existing, audio, root / "audio" / "new.jsonl", root / "transcripts" / "1.json"):
        with pytest.raises((ValueError, OSError)):
            transcribe_batch(
                root, root / "transcripts", model_factory=lambda: None, report_path=path
            )
    assert audio.read_bytes() == original and existing.read_text(encoding="utf-8") == "keep"


def test_report_rejects_dangling_symlink(tmp_path):
    from ml.transcribe import transcribe_batch

    root = tmp_path / "input"
    make_wav(root / "audio" / "1.wav")
    target = tmp_path / "nonexistent"
    link = tmp_path / "report.jsonl"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(OSError):
        transcribe_batch(root, root / "transcripts", model_factory=lambda: None, report_path=link)
    assert not target.exists()


def test_cli_persists_errors_and_refuses_existing_report(tmp_path, capsys):
    from ml.transcribe import main

    root = tmp_path / "input"
    (root / "clips").mkdir(parents=True)
    (root / "clips" / "1.json").write_text("{}")
    report = tmp_path / "report.jsonl"
    args = [
        "--input",
        str(root),
        "--out",
        str(root / "transcripts"),
        "--include-clips",
        "--report",
        str(report),
    ]
    assert main(args) == 1
    record = json.loads(report.read_text(encoding="utf-8"))
    assert record["status"] == "missing_audio"
    assert json.loads(capsys.readouterr().out) == [record]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert json.loads(report.read_text(encoding="utf-8")) == record


def test_report_cannot_be_placed_in_aliased_audio_directory(tmp_path):
    from ml.transcribe import transcribe_batch

    root = tmp_path / "input"
    root.mkdir()
    audio = tmp_path / "external_audio"
    audio.mkdir()
    make_wav(audio / "1.wav")
    try:
        (root / "audio").symlink_to(audio, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    with pytest.raises(ValueError):
        transcribe_batch(
            root,
            root / "transcripts",
            model_factory=lambda: None,
            report_path=audio / "report.jsonl",
        )
    assert not (audio / "report.jsonl").exists()


def test_save_race_validates_winner_and_preserves_no_speech(tmp_path, monkeypatch):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")
    out = tmp_path / "transcripts"

    class Model:
        def transcribe(self, path, **kwargs):
            return iter([SimpleNamespace(start=0, end=0.1, text="ok")]), (
                SimpleNamespace(language="ru", duration=0.1)
            )

    def race(dest, payload, overwrite):
        dest.write_text(
            json.dumps(
                {"clip_id": 1, "model": "other", "language": "ru", "duration": 1, "segments": []}
            ),
            encoding="utf-8",
        )
        raise FileExistsError()

    monkeypatch.setattr("ml.transcribe._save_json", race)
    result = transcribe_batch(tmp_path, out, model_factory=Model)[0]
    assert result["status"] == "skipped" and result["result_status"] == "no_speech"
    assert result["no_speech"] and result["review_required"]


def test_duplicate_keys_in_existing_json_are_rejected(tmp_path):
    from ml.transcribe import transcribe_batch

    make_wav(tmp_path / "audio" / "1.wav")
    out = tmp_path / "transcripts"
    out.mkdir()
    original = '{"clip_id":1,"clip_id":1,"model":"m","language":"ru","duration":1,"segments":[]}'
    (out / "1.json").write_text(original, encoding="utf-8")
    record = transcribe_batch(tmp_path, out, model_factory=lambda: None)[0]
    assert record["status"] == "error" and "duplicate" in record["error"]
    assert (out / "1.json").read_text(encoding="utf-8") == original


def test_report_sync_failure_stops_batch(tmp_path, monkeypatch):
    from ml.transcribe import transcribe_batch

    root = tmp_path / "input"
    for i in (1, 2):
        make_wav(root / "audio" / f"{i}.wav")

    class Model:
        def transcribe(self, audio, **kwargs):
            return iter([SimpleNamespace(start=0, end=0.1, text="ok")]), (
                SimpleNamespace(language="ru", duration=0.1)
            )

    def fail(_fd):
        raise OSError("sync failed")

    monkeypatch.setattr("ml.transcribe.os.fsync", fail)
    with pytest.raises(OSError, match="sync failed"):
        transcribe_batch(
            root, root / "transcripts", model_factory=Model, report_path=tmp_path / "report.jsonl"
        )
    assert (root / "transcripts" / "1.json").exists()
    assert not (root / "transcripts" / "2.json").exists()
