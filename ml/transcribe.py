"""Batch ASR for the research/download layout (no model is loaded at import)."""

import argparse
import importlib
import json
import math
import os
import re
import subprocess
import tempfile
import wave
from contextlib import ExitStack
from pathlib import Path

_ID = re.compile(r"[1-9][0-9]*\Z", re.ASCII)


def _check_wav(path):
    """Read all PCM frames to detect invalid, empty, and truncated WAVs."""
    with wave.open(str(path), "rb") as wav:
        remaining = wav.getnframes()
        if remaining <= 0:
            raise ValueError("empty WAV")
        frame_bytes = wav.getnchannels() * wav.getsampwidth()
        while remaining:
            count = min(remaining, 65536)
            data = wav.readframes(count)
            if len(data) != count * frame_bytes:
                raise ValueError("truncated WAV")
            remaining -= count


def _decode_audio(path, ffmpeg):
    cmd = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"ffmpeg failed: {result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    if not result.stdout or len(result.stdout) % 4:
        raise ValueError("ffmpeg returned empty or invalid audio")
    np = importlib.import_module("numpy")
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def _segments(model, audio, vad, language):
    options = {
        "vad_filter": vad,
        "beam_size": 5,
        "language": None if language == "auto" else language,
    }
    if vad:
        options["vad_parameters"] = {"threshold": 0.3}
    segments, info = model.transcribe(audio, **options)
    result = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        start, end = float(segment.start), float(segment.end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < 0:
            raise ValueError("invalid segment timestamps")
        result.append({"start": start, "end": end, "text": text})
    return result, info


def _timestamp_anomalies(segments, duration):
    anomalies = []
    previous = None
    for index, segment in enumerate(segments):
        start, end = segment["start"], segment["end"]
        if end < start:
            anomalies.append(
                {"type": "end_before_start", "segment": index, "start": start, "end": end}
            )
        if end > duration:
            anomalies.append(
                {"type": "end_after_duration", "segment": index, "end": end, "duration": duration}
            )
        if previous is not None and start < previous:
            anomalies.append(
                {
                    "type": "out_of_order",
                    "segment": index,
                    "start": start,
                    "previous_start": previous,
                }
            )
        previous = start
    return anomalies


def _load_transcript(path, clip_id):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    if path.is_symlink():
        raise ValueError("transcript must not be a symlink")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream, object_pairs_hook=unique_object)
    if not isinstance(payload, dict) or set(payload) != {
        "clip_id",
        "model",
        "language",
        "duration",
        "segments",
    }:
        raise ValueError("invalid transcript fields")
    if type(payload["clip_id"]) is not int or payload["clip_id"] != clip_id:
        raise ValueError("invalid transcript clip_id")
    for key in ("model", "language"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"invalid transcript {key}")

    def valid_number(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    if not valid_number(payload["duration"]) or not isinstance(payload["segments"], list):
        raise ValueError("invalid transcript duration or segments")
    for segment in payload["segments"]:
        if not isinstance(segment, dict) or set(segment) != {"start", "end", "text"}:
            raise ValueError("invalid segment fields")
        if not valid_number(segment["start"]) or not valid_number(segment["end"]):
            raise ValueError("invalid segment timestamps")
        if not isinstance(segment["text"], str) or not segment["text"].strip():
            raise ValueError("invalid segment text")
    return payload


def _skipped_record(path, clip_id):
    payload = _load_transcript(path, clip_id)
    segments = payload["segments"]
    return {
        "status": "skipped",
        "result_status": "ok" if segments else "no_speech",
        "no_speech": not bool(segments),
        "review_required": True,
        "anomalies": _timestamp_anomalies(segments, payload["duration"]),
        "path": str(path),
    }


def _save_json(path, payload, overwrite):
    # Temp + link/replace avoids partial results and does not follow destination symlinks.
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".transcript-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def transcribe_batch(
    input_dir,
    output_dir,
    *,
    model_factory,
    model_name="large-v3",
    include_clips=False,
    overwrite=False,
    language="auto",
    ffmpeg="ffmpeg",
    report_path=None,
):
    """Return ordered per-file status records. model_factory lazily creates a model."""
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if report_path is not None:
        report_file = Path(report_path)
        resolved_report = report_file.resolve()
        protected = (root, output, (root / "audio").resolve(), (root / "clips").resolve())
        if any(resolved_report.is_relative_to(path) for path in protected):
            raise ValueError("report must be outside input and output directories")
        if os.path.lexists(report_file):
            raise FileExistsError(f"report already exists: {report_file}")
    else:
        report_file = None
    with ExitStack() as stack:
        stream = None

        def begin_report():
            nonlocal stream
            if report_file is not None:
                stream = stack.enter_context(open(report_file, "x", encoding="utf-8", newline="\n"))

        def emit(record):
            if stream is not None:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        return _transcribe_batch(
            input_dir,
            output_dir,
            model_factory=model_factory,
            model_name=model_name,
            include_clips=include_clips,
            overwrite=overwrite,
            language=language,
            ffmpeg=ffmpeg,
            emit=emit,
            begin_report=begin_report,
        )


def _transcribe_batch(
    input_dir,
    output_dir,
    *,
    model_factory,
    model_name,
    include_clips,
    overwrite,
    language,
    ffmpeg,
    emit,
    begin_report,
):
    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if language not in ("auto", "ru"):
        raise ValueError("language must be auto or ru")
    if not root.is_dir():
        raise ValueError(f"input directory does not exist: {root}")
    if output == root or (
        output.is_relative_to(root) and not output.is_relative_to(root / "transcripts")
    ):
        raise ValueError("output inside input must be under transcripts/")
    audio_dir, clips_dir = root / "audio", root / "clips"
    if any(output.is_relative_to(path.resolve()) for path in (audio_dir, clips_dir)):
        raise ValueError("output must not be inside audio/ or clips/")
    if include_clips and not clips_dir.is_dir():
        raise ValueError(f"clip manifest directory does not exist: {clips_dir}")
    if not include_clips and not audio_dir.is_dir():
        raise ValueError(f"audio directory does not exist: {audio_dir}")
    invalid = []
    ids = set()
    paths = list(audio_dir.glob("*.wav")) if audio_dir.is_dir() else []
    if include_clips:
        paths += list(clips_dir.glob("*.json"))
    for path in paths:
        if not _ID.fullmatch(path.stem):
            invalid.append(
                {
                    "clip_id": None,
                    "status": "error",
                    "path": str(path),
                    "error": "filename must be a canonical positive integer",
                }
            )
        else:
            ids.add(int(path.stem))
    if not ids and not invalid:
        raise ValueError("no audio files or clip manifests found")
    begin_report()
    output.mkdir(parents=True, exist_ok=True)
    report = sorted(invalid, key=lambda item: item["path"])
    for record in report:
        emit(record)
    model = None
    model_error = None
    for clip_id in sorted(ids):
        audio, dest = audio_dir / f"{clip_id}.wav", output / f"{clip_id}.json"
        record = {"clip_id": clip_id, "status": "error"}
        try:
            if not audio.is_file():
                record.update(status="missing_audio", error=f"audio not found: {audio}")
            elif not overwrite and os.path.lexists(dest):
                record.update(_skipped_record(dest, clip_id))
            else:
                _check_wav(audio)
                samples = _decode_audio(audio, ffmpeg)
                if model is None and model_error is None:
                    try:
                        model = model_factory()
                        if model is None:
                            raise ValueError("model factory returned None")
                    except Exception as exc:
                        model_error = f"model initialization failed: {exc}"
                if model_error:
                    raise RuntimeError(model_error)
                segments, info = _segments(model, samples, True, language)
                if not segments:
                    segments, info = _segments(model, samples, False, language)
                duration = float(info.duration)
                if not math.isfinite(duration) or duration < 0:
                    raise ValueError("invalid audio duration")
                if not isinstance(info.language, str) or not info.language:
                    raise ValueError("invalid detected language")
                anomalies = _timestamp_anomalies(segments, duration)
                rounded_segments = [
                    {
                        **segment,
                        "start": round(segment["start"], 2),
                        "end": round(segment["end"], 2),
                    }
                    for segment in segments
                ]
                rounded_end = max((segment["end"] for segment in rounded_segments), default=0)
                saved_duration = duration
                if rounded_end > duration and not any(s["end"] > duration for s in segments):
                    saved_duration = rounded_end
                payload = {
                    "clip_id": clip_id,
                    "model": f"faster-whisper {model_name}",
                    "language": info.language,
                    "duration": saved_duration,
                    "segments": rounded_segments,
                }
                try:
                    _save_json(dest, payload, overwrite)
                    if saved_duration != duration:
                        record["repair_metadata"] = {
                            "original_duration": duration,
                            "duration": saved_duration,
                        }
                    record.update(
                        status="ok" if segments else "no_speech",
                        no_speech=not bool(segments),
                        review_required=True,
                        anomalies=anomalies,
                        path=str(dest),
                    )
                except FileExistsError:
                    record.update(_skipped_record(dest, clip_id))
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        report.append(record)
        emit(record)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="download root containing audio/ and clips/")
    parser.add_argument("--out", required=True, help="directory for transcript JSON files")
    parser.add_argument(
        "--include-clips", action="store_true", help="report missing audio from clips/"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace existing transcripts")
    parser.add_argument("--model", choices=("large-v3", "small"), default="large-v3")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--language", choices=("auto", "ru"), default="auto")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path")
    parser.add_argument("--report", help="new JSONL report outside input and output")
    args = parser.parse_args(argv)

    def factory():
        whisper = importlib.import_module("faster_whisper")
        return whisper.WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    try:
        report = transcribe_batch(
            args.input,
            args.out,
            model_factory=factory,
            model_name=args.model,
            include_clips=args.include_clips,
            overwrite=args.overwrite,
            language=args.language,
            ffmpeg=args.ffmpeg,
            report_path=args.report,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(any(item["status"] in ("error", "missing_audio") for item in report))


if __name__ == "__main__":
    raise SystemExit(main())
