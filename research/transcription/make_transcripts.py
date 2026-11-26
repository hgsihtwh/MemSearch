"""MEM-30: итоговые транскрипты коллекции с таймкодами и отчёт о распознавании.

Источник — транскрипты прогона faster-whisper large-v3 из MEM-76 (Colab, T4, 29.10.2026):
тот же алгоритм, что в ``ml/transcribe.py`` (VAD с порогом 0,3, при пустом результате —
повтор без VAD), с ``language='ru'``. Скрипт ничего не распознаёт заново, а:

- удаляет известные галлюцинации Whisper на тишине и музыке (титры «Субтитры сделал …»,
  «Продолжение следует…», одиночное «Музыка») — без них такие фразы попадали бы в поиск;
- сохраняет транскрипты в формате MEM-79 (``transcripts/{clip_id}.json``); если после
  очистки сегментов не осталось — ``segments: []``, т. е. речь не обнаружена;
- пишет ``report.csv``: статус каждого ролика коллекции и флаг ``no_speech``.

Исходные файлы MEM-76 не изменяются. Запуск из корня репозитория:

    python research/transcription/make_transcripts.py
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.chunking import chunk_transcript  # noqa: E402

PASSPORT = ROOT / "research/collection/passport.csv"
DOWNLOAD_REPORT = ROOT / "research/download/results/report.csv"
SOURCE = ROOT / "research/embeddings_check/results/transcripts"
HERE = Path(__file__).resolve().parent
OUT = HERE / "transcripts"
REPORT = HERE / "report.csv"

# Сегмент целиком состоит из такой фразы — это галлюцинация, а не речь ролика.
HALLUCINATIONS = [
    re.compile(r"субтитры (сделал|делал|создавал|подогнал|подготовил)\w* \w+", re.IGNORECASE),
    re.compile(r"продолжение следует", re.IGNORECASE),
    re.compile(r"(динамичная |спокойная |весёлая |веселая )?музыка", re.IGNORECASE),
]


def is_hallucination(text):
    phrase = re.sub(r"[^\w ]+", " ", text).strip()
    phrase = re.sub(r"\s+", " ", phrase)
    return any(pattern.fullmatch(phrase) for pattern in HALLUCINATIONS)


def load_passport():
    with PASSPORT.open(encoding="utf-8-sig", newline="") as f:
        return {int(row["№"]): row for row in csv.DictReader(f)}


def load_downloads():
    with DOWNLOAD_REPORT.open(encoding="utf-8-sig", newline="") as f:
        return {int(row["clip_id"]): row for row in csv.DictReader(f)}


def load_sources():
    sources = {}
    for path in SOURCE.glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item["clip_id"] != int(path.stem):
            raise ValueError(f"clip_id не совпадает с именем файла: {path}")
        sources[item["clip_id"]] = item
    return sources


def clean(transcript):
    """Remove hallucinations and fix timestamps past the audio end.

    Whisper pads the last 30-second window: segments may end at «…9.98 s» after the
    real end of audio or even start beyond it. A segment starting at or after the
    duration cannot contain speech of the clip and is dropped; one starting inside
    the audio is clipped to the duration. Returns counts of each change.
    """
    duration = transcript["duration"]
    kept, stats = [], {"hallucinations": 0, "beyond_audio": 0, "clipped": 0}
    for segment in transcript["segments"]:
        if is_hallucination(segment["text"]):
            stats["hallucinations"] += 1
        elif segment["start"] >= duration:
            stats["beyond_audio"] += 1
        elif segment["end"] > duration:
            stats["clipped"] += 1
            kept.append({**segment, "end": duration})
        else:
            kept.append(segment)
    return {**transcript, "segments": kept}, stats


def main():
    passport, downloads, sources = load_passport(), load_downloads(), load_sources()
    unknown = set(sources) - set(passport)
    if unknown:
        raise ValueError(f"транскрипты роликов не из паспорта: {sorted(unknown)}")
    OUT.mkdir(exist_ok=True)
    rows = []
    for clip_id, info in sorted(passport.items()):
        source = sources.get(clip_id)
        speech_label = info["Речь"].strip().lower()
        row = {
            "clip_id": clip_id,
            "platform": info["Платформа"],
            "status": "unavailable",
            "no_speech": "",
            "segments": 0,
            "removed_hallucinations": 0,
            "removed_beyond_audio": 0,
            "clipped_to_duration": 0,
            "words": 0,
            "duration": "",
            "passport_speech": speech_label,
            "note": "",
        }
        if source is None:
            error = downloads.get(clip_id, {}).get("error", "")
            row["note"] = f"ролик не скачан: {error}"[:200]
            rows.append(row)
            continue
        transcript, stats = clean(source)
        chunk_transcript(transcript)  # проверка таймкодов по правилам MEM-79/MEM-31
        segments = transcript["segments"]
        no_speech = not segments
        row.update(
            status="no_speech" if no_speech else "ok",
            no_speech="да" if no_speech else "нет",
            segments=len(segments),
            removed_hallucinations=stats["hallucinations"],
            removed_beyond_audio=stats["beyond_audio"],
            clipped_to_duration=stats["clipped"],
            words=sum(len(s["text"].split()) for s in segments),
            duration=transcript["duration"],
        )
        if not no_speech and speech_label == "нет":
            row["note"] = "по паспорту речи нет: распознан текст песни или закадровый текст"
        elif no_speech and speech_label == "да":
            row["note"] = "по паспорту речь есть, а сегментов нет — проверить вручную"
        target = OUT / f"{clip_id}.json"
        target.write_text(
            json.dumps(transcript, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        rows.append(row)
    with REPORT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "clips": len(rows),
        "ok": sum(r["status"] == "ok" for r in rows),
        "no_speech": sum(r["status"] == "no_speech" for r in rows),
        "unavailable": sum(r["status"] == "unavailable" for r in rows),
        "removed_hallucinations": sum(r["removed_hallucinations"] for r in rows),
        "removed_beyond_audio": sum(r["removed_beyond_audio"] for r in rows),
        "clipped_to_duration": sum(r["clipped_to_duration"] for r in rows),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
