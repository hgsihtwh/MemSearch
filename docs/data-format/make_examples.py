"""MEM-79: JSON-схемы файлов ML-конвейера и примеры на реальном ролике №77 (VK) коллекции.
Схемы пишутся в schemas/, примеры — в examples/77/; примеры проверяются по схемам."""

import csv
import json
import os
import re

import jsonschema

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
CLIP = 77
WINDOW, STRIDE = 15, 7  # выбрано в MEM-76 (decisions.md §14)

S = "https://json-schema.org/draft/2020-12/schema"
CLIP_ID = {"type": "integer", "minimum": 1, "description": "Внутренний номер ролика в MemSearch"}
SEC = {
    "type": "number",
    "minimum": 0,
    "description": "Секунды от начала ролика, 2 знака после запятой",
}

schemas = {
    "clip": {
        "$schema": S,
        "title": "Ролик",
        "description": "clips/{clip_id}.json — создаётся при добавлении ролика",
        "type": "object",
        "additionalProperties": False,
        "required": ["clip_id", "platform", "url", "external_id", "status", "added_at"],
        "properties": {
            "clip_id": CLIP_ID,
            "platform": {"enum": ["tiktok", "vk"]},
            "url": {"type": "string", "format": "uri"},
            "external_id": {
                "type": "string",
                "description": "ID ролика на платформе; пара (platform, external_id) уникальна — по ней ловим дубли",
            },
            "author": {"type": ["string", "null"], "description": "Ник автора с @"},
            "caption": {
                "type": ["string", "null"],
                "description": "Подпись (описание) ролика с платформы, как есть",
            },
            "duration": {
                "type": ["number", "null"],
                "minimum": 0,
                "description": "Длительность, с",
            },
            "thumbnail": {
                "type": ["string", "null"],
                "description": "Путь к уменьшенному превью (thumbnails/{clip_id}.jpg)",
            },
            "added_at": {"type": "string", "format": "date-time"},
            "status": {
                "enum": [
                    "new",
                    "downloading",
                    "transcribing",
                    "indexing",
                    "enriching",
                    "done",
                    "error",
                ]
            },
            "error": {
                "type": ["string", "null"],
                "description": "Текст ошибки, если status = error",
            },
        },
    },
    "transcript": {
        "$schema": S,
        "title": "Транскрипт",
        "description": "transcripts/{clip_id}.json — результат распознавания речи",
        "type": "object",
        "additionalProperties": False,
        "required": ["clip_id", "model", "language", "duration", "segments"],
        "properties": {
            "clip_id": CLIP_ID,
            "model": {
                "type": "string",
                "description": "Модель ASR, например «faster-whisper large-v3»",
            },
            "language": {"type": "string", "description": "Код языка ISO 639-1"},
            "duration": {"type": "number", "minimum": 0},
            "segments": {
                "type": "array",
                "description": "Фразы по порядку; пустой массив — речи нет",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["start", "end", "text"],
                    "properties": {
                        "start": SEC,
                        "end": SEC,
                        "text": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    },
    "ocr": {
        "$schema": S,
        "title": "Текст в кадре",
        "description": "ocr/{clip_id}.json — текст с ключевых кадров (версия 1.0)",
        "type": "object",
        "additionalProperties": False,
        "required": ["clip_id", "model", "frames"],
        "properties": {
            "clip_id": CLIP_ID,
            "model": {"type": "string"},
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["t", "text"],
                    "properties": {"t": SEC, "text": {"type": "string"}},
                },
            },
        },
    },
    "chunks": {
        "$schema": S,
        "title": "Фрагменты",
        "description": "chunks/{clip_id}.json — единицы поиска; i-й фрагмент соответствует i-й строке embeddings/{clip_id}.npy",
        "type": "object",
        "additionalProperties": False,
        "required": ["clip_id", "chunking", "chunks"],
        "properties": {
            "clip_id": CLIP_ID,
            "chunking": {
                "type": "object",
                "additionalProperties": False,
                "required": ["window", "stride"],
                "description": "Параметры разбиения речи на окна, с",
                "properties": {
                    "window": {"type": "number", "exclusiveMinimum": 0},
                    "stride": {"type": "number", "exclusiveMinimum": 0},
                },
            },
            "chunks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["chunk_id", "source", "start", "end", "text"],
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "pattern": "^[0-9]+_[0-9]{3}$",
                            "description": "{clip_id}_{порядковый номер из 3 цифр}",
                        },
                        "source": {
                            "enum": ["caption", "speech", "ocr", "scene"],
                            "description": "caption — подпись; speech — речь; ocr — текст в кадре; scene — описание кадра (версия 1.0)",
                        },
                        "start": {
                            "type": ["number", "null"],
                            "minimum": 0,
                            "description": "null для подписи",
                        },
                        "end": {"type": ["number", "null"], "minimum": 0},
                        "text": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    },
    "embeddings_meta": {
        "$schema": S,
        "title": "Векторы фрагментов (описание)",
        "description": "embeddings/{clip_id}.json рядом с embeddings/{clip_id}.npy (float32, [число фрагментов × dim], нормированы по L2)",
        "type": "object",
        "additionalProperties": False,
        "required": ["clip_id", "model", "dim", "count", "chunk_ids"],
        "properties": {
            "clip_id": CLIP_ID,
            "model": {"type": "string", "description": "Модель эмбеддингов; выбирается в MEM-76"},
            "dim": {"type": "integer", "const": 1024},
            "count": {"type": "integer", "minimum": 0},
            "chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Порядок строк матрицы",
            },
        },
    },
    "enrichment": {
        "$schema": S,
        "title": "Описание, теги и тема ролика",
        "description": "enrichment/{clip_id}.json",
        "type": "object",
        "additionalProperties": False,
        "required": ["clip_id", "summary", "tags", "topic_id"],
        "properties": {
            "clip_id": CLIP_ID,
            "summary": {
                "type": ["string", "null"],
                "description": "Краткое описание, 1–2 предложения (языковая модель)",
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "source"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "source": {"enum": ["auto", "user"]},
                    },
                },
            },
            "topic_id": {
                "type": ["integer", "null"],
                "description": "Тема из topics.json; null — ролик не попал ни в одну тему",
            },
        },
    },
    "topics": {
        "$schema": S,
        "title": "Темы коллекции",
        "description": "topics.json — один файл на коллекцию, пересчитывается BERTopic",
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["topic_id", "name", "keywords", "clip_ids"],
            "properties": {
                "topic_id": {"type": "integer", "minimum": 0},
                "name": {
                    "type": "string",
                    "description": "Название темы; пользователь может переименовать",
                },
                "name_source": {"enum": ["auto", "user"]},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "clip_ids": {"type": "array", "items": CLIP_ID},
            },
        },
    },
}

# ---------- примеры на ролике №77
row = next(
    r
    for r in csv.DictReader(
        open(os.path.join(ROOT, "research", "collection", "passport.csv"), encoding="utf-8-sig")
    )
    if r["№"] == str(CLIP)
)
m = re.search(r"clip(-?\d+_\d+)", row["Ссылка"])
clip = dict(
    clip_id=CLIP,
    platform="vk",
    url=row["Ссылка"],
    external_id=m[1],
    author=row["Автор"],
    caption=None,
    duration=126.0,
    thumbnail=f"thumbnails/{CLIP}.jpg",
    added_at="2026-10-20T19:00:00+07:00",
    status="done",
    error=None,
)

segs = []
for line in (
    open(
        os.path.join(
            ROOT, "research", "asr_check", "results", "transcripts", f"{CLIP}_large-v3.txt"
        ),
        encoding="utf-8",
    )
    .read()
    .splitlines()[1:]
):
    t = re.match(r"\[(\d+):([\d.]+) – (\d+):([\d.]+)\] (.*)", line)
    if t:
        segs.append(
            dict(
                start=round(int(t[1]) * 60 + float(t[2]), 2),
                end=round(int(t[3]) * 60 + float(t[4]), 2),
                text=t[5],
            )
        )
transcript = dict(
    clip_id=CLIP, model="faster-whisper large-v3", language="ru", duration=126.0, segments=segs
)

chunks, t = [], 0.0
while True:  # то же разбиение, что в ноутбуке MEM-76
    part = [s for s in segs if s["end"] > t and s["start"] < t + WINDOW]
    text = " ".join(s["text"] for s in part)
    if part and (not chunks or chunks[-1]["text"] != text):
        chunks.append(
            dict(
                chunk_id=f"{CLIP}_{len(chunks):03d}",
                source="speech",
                start=part[0]["start"],
                end=part[-1]["end"],
                text=text,
            )
        )
    if t + WINDOW >= segs[-1]["end"]:
        break
    t += STRIDE
chunks_doc = dict(
    clip_id=CLIP, chunking=dict(window=WINDOW, stride=STRIDE), chunks=chunks
)  # у ролика VK нет подписи — фрагмента caption нет

emb_meta = dict(
    clip_id=CLIP,
    model="intfloat/multilingual-e5-large",
    dim=1024,
    count=len(chunks),
    chunk_ids=[c["chunk_id"] for c in chunks],
)
enrichment = dict(
    clip_id=CLIP,
    summary="Девушка вспоминает бедное детство в Сибири: донашивала чужие вещи, ходила зимой в кроссовках, "
    "бесплатно занималась конным спортом; рассказывает, как проработала финансовые травмы.",
    tags=[
        dict(name="детство", source="auto"),
        dict(name="бедность", source="auto"),
        dict(name="деньги", source="auto"),
        dict(name="конный спорт", source="auto"),
        dict(name="пересмотреть", source="user"),
    ],
    topic_id=3,
)
topics = [
    dict(
        topic_id=3,
        name="Истории из жизни",
        name_source="auto",
        keywords=["детство", "семья", "деньги", "травма"],
        clip_ids=[13, 52, 77],
    )
]

examples = {
    "clip": clip,
    "transcript": transcript,
    "chunks": chunks_doc,
    "embeddings_meta": emb_meta,
    "enrichment": enrichment,
    "topics": topics,
}
os.makedirs(os.path.join(HERE, "schemas"), exist_ok=True)
os.makedirs(os.path.join(HERE, "examples", str(CLIP)), exist_ok=True)
for name, sch in schemas.items():
    jsonschema.Draft202012Validator.check_schema(sch)
    json.dump(
        sch,
        open(os.path.join(HERE, "schemas", f"{name}.schema.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
for name, ex in examples.items():
    jsonschema.validate(ex, schemas[name], format_checker=jsonschema.FormatChecker())
    fname = "topics.json" if name == "topics" else f"{name}.json"
    json.dump(
        ex,
        open(os.path.join(HERE, "examples", str(CLIP), fname), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
print("ok:", len(schemas), "схем,", len(examples), "примеров; фрагментов речи", len(chunks))
