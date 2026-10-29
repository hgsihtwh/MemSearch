"""Пакетная загрузка коллекции роликов TikTok и VK (MEM-20).

Для каждой ссылки: yt-dlp извлекает метаданные и аудиодорожку, ffmpeg переводит её
в WAV 16 кГц моно. Метаданные пишутся в clips/{clip_id}.json по схеме
docs/data-format/schemas/clip.schema.json, расширенные — в meta/{clip_id}.json.
Готовые субтитры платформы (если yt-dlp их отдаёт) сохраняются в subs/.
Недоступные ролики логируются в errors.log и report.csv, прогон не прерывается.

Пример:
    python download.py --input ../collection/passport.csv --out D:/memsearch_dl --workers 4

Зависимости (вне проекта): yt-dlp, curl_cffi (имитация браузера для TikTok),
ffmpeg в PATH или пакет imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yt_dlp

PROXY_ENV = "MEMSEARCH_PROXY"
REPORT_FIELDS = [
    "clip_id",
    "platform",
    "url",
    "status",
    "duration",
    "has_subtitles",
    "subtitle_langs",
    "auto_caption_langs",
    "error",
]
# TikTok: формат «audio» — это трек звука (часто чужая музыка), а не дорожка ролика,
# поэтому берём лучший аудиоформат, кроме него, а если такого нет — видео со звуком.
FORMAT = (
    "bestaudio[format_id!=audio][protocol!*=m3u8]/bestaudio[format_id!=audio]"
    "/best[acodec!=none]/best"
)


@dataclass
class Item:
    clip_id: int
    platform: str
    url: str


class _Log:
    """Логгер для yt-dlp: предупреждения и ошибки собираются, в консоль не льются."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        self.messages.append(msg)

    def error(self, msg: str) -> None:
        self.messages.append(msg)


def detect_platform(url: str) -> str | None:
    host = re.sub(r"^https?://", "", url.strip().lower()).split("/")[0]
    if "tiktok.com" in host:
        return "tiktok"
    if host.endswith(("vk.com", "vk.ru", "vkvideo.ru")):
        return "vk"
    return None


def read_input(path: Path) -> list[Item]:
    """CSV паспорта (колонки «№», «Ссылка») или txt со ссылками по одной в строке."""
    items: list[Item] = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("Ссылка") or row.get("url") or "").strip()
                num = (row.get("№") or row.get("clip_id") or "").strip()
                if not url or not num:
                    continue
                platform = detect_platform(url) or (row.get("Платформа") or "").strip().lower()
                items.append(Item(int(num), platform, url))
    else:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        urls = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
        for i, url in enumerate(urls, start=1):
            items.append(Item(i, detect_platform(url) or "", url))
    return items


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg не найден: установите его в PATH или пакет imageio-ffmpeg")


def run_ffmpeg(ffmpeg: str, args: list[str]) -> None:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *args]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg: {res.stderr.strip()[-300:]}")


def short_error(exc: BaseException) -> str:
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).replace("\n", " ").strip()
    return re.sub(r"^ERROR:\s*", "", msg)[:300]


def author_of(info: dict[str, Any], platform: str) -> str | None:
    if platform == "tiktok":
        handle = info.get("uploader") or info.get("channel")
        return f"@{handle.lstrip('@')}" if handle else None
    return info.get("uploader") or info.get("channel")


def caption_of(info: dict[str, Any], platform: str) -> str | None:
    text = (info.get("description") or "").strip()
    if not text and platform == "tiktok":
        text = (info.get("title") or "").strip()
    return text or None


def published_at(info: dict[str, Any]) -> str | None:
    ts = info.get("timestamp")
    if ts:
        return datetime.fromtimestamp(ts, UTC).isoformat()
    date = info.get("upload_date")
    return f"{date[:4]}-{date[4:6]}-{date[6:]}" if date else None


def sub_langs(subs: dict[str, Any] | None) -> list[str]:
    return sorted(k for k in (subs or {}) if k != "live_chat")


class Downloader:
    def __init__(
        self, out: Path, proxy: str | None, cookies: Path | None, save_subs: bool, retries: int
    ) -> None:
        self.out = out
        self.proxy = proxy
        self.cookies = cookies
        self.save_subs = save_subs
        self.retries = retries
        self.ffmpeg = find_ffmpeg()
        self.lock = threading.Lock()
        for sub in ("audio", "clips", "meta", "subs", "thumbnails", "tmp"):
            (out / sub).mkdir(parents=True, exist_ok=True)

    def ydl_opts(self, item: Item, logger: _Log) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "format": FORMAT,
            "outtmpl": str(self.out / "tmp" / f"{item.clip_id}.%(ext)s"),
            "quiet": True,
            "noprogress": True,
            "logger": logger,
            "ffmpeg_location": self.ffmpeg,
            "retries": self.retries,
            "socket_timeout": 30,
            "noplaylist": True,
        }
        if self.proxy:
            opts["proxy"] = self.proxy
        if self.cookies:
            opts["cookiefile"] = str(self.cookies)
        return opts

    def log_error(self, item: Item, error: str) -> None:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with self.lock, (self.out / "errors.log").open("a", encoding="utf-8") as f:
            f.write(f"{stamp}\t{item.clip_id}\t{item.platform}\t{item.url}\t{error}\n")

    def done(self, item: Item) -> bool:
        return (self.out / "audio" / f"{item.clip_id}.wav").exists() and (
            self.out / "meta" / f"{item.clip_id}.json"
        ).exists()

    def process(self, item: Item) -> dict[str, Any]:
        meta_path = self.out / "meta" / f"{item.clip_id}.json"
        if self.done(item):
            return row_from_meta(json.loads(meta_path.read_text(encoding="utf-8")))
        logger = _Log()
        try:
            if item.platform not in ("tiktok", "vk"):
                raise ValueError("неизвестная платформа (ожидается TikTok или VK)")
            with yt_dlp.YoutubeDL(self.ydl_opts(item, logger)) as ydl:
                info = ydl.sanitize_info(ydl.extract_info(item.url, download=True))
                subs = sub_langs(info.get("subtitles"))
                auto = sub_langs(info.get("automatic_captions"))
                saved_subs = self.fetch_subs(ydl, item, info) if self.save_subs else []
                thumb = self.fetch_thumbnail(ydl, item, info)
            audio = self.to_wav(item)
        except Exception as exc:  # недоступный ролик не должен ломать прогон
            error = short_error(exc)
            self.log_error(item, error)
            self.write_clip(item, None, None, error)
            return {
                "clip_id": item.clip_id,
                "platform": item.platform,
                "url": item.url,
                "status": "error",
                "duration": "",
                "has_subtitles": "",
                "subtitle_langs": "",
                "auto_caption_langs": "",
                "error": error,
            }
        duration = info.get("duration")
        meta = {
            "clip_id": item.clip_id,
            "platform": item.platform,
            "url": item.url,
            "external_id": str(info.get("id") or ""),
            "author": author_of(info, item.platform),
            "author_id": info.get("uploader_id") or info.get("channel_id"),
            "title": info.get("title"),
            "description": info.get("description"),
            "published_at": published_at(info),
            "duration": float(duration) if duration is not None else None,
            "thumbnail_url": info.get("thumbnail"),
            "thumbnail": thumb,
            "audio": audio,
            "subtitles": subs,
            "automatic_captions": auto,
            "subtitle_files": saved_subs,
            "format_id": info.get("format_id"),
            "downloaded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "warnings": logger.messages[-5:],
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.write_clip(item, info, thumb, None)
        return row_from_meta(meta)

    def to_wav(self, item: Item) -> str:
        tmp = sorted((self.out / "tmp").glob(f"{item.clip_id}.*"))
        tmp = [p for p in tmp if not p.name.endswith((".part", ".ytdl"))]
        if not tmp:
            raise RuntimeError("yt-dlp не сохранил файл")
        wav = self.out / "audio" / f"{item.clip_id}.wav"
        part = wav.with_suffix(".part.wav")
        try:
            run_ffmpeg(
                self.ffmpeg,
                [
                    "-i",
                    str(tmp[0]),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(part),
                ],
            )
            part.replace(wav)
        finally:
            for p in tmp:
                p.unlink(missing_ok=True)
        return f"audio/{item.clip_id}.wav"

    def fetch_subs(self, ydl: yt_dlp.YoutubeDL, item: Item, info: dict[str, Any]) -> list[str]:
        saved: list[str] = []
        for kind in ("subtitles", "automatic_captions"):
            for lang, tracks in (info.get(kind) or {}).items():
                if not tracks:
                    continue
                track = tracks[0]
                ext = track.get("ext") or "vtt"
                name = f"{item.clip_id}.{lang}.{ext}"
                try:
                    if track.get("data"):
                        data = track["data"].encode("utf-8")
                    else:
                        data = ydl.urlopen(track["url"]).read()
                    (self.out / "subs" / name).write_bytes(data)
                    saved.append(f"subs/{name}")
                except Exception as exc:
                    self.log_error(item, f"субтитры {lang}: {short_error(exc)}")
        return saved

    def fetch_thumbnail(
        self, ydl: yt_dlp.YoutubeDL, item: Item, info: dict[str, Any]
    ) -> str | None:
        url = info.get("thumbnail")
        if not url:
            return None
        raw = self.out / "tmp" / f"thumb_{item.clip_id}"
        jpg = self.out / "thumbnails" / f"{item.clip_id}.jpg"
        try:
            raw.write_bytes(ydl.urlopen(url).read())
            run_ffmpeg(
                self.ffmpeg, ["-i", str(raw), "-vf", "scale=320:-2", "-frames:v", "1", str(jpg)]
            )
            return f"thumbnails/{item.clip_id}.jpg"
        except Exception as exc:
            self.log_error(item, f"превью: {short_error(exc)}")
            return None
        finally:
            raw.unlink(missing_ok=True)

    def write_clip(
        self, item: Item, info: dict[str, Any] | None, thumb: str | None, error: str | None
    ) -> None:
        """clips/{clip_id}.json строго по clip.schema.json."""
        info = info or {}
        duration = info.get("duration")
        external_id = str(info.get("id") or "") or external_id_from_url(item.url)
        clip = {
            "clip_id": item.clip_id,
            "platform": item.platform if item.platform in ("tiktok", "vk") else "tiktok",
            "url": item.url,
            "external_id": external_id,
            "author": author_of(info, item.platform) if info else None,
            "caption": caption_of(info, item.platform) if info else None,
            "duration": round(float(duration), 2) if duration is not None else None,
            "thumbnail": thumb,
            "added_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            # аудио готово — следующий шаг конвейера распознавание речи
            "status": "error" if error else "transcribing",
            "error": error,
        }
        path = self.out / "clips" / f"{item.clip_id}.json"
        path.write_text(json.dumps(clip, ensure_ascii=False, indent=2), encoding="utf-8")


def external_id_from_url(url: str) -> str:
    m = re.search(r"/video/(\d+)", url) or re.search(r"clip(-?\d+_\d+)", url)
    return m.group(1) if m else ""


def row_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    subs, auto = meta.get("subtitles") or [], meta.get("automatic_captions") or []
    return {
        "clip_id": meta["clip_id"],
        "platform": meta["platform"],
        "url": meta["url"],
        "status": "ok",
        "duration": meta.get("duration") if meta.get("duration") is not None else "",
        "has_subtitles": "да" if subs or auto else "нет",
        "subtitle_langs": " ".join(subs),
        "auto_caption_langs": " ".join(auto),
        "error": "",
    }


def summary(rows: list[dict[str, Any]]) -> str:
    lines = []
    for platform in ("tiktok", "vk"):
        part = [r for r in rows if r["platform"] == platform]
        if not part:
            continue
        ok = [r for r in part if r["status"] == "ok"]
        with_subs = [r for r in ok if r["has_subtitles"] == "да"]
        langs: dict[str, int] = {}
        for r in with_subs:
            for lang in f"{r['subtitle_langs']} {r['auto_caption_langs']}".split():
                langs[lang] = langs.get(lang, 0) + 1
        lang_str = ", ".join(f"{k}: {v}" for k, v in sorted(langs.items(), key=lambda x: -x[1]))
        lines.append(
            f"{platform}: скачано {len(ok)}/{len(part)}, с субтитрами {len(with_subs)}/{len(ok)}"
            + (f" ({lang_str})" if lang_str else "")
        )
    failed = [str(r["clip_id"]) for r in rows if r["status"] != "ok"]
    lines.append("недоступны: " + (", ".join(failed) if failed else "нет"))
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Пакетная загрузка роликов TikTok/VK: WAV + метаданные")
    p.add_argument("--input", type=Path, required=True, help="passport.csv или txt со ссылками")
    p.add_argument("--out", type=Path, required=True, help="каталог результата (вне репозитория)")
    p.add_argument(
        "--proxy", default=os.environ.get(PROXY_ENV), help=f"прокси yt-dlp, или {PROXY_ENV}"
    )
    p.add_argument(
        "--cookies", type=Path, help="cookies.txt (Netscape) для роликов, требующих входа"
    )
    p.add_argument("--workers", type=int, default=1, help="параллельных загрузок")
    p.add_argument("--limit", type=int, default=0, help="обработать только первые N роликов")
    p.add_argument("--only", choices=["tiktok", "vk"], help="только одна платформа")
    p.add_argument("--retries", type=int, default=3, help="повторов запроса в yt-dlp")
    p.add_argument("--no-subs", action="store_true", help="не сохранять файлы субтитров")
    args = p.parse_args()

    items = read_input(args.input)
    if args.only:
        items = [it for it in items if it.platform == args.only]
    if args.limit:
        items = items[: args.limit]
    dl = Downloader(args.out, args.proxy, args.cookies, not args.no_subs, args.retries)
    print(f"роликов: {len(items)}, прокси: {'да' if args.proxy else 'нет'}, ffmpeg: {dl.ffmpeg}")

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(dl.process, it): it for it in items}
        for n, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            rows.append(row)
            tail = f" — {row['error'][:120]}" if row["error"] else ""
            print(f"[{n}/{len(items)}] №{row['clip_id']} {row['platform']}: {row['status']}{tail}")

    # строки прошлых прогонов (другие ролики) сохраняются, свежие их заменяют
    report = args.out / "report.csv"
    merged: dict[int, dict[str, Any]] = {}
    if report.exists():
        with report.open(encoding="utf-8-sig", newline="") as f:
            merged = {int(r["clip_id"]): r for r in csv.DictReader(f)}
    merged.update({int(r["clip_id"]): r for r in rows})
    rows = [merged[k] for k in sorted(merged)]
    with report.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    shutil.rmtree(args.out / "tmp", ignore_errors=True)
    print(f"\nготово за {time.monotonic() - started:.0f} с\n{summary(rows)}")


if __name__ == "__main__":
    main()
