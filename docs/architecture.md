# Описание архитектуры и модели данных

**Задача:** MEM-26 · **Дата:** 29.10.2026 · **Основание:** [ADR-001](adr/001-technology-stack.md), спецификация требований (SRS v0.1), [решения по продукту](decisions.md), [формат данных ML-конвейера](data-format/README.md)

## 1. Обзор

MemSearch — веб-приложение (PWA, mobile-first) для поиска сохранённых роликов TikTok и VK по смыслу сказанного, тексту в кадре и описанию сцены. Пользователь добавляет ссылки, система в фоне скачивает аудио и кадры, распознаёт речь и текст, строит векторы фрагментов и ищет по ним гибридным поиском (векторный + полнотекстовый). Выдача показывает ролики целиком (решение 11): поиск ведётся по фрагментам, но один ролик — одна карточка.

Ключевые свойства, определяющие архитектуру:

| Требование | Как обеспечивается |
|---|---|
| Поиск ≤ 2 с на 10 000 роликов (НФТ-1) | pgvector с индексом HNSW и полнотекстовый индекс GIN в одной СУБД; вектор запроса считает ML-сервис синхронно |
| Обработка ролика ≤ 1 мин на GPU (НФТ-2) | Длительная обработка — в Celery worker, вне HTTP-запроса; ML-модели в отдельном сервисе |
| Перенос 1000 роликов, поиск по названию через 10 мин (НФТ-6, ФТ-2.5) | Два прохода: быстрый (метаданные, готовые субтитры) и полный; две очереди с приоритетом новых роликов (ФТ-2.6) |
| Видео и аудио не хранятся (НФТ-8, ФТ-2.1) | Аудио и кадры — во временном каталоге worker и удаляются после обработки; в S3 — только превью |
| Без учётных записей (ФТ-7.5 в бэклоге, вопрос В-1) | Коллекция идентифицируется секретным ключом, который хранится на устройстве; Telegram-бот привязывается одноразовым кодом |
| Модели заменяются настройкой (ФТ-8.3) | Модели ASR, эмбеддингов и языковая модель задаются переменными окружения ML-сервиса |

## 2. Компоненты

```mermaid
flowchart LR
    subgraph Клиенты
        PWA["Веб-приложение<br/>React + Vite, PWA"]
        EXT["Расширение браузера<br/>Chrome MV3"]
        BOT["Telegram-бот"]
    end
    subgraph Сервер
        API["Backend API<br/>FastAPI"]
        WRK["Celery worker<br/>оркестрация обработки"]
        ML["ML-сервис<br/>REST API"]
        PG[("PostgreSQL 16<br/>+ pgvector")]
        RD[("Redis<br/>брокер Celery")]
        S3[("S3-хранилище<br/>превью")]
    end
    subgraph Внешние
        SRC["TikTok / VK<br/>yt-dlp, oEmbed"]
        LLM["Языковая модель<br/>Kimi через агрегатор"]
    end
    PWA -- HTTPS / JSON --> API
    EXT -- HTTPS / JSON --> API
    BOT -- HTTPS / JSON --> API
    API --> PG
    API -- задачи --> RD
    API -- "вектор запроса" --> ML
    RD --> WRK
    WRK --> PG
    WRK --> S3
    WRK -- "аудио, кадры" --> SRC
    WRK -- "ASR, OCR, векторы" --> ML
    WRK -- "описание, теги" --> LLM
    PWA -. превью .-> S3
```

| Компонент | Ответственность | Технологии |
|---|---|---|
| Веб-приложение | Поиск, выдача, карточка ролика, добавление ссылок, темы, профиль; приём ссылок из «Поделиться» на Android (Web Share Target) | React, TypeScript, Vite, PWA |
| Расширение браузера | Сбор ссылок со страницы «Избранное» TikTok (затем VK) и пакетная отправка в коллекцию с возобновлением (ФТ-1.6) | Chrome Manifest V3 |
| Telegram-бот | Приём ссылок с iPhone и пересылка в API от имени привязанной коллекции (ФТ-1.5, ФТ-7.2) | aiogram; размещается вне России (вопрос В-4) |
| Backend API | REST API ([OpenAPI](api/openapi.yaml)), проверка ссылок и дубликатов, постановка задач, поиск, теги, темы, оценки | FastAPI, SQLAlchemy 2 (async, asyncpg), Alembic, Pydantic |
| Celery worker | Конвейер обработки ролика: скачивание → распознавание → фрагменты → векторы → обогащение; статусы и повторы | Celery, yt-dlp, ffmpeg |
| ML-сервис | Модели: распознавание речи, эмбеддинги текста, OCR, векторы кадров, тематическое моделирование | FastAPI, faster-whisper large-v3, multilingual-e5-large, BERTopic, OCR, CLIP/SigLIP |
| PostgreSQL + pgvector | Все предметные данные, векторы фрагментов (HNSW), полнотекстовый индекс (GIN, `russian`), триграммы (`pg_trgm`) для нечёткого поиска | PostgreSQL 16, pgvector 0.8 |
| Redis | Брокер и хранилище результатов Celery | Redis 7 |
| S3-хранилище | Уменьшенные превью и ключевые кадры для карточек | MinIO (локально), любое S3-совместимое |

## 3. Развёртывание

```mermaid
flowchart TB
    U["Телефон / компьютер пользователя"] -- HTTPS --> N
    subgraph VPS["Linux-сервер (Docker Compose)"]
        N["Caddy / Nginx<br/>TLS, статика PWA"]
        N --> A["backend"]
        A --- W["worker-fast<br/>очередь fast"]
        A --- W2["worker-full<br/>очередь full"]
        A & W & W2 --> P[("postgres")]
        A & W & W2 --> R[("redis")]
        W & W2 --> M["ml<br/>(GPU при наличии)"]
        W & W2 --> O[("minio")]
    end
    G["Colab GPU<br/>обработка демо-коллекции"] -. "файлы в формате data-format,<br/>импорт (MEM-34)" .-> P
    B["Telegram-бот<br/>зарубежный хостинг"] -- HTTPS --> N
```

- Все сервисы описаны в одном `compose.yaml` и поднимаются одной командой (НФТ-12). Backend и оба worker используют один образ `backend` с разными командами запуска.
- Сервер, скорее всего, без GPU. Распознавание large-v3 на CPU в разы медленнее, поэтому демо-коллекция обрабатывается в Colab на GPU и импортируется файлами ([data-format](data-format/README.md)); новые ролики обрабатываются на сервере (на CPU — модель small как запасной вариант, решение 13).
- Скачивание TikTok с российского IP может быть заблокировано (вопрос В-3) — worker поддерживает HTTP-прокси для yt-dlp через переменную окружения.
- Резервная копия базы — ежедневный `pg_dump` в отдельный том (ФТ-8.4).

## 4. Обработка ролика

```mermaid
sequenceDiagram
    participant C as Клиент
    participant A as Backend API
    participant Q as Redis (Celery)
    participant W as Worker
    participant M as ML-сервис
    participant DB as PostgreSQL
    C->>A: POST /clips {urls}
    A->>DB: clip(status=queued), job(fast)
    A->>Q: process_fast(clip_id)
    A-->>C: 202 + статусы
    Q->>W: process_fast
    W->>W: oEmbed / yt-dlp: метаданные, субтитры
    W->>M: эмбеддинги подписи и субтитров
    W->>DB: чанки caption/subtitles, status=indexed_fast
    W->>Q: process_full(clip_id)
    Q->>W: process_full
    W->>W: yt-dlp + ffmpeg: аудио 16 кГц, ключевые кадры (временно)
    W->>M: ASR → транскрипт
    W->>M: OCR кадров, векторы кадров (1.0)
    W->>M: эмбеддинги фрагментов (окно 15 с, шаг 7 с)
    W->>DB: сегменты, чанки, векторы, status=ready
    W->>W: удалить аудио и кадры
    W->>DB: описание, теги (языковая модель, 1.0)
    C->>A: GET /clips/{id} (опрос статуса)
```

- **Статусы ролика:** `queued → downloading → transcribing → indexing → enriching → ready`; при сбое — `failed` с причиной (`error`). Повтор — `POST /clips/{id}/retry` (ФТ-2.7).
- **Очереди:** `fast` — первый проход и новые ролики, добавленные по одному (ФТ-2.6); `full` — полная обработка при массовом переносе. Ошибка одного ролика не останавливает очередь.
- **Темы** пересчитываются отдельной периодической задачей при изменении коллекции на 10 % и более (BERTopic по векторам роликов); пользовательские названия тем сохраняются (ФТ-5.4).

## 5. Поиск

1. Backend получает запрос, ML-сервис возвращает вектор `query: <запрос>` (multilingual-e5-large, 1024, решение 14).
2. Два запроса в PostgreSQL по фрагментам коллекции: векторный (`embedding <=> :q`, индекс HNSW, топ-200) и полнотекстовый (`tsv @@ websearch_to_tsquery('russian', :q)` + триграммы `pg_trgm` для опечаток и ошибок ASR, ФТ-3.3).
3. Результаты объединяются методом Reciprocal Rank Fusion; оценка ролика — лучший из его фрагментов (ФТ-3.4).
4. Для каждого ролика возвращается лучший фрагмент как «почему найден»: источник (`speech`, `caption`, `ocr`, `scene`) и цитата до 100 символов (ФТ-3.5).
5. Фильтр по теме — условие `clips.topic_id` (ФТ-3.6). Оценки «подходит / не подходит» сохраняются с позицией в выдаче (ФТ-3.8).

Реранкер не используется: на CPU он тратит ~40 с на запрос при прибавке +0,01 MRR (решение 14).

## 6. Безопасность и приватность

- **Доступ к коллекции.** При создании коллекции клиент получает ключ (32 случайных байта, base64url), в базе хранится только его SHA-256. Ключ передаётся в заголовке `Authorization: Bearer <ключ>`. Перенос на другое устройство — по QR-коду или ссылке с ключом. Telegram-бот привязывается одноразовым кодом со сроком 10 минут (ФТ-7.2). При появлении учётных записей (ФТ-7.5) коллекция получит владельца без изменения остальных таблиц.
- Видео и аудио не хранятся; во внешнюю языковую модель передаются только транскрипт, текст в кадре и подпись ролика, без идентификаторов пользователя (НФТ-8).
- Журнал (ФТ-8.2) пишет методы, коды ответов и время, но не тексты поисковых запросов.
- Удаление всех данных (ФТ-7.3) — каскадное удаление коллекции.

## 7. Модель данных

```mermaid
erDiagram
    COLLECTION ||--o{ CLIP : содержит
    COLLECTION ||--o{ TOPIC : "темы коллекции"
    COLLECTION ||--o{ TAG : "теги коллекции"
    COLLECTION ||--o{ IMPORT_SESSION : "перенос расширением"
    COLLECTION ||--o{ TELEGRAM_LINK : "привязки бота"
    COLLECTION ||--o{ SEARCH_FEEDBACK : оценки
    CLIP ||--o{ TRANSCRIPT_SEGMENT : "фразы речи"
    CLIP ||--o{ FRAME : "ключевые кадры"
    CLIP ||--o{ CHUNK : "фрагменты поиска"
    CLIP ||--o{ PROCESSING_JOB : "задачи обработки"
    CLIP ||--o{ CLIP_TAG : ""
    TAG ||--o{ CLIP_TAG : ""
    TOPIC |o--o{ CLIP : "тема ролика"
    CLIP |o--o{ SEARCH_FEEDBACK : ""

    COLLECTION {
        uuid id PK
        text key_hash UK "SHA-256 ключа доступа"
        timestamptz created_at
        timestamptz onboarding_done_at
    }
    CLIP {
        bigint id PK "clip_id"
        uuid collection_id FK
        text platform "tiktok | vk"
        text external_id "UK с collection_id и platform"
        text url
        text author
        text title
        text caption
        real duration
        timestamptz published_at
        text thumbnail_key "ключ превью в S3"
        text status "queued … ready | failed"
        text error
        int topic_id FK
        text summary "краткое описание (1.0)"
        timestamptz added_at
    }
    TRANSCRIPT_SEGMENT {
        bigint id PK
        bigint clip_id FK
        real start
        real end
        text text
        text source "asr | subtitles"
    }
    FRAME {
        bigint id PK
        bigint clip_id FK
        real t
        text ocr_text
        vector embedding "512, модель картинка-текст (1.0)"
    }
    CHUNK {
        bigint id PK
        bigint clip_id FK
        int ordinal
        text source "caption | speech | ocr | scene"
        real start
        real end
        text text
        vector embedding "1024, multilingual-e5-large"
        tsvector tsv "russian, GIN"
    }
    TAG {
        int id PK
        uuid collection_id FK
        text name "UK с collection_id"
    }
    CLIP_TAG {
        bigint clip_id PK
        int tag_id PK
        text source "auto | user"
    }
    TOPIC {
        int id PK
        uuid collection_id FK
        text name
        text name_source "auto | user"
        text_array keywords
        timestamptz computed_at
    }
    PROCESSING_JOB {
        bigint id PK
        bigint clip_id FK
        text stage "fast | full | enrich"
        text queue
        text status "queued | running | done | failed"
        int attempts
        text error
        timestamptz started_at
        timestamptz finished_at
    }
    IMPORT_SESSION {
        uuid id PK
        uuid collection_id FK
        text platform
        text scope "last200 | year | all"
        int total
        int sent
        text cursor "место остановки"
        timestamptz updated_at
    }
    TELEGRAM_LINK {
        bigint id PK
        uuid collection_id FK
        bigint telegram_user_id UK
        text link_code
        timestamptz code_expires_at
        timestamptz linked_at
    }
    SEARCH_FEEDBACK {
        bigint id PK
        uuid collection_id FK
        text query
        bigint clip_id FK
        int position
        bool relevant
        timestamptz created_at
    }
```

### Соответствие требованиям SRS

| Сущность | Требования | Примечания |
|---|---|---|
| `collection` | ФТ-6.1, ФТ-7.3, НФТ-8, В-1 | Вместо пользователя: учётные записи в бэклоге (ФТ-7.5) |
| `clip` | ФТ-1.1–1.3, ФТ-1.7, ФТ-1.8, ФТ-2.8, ФТ-4.1–4.3, ФТ-7.1 | Уникальность `(collection_id, platform, external_id)` — защита от дубликатов (ФТ-1.2) |
| `transcript_segment` | ФТ-2.2, ФТ-2.3, ФТ-4.4, ФТ-4.6 | Фразы с таймкодами; источник — распознавание или готовые субтитры платформы |
| `frame` | ФТ-2.9, ФТ-2.10, ФТ-3.9 | Версия 1.0; сами кадры не хранятся, только текст и вектор |
| `chunk` | ФТ-2.4, ФТ-3.2–3.5 | Единица поиска; HNSW по `embedding` (`vector_cosine_ops`), GIN по `tsv`, GIN `gin_trgm_ops` по `text` |
| `tag`, `clip_tag` | ФТ-5.1, ФТ-5.2 | Ручные теги (`user`) не удаляются при пересчёте |
| `topic` | ФТ-3.6, ФТ-5.3–5.6 | Пользовательское название сохраняется при пересчёте |
| `processing_job` | ФТ-2.5–2.7, ФТ-8.1 | Состояние обработки не зависит от Celery (ADR-001) |
| `import_session` | ФТ-1.6 | Возобновление переноса с места остановки |
| `telegram_link` | ФТ-1.5, ФТ-7.2 | Одноразовый код привязки |
| `search_feedback` | ФТ-3.8 | Выгрузка для администратора и для оценки качества |

Размер индекса: на ролик в среднем ~9 фрагментов (решение 14), 10 000 роликов → ~90 000 векторов × 4 КБ ≈ 370 МБ плюс индекс HNSW; проверка скорости — MEM-78.

## 8. Открытые вопросы

| Вопрос | Где решается |
|---|---|
| Прокси для скачивания TikTok с российского сервера | MEM-20 (В-3) |
| Хостинг Telegram-бота за рубежом | созвон с заказчиком (В-4) |
| Отличать запросы без ответа («ничего не найдено»): порог или гибрид с BM25 | MEM-41 |
| Модель OCR и «картинка — текст» | MEM-58, MEM-87 |
