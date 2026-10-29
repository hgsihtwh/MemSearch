"""MEM-76: собирает ноутбук Colab для сравнения моделей эмбеддингов.
Паспорт коллекции и контрольные запросы вшиваются в ноутбук, чтобы он воспроизводился с нуля без загрузки файлов."""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..")
clips = [
    dict(
        n=int(r["№"]),
        platform=r["Платформа"],
        url=r["Ссылка"],
        caption=r["Название / подпись"],
        speech=r["Речь"],
    )
    for r in csv.DictReader(
        open(os.path.join(DATA, "collection", "passport.csv"), encoding="utf-8-sig")
    )
]
queries = [
    dict(
        q=r["Запрос"],
        kind=r["Вид запроса"],
        where=r["Где ответ в ролике"],
        author=r["Автор запроса"],
        answers=[int(x) for x in r["Правильные ролики (№ по паспорту)"].split(",") if x.strip()],
    )
    for r in csv.DictReader(
        open(os.path.join(DATA, "queries", "control_queries.csv"), encoding="utf-8-sig")
    )
]

cells = []


def md(s):
    cells.append(dict(cell_type="markdown", metadata={}, source=s.strip("\n")))


def code(s):
    cells.append(
        dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=s.strip("\n"))
    )


md("""
# MEM-76 · Выбор модели эмбеддингов: multilingual-e5 или BGE-M3

1. **Среда выполнения → Сменить среду → GPU (T4)**.
2. **Среда выполнения → Выполнить все** (≈ 30–35 мин: скачивание и распознавание 91 ролика — самое долгое).
3. В конце скачается `embeddings_check_results.zip` — пришли его в чат.

Что делает ноутбук:
1. Скачивает аудио всех роликов коллекции и распознаёт речь faster-whisper large-v3 (выбрана в MEM-16).
2. Режет транскрипт на перекрывающиеся фрагменты; подпись ролика — отдельный фрагмент.
3. Строит векторы фрагментов двумя моделями и ищет по косинусной близости в памяти, без БД. Оценка ролика — лучший из его фрагментов.
4. Пробует реранкер bge-reranker-v2-m3 поверх каждой модели: пересортировывает 50 лучших фрагментов (и меряет его скорость на CPU).
5. Прогоняет контрольные запросы MEM-17 и считает Recall@1/5/10, MRR, скорость и размер вектора. Для сравнения — простой поиск по словам (BM25).

Поиск идёт по всей коллекции из 91 ролика, а не только по тем, к которым есть запросы, — остальные ролики служат «шумом», как у настоящего пользователя.
""")

code("""
!pip -q install faster-whisper yt-dlp sentence-transformers rank_bm25 nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "GPU нет — будет очень медленно"
# faster-whisper (CTranslate2) собран под CUDA 12, а в Colab может стоять CUDA 13 — подгружаем библиотеки CUDA 12 вручную
import ctypes, glob, nvidia.cublas, nvidia.cudnn
for lib_dir in [nvidia.cublas.__path__[0] + '/lib', nvidia.cudnn.__path__[0] + '/lib']:
    for so in sorted(glob.glob(lib_dir + '/*.so*'), key=lambda p: 'Lt' not in p):
        try: ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError: pass
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)
""")

code(
    "# Коллекция (research/collection/passport.csv) и контрольные запросы (research/queries/control_queries.csv)\n"
    "CLIPS = " + json.dumps(clips, ensure_ascii=False, indent=0) + "\n\n"
    "QUERIES = " + json.dumps(queries, ensure_ascii=False, indent=0) + "\n\n"
    "print(len(CLIPS), 'роликов,', len(QUERIES), 'запросов')"
)

code("""
# Скачиваем только аудио (уже скачанные пропускаются — ячейку можно перезапускать)
import os, subprocess
os.makedirs('audio', exist_ok=True)
failed = []
for c in CLIPS:
    n = c['n']
    if os.path.exists(f'audio/{n:02d}.mp3'): continue
    r = subprocess.run(['yt-dlp', '-q', '-f', 'bestaudio/best', '-x', '--audio-format', 'mp3', '-o', f'audio/{n:02d}.%(ext)s', c['url']],
                       capture_output=True, text=True)
    if not os.path.exists(f'audio/{n:02d}.mp3'):
        failed.append(n); print(f'{n:>3} ОШИБКА: {(r.stderr.strip().splitlines() or ["?"])[-1][:120]}')
print('скачано', len(CLIPS) - len(failed), 'из', len(CLIPS), '; без аудио:', failed)
""")

code("""
# Распознавание речи: faster-whisper large-v3
# Аудио читаем через ffmpeg: свежий PyAV в Colab несовместим с faster-whisper.
# Фильтр тишины (VAD) на речи поверх музыки вырезает голос (MEM-16) — порог ослаблен, при пустом результате распознаём без фильтра.
import numpy as np, time, json, torch
from faster_whisper import WhisperModel
def load_audio(path, sr=16000):
    out = subprocess.run(['ffmpeg', '-nostdin', '-i', path, '-f', 's16le', '-ac', '1', '-ar', str(sr), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

asr = WhisperModel('large-v3', device=device, compute_type='float16' if device == 'cuda' else 'int8')
os.makedirs('transcripts', exist_ok=True)
TR = {}
t_all = time.time()
for c in CLIPS:
    n = c['n']; path = f'transcripts/{n:02d}.json'
    if os.path.exists(path): TR[n] = json.load(open(path, encoding='utf-8')); continue
    if not os.path.exists(f'audio/{n:02d}.mp3'): continue
    audio = load_audio(f'audio/{n:02d}.mp3')
    segs, info = asr.transcribe(audio, language='ru', beam_size=5, vad_filter=True, vad_parameters=dict(threshold=0.3))
    segs = list(segs)
    if not segs:
        segs, info = asr.transcribe(audio, language='ru', beam_size=5, vad_filter=False); segs = list(segs)
    TR[n] = dict(clip_id=n, model='faster-whisper large-v3', language=info.language, duration=round(info.duration, 2),
                 segments=[dict(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip()) for s in segs])
    json.dump(TR[n], open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{n:>3} {info.duration:5.0f} с | {" ".join(s.text.strip() for s in segs)[:100]}')
print(f'распознано {len(TR)} роликов за {time.time() - t_all:.0f} с')
# освобождаем память под модели эмбеддингов и реранкер (иначе Colab падает от нехватки памяти)
import gc
del asr; gc.collect(); torch.cuda.empty_cache()
""")

code("""
# Разбиение на фрагменты: окно по времени с перекрытием; подпись ролика — отдельный фрагмент
import re
def clean_caption(s):
    s = re.sub(r'#\\S+', ' ', s or ''); s = re.sub(r'^Clip by @\\S+$', '', s.strip())
    return re.sub(r'\\s+', ' ', s).strip()

def chunks_for(n, window, stride):
    out = []
    cap = clean_caption(next(c['caption'] for c in CLIPS if c['n'] == n))
    if cap: out.append(dict(clip=n, kind='подпись', start=None, end=None, text=cap[:1000]))
    segs = TR.get(n, {}).get('segments', [])
    if not segs: return out
    if window is None:  # весь транскрипт одним фрагментом
        return out + [dict(clip=n, kind='речь', start=segs[0]['start'], end=segs[-1]['end'], text=' '.join(s['text'] for s in segs))]
    t, last = 0.0, segs[-1]['end']
    while True:
        part = [s for s in segs if s['end'] > t and s['start'] < t + window]
        if part and (not out or out[-1]['text'] != ' '.join(s['text'] for s in part)):  # длинные сегменты дают одинаковые окна
            out.append(dict(clip=n, kind='речь', start=part[0]['start'], end=part[-1]['end'], text=' '.join(s['text'] for s in part)))
        if t + window >= last: break
        t += stride
    return out

# варианты разбиения: (название, окно, шаг), секунды
CHUNKINGS = [('окно 15 с / шаг 7 с', 15, 7), ('окно 30 с / шаг 15 с', 30, 15), ('окно 60 с / шаг 30 с', 60, 30), ('весь транскрипт', None, None)]
CHUNKS = {name: [ch for c in CLIPS for ch in chunks_for(c['n'], w, s)] for name, w, s in CHUNKINGS}
for name, ch in CHUNKS.items():
    print(f'{name:22} {len(ch):4} фрагментов, в среднем {np.mean([len(x["text"].split()) for x in ch]):.0f} слов')
""")

code("""
# Метрики: ролик считается найденным, если любой из правильных роликов попал в первые k
ALL = [c['n'] for c in CLIPS]
EVAL = [q for q in QUERIES if q['answers']]
NOANS = [q for q in QUERIES if not q['answers']]

def rank_clips(chunk_scores, chunks):
    best = {}
    for s, ch in zip(chunk_scores, chunks, strict=True):
        if s > best.get(ch['clip'], -1e9): best[ch['clip']] = float(s)
    order = sorted(ALL, key=lambda n: -best.get(n, -1e9))
    return order, best

def metrics(ranks):
    r = np.array([x if x else 10**6 for x in ranks])
    return {'Recall@1': np.mean(r <= 1), 'Recall@5': np.mean(r <= 5), 'Recall@10': np.mean(r <= 10), 'MRR': np.mean(1 / r)}

def first_rank(order, answers):
    return next((i + 1 for i, n in enumerate(order) if n in answers), None)
""")

code("""
# Модели эмбеддингов. У e5 обязательны префиксы «query: » и «passage: », у BGE-M3 префиксов нет.
import gc
from sentence_transformers import SentenceTransformer
MODELS = {
    'multilingual-e5-large': dict(name='intfloat/multilingual-e5-large', q='query: ', p='passage: '),
    'bge-m3':                dict(name='BAAI/bge-m3', q='', p=''),
}
rows, per_query, SIMS = [], [], {}
for mname, cfg in MODELS.items():
    t0 = time.time(); model = SentenceTransformer(cfg['name'], device=device); load = time.time() - t0
    if device == 'cuda': model.half()
    dim = model.get_sentence_embedding_dimension()
    t0 = time.time()
    qv = model.encode([cfg['q'] + q['q'] for q in QUERIES], normalize_embeddings=True, batch_size=64)
    q_ms = (time.time() - t0) / len(QUERIES) * 1000
    for cname, chunks in CHUNKS.items():
        t0 = time.time()
        pv = model.encode([cfg['p'] + ch['text'] for ch in chunks], normalize_embeddings=True, batch_size=32)
        enc = time.time() - t0
        sims = qv @ pv.T; SIMS[(mname, cname)] = sims
        ranks, top_noans, top_ans = [], [], []
        for qi, q in enumerate(QUERIES):
            order, best = rank_clips(sims[qi], chunks)
            if q['answers']:
                rk = first_rank(order, q['answers']); ranks.append(rk); top_ans.append(best[order[0]])
                per_query.append(dict(model=mname, chunking=cname, query=q['q'], author=q['author'], where=q['where'],
                                      answers=q['answers'], rank=rk, top3=order[:3]))
            else:
                top_noans.append(best[order[0]])
        m = metrics(ranks)
        rows.append(dict(модель=mname, разбиение=cname, **{k: round(v, 3) for k, v in m.items()},
                         **{'размер вектора': dim, 'фрагментов': len(chunks), 'векторы фрагментов, с': round(enc, 1),
                            'фрагментов в с': round(len(chunks) / enc), 'запрос, мс': round(q_ms, 1), 'загрузка модели, с': round(load),
                            'ср. сходство топ-1 (есть ответ)': round(float(np.mean(top_ans)), 3),
                            'ср. сходство топ-1 (нет ответа)': round(float(np.mean(top_noans)), 3)}))
        print(f'{mname:22} {cname:22} Recall@10={m["Recall@10"]:.3f} MRR={m["MRR"]:.3f}  ({enc:.1f} с на {len(chunks)} фрагм.)')
    del model, pv; gc.collect(); torch.cuda.empty_cache()
""")

code("""
# Для сравнения: поиск по словам (BM25) — то, что даст полнотекстовый поиск без моделей
from rank_bm25 import BM25Okapi
def tok(s):
    return re.findall(r'\\w+', s.lower())
for cname, chunks in CHUNKS.items():
    bm = BM25Okapi([tok(ch['text']) for ch in chunks])
    ranks = [first_rank(rank_clips(bm.get_scores(tok(q['q'])), chunks)[0], q['answers']) for q in EVAL]
    m = metrics(ranks)
    rows.append(dict(модель='BM25 (по словам)', разбиение=cname, **{k: round(v, 3) for k, v in m.items()}))
    print(f'{"BM25":22} {cname:22} Recall@10={m["Recall@10"]:.3f} MRR={m["MRR"]:.3f}')
""")

code("""
# Реранкер: берём 50 лучших фрагментов векторного поиска и пересортировываем их моделью bge-reranker-v2-m3,
# которая читает запрос и фрагмент вместе. Ролики вне этих 50 идут следом в порядке векторного поиска.
from sentence_transformers import CrossEncoder
K = 50
rr = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device=device)
if device == 'cuda': rr.model.half()
for (mname, cname), sims in SIMS.items():
    chunks = CHUNKS[cname]
    tops = {qi: np.argsort(-sims[qi])[:K] for qi, q in enumerate(QUERIES) if q['answers']}
    pairs = [(QUERIES[qi]['q'], chunks[i]['text']) for qi, top in tops.items() for i in top]
    t0 = time.time(); scores = rr.predict(pairs, batch_size=16, show_progress_bar=False); dt = time.time() - t0
    ranks, pos = [], 0
    for qi, top in tops.items():
        sc = scores[pos:pos + len(top)]; pos += len(top)
        best = {}
        for s, i in zip(sc, top, strict=True):
            n = chunks[i]['clip']
            if s > best.get(n, -1e9): best[n] = float(s)
        vec_order, _ = rank_clips(sims[qi], chunks)
        order = sorted(best, key=lambda n: -best[n]) + [n for n in vec_order if n not in best]
        q = QUERIES[qi]; rk = first_rank(order, q['answers']); ranks.append(rk)
        per_query.append(dict(model=mname + ' + реранкер', chunking=cname, query=q['q'], author=q['author'], where=q['where'],
                              answers=q['answers'], rank=rk, top3=order[:3]))
    m = metrics(ranks)
    rows.append(dict(модель=mname + ' + реранкер', разбиение=cname, **{k: round(v, 3) for k, v in m.items()},
                     **{'реранкер на запрос (GPU), мс': round(dt / len(tops) * 1000)}))
    print(f'{mname + " + реранкер":34} {cname:22} Recall@10={m["Recall@10"]:.3f} MRR={m["MRR"]:.3f}  ({dt / len(tops) * 1000:.0f} мс на запрос)')
del rr; gc.collect(); torch.cuda.empty_cache()

# Сервер, скорее всего, без GPU: меряем реранкер на CPU на 5 запросах (требование — ответ за 2 с)
rr_cpu = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device='cpu')
chunks = CHUNKS['окно 30 с / шаг 15 с']; sims = SIMS[('bge-m3', 'окно 30 с / шаг 15 с')]
t0 = time.time()
for qi in range(5):
    rr_cpu.predict([(QUERIES[qi]['q'], chunks[i]['text']) for i in np.argsort(-sims[qi])[:K]], batch_size=16, show_progress_bar=False)
RERANK_CPU_MS = round((time.time() - t0) / 5 * 1000)
print(f'реранкер на CPU: {RERANK_CPU_MS} мс на запрос ({K} фрагментов)')
rows.append(dict(модель='реранкер на CPU (замер)', разбиение='окно 30 с / шаг 15 с', **{'реранкер на запрос (CPU), мс': RERANK_CPU_MS}))
del rr_cpu; gc.collect()
""")

code("""
# Итоговая таблица и разрезы
import pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
res = pd.DataFrame(rows)
display(res.sort_values(['Recall@10', 'MRR'], ascending=False))

pq = pd.DataFrame(per_query)
pq['hit10'] = pq['rank'].apply(lambda r: r is not None and r <= 10)
print('\\nRecall@10 по тому, где ответ в ролике:')
display(pq.pivot_table(index=['model', 'chunking'], columns='where', values='hit10', aggfunc='mean').round(3))
print('\\nRecall@10 по автору запроса:')
display(pq.pivot_table(index=['model', 'chunking'], columns='author', values='hit10', aggfunc='mean').round(3))
""")

code("""
# Сохраняем результаты и транскрипты (пригодятся для MEM-30 и MEM-79) и скачиваем архив
import shutil, zipfile
res.to_csv('comparison.csv', index=False, encoding='utf-8-sig')
pq.to_csv('per_query.csv', index=False, encoding='utf-8-sig')
shutil.make_archive('embeddings_check_results', 'zip', '.', base_dir='transcripts')
with zipfile.ZipFile('embeddings_check_results.zip', 'a') as z:
    z.write('comparison.csv'); z.write('per_query.csv')
try:
    from google.colab import files; files.download('embeddings_check_results.zip')
except Exception:
    print('Скачай embeddings_check_results.zip из панели файлов слева')
""")

nb = dict(
    cells=cells,
    metadata=dict(
        accelerator="GPU",
        colab=dict(provenance=[], gpuType="T4"),
        kernelspec=dict(name="python3", display_name="Python 3"),
        language_info=dict(name="python"),
    ),
    nbformat=4,
    nbformat_minor=0,
)
json.dump(
    nb,
    open(os.path.join(HERE, "embeddings_check.ipynb"), "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)
print("ok", len(clips), "роликов", len(queries), "запросов")
