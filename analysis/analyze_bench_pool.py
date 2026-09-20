"""Реальный потолок пула на корпусе BENCHMARK (настоящий корпус!).

Val-корпус оказался нерепрезентативным (BM25 standalone recall 0.0007
против 0.531 на benchmark), поэтому решения принимаем только по данным
настоящего корпуса. Используем train-сигнал как (шумную) разметку:
для benchmark-запросов, которые есть в train, известны кликнутые айтемы.

Измеряем: сколько известных релевантных айтемов попадает в пул
при разных стратегиях кандидатогенерации. Потолок пула = верхняя граница
Recall@50: чего нет в пуле, то не спасёт никакое ранжирование.
"""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
import pandas as pd
from src.utils.logging import log
from src.utils.loader import load_raw
from src.retrieval.bm25 import BM25Retriever
from src.utils.text import Lemmatizer, norm_text
from src.retrieval import dense as dense_mod
from src.config import CAND_BM25, CAND_DENSE

t0 = time.time()
tr, q, it = load_raw()
log(f'loaded {time.time() - t0:.0f}s')

lem = Lemmatizer()
retriever = BM25Retriever(it, lem)  # кэш под benchmark-корпус — корректный
log('bm25 ready')

bm_item_set = set(it['item_id'].values)
loc_codes = it['item_location_id'].values
loc_to_positions = {}
for pos, lc in enumerate(loc_codes):
    loc_to_positions.setdefault(lc, []).append(pos)
log(f'локаций: {len(loc_to_positions)}')

# train-сигнал для benchmark-запросов
counts = tr.groupby(['search_query', 'item_id']).size()
train_signal = {}
for (qtext, iid), c in counts.items():
    if iid in bm_item_set:
        train_signal.setdefault(qtext, {})[iid] = int(c)

# только запросы с надёжным сигналом (count>=2) — меньше шума
cand_queries = []
for _, row in q.iterrows():
    sig = train_signal.get(row['search_query'])
    if sig and max(sig.values()) >= 2:
        cand_queries.append(row)
log(f'benchmark-запросов с train-сигналом count>=2: {len(cand_queries)}')
# ограничим для скорости
rng = np.random.default_rng(0)
if len(cand_queries) > 700:
    idx = rng.choice(len(cand_queries), 700, replace=False)
    cand_queries = [cand_queries[i] for i in idx]
log(f'подвыборка: {len(cand_queries)}')

from sentence_transformers import SentenceTransformer
from src.config import MODEL_DIR, MAX_SEQ
from src.utils.device import get_device
st = SentenceTransformer(MODEL_DIR, device=get_device('cpu'))
st.max_seq_length = MAX_SEQ
item_emb = dense_mod.encode_items(st, it)

q_texts = [dense_mod.query_text(r) for r in cand_queries]
q_emb = np.asarray(st.encode(q_texts, batch_size=128, show_progress_bar=False,
                             normalize_embeddings=True), dtype=np.float32)
order = dense_mod.topk_cosine(q_emb, item_emb, k=max(CAND_DENSE, 300))
ids = it['item_id'].values
log(f'dense готов, {time.time() - t0:.0f}s')

weights = {'title': 3.0, 'desc': 1.0, 'params': 2.0}
stats = {k: [] for k in ('dense300', 'bm25_300', 'union500', 'local150',
                         'union_loc', 'loc_in_union', 'loc_total')}
n_no_local = 0
for i, row in enumerate(cand_queries):
    sig = train_signal[row['search_query']]
    # надёжные = count>=2; считаем по ним
    rel = {iid for iid, c in sig.items() if c >= 2}
    if not rel:
        continue
    qloc = row['search_location_id']

    def retrieve_local(topn):
        qt = lem.doc_tokens(row['search_query'])
        if not qt or pd.isna(qloc) or qloc not in loc_to_positions:
            return []
        pos_list = np.asarray(loc_to_positions[qloc])
        mask = np.zeros(len(ids), dtype=bool)
        mask[pos_list] = True
        combined = None
        for field, w in weights.items():
            o, sc = retriever.indexes[field].score(qt, topn=None)
            if len(o) == 0:
                continue
            vec = np.zeros(len(ids), dtype=np.float32)
            vec[o] = sc * w
            combined = vec if combined is None else combined + vec
        if combined is None:
            return []
        combined = np.where(mask, combined, -1e9)
        o = np.argsort(combined)[::-1][:topn]
        return [ids[j] for j in o]

    dense_ids = [ids[j] for j in order[i]]
    bm_ids = retriever.retrieve(row['search_query'], topn=300)
    local_ids = retrieve_local(150)

    stats['dense300'].append(len(set(dense_ids[:300]) & rel) / len(rel))
    stats['bm25_300'].append(len(set(bm_ids) & rel) / len(rel))
    seen = set(dense_ids)
    union = list(dense_ids) + [x for x in bm_ids if x not in seen]
    stats['union500'].append(len(set(union[:500]) & rel) / len(rel))
    stats['local150'].append(len(set(local_ids) & rel) / len(rel))
    un_loc = set(union) | set(local_ids)
    stats['union_loc'].append(len(un_loc & rel) / len(rel))
    stats['loc_in_union'].append(len(set(local_ids) & rel & set(union)) / max(1, len(rel)))
    stats['loc_total'].append(len(rel & set(local_ids)) / len(rel))
    if not local_ids:
        n_no_local += 1

    if (i + 1) % 200 == 0:
        log(f'обработано {i + 1}/{len(cand_queries)}, {time.time() - t0:.0f}s')

print('=' * 78)
print('ПОТОЛОК ПУЛА (доля релевантных, попавших в пул), корпус BENCHMARK')
print('=' * 78)
for k, v in stats.items():
    if v:
        log(f'{k:14s} n={len(v):4d}  потолок = {np.mean(v):.4f}  '
            f'(p50={np.median(v):.2f})')
log(f'запросов без локальных кандидатов: {n_no_local}/{len(cand_queries)}')
