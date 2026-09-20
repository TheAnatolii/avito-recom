"""ЧЕСТНАЯ симуляция production на настоящем корпусе BENCHMARK.

Метод: benchmark-запросы, которые есть в train. Их клики делим по строккам:
  signal = половина кликов  -> имитирует train (для буста и фичей)
  truth  = вторая половина  -> имитирует разметку бенчмарка
Кандидаты и ранкер — настоящие, на настоящем корпусе. Так можно
сравнивать стратегии для финальной попытки без отправки на проверку.
"""
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.utils.logging import log
from src.utils.loader import load_raw
from src.retrieval.bm25 import BM25Retriever
from src.utils.text import Lemmatizer, norm_text
from src.ranking.features import build_features
from src.ranking.model import load_model
from src.pipeline.candidates import get_candidates
from src.pipeline.predict import build_train_counts
from src.config import TOP_K

K = 50
t0 = time.time()
tr, q, it = load_raw()
log(f'loaded {time.time() - t0:.0f}s')

model = load_model()
if model is None:
    raise RuntimeError('модель не обучена')
retriever = BM25Retriever(it, Lemmatizer())
log('bm25 ready')

bm_set = set(it['item_id'].values)
item_lookup = {iid: i for i, iid in enumerate(it['item_id'].values)}

# какие benchmark-запросы есть в train
tr_q = set(tr.search_query.apply(norm_text))
q['qn'] = q.search_query.apply(norm_text)
cand_q = q[q.qn.isin(tr_q)].reset_index(drop=True)
log(f'benchmark-запросов в train: {len(cand_q)}')

# делим клики каждого такого запроса пополам
rng = np.random.default_rng(42)
tr_c = tr[tr.search_query.isin(set(cand_q.search_query))].copy()
tr_c['_r'] = rng.random(len(tr_c))
signal_df = tr_c[tr_c._r < 0.5]
truth_df = tr_c[tr_c._r >= 0.5]
truth_sets = truth_df.groupby('search_query')['item_id'].apply(set).to_dict()
signal_counts_raw = signal_df.groupby(['search_query', 'item_id']).size()
log(f'signal строк: {len(signal_df)}, truth запросов: {len(truth_sets)}')

# truth с привязкой к корпусу
truth = {}
for _, row in cand_q.iterrows():
    rel = truth_sets.get(row.search_query, set()) & bm_set
    if rel:
        truth[row.query_id] = rel
log(f'запросов с truth в корпусе: {len(truth)}')

# кандидаты только для этих запросов
q_eval = q[q.query_id.isin(truth)].reset_index(drop=True)
log(f'запросов на эвал: {len(q_eval)}')
cands = get_candidates(q_eval, it, retriever, 'cpu')
log(f'кандидаты готовы, {time.time() - t0:.0f}s')

# signal counts в формате predict.build_train_counts
signal_counts = {(norm_text(qq), i): int(c) for (qq, i), c in signal_counts_raw.items()}

# предпосчитаем фичи и скоры ОДИН раз
cache = {}
for idx, (_, row) in enumerate(q_eval.iterrows()):
    qid = row.query_id
    cand_ids = [iid for iid in cands.get(qid, []) if iid in item_lookup]
    if not cand_ids:
        continue
    feats = [build_features(row, it.iloc[item_lookup[iid]]) for iid in cand_ids]
    X = pd.DataFrame(feats)
    scores = np.asarray(model.predict(X), dtype=np.float64)
    qn = norm_text(row.search_query)
    cnt = np.array([signal_counts.get((qn, iid), 0) for iid in cand_ids])
    # позиция в dense-источнике (нужно для бонуса за dense-ранг)
    cache[qid] = dict(cand_ids=cand_ids, scores=scores, cnt=cnt)
    if (idx + 1) % 100 == 0:
        log(f'предпосчёт {idx + 1}/{len(q_eval)}, {time.time() - t0:.0f}s')
log(f'предпосчёт завершён: {len(cache)} запросов, {time.time() - t0:.0f}s')


def evaluate(name, score_fn, require_pool=True):
    rs = []
    for qid, d in cache.items():
        rel = truth.get(qid)
        if not rel:
            continue
        s = score_fn(d)
        order = np.argsort(-s)
        pred = [d['cand_ids'][i] for i in order[:K]]
        rs.append(len(set(pred) & rel) / len(rel))
    rs = np.array(rs)
    log(f'{name:42s} n={len(rs):4d} Recall@{K} = {rs.mean():.4f}  '
        f'(p25={np.percentile(rs, 25):.2f}, p50={np.percentile(rs, 50):.2f})')
    return rs.mean()


def boost_vec(d, w_by_count):
    return np.array([w_by_count.get(min(c, 6), 0.0) * np.log1p(c) if c > 0 else 0.0
                     for c in d['cnt']])


print('=' * 78)
print(f'СИМУЛЯЦИЯ НА КОРПУСЕ BENCHMARK (n={len(cache)})')
print('=' * 78)

# 0.天花板: всё что в пуле
in_pool = []
for qid, d in cache.items():
    rel = truth.get(qid)
    if not rel:
        continue
    in_pool.append(len(set(d['cand_ids']) & rel) / len(rel))
log(f'{"ПОТОЛОК (всё в пуле)":42s}          Recall@{K} = {np.mean(in_pool):.4f}')

# 1. бейзлайн: как в production, boost = 5.0 * log1p(count)
evaluate('ranker + 5.0*log1p(c) [production]',
         lambda d: d['scores'] + 5.0 * np.log1p(d['cnt'].astype(float)))

# 2. без буста
evaluate('ranker без буста', lambda d: d['scores'])

# 3. адаптивный буст по точности сигнала
#    count=1: 7.5%, count=2: 30.6%, count=3-5: 64.1%, count=6+: 98.7%
for name, wmap in (
    ('адаптивный 0.5/2/5/5', {1: 0.5, 2: 2.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0}),
    ('count=1 не бустим', {1: 0.0, 2: 2.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0}),
    ('только count>=2', {1: 0.0, 2: 5.0, 3: 5.0, 4: 5.0, 5: 5.0, 6: 5.0}),
    ('адаптивный 0.5/3/8/8', {1: 0.5, 2: 3.0, 3: 8.0, 4: 8.0, 5: 8.0, 6: 8.0}),
):
    evaluate(name, lambda d, wm=wmap: d['scores'] + boost_vec(d, wm))

# 4. влияние веса буста
for w in (2.0, 3.0, 8.0, 12.0):
    evaluate(f'буст {w} * log1p(c)',
             lambda d, x=w: d['scores'] + x * np.log1p(d['cnt'].astype(float)))

# 5._dense-ранг и локальность
loc_cache = {}
src_rank_cache = {}
for qid, d in cache.items():
    row = q_eval[q_eval.query_id == qid].iloc[0]
    qloc = row.search_location_id
    loc_cache[qid] = np.array([it.iloc[item_lookup[i]].item_location_id == qloc
                               for i in d['cand_ids']])
    # позиция в итоговом пуле (dense вперёд) — заменитель dense_rank
    src_rank_cache[qid] = np.arange(len(d['cand_ids']))


def evaluate_qid(name, score_fn):
    rs = []
    for qid, d in cache.items():
        rel = truth.get(qid)
        if not rel:
            continue
        s = score_fn(d, qid)
        order = np.argsort(-s)
        pred = [d['cand_ids'][i] for i in order[:K]]
        rs.append(len(set(pred) & rel) / len(rel))
    rs = np.array(rs)
    log(f'{name:42s} n={len(rs):4d} Recall@{K} = {rs.mean():.4f}  '
        f'(p25={np.percentile(rs, 25):.2f}, p50={np.percentile(rs, 50):.2f})')
    return rs.mean()


for bl in (0.25, 0.5, 1.0, 2.0):
    evaluate_qid(f'ranker + boost + {bl} * is_local',
                 lambda d, qid, x=bl: d['scores'] + 5.0 * np.log1p(d['cnt'].astype(float))
                 + x * loc_cache[qid].astype(float))

print()
print('=== плот-приоритет: порядок пула как сигнал ===')
for w in (0.01, 0.02, 0.05, 0.1):
    evaluate_qid(f'ranker + boost - {w} * pool_pos',
                 lambda d, qid, x=w: d['scores'] + 5.0 * np.log1p(d['cnt'].astype(float))
                 - x * src_rank_cache[qid])

print()
print('=== только порядок пула + boost (без ранкера) ===')
for w in (0.0, 5.0):
    evaluate_qid(f'pool-order + boost({w})',
                 lambda d, qid, x=w: -src_rank_cache[qid] + x * np.log1p(d['cnt'].astype(float)))
