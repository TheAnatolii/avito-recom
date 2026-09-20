"""ФИНАЛЬНАЯ проверка на всех benchmark-запросах с train-сигналом.

Две части:
 1. pool_pos-приоритет (post-hoc бонус за позицию в пуле) — НЕ использует
    train-сигнал, поэтому меряем на полном сигнале как истине: чисто.
 2. буст — требует signal/truth-сплита, иначе утечка (мерили в sim_bench).
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
model = load_model()
retriever = BM25Retriever(it, Lemmatizer())
log(f'готово {time.time() - t0:.0f}s')

bm_set = set(it['item_id'].values)
item_lookup = {iid: i for i, iid in enumerate(it['item_id'].values)}
train_counts = build_train_counts(tr)

# все запросы, у которых есть train-сигнал в корпусе
qn2qid = {}
for _, row in q.iterrows():
    qn2qid[norm_text(row['search_query'])] = row['query_id']

has_signal = set()
truth_full = {}
for (qq, iid), c in train_counts.items():
    if iid in bm_set:
        has_signal.add(qq)
        truth_full.setdefault(qq, set()).add(iid)
q_eval = q[q.search_query.apply(norm_text).isin(has_signal)].reset_index(drop=True)
log(f'запросов с сигналом: {len(q_eval)}')

cands = get_candidates(q_eval, it, retriever, 'cpu')
log(f'кандидаты {time.time() - t0:.0f}s')

cache = {}
for idx, (_, row) in enumerate(q_eval.iterrows()):
    qid = row['query_id']
    cand_ids = [iid for iid in cands.get(qid, []) if iid in item_lookup]
    if not cand_ids:
        continue
    feats = [build_features(row, it.iloc[item_lookup[iid]]) for iid in cand_ids]
    X = pd.DataFrame(feats)
    scores = np.asarray(model.predict(X), dtype=np.float64)
    qn = norm_text(row['search_query'])
    cnt = np.array([train_counts.get((qn, iid), 0) for iid in cand_ids])
    cache[qid] = dict(cand_ids=cand_ids, scores=scores, cnt=cnt,
                      truth=truth_full.get(qn, set()) & bm_set,
                      qn=qn)
    if (idx + 1) % 200 == 0:
        log(f'предпосчёт {idx + 1}/{len(q_eval)}, {time.time() - t0:.0f}s')
log(f'кэш: {len(cache)} запросов, {time.time() - t0:.0f}s')


def evaluate(name, score_fn):
    rs = []
    for qid, d in cache.items():
        rel = d['truth']
        if not rel:
            continue
        s = score_fn(d)
        order = np.argsort(-s)
        pred = [d['cand_ids'][i] for i in order[:K]]
        rs.append(len(set(pred) & rel) / len(rel))
    rs = np.array(rs)
    log(f'{name:44s} n={len(rs):4d} Rec@{K}={rs.mean():.4f} '
        f'(p50={np.median(rs):.2f}, queries=0: {(rs == 0).mean():.2f})')
    return rs.mean()


print('=' * 80)
print(f'ПОЛНЫЙ TRAIN-СИГНАЛ КАК ИСТИНА (n={len(cache)})')
print('=' * 80)
in_pool = [len(set(d['cand_ids']) & d['truth']) / max(1, len(d['truth']))
           for d in cache.values() if d['truth']]
log(f'{"ПОТОЛОК пула":44s}           Rec@{K}={np.mean(in_pool):.4f}')

evaluate('ranker (без буста, без приоритета)', lambda d: d['scores'])
evaluate('production: ranker + 5*log1p(c)',
         lambda d: d['scores'] + 5.0 * np.log1p(d['cnt'].astype(float)))

print()
print('=== pool_pos приоритет: post-hoc, без train-сигнала ===')
for w in (0.01, 0.02, 0.03, 0.05):
    evaluate(f'ranker - {w} * pool_pos',
             lambda d, x=w: d['scores'] - x * np.arange(len(d['cand_ids'])))

print()
print('=== pool_pos + boost ===')
for w in (0.01, 0.02, 0.03):
    evaluate(f'ranker + 5*log1p(c) - {w} * pool_pos',
             lambda d, x=w: d['scores'] + 5.0 * np.log1p(d['cnt'].astype(float))
             - x * np.arange(len(d['cand_ids'])))

print()
print('=== grid: pool_pos x boost ===')
best = (None, -1)
for w in (0.01, 0.02, 0.03):
    for b in (3.0, 5.0, 8.0):
        r = evaluate(f'boost={b} - {w}*pool_pos',
                     lambda d, x=w, y=b: d['scores'] + y * np.log1p(d['cnt'].astype(float))
                     - x * np.arange(len(d['cand_ids'])))
        if r > best[1]:
            best = ((w, b), r)
log(f'ЛУЧШИЙ: pool_pos={best[0][0]}, boost={best[0][1]} -> {best[1]:.4f}')
