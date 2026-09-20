"""ФИНАЛЬНАЯ гипотеза: инъекция train-айдемов, отсутствующих в пуле.

Потолок пула 0.7753 → 22.5% релевантных вообще не в пуле,_boost их не
достанет никогда. Если айтем известен по train-сигналу, но не вышел в пул
BM25/dense — добавляем его в пул принудительно и бустим.

Честно: signal-половина кликов -> буст+инъекция, truth-половина -> истина.
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
from src.config import TOP_K

K = 50
POOL_POS_W = 0.02
t0 = time.time()
tr, q, it = load_raw()
model = load_model()
retriever = BM25Retriever(it, Lemmatizer())
bm_set = set(it['item_id'].values)
item_lookup = {iid: i for i, iid in enumerate(it['item_id'].values)}
log(f'готово {time.time() - t0:.0f}s')

tr_q = set(tr.search_query.apply(norm_text))
q['qn'] = q.search_query.apply(norm_text)
cand_q = q[q.qn.isin(tr_q)].reset_index(drop=True)

rng = np.random.default_rng(42)
tr_c = tr[tr.search_query.isin(set(cand_q.search_query))].copy()
tr_c['_r'] = rng.random(len(tr_c))
signal_df = tr_c[tr_c._r < 0.5]
truth_df = tr_c[tr_c._r >= 0.5]
truth_sets = truth_df.groupby('search_query')['item_id'].apply(set).to_dict()
sig_counts = signal_df.groupby(['search_query', 'item_id']).size()
sig_items = signal_df.groupby('search_query')['item_id'].apply(set).to_dict()
log(f'signal: {len(signal_df)} строк, truth: {len(truth_sets)} запросов')

truth = {}
for _, row in cand_q.iterrows():
    rel = truth_sets.get(row.search_query, set()) & bm_set
    if rel:
        truth[row.query_id] = rel
q_eval = q[q.query_id.isin(truth)].reset_index(drop=True)
log(f'на эвал: {len(q_eval)}')

cands = get_candidates(q_eval, it, retriever, 'cpu')
log(f'кандидаты {time.time() - t0:.0f}s')

cache = {}
for idx, (_, row) in enumerate(q_eval.iterrows()):
    qid = row['query_id']
    qtext = row.search_query
    pool = [iid for iid in cands.get(qid, []) if iid in item_lookup]
    pool_set = set(pool)
    # signal-айтемы, которые НЕ вышли в пул — кандидаты на инъекцию
    sig = sig_items.get(qtext, set()) & bm_set
    inject = [iid for iid in sig if iid not in pool_set]
    # скоры для пула
    ids_all = pool + inject
    feats = [build_features(row, it.iloc[item_lookup[i]]) for i in ids_all]
    X = pd.DataFrame(feats)
    scores = np.asarray(model.predict(X), dtype=np.float64)
    qn = norm_text(qtext)
    cnt = np.array([sig_counts.get((qtext, iid), 0) for iid in ids_all])
    is_inject = np.array([iid in inject for iid in ids_all])
    cache[qid] = dict(ids=ids_all, scores=scores, cnt=cnt, is_inject=is_inject,
                      truth=truth.get(qid, set()), n_pool=len(pool))
    if (idx + 1) % 200 == 0:
        log(f'предпосчёт {idx + 1}/{len(q_eval)}, {time.time() - t0:.0f}s')
log(f'кэш: {len(cache)}')


def evaluate(name, score_fn):
    rs, rs_nosig = [], []
    n_inj_used = 0
    for qid, d in cache.items():
        rel = d['truth']
        if not rel:
            continue
        s = score_fn(d)
        order = np.argsort(-s)
        pred = [d['ids'][i] for i in order[:K]]
        r = len(set(pred) & rel) / len(rel)
        rs.append(r)
        if not (d['cnt'] > 0).any():
            rs_nosig.append(r)
        n_inj_used += len(set(pred) & set(list(np.asarray(d['ids'])[d['is_inject']])))
    rs = np.array(rs)
    log(f'{name:46s} n={len(rs):4d} Rec@{K}={rs.mean():.4f} '
        f'(p50={np.median(rs):.2f}, 0-запросов={(rs == 0).mean():.2f}) '
        f'[без-сигнала n={len(rs_nosig)}: {np.mean(rs_nosig) if rs_nosig else 0:.4f}]')
    return rs.mean()


print('=' * 84)
print(f'СИГНАЛ/TRUTH СПЛИТ, инъекция (n={len(cache)})')
print('=' * 84)
in_pool = [len(set(d['ids']) & d['truth']) / len(d['truth'])
           for d in cache.values() if d['truth']]
in_pool_no_inj = [len(set(np.asarray(d['ids'])[~d['is_inject']]) & d['truth']) / len(d['truth'])
                  for d in cache.values() if d['truth']]
log(f'{"потолок БЕЗ инъекции":46s} Rec@{K}={np.mean(in_pool_no_inj):.4f}')
log(f'{"потолок С инъекцией":46s} Rec@{K}={np.mean(in_pool):.4f}')

evaluate('production (без инъекции, без pool_pos)',
         lambda d: d['scores'] + 5.0 * np.log1p(np.where(d['is_inject'], 0, d['cnt']).astype(float)))
evaluate('production + pool_pos',
         lambda d: (d['scores'] + 5.0 * np.log1p(np.where(d['is_inject'], 0, d['cnt']).astype(float))
                    - POOL_POS_W * np.arange(len(d['ids']))))

print()
print('=== ИНЪЕКЦИЯ train-айдемов в пул ===')
for inj_w in (3.0, 5.0, 8.0):
    evaluate(f'инъекция boost={inj_w}',
             lambda d, x=inj_w: (d['scores']
                                 + x * np.log1p(d['cnt'].astype(float))))
for inj_w in (3.0, 5.0, 8.0):
    evaluate(f'инъекция boost={inj_w} + pool_pos',
             lambda d, x=inj_w: (d['scores'] + x * np.log1p(d['cnt'].astype(float))
                                 - POOL_POS_W * np.arange(len(d['ids']))))
