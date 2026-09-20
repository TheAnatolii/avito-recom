"""Устойчивость pool_pos-приоритета на полном наборе (n=364).

sim_bench/inject_test использовали signal/truth-сплит (241 запрос).
Здесь — все 364 запроса с train-сигналом, несколько сидов разделения,
плюс проверка на подгруппе без сигнала (главный риск: pool_pos
опирается только на порядок dense, а он может быть хуже ранкера).
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
t0 = time.time()
tr, q, it = load_raw()
model = load_model()
retriever = BM25Retriever(it, Lemmatizer())
bm_set = set(it['item_id'].values)
item_lookup = {iid: i for i, iid in enumerate(it['item_id'].values)}

tr_q = set(tr.search_query.apply(norm_text))
q['qn'] = q.search_query.apply(norm_text)
cand_q = q[q.qn.isin(tr_q)].reset_index(drop=True)
log(f'запросов с сигналом: {len(cand_q)}')

cands = get_candidates(cand_q, it, retriever, 'cpu')
log(f'кандидаты {time.time() - t0:.0f}s')

# Предпосчёт скоров ранкера для каждого запроса (не зависит от сида)
base = {}
for idx, (_, row) in enumerate(cand_q.iterrows()):
    qid = row['query_id']
    pool = [iid for iid in cands.get(qid, []) if iid in item_lookup]
    if not pool:
        continue
    feats = [build_features(row, it.iloc[item_lookup[i]]) for i in pool]
    scores = np.asarray(model.predict(pd.DataFrame(feats)), dtype=np.float64)
    base[qid] = (pool, scores)
log(f'предпосчёт: {len(base)} запросов, {time.time() - t0:.0f}s')

tr_c = tr[tr.search_query.isin(set(cand_q.search_query))].copy()


def run_split(seed):
    rng = np.random.default_rng(seed)
    r = rng.random(len(tr_c))
    sig = tr_c[r < 0.5]
    tru = tr_c[r >= 0.5]
    sig_cnt = sig.groupby(['search_query', 'item_id']).size()
    tru_sets = tru.groupby('search_query')['item_id'].apply(set).to_dict()
    return sig_cnt, tru_sets


def evaluate(name, score_fn, sig_cnt, tru_sets):
    rs = []
    for _, row in cand_q.iterrows():
        qid = row['query_id']
        if qid not in base:
            continue
        pool, scores = base[qid]
        rel = tru_sets.get(row.search_query, set()) & bm_set
        if not rel:
            continue
        qn = norm_text(row.search_query)
        cnt = np.array([sig_cnt.get((row.search_query, i), 0) for i in pool])
        s = score_fn(scores, cnt)
        order = np.argsort(-s)
        pred = [pool[i] for i in order[:K]]
        rs.append(len(set(pred) & rel) / len(rel))
    return float(np.mean(rs)), len(rs)


print('=' * 80)
print('УСТОЙЧИВОСТЬ pool_pos (3 сида, n≈360)')
print('=' * 80)
for seed in (42, 7, 2024):
    sig_cnt, tru_sets = run_split(seed)
    prod = evaluate('prod', lambda s, c: s + 5.0 * np.log1p(c.astype(float)), sig_cnt, tru_sets)
    pp1 = evaluate('pp0.01', lambda s, c: s - 0.01 * np.arange(len(s)), sig_cnt, tru_sets)
    pp2 = evaluate('pp0.02', lambda s, c: s - 0.02 * np.arange(len(s)), sig_cnt, tru_sets)
    both = evaluate('both', lambda s, c: s + 5.0 * np.log1p(c.astype(float)) - 0.02 * np.arange(len(s)),
                    sig_cnt, tru_sets)
    log(f'seed={seed}: prod={prod[0]:.4f}  pp0.01={pp1[0]:.4f}  '
        f'pp0.02={pp2[0]:.4f}  boost+pp0.02={both[0]:.4f}  (n={prod[1]})')

print()
print('=== ВЛИЯНИЕ pool_pos НА ЗАПРОСЫ БЕЗ СИГНАЛА (чистый эффект) ===')
# Запросы, у которых в signal-половине не было кликов
sig_cnt, tru_sets = run_split(42)
rs_a, rs_b = [], []
for _, row in cand_q.iterrows():
    qid = row['query_id']
    if qid not in base:
        continue
    pool, scores = base[qid]
    rel = tru_sets.get(row.search_query, set()) & bm_set
    if not rel:
        continue
    cnt = np.array([sig_cnt.get((row.search_query, i), 0) for i in pool])
    if (cnt > 0).any():
        continue
    o1 = np.argsort(-scores)
    o2 = np.argsort(-(scores - 0.02 * np.arange(len(scores))))
    p1 = [pool[i] for i in o1[:K]]
    p2 = [pool[i] for i in o2[:K]]
    rs_a.append(len(set(p1) & rel) / len(rel))
    rs_b.append(len(set(p2) & rel) / len(rel))
log(f'без сигнала: n={len(rs_a)}, ранкер={np.mean(rs_a):.4f}, '
    f'ранкер+pool_pos={np.mean(rs_b):.4f}, дельта={np.mean(rs_b) - np.mean(rs_a):+.4f}')

print()
print('=== pool_pos: порядок vs ранкер, где они расходятся ===')
agree = disagree = 0
gain_rank = gain_pos = 0
for _, row in cand_q.iterrows():
    qid = row['query_id']
    if qid not in base:
        continue
    pool, scores = base[qid]
    rel = tru_sets.get(row.search_query, set()) & bm_set
    if not rel:
        continue
    o1 = np.argsort(-scores)[:K]
    o2 = np.argsort(-(scores - 0.02 * np.arange(len(scores))))[:K]
    s1, s2 = set(pool[i] for i in o1), set(pool[i] for i in o2)
    if s1 == s2:
        agree += 1
    else:
        disagree += 1
        gain_rank += len(s1 & rel)
        gain_pos += len(s2 & rel)
log(f'одинаковый топ-50: {agree}, разный: {disagree}')
log(f'среди расходящихся: ранкер нашёл релевантных {gain_rank}, '
    f'pool_pos нашёл {gain_pos}')
