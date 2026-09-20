"""Глубокий анализ: где релевантные айтемы в ранжировании и что их отличает.

Работает на предвычисленном пуле из build_eval.py.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.utils.cache import load_cached
from src.utils.logging import log

K = 50
data = load_cached('val_candidates.pkl')
log(f'запружено {len(data)} запросов')

rows = []
for qid, d in data.items():
    rel = set(d['relevant_ids'])
    if not rel:
        continue
    cand = d['cand_ids']
    scores = d['scores']
    n = len(cand)
    order = np.argsort(-scores)
    # позиция каждого кандидата в ранжире ранкера
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    # позиция в dense-источнике
    dense_rank = np.where(d['src'] == 'dense', d['rank_in_src'], 10 ** 6)
    for i, iid in enumerate(cand):
        rows.append(dict(
            qid=qid,
            is_rel=int(iid in rel),
            ranker_rank=rank[i],
            src=d['src'][i],
            rank_in_src=d['rank_in_src'][i],
            is_local=bool(d['is_local'][i]),
            score=scores[i],
            n_rel=len(rel),
        ))

df = pd.DataFrame(rows)
log(f'всего пар: {len(df)}, релевантных: {df.is_rel.sum()}')

print('=' * 78)
print('1. ГДЕ РЕЛЕВАНТНЫЕ: позиция в dense-ранжире')
print('=' * 78)
rel_d = df[df.is_rel == 1]
print('распределение dense-rank релевантных:')
q = rel_d[rel_d.src == 'dense'].rank_in_src
print(q.describe().to_string())
print(f'доля релевантных, найденных dense: {(rel_d.src == "dense").mean():.3f}')
print(f'  из них в top-50 dense: {(q < 50).mean():.3f}')
print(f'  из них в top-100 dense: {(q < 100).mean():.3f}')
print(f'  из них в top-300 dense: {(q < 300).mean():.3f}')

print()
print('=' * 78)
print('2. ЛОКАЛЬНОСТЬ релевантных vs нерелевантных (в пуле)')
print('=' * 78)
print(f'P(is_local | rel)     = {rel_d.is_local.mean():.3f}')
print(f'P(is_local | non-rel) = {df[df.is_rel == 0].is_local.mean():.3f}')

print()
print('=' * 78)
print('3. РАНКЕР: куда кладёт релевантные')
print('=' * 78)
print('распределение ranker_rank релевантных:')
print(rel_d.ranker_rank.describe().to_string())
print(f'доля релевантных в top-50 ранкера: {(rel_d.ranker_rank < 50).mean():.3f}')

print()
print('=' * 78)
print('4. ИДЕАЛЬНЫЙ Recall@50 для разных правил отбора')
print('=' * 78)


def recall_of(pred_rank_mask_fn, name):
    tot = []
    for qid, d in data.items():
        rel = set(d['relevant_ids'])
        if not rel:
            continue
        cand = d['cand_ids']
        n = len(cand)
        order = np.argsort(-d['scores'])
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        dense_rank = np.where(d['src'] == 'dense', d['rank_in_src'], 10 ** 6)
        bm_rank = np.where(d['src'] == 'bm25', d['rank_in_src'], 10 ** 6)
        m = pred_rank_mask_fn(rank, dense_rank, bm_rank, d)
        sel = [cand[i] for i in np.where(m)[0][:K]]
        tot.append(len(set(sel) & rel) / len(rel))
    log(f'{name:44s} Recall@50 = {np.mean(tot):.4f}')
    return np.mean(tot)


recall_of(lambda r, dr, br, d: r < K, 'ранкер top-50')
recall_of(lambda r, dr, br, d: dr < K, 'dense top-50')
recall_of(lambda r, dr, br, d: (dr < K) | (br < 5), 'dense top-50 | bm25 top-5')
recall_of(lambda r, dr, br, d: dr < 100, 'dense top-100')
recall_of(lambda r, dr, br, d: (dr < 40) | (br < 10), 'dense top-40 | bm25 top-10')

print()
print('=' * 78)
print('5. ОРТОГОНАЛЬНОСТЬ dense и bm25 (на релевантных)')
print('=' * 78)
found_dense = set()
found_bm25 = set()
found_local = set()
for qid, d in data.items():
    rel = set(d['relevant_ids'])
    if not rel:
        continue
    for i, iid in enumerate(d['cand_ids']):
        if iid in rel:
            if d['src'][i] == 'dense':
                found_dense.add((qid, iid))
            if d['src'][i] == 'bm25':
                found_bm25.add((qid, iid))
            if d['src'][i] == 'local':
                found_local.add((qid, iid))
all_rel = set()
for qid, d in data.items():
    for iid in set(d['relevant_ids']) & set(d['cand_ids']):
        all_rel.add((qid, iid))
log(f'всего релевантных в пуле: {len(all_rel)}')
log(f'  в dense:  {len(found_dense)} ({len(found_dense) / len(all_rel):.3f})')
log(f'  в bm25:   {len(found_bm25)} ({len(found_bm25) / len(all_rel):.3f})')
log(f'  в local:  {len(found_local)} ({len(found_local) / len(all_rel):.3f})')
log(f'  ТОЛЬКО bm25 (не dense): {len(found_bm25 - found_dense)}')
log(f'  ТОЛЬКО local: {len(found_local - found_dense - found_bm25)}')

print()
print('=' * 78)
print('6. ЗАВИСИМОСТЬ ОТ ЧИСЛА РЕЛЕВАНТНЫХ')
print('=' * 78)
per_q = []
for qid, d in data.items():
    rel = set(d['relevant_ids'])
    if not rel:
        continue
    cand = d['cand_ids']
    order = np.argsort(-d['scores'])
    rank = np.empty(len(cand), dtype=int)
    rank[order] = np.arange(len(cand))
    dense_rank = np.where(d['src'] == 'dense', d['rank_in_src'], 10 ** 6)
    sel_r = set(cand[i] for i in np.where(rank < K)[0])
    sel_d = set(cand[i] for i in np.where(dense_rank < K)[0])
    per_q.append(dict(n_rel=len(rel), rec_ranker=len(sel_r & rel) / len(rel),
                      rec_dense=len(sel_d & rel) / len(rel)))
pq = pd.DataFrame(per_q)
print(pq.groupby(pd.cut(pq.n_rel, [0, 1, 2, 5, 10, 1000])).agg(
    n=('n_rel', 'size'), rec_ranker=('rec_ranker', 'mean'),
    rec_dense=('rec_dense', 'mean')).to_string())
