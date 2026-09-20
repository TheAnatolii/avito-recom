"""Сравнение стратегий ранжирования на офлайн-эвале (val-сплит).

Все стратегии работают на ОДНОМ и том же предвычисленном пуле кандидатов
из build_eval.py, поэтому сравнение честное и быстрое.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.utils.cache import load_cached
from src.utils.logging import log

K = 50
data = load_cached('val_candidates.pkl')
log(f'загружено {len(data)} запросов')


def recall(pred_ids, rel):
    rel = set(rel)
    if not rel:
        return None
    return len(set(pred_ids[:K]) & rel) / len(rel)


def evaluate(name, select_fn):
    """select_fn(d) -> list[item_id] длиной K."""
    rs, rs_local = [], []
    for qid, d in data.items():
        rel = set(d['relevant_ids'])
        if not rel:
            continue
        pred = select_fn(d)
        r = recall(pred, rel)
        if r is None:
            continue
        rs.append(r)
        # локальная разбивка: есть ли хотя бы один релевантный в локации запроса
        rs_local.append(r)
    rs = np.array(rs)
    log(f'{name:32s} n={len(rs):5d} Recall@{K} = {rs.mean():.4f}  '
        f'(p25={np.percentile(rs, 25):.2f}, p50={np.percentile(rs, 50):.2f}, '
        f'p75={np.percentile(rs, 75):.2f})')
    return rs.mean()


def take_src(d, src_name):
    cand = d['cand_ids']
    mask = d['src'] == src_name
    return [cand[i] for i in np.where(mask)[0]] + [
        cand[i] for i in np.where(~mask)[0]]


print('=' * 78)
print('A. ПОТОЛОК: релевантные ВООБЩЕ есть в пуле?')
print('=' * 78)
in_pool = []
n_rel_total = 0
n_pool_total = 0
for qid, d in data.items():
    rel = set(d['relevant_ids'])
    if not rel:
        continue
    pool = set(d['cand_ids'])
    in_pool.append(len(rel & pool) / len(rel))
    n_rel_total += len(rel)
    n_pool_total += len(rel & pool)
in_pool = np.array(in_pool)
log(f'запросов с релевантными: {len(in_pool)}')
log(f'ПОТОЛОК Recall@50 (всё что в пуле): {in_pool.mean():.4f}')
log(f'агрегатно: {n_pool_total}/{n_rel_total} = {n_pool_total / n_rel_total:.4f}')

print()
print('=' * 78)
print('B. ИСТОЧНИКИ КАНДИДАТОВ по отдельности')
print('=' * 78)
evaluate('dense@300 only', lambda d: take_src(d, 'dense'))
evaluate('bm25@300 only', lambda d: take_src(d, 'bm25'))
evaluate('local@150 only', lambda d: take_src(d, 'local'))

print()
print('=' * 78)
print('C. РАНКЕР: бейзлайн (текущая логика)')
print('=' * 78)


def baseline(d):
    order = np.argsort(-d['scores'])
    return [d['cand_ids'][i] for i in order]


evaluate('ranker (baseline)', baseline)

print()
print('=' * 78)
print('D. ЛОКАЛЬНЫЙ ПРИОРИТЕТ: ранкер +者优先 local')
print('=' * 78)


def local_first(d, bonus):
    s = d['scores'].copy() + bonus * d['is_local'].astype(float)
    order = np.argsort(-s)
    return [d['cand_ids'][i] for i in order]


for bonus in (0.1, 0.25, 0.5, 1.0, 2.0):
    evaluate(f'ranker + {bonus} * is_local', lambda d, b=bonus: local_first(d, b))

print()
print('=' * 78)
print('E. ПОЛОЖЕНИЕ В РЕЙТИНГЕ ИСТОЧНИКА (rank_in_src)')
print('=' * 78)


def src_rank_bonus(d, w):
    s = d['scores'].copy() - w * d['rank_in_src'].astype(float)
    order = np.argsort(-s)
    return [d['cand_ids'][i] for i in order]


for w in (0.01, 0.05, 0.1):
    evaluate(f'ranker - {w} * rank_in_src', lambda d, x=w: src_rank_bonus(d, x))

print()
print('=' * 78)
print('F. КОМБИНАЦИЯ: local + rank_in_src')
print('=' * 78)


def combo(d, bl, wr):
    s = (d['scores'].copy()
         + bl * d['is_local'].astype(float)
         - wr * d['rank_in_src'].astype(float))
    order = np.argsort(-s)
    return [d['cand_ids'][i] for i in order]


for bl in (0.25, 0.5, 1.0):
    for wr in (0.02, 0.05):
        evaluate(f'ranker + {bl}*local - {wr}*rank', lambda d, x=bl, y=wr: combo(d, x, y))
