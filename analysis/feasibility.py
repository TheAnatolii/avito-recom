"""Проверка реализуемости офлайн-эвала на production-корпусе.

Идея: корпус benchmark_items (189212) уже имеет закэшированные BM25-индекс
и dense-эмбеддинги. Если много train-запросов имеют свои кликнутые айтемы
внутри этого корпуса, можно построить псевдо-бенчмарк:
  signal  = половина кликов по запросу (имитирует train)
  truth   = вторая половина       (имитирует разметку бенчмарка)
и измерять Recall@50 честно, без траты попыток.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

tr = pd.read_parquet('dataset/train.parquet')
q = pd.read_parquet('dataset/benchmark_queries.parquet')
it = pd.read_parquet('dataset/benchmark_items.parquet')
bm_set = set(it['item_id'].values)

pairs = tr[['search_query', 'item_id']].drop_duplicates()
inb = pairs['item_id'].isin(bm_set)
print(f'unique (query,item) pairs: {len(pairs)}')
print(f'  item в корпусе benchmark: {int(inb.sum())} ({inb.mean():.3f})')

grp = pairs.assign(inb=inb).groupby('search_query')['inb'].agg(all_in='mean', n='count')
allin = grp[grp['all_in'] == 1.0]
print(f'запросов, у которых ВСЕ айтемы в корпусе: {len(allin)}')
for thr in (1, 2, 3, 5):
    print(f'  из них с >= {thr} айтемов: {int((allin["n"] >= thr).sum())}')

cnt = tr.groupby(['search_query', 'item_id']).size()
print('\nРаспределение count (query,item):')
print(cnt.value_counts().sort_index().head(10).to_string())

from src.utils.text import norm_text
tr_q = set(pairs['search_query'].apply(norm_text))
bm_q = set(q['search_query'].apply(norm_text))
print(f'\nbenchmark запросов: {len(bm_q)}, есть train-сигнал: {len(tr_q & bm_q)}')
