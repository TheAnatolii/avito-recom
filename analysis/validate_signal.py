"""Валидация train-сигнала: расщепление по СТРОКАМ, не по запросам.

Если айтем выбран одной половиной кликов, с какой вероятностью
он присутствует в релевантных другой половины для того же запроса?
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

tr = pd.read_parquet('dataset/train.parquet')
rng = np.random.default_rng(42)

# Расщепление по строкам (кликам), запросы общие
mask = rng.random(len(tr)) < 0.5
tr_a = tr[mask]
tr_b = tr[~mask]

rel_a = tr_a.groupby('search_query')['item_id'].apply(set).to_dict()
rel_b = tr_b.groupby('search_query')['item_id'].apply(set).to_dict()

common = set(rel_a) & set(rel_b)
print(f'Запросов в A: {len(rel_a)}, в B: {len(rel_b)}, общих: {len(common)}')

n_items = 0
n_hit = 0
per_count = {}

# Для каждого общего запроса: count в A, присутствие в B
cnt_a = tr_a.groupby(['search_query', 'item_id']).size()
for (qtext, iid), c in cnt_a.items():
    if qtext not in rel_b:
        continue
    hit = int(iid in rel_b[qtext])
    n_items += 1
    n_hit += hit
    bucket = 1 if c == 1 else (2 if c == 2 else (3 if c <= 5 else 6))
    if bucket not in per_count:
        per_count[bucket] = [0, 0]
    per_count[bucket][0] += hit
    per_count[bucket][1] += 1

print(f'Проверено (query,item) пар: {n_items}')
print(f'Присутствуют в B: {n_hit} ({n_hit / max(1, n_items):.4f})')
print()
print('Точность по count в A:')
for b in sorted(per_count):
    h, t = per_count[b]
    label = {1: 'count=1', 2: 'count=2', 3: 'count=3-5', 6: 'count=6+'}[b]
    print(f'  {label:10s}: {h}/{t} = {h / max(1, t):.4f}')
