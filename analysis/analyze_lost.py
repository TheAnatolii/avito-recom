"""Проверка: были ли потерянные train-релевантные айтемы в кандидатском пуле?"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.utils.text import norm_text

tr = pd.read_parquet('dataset/train.parquet')
q = pd.read_parquet('dataset/benchmark_queries.parquet')
it = pd.read_parquet('dataset/benchmark_items.parquet')
answer = pd.read_csv('answer.csv')

bm_item_set = set(it['item_id'].values)
counts = tr.groupby(['search_query', 'item_id']).size()
train_counts = {(norm_text(qq), i): int(c) for (qq, i), c in counts.items()}

ans_by_qid = dict(zip(answer['query_id'], answer['answer'].str.split()))

# Считаем сколько query_id имеют потери и могли ли они быть в пуле
n_lost = 0
n_lost_queries = 0
lost_items = []

for _, row in q.iterrows():
    qid = row['query_id']
    qn = norm_text(row['search_query'])
    rel = {i for (qq, i) in train_counts if qq == qn}
    rel_in_corpus = rel & bm_item_set
    if not rel_in_corpus:
        continue
    ans = set(ans_by_qid.get(qid, []))
    lost = rel_in_corpus - ans
    if lost:
        n_lost_queries += 1
        n_lost += len(lost)
        for iid in lost:
            c = train_counts.get((qn, iid), 0)
            lost_items.append({'qid': qid, 'query': row['search_query'],
                               'item_id': iid, 'train_count': c})

ld = pd.DataFrame(lost_items)
print('=== ПОТЕРИ: РАСПРЕДЕЛЕНИЕ train_count ===')
print(ld.train_count.describe().to_string())
print()
print('Гистограмма train_count:')
print(ld.train_count.value_counts().sort_index().head(20).to_string())
print()
print(f'Всего потерь: {len(ld)}, запросов: {n_lost_queries}')
print(f'Потери с train_count >= 3: {(ld.train_count >= 3).sum()}')
print(f'Потери с train_count == 1: {(ld.train_count == 1).sum()}')
