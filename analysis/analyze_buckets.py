"""Сколько benchmark-запросов имеют НАДЕЖНЫЙ train-сигнал (count>=3)?"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.utils.text import norm_text

tr = pd.read_parquet('dataset/train.parquet')
q = pd.read_parquet('dataset/benchmark_queries.parquet')
it = pd.read_parquet('dataset/benchmark_items.parquet')

bm_item_set = set(it['item_id'].values)
counts = tr.groupby(['search_query', 'item_id']).size()
train_counts = {(norm_text(qq), i): int(c) for (qq, i), c in counts.items()}

answer = pd.read_csv('answer.csv')
ans_by_qid = dict(zip(answer['query_id'], answer['answer'].str.split()))

stats = {'any': 0, 'c1': 0, 'c2': 0, 'c3': 0}
in_ans = {'c1': 0, 'c2': 0, 'c3': 0}
not_in_ans = {'c1': 0, 'c2': 0, 'c3': 0}

for _, row in q.iterrows():
    qn = norm_text(row['search_query'])
    rel = {i: c for (qq, i), c in train_counts.items() if qq == qn and i in bm_item_set}
    if not rel:
        continue
    stats['any'] += 1
    ans = set(ans_by_qid.get(row['query_id'], []))
    for iid, c in rel.items():
        bucket = 'c1' if c == 1 else ('c2' if c == 2 else 'c3')
        stats[bucket] += 1
        if iid in ans:
            in_ans[bucket] += 1
        else:
            not_in_ans[bucket] += 1

print('=== TRAIN-СИГНАЛ В BENCHMARK (по buckets) ===')
print(f'Запросов с любым сигналом: {stats["any"]}')
print(f'  count=1:   {stats["c1"]:5d}  (в answer: {in_ans["c1"]}, потеряно: {not_in_ans["c1"]})')
print(f'  count=2:   {stats["c2"]:5d}  (в answer: {in_ans["c2"]}, потеряно: {not_in_ans["c2"]})')
print(f'  count=3+:  {stats["c3"]:5d}  (в answer: {in_ans["c3"]}, потеряно: {not_in_ans["c3"]})')
print()
total_c3_queries = set()
for _, row in q.iterrows():
    qn = norm_text(row['search_query'])
    rel = {i: c for (qq, i), c in train_counts.items() if qq == qn and i in bm_item_set and c >= 3}
    if rel:
        total_c3_queries.add(row['query_id'])
print(f'Запросов с count>=3 сигналом: {len(total_c3_queries)}')
print(f'  точность 64-99%,这些都是 надёжные')
print()

# Если поднять буст для count>=3 до попадания в top-50
# и убрать буст для count=1
print('=== ОЦЕНКА АДАПТИВНОГО БУСТА ===')
print(f'count>=3 потеряно: {not_in_ans["c3"]} айтемов')
print(f'  при точности 64-99%: ожидаю +{not_in_ans["c3"] * 0.8 / 2452:.4f} к recall')
print(f'count=1 в answer: {in_ans["c1"]} (точность 7.5% → ~{in_ans["c1"] * 0.075:.0f} правильных)')
print(f'  освободив слоты от count=1 шума, ранкер найдёт больше')
