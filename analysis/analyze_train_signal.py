"""Анализ: насколько train-сигнал уже использован в answer.csv."""
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
# {(norm_query, item_id): count}
counts = tr.groupby(['search_query', 'item_id']).size()
train_counts = {(norm_text(qq), i): int(c) for (qq, i), c in counts.items()}

ans_by_qid = dict(zip(answer['query_id'], answer['answer'].str.split()))

n_in_train = 0
n_relevant_known = 0
n_already_in_answer = 0
n_missing_from_answer = 0
n_not_in_pool = 0

missing_details = []

for _, row in q.iterrows():
    qid = row['query_id']
    qn = norm_text(row['search_query'])
    # Все айтемы из train для этого запроса
    rel = {i for (qq, i) in train_counts if qq == qn}
    if not rel:
        continue
    n_in_train += 1
    # Только те, что в корпусе
    rel_in_corpus = rel & bm_item_set
    if not rel_in_corpus:
        continue
    n_relevant_known += 1

    ans = set(ans_by_qid.get(qid, []))
    in_ans = rel_in_corpus & ans
    not_in_ans = rel_in_corpus - ans

    n_already_in_answer += len(in_ans)
    n_missing_from_answer += len(not_in_ans)
    if not_in_ans:
        n_not_in_pool += 1
        missing_details.append((qid, row['search_query'], len(rel_in_corpus), len(in_ans), len(not_in_ans)))

print('=== TRAIN-СИГНАЛ В ANSWER.CSV ===')
print(f'benchmark запросов в train: {n_in_train}')
print(f'из них с релевантными в корпусе: {n_relevant_known}')
print(f'  уже в answer: {n_already_in_answer}')
print(f'  НЕ в answer:  {n_missing_from_answer}')
print(f'запросов с хотя бы 1 потерянным: {n_not_in_pool}')
print()
if missing_details:
    md = pd.DataFrame(missing_details, columns=['qid', 'query', 'n_rel', 'in_ans', 'not_in_ans'])
    print('Топ-20 потерь:')
    print(md.sort_values('not_in_ans', ascending=False).head(20).to_string(index=False))
    print()
    print(f'Среднее потерь на проблемный запрос: {md.not_in_ans.mean():.2f}')
    print(f'Если все потери релевантны → потенциал +{md.not_in_ans.sum() / len(q):.4f} к recall')
