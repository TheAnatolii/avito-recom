"""Диагностика: разброс скоров ранкера и попадание train-релевантных в пул."""
import sys, time
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.utils.logging import log
from src.utils.loader import load_raw
from src.utils.cache import load_cached
from src.retrieval.bm25 import BM25Retriever
from src.utils.text import Lemmatizer, norm_text
from src.ranking.features import build_features
from src.ranking.model import load_model
from src.pipeline.candidates import get_candidates
from src.pipeline.predict import build_train_counts
from src.config import TOP_K, TRAIN_BOOST_WEIGHT, POOL_SIZE

tr, q, it = load_raw()
model = load_model()
retriever = BM25Retriever(it, Lemmatizer())
train_counts = build_train_counts(tr)

# Кандидаты только для подвыборки проблемных запросов
problem_qids = set()
answer = pd.read_csv('answer.csv')
ans_by_qid = dict(zip(answer['query_id'], answer['answer'].str.split()))
bm_item_set = set(it['item_id'].values)
for _, row in q.iterrows():
    qn = norm_text(row['search_query'])
    rel = {i for (qq, i) in train_counts if qq == qn} & bm_item_set
    if not rel:
        continue
    ans = set(ans_by_qid.get(row['query_id'], []))
    if rel - ans:
        problem_qids.add(row['query_id'])

q_prob = q[q['query_id'].isin(problem_qids)].head(50)
log(f'Проблемных запросов для диагностики: {len(q_prob)}')

cands = get_candidates(q_prob, it, retriever, 'cpu')

item_lookup = {iid: i for i, iid in enumerate(it['item_id'].values)}
n_in_pool = 0
n_not_in_pool = 0
score_ranges = []

for _, row in q_prob.iterrows():
    qid = row['query_id']
    qn = norm_text(row['search_query'])
    cand_ids = [iid for iid in cands.get(qid, []) if iid in item_lookup]
    if not cand_ids:
        continue
    feats = [build_features(row, it.iloc[item_lookup[iid]]) for iid in cand_ids]
    X = pd.DataFrame(feats)
    scores = np.asarray(model.predict(X), dtype=np.float64)
    score_ranges.append((scores.min(), scores.max(), np.sort(scores)[-50], np.sort(scores)[-1]))
    
    pool_set = set(cand_ids)
    rel = {i for (qq, i) in train_counts if qq == qn} & bm_item_set
    for iid in rel:
        if iid in pool_set:
            n_in_pool += 1
        else:
            n_not_in_pool += 1

sr = pd.DataFrame(score_ranges, columns=['min', 'max', 'top50_threshold', 'top1'])
log('=== РАЗБРОС СКОРОВ РАНКЕРА (50 проблемных запросов) ===')
log(sr.describe().to_string())
log('')
log(f'top-50 порог (медиана): {sr.top50_threshold.median():.3f}')
log(f'max скор (медиана):    {sr.top1.median():.3f}')
log(f'boost для count=1:     {TRAIN_BOOST_WEIGHT * np.log1p(1):.3f}')
log('')
log(f'train-релевантных в пуле:   {n_in_pool}')
log(f'train-релевантных НЕ в пуле: {n_not_in_pool}')
