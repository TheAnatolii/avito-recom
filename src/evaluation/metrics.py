"""Метрики оценки."""
import os
import pandas as pd
from src.utils.logging import log
from src.config import DATA_DIR


def recall_at_k(ranked_lists, val_queries, k):
    rel_map = dict(zip(val_queries['search_query'], val_queries['relevant_ids']))
    new_q = set(val_queries[val_queries.has_new_item]['search_query'])
    hits = hits_new = 0
    for qtext, ranked in ranked_lists.items():
        if set(rel_map[qtext]) & set(ranked[:k]):
            hits += 1
            if qtext in new_q:
                hits_new += 1
    n = len(ranked_lists)
    n_new = len(new_q)
    return hits / max(1, n), (hits_new / max(1, n_new)) if n_new else 0.0


def evaluate_predict(answer_path='answer.csv', benchmark_path='benchmark_queries.parquet'):
    answer = pd.read_csv(answer_path)
    bm = pd.read_parquet(os.path.join(DATA_DIR, benchmark_path))
    log(f'Benchmark queries: {len(bm)}, predictions: {len(answer)}')
    log('Predict validation passed (format only)')
