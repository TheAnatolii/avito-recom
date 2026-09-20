"""ДИАГНОСТИКА 1: репрезентативен ли val-корпус? Standalone recall BM25 vs dense.
Плюс ЧЕСТНАЯ симуляция production: signal-половина кликов -> буст,
truth-половина -> разметка. Это точно повторяет сценарий benchmark.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.utils.cache import load_cached
from src.utils.logging import log
from src.data.split import get_or_build_split
from src.retrieval.bm25 import BM25Index
from src.utils.text import Lemmatizer, norm_text

K = 50
train_pairs, val_queries, corpus, seen_ids = get_or_build_split()
idx_obj = load_cached('val_bm25_index.pkl')
data = load_cached('val_candidates.pkl')
log(f'corpus={len(corpus)}, val={len(val_queries)}, eval={len(data)}')

# сопоставление: qid в data -> search_query
qtext_by_qid = {qid: d['search_query'] for qid, d in data.items()}
corpus_set = set(corpus['item_id'].values)

print('=' * 78)
print('ДИАГНОСТИКА: standalone召回 BM25 vs dense на val-корпусе')
print('=' * 78)
# Восстановим тексты запросов и эталон
rng = np.random.default_rng(7)
sample = list(data.items())[:1500]

lem = Lemmatizer()
weights = {'title': 3.0, 'desc': 1.0, 'params': 2.0}
ids = idx_obj['ids']

res = {'bm25': [], 'dense': [], 'pool': []}
for qid, d in sample:
    rel = set(d['relevant_ids'])
    if not rel:
        continue
    rel_c = rel & corpus_set
    if not rel_c:
        continue
    # BM25 standalone top-50
    qt = lem.doc_tokens(d['search_query'])
    if qt:
        combined = None
        for field, w in weights.items():
            order, sc = idx_obj['indexes'][field].score(qt, topn=None)
            if len(order) == 0:
                continue
            vec = np.zeros(len(ids), dtype=np.float32)
            vec[order] = sc * w
            combined = vec if combined is None else combined + vec
        if combined is not None:
            top = [ids[i] for i in np.argsort(combined)[::-1][:K]]
            res['bm25'].append(len(set(top) & rel_c) / len(rel_c))
    # dense standalone top-50
    dmask = (d['src'] == 'dense')
    dense_ids = [d['cand_ids'][i] for i in np.where(dmask)[0]]
    res['dense'].append(len(set(dense_ids[:K]) & rel_c) / len(rel_c))
    # полный пул
    res['pool'].append(len(set(d['cand_ids'][:K]) & rel_c) / len(rel_c))

for k in ('bm25', 'dense', 'pool'):
    if res[k]:
        log(f'{k:8s} standalone Recall@50 = {np.mean(res[k]):.4f} (n={len(res[k])})')

print()
print('=' * 78)
print('ЧЕСТНАЯ СИМУЛЯЦИЯ: signal/truth разбиение по строкам')
print('=' * 78)
# Полный train: для каждого val-запроса его клики делим пополам
tr = pd.read_parquet('dataset/train.parquet')
val_qtexts = set(qtext_by_qid.values())
tr_val = tr[tr.search_query.isin(val_qtexts)]
log(f'строк train по val-запросам: {len(tr_val)}')

mask = rng.random(len(tr_val)) < 0.5
signal = tr_val[mask]
truth = tr_val[~mask]
signal_counts = signal.groupby(['search_query', 'item_id']).size()
truth_sets = truth.groupby('search_query')['item_id'].apply(set).to_dict()
log(f'signal пар: {len(signal_counts)}, truth запросов: {len(truth_sets)}')

TRAIN_BOOST = 5.0


def evaluate_strategy(name, score_fn):
    rs = []
    n_with_signal = 0
    for qid, d in data.items():
        qtext = d['search_query']
        truth_rel = truth_sets.get(qtext)
        if not truth_rel:
            continue
        truth_rel = truth_rel & corpus_set
        if not truth_rel:
            continue
        qn = norm_text(qtext)
        boost = np.zeros(len(d['cand_ids']))
        for i, iid in enumerate(d['cand_ids']):
            tc = signal_counts.get((qtext, iid), 0)
            if tc > 0:
                boost[i] = TRAIN_BOOST * np.log1p(tc)
        if (boost > 0).any():
            n_with_signal += 1
        s = score_fn(d, boost)
        order = np.argsort(-s)
        pred = [d['cand_ids'][i] for i in order[:K]]
        rs.append(len(set(pred) & truth_rel) / len(truth_rel))
    rs = np.array(rs)
    log(f'{name:38s} n={len(rs):5d} Rec@50={rs.mean():.4f}  запросов_с_сигналом={n_with_signal}')
    return rs.mean()


# бейз: ранкер + буст 5.0 (как в production)
evaluate_strategy('ranker + boost(5.0)', lambda d, b: d['scores'] + b)
# только dense-порядок + буст
evaluate_strategy('dense-order + boost(5.0)',
                  lambda d, b: -d['rank_in_src'].astype(float) + b)
# ранкер + буст + небольшой dense-rank приоритет
for w in (0.01, 0.02, 0.05):
    evaluate_strategy(f'ranker + boost - {w}*dense_rank',
                      lambda d, b, x=w: d['scores'] + b - x * d['rank_in_src'].astype(float))
# разные веса буста
for tb in (2.0, 3.0, 8.0):
    evaluate_strategy(f'ranker + boost({tb})',
                      lambda d, b, x=tb: d['scores'] + b * (x / TRAIN_BOOST))
# без буста
evaluate_strategy('ranker без буста', lambda d, b: d['scores'])
