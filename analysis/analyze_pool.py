"""Анализ пула кандидатов: представлена ли локация в топе BM25/dense?

loc_match — сильнейший разделитель (0.83 у кликнутых vs 0.027 у случайных),
но кандидаты строятся по всему корпусу без учёта локации. Если локальных
айтемов мало в топе — это узкое горлышко召回.
"""
import sys
sys.path.insert(0, '.')

import time
import numpy as np
import pandas as pd
from src.utils.logging import log
from src.utils.loader import load_raw
from src.retrieval.bm25 import BM25Retriever
from src.utils.text import Lemmatizer
from src.retrieval import dense as dense_mod
from src.config import CAND_BM25, CAND_DENSE

t0 = time.time()
tr, q, it = load_raw()
log(f'loaded {time.time() - t0:.0f}s')

answer = pd.read_csv('answer.csv')
ans_by_qid = dict(zip(answer.query_id, answer.answer.str.split()))

retriever = BM25Retriever(it, Lemmatizer())
log('bm25 ready')

# dense: кэш эмбеддингов айтемов
from sentence_transformers import SentenceTransformer
from src.config import MODEL_DIR, MAX_SEQ
from src.utils.device import get_device
device = get_device('cpu')
model = SentenceTransformer(MODEL_DIR, device=device)
model.max_seq_length = MAX_SEQ
item_emb = dense_mod.encode_items(model, it)
log(f'item emb {item_emb.shape}')

rng = np.random.default_rng(0)
sample = q.sample(min(600, len(q)), random_state=0).reset_index(drop=True)

q_texts = [dense_mod.query_text(r) for _, r in sample.iterrows()]
q_emb = np.asarray(model.encode(q_texts, batch_size=128, show_progress_bar=False,
                                normalize_embeddings=True), dtype=np.float32)
order = dense_mod.topk_cosine(q_emb, item_emb, k=CAND_DENSE)
ids = it.item_id.values

loc_of = dict(zip(it.item_id.values, it.item_location_id.values))
loc_pop = it.item_location_id.value_counts()

rows = []
for i, (_, row) in enumerate(sample.iterrows()):
    qid = row.query_id
    bm = retriever.retrieve(row.search_query, topn=CAND_BM25)
    dn = [ids[j] for j in order[i]]
    qloc = row.search_location_id

    def loc_rate(lst):
        if not lst:
            return np.nan, 0
        m = sum(1 for x in lst if loc_of.get(x) == qloc)
        return m / len(lst), m

    bm_rate, bm_n = loc_rate(bm)
    dn_rate, dn_n = loc_rate(dn)
    ans = ans_by_qid.get(qid, [])
    ans_rate, ans_n = loc_rate(ans)
    seen = set()
    union = []
    for x in dn:
        if x not in seen:
            seen.add(x)
            union.append(x)
    for x in bm:
        if x not in seen:
            seen.add(x)
            union.append(x)
    un_rate, un_n = loc_rate(union)
    # сколько вообще айтемов в этой локации
    pool_loc = int(loc_pop.get(qloc, 0))
    rows.append(dict(qid=qid, bm_loc=bm_rate, dn_loc=dn_rate, ans_loc=ans_rate,
                     un_loc=un_rate, pool_loc=pool_loc, bm_n=bm_n, dn_n=dn_n))

df = pd.DataFrame(rows)
log('=== ДОЛЯ АЙТЕМОВ В ЛОКАЦИИ ЗАПРОСА ===')
log(df[['bm_loc', 'dn_loc', 'un_loc', 'ans_loc']].describe().to_string())
log('')
log(f'медиана размера локального пула: {df.pool_loc.median():.0f}')
log(f'запросов, где в BM25@{CAND_BM25} НЕТ ни одного локального: {(df.bm_n == 0).sum()}/{len(df)}')
log(f'запросов, где в dense@{CAND_DENSE} НЕТ ни одного локального: {(df.dn_n == 0).sum()}/{len(df)}')
log('')
log('децили доли локальных в ответе (текущий answer.csv):')
log(df.ans_loc.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
log(f'ответов с ans_loc < 0.1: {(df.ans_loc < 0.1).sum()}')
log(f'ответов с ans_loc == 0:   {(df.ans_loc == 0).sum()}')
df.to_csv('pool_loc_analysis.csv', index=False)
log('saved pool_loc_analysis.csv')
