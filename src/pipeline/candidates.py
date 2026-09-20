"""Генерация кандидатов: BM25 + dense retrieval, простое объединение.

RRF убран: он менял распределение кандидатов и приводил к падению метрики.
Объединение как в оригинальном коде: dense вперёд, затем BM25 (dedup).
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.logging import log
from src.utils.device import get_device
from src.retrieval.bm25 import BM25Retriever
from src.utils.text import Lemmatizer
from src.retrieval import dense as dense_mod
from src.config import MODEL_DIR, MAX_SEQ, CAND_BM25, CAND_DENSE, POOL_SIZE


def get_candidates(q, it, retriever, device_pref='auto'):
    """Возвращает {qid: [item_id, ...]} — объединение dense + BM25."""
    log('[CANDIDATES] Начало генерации кандидатов...')

    # --- BM25 ---
    log(f'[CANDIDATES] BM25 retrieval (top-{CAND_BM25})...')
    bm25_cands = {}
    total_q = len(q)
    for idx, (_, row) in enumerate(q.iterrows(), 1):
        bm25_cands[row['query_id']] = retriever.retrieve(row['search_query'], topn=CAND_BM25)
        if idx % 500 == 0 or idx == total_q:
            log(f'[CANDIDATES] BM25: обработано запросов {idx}/{total_q}')
    log(f'[CANDIDATES] BM25 готово: {len(bm25_cands)} запросов')

    # --- Dense ---
    device = get_device(device_pref)
    log(f'[CANDIDATES] Dense retrieval (device={device}, top-{CAND_DENSE})...')
    log('[CANDIDATES] Загрузка модели SentenceTransformer...')
    model = SentenceTransformer(MODEL_DIR, device=device)
    model.max_seq_length = MAX_SEQ
    log('[CANDIDATES] Кодирование эмбеддингов айтемов...')
    item_emb = dense_mod.encode_items(model, it)
    log(f'[CANDIDATES] Item embeddings: {item_emb.shape}')
    log('[CANDIDATES] Кодирование эмбеддингов запросов...')
    q_emb = dense_mod.encode_queries(model, q)
    log(f'[CANDIDATES] Query embeddings: {q_emb.shape}')
    log('[CANDIDATES] Вычисление cosine similarity...')
    order = dense_mod.topk_cosine(q_emb, item_emb, k=CAND_DENSE)
    ids = it['item_id'].values
    dense_cands = {q.iloc[i]['query_id']: [ids[j] for j in order[i]] for i in range(len(q))}
    log(f'[CANDIDATES] Dense готово')

    # --- Объединение (dense вперёд, затем BM25) ---
    log('[CANDIDATES] Объединение dense + bm25...')
    merged = {}
    n_bigger_pool = 0
    for _, row in q.iterrows():
        qid = row['query_id']
        dense_list = dense_cands.get(qid, [])
        bm25_list = bm25_cands.get(qid, [])
        seen = set()
        combined = []
        # Dense вперёд — он точнее на top
        for iid in dense_list:
            if iid not in seen:
                combined.append(iid)
                seen.add(iid)
        for iid in bm25_list:
            if iid not in seen:
                combined.append(iid)
                seen.add(iid)
        # Добиваем популярными если < POOL_SIZE
        if len(combined) < POOL_SIZE:
            popular = it['item_id'].head(POOL_SIZE - len(combined)).tolist()
            for iid in popular:
                if iid not in seen:
                    combined.append(iid)
                    seen.add(iid)
        merged[qid] = combined[:POOL_SIZE]
        if len(merged[qid]) > 300:
            n_bigger_pool += 1

    log(f'[CANDIDATES] Кандидаты готовы для {len(merged)} запросов '
        f'(пул >300 у {n_bigger_pool}).')
    return merged
