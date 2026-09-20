"""Обучение CatBoost ранкера.

Кандидаты и фичи соответствуют предсказанию (predict.py):
- Кандидаты: dense top-N + BM25 top-N, объединение (dense вперёд)
- Фичи: оригинальный набор (текст/гео/числовые), БЕЗ рангов и скоров
- Негативы: из топа кандидатов (hard negatives), как в оригинале

train_count НЕ используется как фича модели — для него утечка:
все позитивы в train имеют count >= 1 по определению.
Вместо него post-hoc буст в predict.py.
"""
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from src.utils.logging import log
from src.utils.loader import load_raw
from src.utils.cache import save_cached
from src.retrieval.bm25 import BM25Retriever
from src.utils.text import Lemmatizer
from src.ranking.features import build_features
from src.config import (MAX_NEG, MAX_GROUPS, MODEL_NAME, MODEL_DIR, MAX_SEQ,
                        CAND_DENSE, POOL_SIZE)


def prepare_train_data(tr, retriever, max_neg=MAX_NEG, max_groups=MAX_GROUPS):
    """Генерируем пары: кандидаты из dense+BM25, разметка по train."""
    corpus_df = retriever.corpus
    item_lookup = {iid: i for i, iid in enumerate(retriever.ids)}
    corpus_ids = retriever.ids

    # Берём уникальные запросы из train
    groups = list(tr.groupby('search_query'))
    if max_groups and len(groups) > max_groups:
        groups = groups[:max_groups]
    log(f'[RANKER] Запросов к обработке: {len(groups)}')

    # Dense: кодируем айтемы и запросы, берём top-N
    from sentence_transformers import SentenceTransformer
    from src.retrieval import dense as dense_mod
    from src.utils.device import get_device

    log('[RANKER] Загрузка dense-модели...')
    device = get_device('auto')
    st_model = SentenceTransformer(MODEL_DIR, device=device)
    st_model.max_seq_length = MAX_SEQ
    item_emb = dense_mod.encode_items(st_model, corpus_df)
    log(f'[RANKER] Item embeddings: {item_emb.shape}')

    query_rows = [g.iloc[0] for _, g in groups]
    q_texts = [dense_mod.query_text(r) for r in query_rows]
    log(f'[RANKER] Кодирую {len(q_texts)} запросов...')
    q_emb = np.asarray(st_model.encode(q_texts, batch_size=128,
                                        show_progress_bar=True, normalize_embeddings=True),
                       dtype=np.float32)
    log(f'[RANKER] Query embeddings: {q_emb.shape}')

    log(f'[RANKER] Вычисление dense top-{CAND_DENSE}...')
    dense_order = dense_mod.topk_cosine(q_emb, item_emb, k=CAND_DENSE)
    log('[RANKER] Dense готов')

    features, labels, group_sizes = [], [], []
    t0 = time.time()
    n_queries_with_relevant = 0

    for gi, (qtext, group) in enumerate(groups):
        pos_ids = set(group['item_id'].tolist())
        query_row = group.iloc[0]

        # --- Dense top-N ---
        d_order = dense_order[gi]
        dense_ids = [corpus_ids[j] for j in d_order]

        # --- BM25 top-N ---
        bm25_ids = retriever.retrieve(qtext, topn=300)

        # Объединение (dense вперёд, затем BM25) — идентично predict
        seen = set()
        pool = []
        for iid in dense_ids:
            if iid not in seen:
                seen.add(iid)
                pool.append(iid)
        for iid in bm25_ids:
            if iid not in seen:
                seen.add(iid)
                pool.append(iid)
        pool = pool[:POOL_SIZE]

        # Разметка
        pos_in_pool = [iid for iid in pos_ids if iid in item_lookup][:20]
        neg_ids = [iid for iid in pool if iid not in pos_ids and iid in item_lookup][:max_neg]
        if not neg_ids:
            continue
        if pos_in_pool:
            n_queries_with_relevant += 1

        n_before = len(features)
        for item_id in pos_in_pool:
            features.append(build_features(query_row, corpus_df.iloc[item_lookup[item_id]]))
            labels.append(1)
        for item_id in neg_ids:
            features.append(build_features(query_row, corpus_df.iloc[item_lookup[item_id]]))
            labels.append(0)
        group_sizes.append(len(features) - n_before)

        if (gi + 1) % 2000 == 0:
            log(f'  обработано {gi + 1} групп, {time.time() - t0:.1f}s')

    X = pd.DataFrame(features)
    y = np.array(labels, dtype=np.int8)
    log(f'Пар: {len(X)} (pos={int(y.sum())}, neg={int((y == 0).sum())})')
    log(f'Запросов с позитивами в пуле: {n_queries_with_relevant}')
    return X, y, np.array(group_sizes, dtype=np.int32)


def train():
    t0 = time.time()
    log('[RANKER] Начало обучения ранкера...')
    log('[RANKER] Загрузка данных...')
    tr, q, it = load_raw()
    log(f'[RANKER] Данные загружены: train={len(tr)}, items={len(it)}')
    log('[RANKER] Построение BM25 индекса...')
    retriever = BM25Retriever(it, Lemmatizer())
    log(f'[RANKER] BM25 готов: {time.time() - t0:.1f}s')
    log('[RANKER] Подготовка пар (позитивы + негативы)...')
    X, y, groups = prepare_train_data(tr, retriever)
    log(f'[RANKER] Пары готовы: {len(X)} строк, групп={len(groups)}')

    log('[RANKER] Обучение CatBoost...')
    train_pool = Pool(X, y, group_id=np.repeat(np.arange(len(groups)), groups))
    model = CatBoostRanker(
        loss_function='QuerySoftMax',
        iterations=2500,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
    )
    model.fit(train_pool)
    log(f'[RANKER] Модель обучена: {time.time() - t0:.1f}s')
    save_cached(model, MODEL_NAME)
    log(f'[RANKER] Модель сохранена в work/{MODEL_NAME}')
    imp = pd.Series(model.get_feature_importance(train_pool), index=X.columns).sort_values(ascending=False)
    log('Feature importances:\n' + imp.to_string())
