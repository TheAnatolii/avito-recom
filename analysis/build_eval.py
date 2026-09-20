"""Офлайн-эвал на val-сплите: кандидаты + скоры ранкера, кэширование.

Цель — иметь ЧЕСТНУЮ локальную метрику Recall@50 до отправки попытки.
Важно: корпус val-сплита (189000) отличается от корпуса benchmark (189212),
поэтому BM25-индекс и эмбеддинги строятся отдельно с префиксом val_.
Нельзя использовать work/bm25_index.pkl и work/item_embeddings.npy —
 они построены для другого корпуса и дадут бесшумно неверный результат.

Пул намеренно ШИРЕ продукционного: dense@300 ∪ bm25@300 ∪ local_bm25@150.
Локальные кандидаты — главная находка анализа: 83% кликнутых лежат в локации
запроса, а в обычном пуле их лишь ~5%.
"""
import os
import sys
import time
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.utils.logging import log
from src.utils.cache import load_cached, save_cached
from src.data.split import get_or_build_split
from src.retrieval.bm25 import BM25Index
from src.utils.text import Lemmatizer
from src.ranking.features import build_features
from src.ranking.model import load_model
from src.config import WORK_DIR, MAX_SEQ, MODEL_DIR, CAND_BM25, CAND_DENSE

N_VAL = int(os.environ.get('N_VAL', '4000'))
LOCAL_TOPN = 150

VAL_BM25 = 'val_bm25_index.pkl'
VAL_CANDS = 'val_candidates.pkl'


def build_val_bm25(corpus, lem):
    """BM25 по полям для val-корпуса. Кэш отдельный, не продукционный."""
    cached = load_cached(VAL_BM25)
    if cached is not None:
        log('val bm25: кэш найден')
        return cached
    t0 = time.time()
    indexes = {}
    fields = {
        'title': corpus['item_title_raw'].fillna('').astype(str),
        'desc': corpus['item_description_raw'].fillna('').astype(str),
        'params': corpus['item_infm_params_text'].fillna('').astype(str),
    }
    for field, series in fields.items():
        toks = [lem.doc_tokens(s) for s in series]
        indexes[field] = BM25Index().fit(toks)
        log(f'val BM25 "{field}": vocab={len(indexes[field].vocab)}, {time.time() - t0:.0f}s')
    obj = {'indexes': indexes, 'ids': corpus['item_id'].values}
    save_cached(obj, VAL_BM25)
    return obj


def retrieve(idx_obj, lem, query_text, topn, weights, restrict_mask=None):
    """BM25 по полям с возможностью ограничения множества документов."""
    qt = lem.doc_tokens(query_text)
    if not qt:
        return []
    combined = None
    for field, w in weights.items():
        idx = idx_obj['indexes'][field]
        order, sc = idx.score(qt, topn=None)
        if len(order) == 0:
            continue
        vec = np.zeros(len(idx_obj['ids']), dtype=np.float32)
        vec[order] = sc * w
        combined = vec if combined is None else combined + vec
    if combined is None:
        return []
    if restrict_mask is not None:
        combined = np.where(restrict_mask, combined, -1e9)
    order = np.argsort(combined)[::-1][:topn]
    return [idx_obj['ids'][i] for i in order]


def encode_val_items(corpus):
    """Dense-эмбеддинги val-корпуса в отдельный кэш."""
    from sentence_transformers import SentenceTransformer
    from src.retrieval import dense as dense_mod
    from src.utils.device import get_device
    cache = os.path.join(WORK_DIR, 'val_item_embeddings.npy')
    if os.path.exists(cache):
        log('val emb: кэш найден')
        return np.load(cache)
    model = SentenceTransformer(MODEL_DIR, device=get_device('cpu'))
    model.max_seq_length = MAX_SEQ
    texts = [dense_mod.item_text(r) for _, r in corpus.iterrows()]
    log(f'val emb: кодирую {len(texts)} айтемов...')
    t0 = time.time()
    emb = np.asarray(model.encode(texts, batch_size=128, show_progress_bar=True,
                                  normalize_embeddings=True), dtype=np.float32)
    np.save(cache, emb)
    log(f'val emb готово: {emb.shape}, {time.time() - t0:.0f}s')
    return emb


def main():
    t0 = time.time()
    log(f'=== ОФЛАЙН-ЭВАЛ (N_VAL={N_VAL}) ===')
    train_pairs, val_queries, corpus, seen_ids = get_or_build_split()
    log(f'val={len(val_queries)}, corpus={len(corpus)}')

    lem = Lemmatizer()
    idx_obj = build_val_bm25(corpus, lem)
    item_emb = encode_val_items(corpus)
    model = load_model()
    if model is None:
        raise RuntimeError('модель не обучена')

    item_lookup = {iid: i for i, iid in enumerate(corpus['item_id'].values)}
    loc_codes = corpus['item_location_id'].values
    # позиции айтемов по каждой локации
    log('val: группировка по локациям...')
    loc_to_positions = {}
    for pos, lc in enumerate(loc_codes):
        loc_to_positions.setdefault(lc, []).append(pos)
    log(f'val: локаций={len(loc_to_positions)}, {time.time() - t0:.0f}s')

    # подвыборка val-запросов
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(val_queries), size=min(N_VAL, len(val_queries)), replace=False)
    sample = val_queries.iloc[sample_idx].reset_index(drop=True)
    log(f'val: выбрано {len(sample)} запросов')

    from sentence_transformers import SentenceTransformer
    from src.retrieval import dense as dense_mod
    from src.utils.device import get_device
    st = SentenceTransformer(MODEL_DIR, device=get_device('cpu'))
    st.max_seq_length = MAX_SEQ
    q_texts = [dense_mod.query_text(r) for _, r in sample.iterrows()]
    q_emb = np.asarray(st.encode(q_texts, batch_size=128, show_progress_bar=False,
                                 normalize_embeddings=True), dtype=np.float32)
    order = dense_mod.topk_cosine(q_emb, item_emb, k=CAND_DENSE)
    log(f'val: dense готов, {time.time() - t0:.0f}s')

    weights = {'title': 3.0, 'desc': 1.0, 'params': 2.0}
    out = {}
    for i, (_, row) in enumerate(sample.iterrows()):
        qtext = row['search_query']
        qid = f'v{i}'
        dense_ids = [corpus['item_id'].values[j] for j in order[i]]
        bm25_ids = retrieve(idx_obj, lem, qtext, topn=CAND_BM25, weights=weights)

        # локальный пул: только айтемы из локации запроса
        qloc = row.get('search_location_id')
        local_ids = []
        if pd.notna(qloc) and qloc in loc_to_positions:
            pos_list = np.asarray(loc_to_positions[qloc])
            mask = np.zeros(len(corpus), dtype=bool)
            mask[pos_list] = True
            local_ids = retrieve(idx_obj, lem, qtext, topn=LOCAL_TOPN, weights=weights,
                                 restrict_mask=mask)

        # объединение с пометками источника
        seen = {}
        for src, lst in (('dense', dense_ids), ('bm25', bm25_ids), ('local', local_ids)):
            for rank, iid in enumerate(lst):
                if iid not in seen:
                    seen[iid] = {'src': src, 'rank': rank}

        cand_ids = list(seen.keys())
        if not cand_ids:
            continue
        feats = [build_features(row, corpus.iloc[item_lookup[iid]]) for iid in cand_ids]
        X = pd.DataFrame(feats)
        scores = np.asarray(model.predict(X), dtype=np.float64)

        out[qid] = {
            'search_query': qtext,
            'relevant_ids': list(row['relevant_ids']),
            'cand_ids': cand_ids,
            'scores': scores,
            'src': np.array([seen[i]['src'] for i in cand_ids], dtype=object),
            'rank_in_src': np.array([seen[i]['rank'] for i in cand_ids]),
            'is_local': np.array([loc_codes[item_lookup[i]] == qloc for i in cand_ids]),
            'query_row': row.drop('relevant_ids').to_dict(),
        }
        if (i + 1) % 500 == 0 or i + 1 == len(sample):
            log(f'val: обработано {i + 1}/{len(sample)}, {time.time() - t0:.0f}s')

    save_cached(out, VAL_CANDS)
    log(f'val: сохранено {len(out)} запросов в {VAL_CANDS}')
    log(f'val: итог {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
