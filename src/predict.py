"""Предсказание: BM25 + dense -> CatBoost -> answer.csv."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import load_raw, load_cached, MODEL_DIR, log, get_device
from ranker import build_features
from bm25 import BM25Retriever, Lemmatizer
import dense as dense_mod
from sentence_transformers import SentenceTransformer

TOP_K = 50
OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'answer.csv')


def get_candidates(q, it, retriever, device_pref):
    bm25_cands = {}
    for _, row in q.iterrows():
        bm25_cands[row['query_id']] = retriever.retrieve(row['search_query'], topn=150)
    log(f'BM25-кандидаты: {len(bm25_cands)} запросов')

    device = get_device(device_pref)
    log(f'Dense retrieval (device={device})...')
    model = SentenceTransformer(MODEL_DIR, device=device)
    model.max_seq_length = dense_mod.MAX_SEQ
    item_emb = dense_mod.encode_items(model, it)
    q_emb = dense_mod.encode_queries(model, q)
    order = dense_mod.topk_cosine(q_emb, item_emb, k=150)
    ids = it['item_id'].values
    dense_cands = {q.iloc[i]['query_id']: [ids[j] for j in order[i]] for i in range(len(q))}
    log(f'Dense-кандидаты: {len(dense_cands)} запросов')

    merged = {}
    for qid in q['query_id']:
        bm25_list = bm25_cands.get(qid, [])
        dense_list = dense_cands.get(qid, [])
        seen = set()
        combined = []
        for iid in dense_list:
            if iid not in seen:
                combined.append(iid)
                seen.add(iid)
        for iid in bm25_list:
            if iid not in seen:
                combined.append(iid)
                seen.add(iid)
        if len(combined) < 300:
            popular = it['item_id'].head(300 - len(combined)).tolist()
            for iid in popular:
                if iid not in seen:
                    combined.append(iid)
                    seen.add(iid)
        merged[qid] = combined
    return merged


def main(top_k=TOP_K, device_pref='auto'):
    t0 = time.time()
    log('Загрузка данных...')
    tr, q, it = load_raw()
    log(f'queries={len(q)}, items={len(it)}')
    log('Загрузка ранкера...')
    model = load_cached('ranker_model_cat.pkl')
    if model is None:
        raise RuntimeError('Запустите ranker.py сначала')
    log('Модель загружена')
    log('Построение BM25...')
    retriever = BM25Retriever(it, Lemmatizer())
    log(f'Индекс готов, {time.time() - t0:.1f}s')
    log('Получение кандидатов...')
    cands = get_candidates(q, it, retriever, device_pref)
    log(f'Кандидаты готовы, {time.time() - t0:.1f}s')

    item_lookup = {iid: i for i, iid in enumerate(it['item_id'].values)}
    log('Ранжирование...')
    predictions, query_ids = [], []
    for _, row in q.iterrows():
        qid = row['query_id']
        cand_ids = cands.get(qid, [])[:300]
        if not cand_ids:
            cand_ids = it['item_id'].head(top_k).tolist()
        feats = [build_features(row, it.iloc[item_lookup[iid]]) for iid in cand_ids if iid in item_lookup]
        cand_ids = [iid for iid in cand_ids if iid in item_lookup]
        if not feats:
            predictions.append([])
            query_ids.append(qid)
            continue
        X = pd.DataFrame(feats)
        scores = model.predict(X)
        order = np.argsort(-scores)[:top_k]
        seen, top = set(), []
        for i in order:
            iid = cand_ids[i]
            if iid not in seen:
                seen.add(iid)
                top.append(iid)
        predictions.append(top[:top_k])
        query_ids.append(qid)
    log(f'Ранжирование завершено, {time.time() - t0:.1f}s')

    log('Сохранение answer.csv...')
    answer = pd.DataFrame({
        'query_id': query_ids,
        'answer': [' '.join(t) for t in predictions],
    })
    answer.to_csv(OUT_CSV, index=False, encoding='utf-8')
    log(f'Сохранено {len(answer)} строк в {OUT_CSV}')

    log('Валидация...')
    assert len(answer) == len(q), f'Неверное число строк: {len(answer)} != {len(q)}'
    assert answer.query_id.nunique() == len(answer), 'Дубли query_id'
    assert set(answer.query_id) == set(q.query_id), 'Несовпадение query_id'
    for ans in answer.answer:
        ids = ans.split()
        assert len(ids) <= top_k, f'Больше {top_k} id: {len(ids)}'
        assert len(set(ids)) == len(ids), 'Дубли внутри строки'
        for iid in ids:
            assert iid in item_lookup, f'Несуществующий item_id: {iid}'
    log('Валидация прошла!')
    n = answer.answer.str.split().apply(len)
    log(f'item_id на запрос: min={n.min()}, median={int(n.median())}, max={n.max()}')
    log(f'Полное время: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
