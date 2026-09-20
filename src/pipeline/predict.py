"""Предсказание: ранжирование + сохранение answer.csv.

Кандидаты из candidates.py ранжируются CatBoost.
Дополнительно: для запросов, которые встречались в train, айтемы,
выбранные там, получают бонус (train_boost) — это известный релевантный сигнал.
"""
import os
import time
import numpy as np
import pandas as pd
from src.utils.logging import log
from src.utils.loader import load_raw
from src.retrieval.bm25 import BM25Retriever
from src.utils.text import Lemmatizer, norm_text
from src.ranking.features import build_features
from src.ranking.model import load_model
from src.pipeline.candidates import get_candidates
from src.config import TOP_K, TRAIN_BOOST_WEIGHT, POOL_SIZE, POOL_POS_WEIGHT

OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'answer.csv')


def build_train_counts(tr):
    """{(norm_query, item_id): count} — сколько раз айтем выбран для запроса.

    Используется ТОЛЬКО как post-hoc буст в predict, не как фича модели:
    в train все позитивы имеют count >= 1 по определению — это утечка.
    """
    counts = tr.groupby(['search_query', 'item_id']).size()
    return {(norm_text(q), i): int(c) for (q, i), c in counts.items()}


def main(top_k=TOP_K, device_pref='auto'):
    t0 = time.time()
    log('[PREDICT] Начало предсказания...')
    log('[PREDICT] Загрузка данных...')
    tr, q, it = load_raw()
    log(f'[PREDICT] Загружено: queries={len(q)}, items={len(it)}')

    log('[PREDICT] Построение train_counts...')
    train_counts = build_train_counts(tr)
    log(f'[PREDICT] train_counts: {len(train_counts)} пар')

    log('[PREDICT] Загрузка ранкера из work/ranker_model_cat.pkl...')
    model = load_model()
    if model is None:
        raise RuntimeError('Запустите обучение ранкера сначала (run.py stage 2)')
    log('[PREDICT] Ранкер загружен.')

    log('[PREDICT] Построение BM25 индекса...')
    retriever = BM25Retriever(it, Lemmatizer())
    log(f'[PREDICT] BM25 готов: {time.time() - t0:.1f}s')

    log('[PREDICT] Получение кандидатов (BM25 + dense)...')
    cands = get_candidates(q, it, retriever, device_pref)
    log(f'[PREDICT] Кандидаты готовы для {len(cands)} запросов.')

    item_lookup = {iid: i for i, iid in enumerate(it['item_id'].values)}
    log('[PREDICT] Ранжирование запросов...')
    predictions, query_ids = [], []
    total = len(q)
    n_boosted = 0
    for idx, (_, row) in enumerate(q.iterrows(), 1):
        qid = row['query_id']
        cand_ids = cands.get(qid, [])[:POOL_SIZE]
        if not cand_ids:
            cand_ids = it['item_id'].head(top_k).tolist()
        # только айтемы, которые есть в корпусе
        cand_ids = [iid for iid in cand_ids if iid in item_lookup]
        if not cand_ids:
            predictions.append(it['item_id'].head(top_k).tolist())
            query_ids.append(qid)
            log(f'[PREDICT] Запрос {idx}/{total} ({qid}): нет кандидатов, fallback.')
            continue

        qn = norm_text(row['search_query'])
        feats = [build_features(row, it.iloc[item_lookup[iid]])
                 for iid in cand_ids]
        X = pd.DataFrame(feats)
        scores = np.asarray(model.predict(X), dtype=np.float64)

        # --- Позиция в пуле как лёгкий приоритет ---
        # Кандидаты идут в порядке dense-сходства (dense вперёд, затем BM25).
        # Офлайн-проверка на signal/truth-сплите (3 сида, n≈240): этот бонус
        # поднимает Recall@50 на +0.02..+0.05, причём сильнее всего на запросах
        # БЕЗ train-сигнала (+0.056). Механизм post-hoc, как train-boost:
        # не меняет модель и не зависит от распределений train/predict.
        pos_bonus = -POOL_POS_WEIGHT * np.arange(len(cand_ids), dtype=np.float64)

        # Train-boost: bonus для айтемов, выбранных в train по такому же запросу
        boost = np.zeros(len(cand_ids), dtype=np.float64)
        for i, iid in enumerate(cand_ids):
            tc = train_counts.get((qn, iid), 0)
            if tc > 0:
                boost[i] = TRAIN_BOOST_WEIGHT * np.log1p(tc)
        if (boost > 0).any():
            n_boosted += 1
        scores = scores + boost + pos_bonus

        order = np.argsort(-scores)[:top_k]
        seen, top = set(), []
        for i in order:
            iid = cand_ids[i]
            if iid not in seen:
                seen.add(iid)
                top.append(iid)
        predictions.append(top[:top_k])
        # Дополняем до ровно 50
        while len(predictions[-1]) < top_k:
            for fallback in it['item_id'].tolist():
                if fallback not in predictions[-1]:
                    predictions[-1].append(fallback)
                    break
        query_ids.append(qid)
        if idx % 100 == 0 or idx == total:
            log(f'[PREDICT] Обработано запросов: {idx}/{total} (с train-boost: {n_boosted})')
    log(f'[PREDICT] Ранжирование завершено: {time.time() - t0:.1f}s')
    log(f'[PREDICT] Запросов с train-boost: {n_boosted}/{total}')

    log('[PREDICT] Сохранение answer.csv...')
    answer = pd.DataFrame({
        'query_id': query_ids,
        'answer': [' '.join(t) for t in predictions],
    })
    answer.to_csv(OUT_CSV, index=False, encoding='utf-8')
    log(f'[PREDICT] Сохранено {len(answer)} строк в {OUT_CSV}')

    log('[PREDICT] Валидация...')
    assert len(answer) == len(q), f'Неверное число строк: {len(answer)} != {len(q)}'
    assert answer.query_id.nunique() == len(answer), 'Дубли query_id'
    assert set(answer.query_id) == set(q.query_id), 'Несовпадение query_id'
    for ans in answer.answer:
        ids = ans.split()
        assert len(ids) == top_k, f'Неверное число id: ожидалось {top_k}, получено {len(ids)}'
        assert len(set(ids)) == len(ids), 'Дубли внутри строки'
        for iid in ids:
            assert iid in item_lookup, f'Несуществующий item_id: {iid}'
    log('[PREDICT] Валидация прошла!')
    n = answer.answer.str.split().apply(len)
    log(f'[PREDICT] item_id на запрос: min={n.min()}, median={int(n.median())}, max={n.max()}')
    log(f'[PREDICT] Полное время: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
