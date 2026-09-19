"""CatBoost ранкер."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import load_raw, load_cached, save_cached, log, norm_text, haversine_np, build_location_lookup
from bm25 import BM25Retriever, Lemmatizer

MAX_NEG = 30
MAX_GROUPS = 20000
MODEL_NAME = 'ranker_model_cat.pkl'

_LOCATION_LOOKUP = None


def _get_location_lookup():
    global _LOCATION_LOOKUP
    if _LOCATION_LOOKUP is None:
        _LOCATION_LOOKUP = build_location_lookup()
    return _LOCATION_LOOKUP


def build_features(query_row, item_row):
    qtext = norm_text(query_row['search_query'])
    title = norm_text(item_row['item_title_raw'])
    desc = norm_text(item_row['item_description_raw'])
    qparams = norm_text(query_row['search_infm_params_text'])
    iparams = norm_text(item_row['item_infm_params_text'])
    qt, tt, dt = set(qtext.split()), set(title.split()), set(desc.split())
    pq, pi = set(qparams.split()), set(iparams.split())
    price = item_row['item_price']
    rating = item_row['item_rating']
    reviews = item_row['item_rating_reviews_count']
    return {
        'q_len': len(qt),
        'title_overlap': len(qt & tt) / max(1, len(qt)),
        'desc_overlap': len(qt & dt) / max(1, len(qt)),
        'title_exact': float(qtext in title),
        'desc_exact': float(qtext in desc),
        'params_overlap': len(pq & pi) / max(1, len(pq)),
        'loc_match': float(query_row['search_location_id'] == item_row['item_location_id']),
        'cat_match': float(query_row['search_category'] == item_row['item_category_id']),
        'microcat_match': float(query_row.get('search_category') == item_row.get('item_microcat_id'))
        if (pd.notna(query_row.get('search_category')) and pd.notna(item_row.get('item_microcat_id')))
        else 0.0,
        'geo_dist_km': float(haversine_np(
            float(_get_location_lookup().get(query_row['search_location_id'], {}).get('item_latitude', 55.75))
            if pd.notna(query_row.get('search_location_id')) else 55.75,
            float(_get_location_lookup().get(query_row['search_location_id'], {}).get('item_longitude', 37.62))
            if pd.notna(query_row.get('search_location_id')) else 37.62,
            float(item_row['item_latitude']) if pd.notna(item_row.get('item_latitude')) else 55.75,
            float(item_row['item_longitude']) if pd.notna(item_row.get('item_longitude')) else 37.62,
        )) if (pd.notna(query_row.get('search_location_id'))
              and pd.notna(item_row.get('item_latitude'))
              and pd.notna(item_row.get('item_longitude')))
        else -1.0,
        'price': np.log1p(max(0, float(price))) if pd.notna(price) else -1.0,
        'rating': float(rating) if pd.notna(rating) else -1.0,
        'reviews': np.log1p(max(0, float(reviews))) if pd.notna(reviews) else 0.0,
        'phone_hidden': float(item_row['item_is_phone_hidden']),
        'msg_forbidden': float(item_row['item_is_message_forbidden']),
    }


def prepare_train_data(tr, retriever, max_neg=MAX_NEG, max_groups=MAX_GROUPS):
    corpus_df = retriever.corpus
    item_lookup = {iid: i for i, iid in enumerate(retriever.ids)}
    groups = tr.groupby('search_query')
    features, labels, group_sizes = [], [], []
    n_done = 0
    t0 = time.time()
    for qtext, group in groups:
        if n_done >= max_groups:
            break
        pos_ids = set(group['item_id'].tolist())
        query_row = group.iloc[0]
        ranked = retriever.retrieve(qtext, topn=max_neg + len(pos_ids))
        neg_ids = [i for i in ranked if i not in pos_ids][:max_neg]
        if not neg_ids:
            continue
        n_before = len(features)
        for item_id in list(pos_ids)[:20]:
            if item_id not in item_lookup:
                continue
            features.append(build_features(query_row, corpus_df.iloc[item_lookup[item_id]]))
            labels.append(1)
        for item_id in neg_ids:
            features.append(build_features(query_row, corpus_df.iloc[item_lookup[item_id]]))
            labels.append(0)
        group_sizes.append(len(features) - n_before)
        n_done += 1
        if n_done % 2000 == 0:
            log(f'  обработано {n_done} групп, {time.time() - t0:.1f}s')
    X = pd.DataFrame(features)
    y = np.array(labels, dtype=np.int8)
    log(f'Пар: {len(X)} (pos={int(y.sum())}, neg={int((y == 0).sum())})')
    return X, y, np.array(group_sizes, dtype=np.int32)


def train():
    from catboost import CatBoostRanker, Pool
    t0 = time.time()
    log('Загрузка данных...')
    tr, q, it = load_raw()
    log(f'train={len(tr)}, items={len(it)}')
    log('Построение BM25...')
    retriever = BM25Retriever(it, Lemmatizer())
    log(f'Индекс готов, {time.time() - t0:.1f}s')
    log('Подготовка пар...')
    X, y, groups = prepare_train_data(tr, retriever)
    log(f'Готово, {time.time() - t0:.1f}s')
    log('Обучение CatBoost...')
    train_pool = Pool(X, y, group_id=np.repeat(np.arange(len(groups)), groups))
    model = CatBoostRanker(
        loss_function='QuerySoftMax',
        iterations=1500,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
    )
    model.fit(train_pool)
    log(f'Модель обучена, {time.time() - t0:.1f}s')
    save_cached(model, MODEL_NAME)
    log(f'Модель сохранена в work/{MODEL_NAME}')
    imp = pd.Series(model.get_feature_importance(train_pool), index=X.columns).sort_values(ascending=False)
    log('Feature importances:\n' + imp.to_string())


if __name__ == '__main__':
    train()
