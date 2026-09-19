"""Валидация: сплит и recall@K."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import load_raw, save_cached, load_cached, log

CORPUS_SIZE = 189000
VAL_FRAC = 0.2
SEED = 42

ITEM_COLS = [
    'item_id', 'item_title_raw', 'item_description_raw',
    'item_infm_params_text', 'item_category_id', 'item_microcat_id',
    'item_price', 'item_rating', 'item_rating_reviews_count',
    'item_location_id', 'item_latitude', 'item_longitude',
    'item_is_phone_hidden', 'item_is_message_forbidden',
]


def build_split(tr, it, corpus_size=CORPUS_SIZE, val_frac=VAL_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    tr_items = tr.drop_duplicates(subset='item_id')[ITEM_COLS].copy()
    bm_items = it.drop_duplicates(subset='item_id')[ITEM_COLS].copy()
    all_items = pd.concat([tr_items, bm_items], ignore_index=True)
    all_items = all_items.drop_duplicates(subset='item_id').reset_index(drop=True)
    log(f'Уникальных айтемов: {len(all_items)}')

    qtexts = tr['search_query'].unique()
    rng.shuffle(qtexts)
    n_val = int(len(qtexts) * val_frac)
    val_qtexts = set(qtexts[:n_val].tolist())
    log(f'Сплит: val={n_val}, train={len(qtexts) - n_val}')

    tr['is_val'] = tr['search_query'].isin(val_qtexts)
    val_pairs = tr[tr['is_val']][['search_query', 'item_id']]
    train_pairs = tr[~tr['is_val']][['search_query', 'item_id']]

    val_pos_ids = set(val_pairs['item_id'].unique())
    pool = all_items['item_id'].values
    pool_non_pos = np.setdiff1d(pool, list(val_pos_ids))
    n_sample = min(corpus_size - len(val_pos_ids), len(pool_non_pos))
    sampled = rng.choice(pool_non_pos, size=n_sample, replace=False)
    corpus_ids = np.concatenate([np.array(list(val_pos_ids)), sampled])
    corpus = all_items[all_items['item_id'].isin(set(corpus_ids.tolist()))]
    corpus = corpus.reset_index(drop=True)
    log(f'Корпус: {len(corpus)} (релевантных val: {len(val_pos_ids)})')

    seen_ids = set(train_pairs['item_id'].unique())
    val_queries = val_pairs.groupby('search_query')['item_id'].apply(list).reset_index()
    val_queries.columns = ['search_query', 'relevant_ids']
    qctx = tr[tr['is_val'].fillna(False)].groupby('search_query').agg({
        'search_location_id': 'first',
        'search_category': 'first',
        'search_infm_params_text': 'first',
        'search_is_delivery_search': 'first',
    }).reset_index()
    val_queries = val_queries.merge(qctx, on='search_query', how='left')
    val_queries['has_new_item'] = val_queries['relevant_ids'].apply(
        lambda ids: any(i not in seen_ids for i in ids))
    log(f'Val-запросов: {len(val_queries)}, новых: {int(val_queries.has_new_item.sum())}')
    return train_pairs, val_queries, corpus, seen_ids


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
    import pandas as pd
    answer = pd.read_csv(answer_path)
    bm = pd.read_parquet(os.path.join(os.path.dirname(__file__), '..', 'dataset', benchmark_path))
    log(f'Benchmark queries: {len(bm)}, predictions: {len(answer)}')
    log('Predict validation passed (format only)')


def get_or_build_split():
    cached = load_cached('split_v1.pkl')
    if cached is not None:
        log('Кэш split_v1.pkl')
        return cached
    log('Загрузка данных...')
    tr, q, it = load_raw()
    log(f'train={len(tr)}, benchmark_items={len(it)}')
    split = build_split(tr, it)
    save_cached(split, 'split_v1.pkl')
    log('Сплит сохранён')
    return split


if __name__ == '__main__':
    t0 = time.time()
    train_pairs, val_queries, corpus, seen_ids = get_or_build_split()
    log(f'Готово. val={len(val_queries)}, corpus={len(corpus)}, {time.time() - t0:.1f}s')
    log(f'Медиана релевантных={int(val_queries.relevant_ids.apply(len).median())}, '
        f'max={int(val_queries.relevant_ids.apply(len).max())}')
