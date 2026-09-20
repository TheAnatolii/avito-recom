"""Построение и кэширование сплита данных."""
import numpy as np
import pandas as pd
from src.utils.logging import log
from src.utils.loader import load_raw
from src.utils.cache import load_cached, save_cached
from src.config import SEED, CORPUS_SIZE, VAL_FRAC

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


def get_or_build_split():
    log('[SPLIT] Проверка кэша split_v1.pkl...')
    cached = load_cached('split_v1.pkl')
    if cached is not None:
        log('[SPLIT] Кэш найден, загружаем.')
        return cached
    log('[SPLIT] Кэш не найден. Загрузка данных...')
    tr, q, it = load_raw()
    log(f'[SPLIT] Загружено: train={len(tr)}, benchmark_items={len(it)}')
    log('[SPLIT] Построение сплита...')
    split = build_split(tr, it)
    log('[SPLIT] Сплит построен. Сохранение в work/split_v1.pkl...')
    save_cached(split, 'split_v1.pkl')
    log('[SPLIT] Сплит сохранён.')
    return split
