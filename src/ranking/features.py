"""Построение признаков для ранжирования.

Оригинальный набор признаков (дал Recall@50 = 0.4828):
текстовые совпадения, гео, числовые характеристики объявлений.

Ранги/скоры ретривалов (bm25_score, dense_sim, bm25_rank, dense_rank)
намеренно убраны — у них разные распределения в train и predict,
что приводило к переобучению и падению метрики до 0.21.

train_count тоже НЕ фича модели — для него утечка: все позитивы в train
имеют count >= 1 по определению. Вместо него используется post-hoc буст
в predict.py только для запросов, которые реально есть в train.
"""
import numpy as np
import pandas as pd
from src.utils.text import norm_text
from src.utils.geo import haversine_np
from src.utils.loader import build_location_lookup

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

    # Локация
    search_loc = query_row.get('search_location_id')
    item_loc = item_row.get('item_location_id')
    loc_match = float(pd.notna(search_loc) and pd.notna(item_loc) and search_loc == item_loc)

    # Гео-дистанция
    geo_dist = -1.0
    if (pd.notna(search_loc) and pd.notna(item_row.get('item_latitude'))
            and pd.notna(item_row.get('item_longitude'))):
        ll = _get_location_lookup().get(search_loc, {})
        geo_dist = float(haversine_np(
            float(ll.get('item_latitude', 55.75)),
            float(ll.get('item_longitude', 37.62)),
            float(item_row['item_latitude']),
            float(item_row['item_longitude']),
        ))

    return {
        # Текстовые
        'q_len': len(qt),
        'title_overlap': len(qt & tt) / max(1, len(qt)),
        'desc_overlap': len(qt & dt) / max(1, len(qt)),
        'title_exact': float(qtext in title),
        'desc_exact': float(qtext in desc),
        'params_overlap': len(pq & pi) / max(1, len(pq)),
        # Гео
        'loc_match': loc_match,
        'geo_dist_km': geo_dist if geo_dist >= 0 else -1.0,
        'geo_dist_bucket': (
            0 if geo_dist < 0 else
            1 if geo_dist < 5 else
            2 if geo_dist < 25 else
            3 if geo_dist < 100 else
            4
        ),
        # Категории
        'cat_match': float(query_row['search_category'] == item_row['item_category_id'])
        if (pd.notna(query_row.get('search_category')) and pd.notna(item_row.get('item_category_id')))
        else 0.0,
        'microcat_in_query': float(pd.notna(item_row.get('item_microcat_id')) and
                                   str(item_row['item_microcat_id']) in str(query_row.get('search_infm_params_text', ''))),
        # Числовые
        'price': np.log1p(max(0, float(price))) if pd.notna(price) else -1.0,
        'rating': float(rating) if pd.notna(rating) else -1.0,
        'reviews': np.log1p(max(0, float(reviews))) if pd.notna(reviews) else 0.0,
        # Флаги
        'phone_hidden': float(item_row['item_is_phone_hidden']),
        'msg_forbidden': float(item_row['item_is_message_forbidden']),
    }
