"""Полный анализ данных: структура запросов, сепарабельность фичей, метрики.

Цель — понять, где вообще лежит потолок Recall@50 и какие сигналы
разделяют кликнутые айтемы от остальных.
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from src.utils.text import norm_text

rng = np.random.default_rng(42)
tr = pd.read_parquet('dataset/train.parquet')
q = pd.read_parquet('dataset/benchmark_queries.parquet')
it = pd.read_parquet('dataset/benchmark_items.parquet')

pd.set_option('display.width', 200)

print('=' * 70)
print('1. РАЗМЕРЫ')
print('=' * 70)
print(f'train rows={len(tr)}, uniq queries={tr.search_query.nunique()}, uniq items={tr.item_id.nunique()}')
print(f'benchmark queries={len(q)} (uniq {q.query_id.nunique()}), corpus={len(it)}')

print()
print('=' * 70)
print('2. |РЕЛЕВАНТНЫЕ| НА ЗАПРОС (структура метрики)')
print('=' * 70)
per_q = tr.groupby('search_query')['item_id'].nunique()
print(per_q.describe().to_string())
print(f'доля запросов с 1 релевантным: {(per_q == 1).mean():.3f}')
print(f'доля с 1-2:                    {per_q.le(2).mean():.3f}')
print(f'запросы с >=5 айтемами несут долю всех пар: {per_q[per_q >= 5].sum() / per_q.sum():.3f}')

print()
print('=' * 70)
print('3. СОВПАДЕНИЕ ЗАПРОСОВ train <-> benchmark')
print('=' * 70)
tr_q = set(tr.search_query.apply(norm_text))
bm_q = set(q.search_query.apply(norm_text))
print(f'benchmark запросов: {len(bm_q)}')
print(f'  точно есть в train: {len(bm_q & tr_q)} ({len(bm_q & tr_q) / len(bm_q):.3f})')
print(f'  НЕТ в train:        {len(bm_q - tr_q)} ({len(bm_q - tr_q) / len(bm_q):.3f})')

print()
print('=' * 70)
print('4. СЕПАРАБЕЛЬНОСТЬ: кликнутые vs случайные айтемы')
print('=' * 70)
item_lookup = {iid: i for i, iid in enumerate(it.item_id.values)}
tr_pairs = tr[['search_query', 'item_id', 'search_location_id',
               'search_category', 'search_infm_params_text']].drop_duplicates(subset=['search_query', 'item_id'])
# только пары, чьи айтемы в корпусе benchmark (чтобы сравнивать в одном пространстве)
tr_pairs = tr_pairs[tr_pairs.item_id.isin(item_lookup)]
print(f'пар под анализ: {len(tr_pairs)}')

loc = it.set_index('item_id')


def stats_for(df, label):
    idx = [item_lookup[i] for i in df.item_id]
    rows = it.iloc[idx]
    loc_match = (rows.item_location_id.values == df.search_location_id.values).astype(float)
    cat_match = (rows.item_category_id.values == df.search_category.values).astype(float)
    return {
        'группа': label,
        'loc_match': np.nanmean(loc_match),
        'geo_dist<25km': np.nanmean(rows.item_latitude.values != -1),
        'cat_match': np.nanmean(cat_match),
        'price_med': np.nanmedian(rows.item_price.values),
        'rating_med': np.nanmedian(rows.item_rating.values),
        'reviews_med': np.nanmedian(rows.item_rating_reviews_count.values),
        'phone_hidden': np.nanmean(rows.item_is_phone_hidden.values),
    }


n = min(len(tr_pairs), 20000)
sample_pos = tr_pairs.sample(n, random_state=42)
# случайные айтемы на те же запросы
rand_items = rng.choice(it.item_id.values, size=n)
sample_neg = sample_pos.copy()
sample_neg['item_id'] = rand_items
# для негативов нужны item-признаки, берём из корпуса
comp = pd.DataFrame({
    'группа': ['кликнутые', 'случайные'],
    'loc_match': [
        (it.iloc[[item_lookup[i] for i in sample_pos.item_id]].item_location_id.values
         == sample_pos.search_location_id.values).mean(),
        (it.iloc[[item_lookup[i] for i in sample_neg.item_id]].item_location_id.values
         == sample_neg.search_location_id.values).mean(),
    ],
})
pos_rows = it.iloc[[item_lookup[i] for i in sample_pos.item_id]]
neg_rows = it.iloc[[item_lookup[i] for i in sample_neg.item_id]]
comp['cat_match'] = [
    (pos_rows.item_category_id.values == sample_pos.search_category.values).mean(),
    (neg_rows.item_category_id.values == sample_neg.search_category.values).mean(),
]
comp['price_med'] = [np.nanmedian(pos_rows.item_price), np.nanmedian(neg_rows.item_price)]
comp['rating_med'] = [np.nanmedian(pos_rows.item_rating), np.nanmedian(neg_rows.item_rating)]
comp['reviews_med'] = [np.nanmedian(pos_rows.item_rating_reviews_count),
                       np.nanmedian(neg_rows.item_rating_reviews_count)]
comp['reviews_mean'] = [np.nanmean(pos_rows.item_rating_reviews_count),
                        np.nanmean(neg_rows.item_rating_reviews_count)]
comp['phone_hidden'] = [pos_rows.item_is_phone_hidden.mean(), neg_rows.item_is_phone_hidden.mean()]
comp['msg_forbidden'] = [pos_rows.item_is_message_forbidden.mean(), neg_rows.item_is_message_forbidden.mean()]
print(comp.to_string(index=False))

print()
print('=' * 70)
print('5. ЗАПРОСЫ: длина и локация')
print('=' * 70)
qlen_tr = tr.search_query.astype(str).str.split().str.len()
qlen_bm = q.search_query.astype(str).str.split().str.len()
print(f'длина запроса train: median={qlen_tr.median()}, p90={qlen_tr.quantile(0.9)}, max={qlen_tr.max()}')
print(f'длина запроса bm:    median={qlen_bm.median()}, p90={qlen_bm.quantile(0.9)}, max={qlen_bm.max()}')
print(f'train запросов с 1 словом: {(qlen_tr == 1).mean():.3f}; bm: {(qlen_bm == 1).mean():.3f}')
print()
top_loc_q = q.search_location_id.value_counts().head(5)
print('топ-5 локаций benchmark-запросов:')
print(top_loc_q.to_string())
print(f'доля запросов в топ-локации: {top_loc_q.iloc[0] / len(q):.3f}')
print()
# сколько айтемов в локации топ-запросов
loc_counts = it.item_location_id.value_counts()
print(f'уникальных локаций в корпусе: {len(loc_counts)}')
med_pool = q.search_location_id.map(loc_counts)
print(f'размер пула айтемов в локации запроса: median={med_pool.median()}, mean={med_pool.mean():.0f}')

print()
print('=' * 70)
print('6. ПОЛЕ / КАТЕГОРИЯ: пересечение query <-> item')
print('=' * 70)
# насколько params текст запроса совпадает с item params
sub = tr_pairs.sample(min(5000, len(tr_pairs)), random_state=1)


def poverlap(qparams, iparams):
    a = set(str(qparams).split())
    b = set(str(iparams).split())
    return len(a & b) / max(1, len(a))


pos_ov = [poverlap(a, b) for a, b in zip(sub.search_infm_params_text.fillna(''),
                                         it.iloc[[item_lookup[i] for i in sub.item_id]].item_infm_params_text.fillna(''))]
neg_ov = [poverlap(a, b) for a, b in zip(sub.search_infm_params_text.fillna(''),
                                         it.iloc[[item_lookup[i] for i in rng.choice(it.item_id.values, size=len(sub))]].item_infm_params_text.fillna(''))]
print(f'params_overlap кликнутые: mean={np.mean(pos_ov):.3f}')
print(f'params_overlap случайные: mean={np.mean(neg_ov):.3f}')

print()
print('=' * 70)
print('7. КОЛИЧЕСТВО КЛИКОВ НА ЗАПРОС (уверенность сигнала)')
print('=' * 70)
q_clicks = tr.groupby('search_query').size()
print(q_clicks.describe().to_string())
print(f'доля запросов с >=3 кликами: {(q_clicks >= 3).mean():.3f}')
