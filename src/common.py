"""Общие утилиты."""
import os, re, time, pickle
import numpy as np, pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')
WORK_DIR = os.path.join(os.path.dirname(__file__), '..', 'work')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'multilingual-e5-small')
os.makedirs(WORK_DIR, exist_ok=True)

_PUNCT = re.compile(r'[^a-zа-я0-9\s\-]')
_WS = re.compile(r'\s+')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def norm_text(s):
    if not isinstance(s, str):
        return ''
    s = s.lower().replace('ё', 'е')
    s = _PUNCT.sub(' ', s)
    return _WS.sub(' ', s).strip()


def load_raw():
    return (
        pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet')),
        pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet')),
        pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_items.parquet')),
    )


def load_cached(name):
    p = os.path.join(WORK_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def save_cached(obj, name):
    with open(os.path.join(WORK_DIR, name), 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_benchmark_queries():
    return pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet'))


def save_benchmark_queries(df):
    df.to_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet'))


def build_location_lookup():
    _, _, it = load_raw()
    lookup = it.groupby('item_location_id')[['item_latitude', 'item_longitude']].mean().to_dict('index')
    return lookup


def get_device(pref='auto'):
    if pref in ('cpu', 'mps'):
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'
