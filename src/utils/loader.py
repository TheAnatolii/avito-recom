"""Загрузка данных."""
import os
import pandas as pd
from src.config import DATA_DIR


def load_raw():
    return (
        pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet')),
        pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet')),
        pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_items.parquet')),
    )


def load_benchmark_queries():
    return pd.read_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet'))


def save_benchmark_queries(df):
    df.to_parquet(os.path.join(DATA_DIR, 'benchmark_queries.parquet'))


def build_location_lookup():
    _, _, it = load_raw()
    lookup = it.groupby('item_location_id')[['item_latitude', 'item_longitude']].mean().to_dict('index')
    return lookup
