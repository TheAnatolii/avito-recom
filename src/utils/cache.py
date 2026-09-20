"""Кэширование артефактов."""
import os
import pickle
from src.config import WORK_DIR


def load_cached(name):
    p = os.path.join(WORK_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return pickle.load(f)


def save_cached(obj, name):
    with open(os.path.join(WORK_DIR, name), 'wb') as f:
        pickle.dump(obj, f, protocol=4)
