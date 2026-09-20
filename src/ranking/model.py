"""Загрузка и сохранение модели ранкера."""
from src.utils.cache import load_cached, save_cached
from src.config import MODEL_NAME


def load_model():
    return load_cached(MODEL_NAME)


def save_model(model):
    save_cached(model, MODEL_NAME)
