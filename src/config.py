"""Конфигурация проекта: пути и константы."""
import os

# Пути относительно корня проекта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')
WORK_DIR = os.path.join(PROJECT_ROOT, 'work')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'multilingual-e5-small')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# Базовые константы
SEED = 42
TOP_K = 50

# --- Кандидатогенерация ---
# Берём BM25 top-N и Dense top-N, объединяем (dense вперёд, затем bm25).
# Анализ на val: потолок recall@50 при 150+150 = 0.427, при 300+300 = 0.62.
CAND_BM25 = 300
CAND_DENSE = 300
# Сколько максимум кандидатов оставляем на запрос после объединения
POOL_SIZE = 500

# --- Обучение ранкера ---
MAX_NEG = 40
MAX_GROUPS = 30000

# --- Модели ---
BATCH = 128
MAX_SEQ = 256
BM25_K = 1.2
BM25_B = 0.75
FIELD_WEIGHTS = {'title': 3.0, 'desc': 1.0, 'params': 2.0}
MODEL_NAME = 'ranker_model_cat.pkl'
CORPUS_SIZE = 189000
VAL_FRAC = 0.2

# --- Буст по train-сигналу ---
# Для запросов, которые встречаются в train, айтемы, выбранные там,
# получают бонус к скору ранкера. Boost = BOOST_WEIGHT * log1p(count).
# Анализ: 37% benchmark-запросов есть в train, для 14.7% известны
# релевантные айтемы в корпусе.
TRAIN_BOOST_WEIGHT = 5.0

# Приоритет позиции в пуле. Кандидаты упорядочены по dense-сходству,
# поэтому позиция в пуле — это информация от ретривера, которую ранкер
# (не видящий скоров ретривала) может недооценивать.
# Офлайн-замеры: 0.01-0.02 оптимально, больше — начинает вредить.
POOL_POS_WEIGHT = 0.02

# Убедимся, что рабочая директория существует
os.makedirs(WORK_DIR, exist_ok=True)
