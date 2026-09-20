"""Единая точка входа: сплит → обучение ранкера → предсказание."""
import sys
from src.utils.logging import log
from src.utils.device import get_device
from src.data.split import get_or_build_split
from src.ranking.train import train
from src.pipeline.predict import main as predict_main

sys.path.insert(0, '.')


def main():
    device = get_device('auto')
    log(f'[START] Пайплайн запущен, device={device}')

    log('[STAGE 1/3] Построение сплита данных...')
    train_pairs, val_queries, corpus, seen_ids = get_or_build_split()
    log(f'[STAGE 1/3] Сплит готов: val_queries={len(val_queries)}, '
        f'corpus_items={len(corpus)}, seen_ids={len(seen_ids)}')

    log('[STAGE 2/3] Обучение CatBoost ранкера...')
    train()
    log('[STAGE 2/3] Ранкер обучен и сохранён в work/ranker_model_cat.pkl')

    log('[STAGE 3/3] Генерация кандидатов и ранжирование...')
    predict_main()
    log('[STAGE 3/3] Предсказание завершено, answer.csv готов')

    log('[FINISH] Пайплайн завершён успешно')


if __name__ == '__main__':
    main()
