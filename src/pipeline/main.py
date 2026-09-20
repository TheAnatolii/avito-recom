"""Оркестрация пайплайна."""
from src.utils.logging import log
from src.utils.device import get_device
from src.data.split import get_or_build_split
from src.pipeline.predict import main as predict_main


def stage_split():
    train_pairs, val_queries, corpus, seen_ids = get_or_build_split()
    log(f'val={len(val_queries)}, corpus={len(corpus)}')


def stage_predict():
    predict_main()


def main():
    device = get_device('auto')
    log(f'Пайплайн: device={device}')
    stage_split()
    stage_predict()
    log('Пайплайн завершён')


if __name__ == '__main__':
    main()
