"""Единая точка входа."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, get_device

T0 = time.time()


def stage_split():
    from valid import get_or_build_split
    train_pairs, val_queries, corpus, seen_ids = get_or_build_split()
    log(f'val={len(val_queries)}, corpus={len(corpus)}')


def stage_bm25():
    import bm25
    bm25.evaluate()


def stage_dense():
    import dense
    dense.evaluate()


def stage_ranker():
    from ranker import train as train_ranker
    train_ranker()


def stage_predict():
    from predict import main as predict_main
    predict_main()


def main():
    device = get_device('auto')
    log(f'Пайплайн: device={device}')
    stage_split()
    stage_bm25()
    stage_dense()
    stage_ranker()
    stage_predict()
    log(f'Полное время: {time.time() - T0:.1f}s')


if __name__ == '__main__':
    main()
