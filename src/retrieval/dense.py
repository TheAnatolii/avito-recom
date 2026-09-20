"""Dense retrieval на multilingual-e5-small."""
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from src.utils.logging import log
from src.utils.cache import load_cached
from src.config import WORK_DIR, MODEL_DIR, BATCH, MAX_SEQ

EMB_CACHE = 'item_embeddings.npy'


def item_text(row):
    return ('passage: ' + str(row['item_title_raw'] or '') + '. ' +
            str(row['item_description_raw'] or '') + '. ' +
            str(row['item_infm_params_text'] or ''))


def query_text(row):
    return ('query: ' + str(row['search_query'] or '') + '. ' +
            str(row['search_infm_params_text'] or ''))


def encode_items(model, corpus, batch=BATCH):
    cache = os.path.join(WORK_DIR, EMB_CACHE)
    if os.path.exists(cache):
        log('Загружаю кэш эмбеддингов')
        return np.load(cache)
    texts = [item_text(r) for _, r in corpus.iterrows()]
    log(f'Кодирую {len(texts)} айтемов...')
    t0 = time.time()
    emb = model.encode(texts, batch_size=batch, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(cache, emb)
    log(f'Готово: {emb.shape}, {time.time() - t0:.1f}s')
    return emb


def encode_queries(model, queries, batch=BATCH):
    texts = [query_text(r) for _, r in queries.iterrows()]
    emb = model.encode(texts, batch_size=batch, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb, dtype=np.float32)


def fine_tune_model():
    log('Fine-tuning e5 на train...')
    from src.utils.loader import load_raw
    from sentence_transformers import InputExample, losses
    from torch.utils.data import DataLoader
    from src.utils.device import get_device
    tr, _, _ = load_raw()
    examples = []
    for _, row in tr.head(50000).iterrows():
        query_text_str = 'query: ' + str(row['search_query'] or '') + '. ' + str(row['search_infm_params_text'] or '')
        item_text_str = ('passage: ' + str(row['item_title_raw'] or '') + '. ' +
                         str(row['item_description_raw'] or '') + '. ' +
                         str(row['item_infm_params_text'] or ''))
        examples.append(InputExample(texts=[query_text_str, item_text_str], label=1.0))
    model = SentenceTransformer(MODEL_DIR, device=get_device('auto'))
    model.max_seq_length = MAX_SEQ
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    model.fit(
        train_objectives=[(train_dataloader, losses.MultipleNegativesRankingLoss(model))],
        epochs=1, warmup_steps=100, show_progress_bar=True,
    )
    model.save('models/multilingual-e5-small-finetuned')
    log('Fine-tuning завершён')


def topk_cosine(q_emb, item_emb, k=300, chunk=200):
    n_q = q_emb.shape[0]
    order_all = np.empty((n_q, k), dtype=np.int64)
    for start in range(0, n_q, chunk):
        stop = min(start + chunk, n_q)
        sims = q_emb[start:stop] @ item_emb.T
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        selected_scores = np.take_along_axis(sims, idx, axis=1)
        sort_order = np.argsort(-selected_scores, axis=1)
        order_all[start:stop] = np.take_along_axis(idx, sort_order, axis=1)
    return order_all
