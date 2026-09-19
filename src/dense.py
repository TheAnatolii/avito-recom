"""Dense retrieval на multilingual-e5-small."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from common import WORK_DIR, MODEL_DIR, load_cached, load_raw, log, get_device
from valid import recall_at_k

EMB_CACHE = 'item_embeddings.npy'
BATCH = 128
MAX_SEQ = 256
SEED = 42


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


def evaluate(sample=500, topn=300, device_pref='auto', seed=SEED):
    t0 = time.time()
    _, val_queries, corpus, _ = load_cached('split_v1.pkl')
    if sample and sample < len(val_queries):
        rng = np.random.default_rng(seed)
        vq = val_queries.iloc[rng.choice(len(val_queries), size=sample, replace=False)].copy()
        log(f'Подвыборка val: {len(vq)}')
    else:
        vq = val_queries
    device = get_device(device_pref)
    log(f'Загрузка модели (device={device})...')
    model = SentenceTransformer(MODEL_DIR, device=device)
    model.max_seq_length = MAX_SEQ
    log(f'Модель загружена, dim={model.get_embedding_dimension()}, {time.time() - t0:.1f}s')
    item_emb = encode_items(model, corpus)
    log(f'Item-эмбеддинги: {item_emb.shape}')
    log(f'Кодирую {len(vq)} запросов...')
    q_emb = encode_queries(model, vq)
    log(f'Query-эмбеддинги: {q_emb.shape}, {time.time() - t0:.1f}s')
    log(f'Считаю топ-{topn}...')
    order = topk_cosine(q_emb, item_emb, k=topn)
    ids = corpus['item_id'].values
    ranked = {row['search_query']: [ids[j] for j in order[i]] for i, (_, row) in enumerate(vq.iterrows())}
    log(f'Готово, {time.time() - t0:.1f}s')
    for k in (10, 50, topn):
        r, rn = recall_at_k(ranked, vq, k)
        log(f'  dense recall@{k}: overall={r:.4f}, new-item={rn:.4f}')


def fine_tune_model():
    log('Fine-tuning e5 на train...')
    tr, _, _ = load_raw()
    examples = []
    for _, row in tr.head(50000).iterrows():
        query_text = 'query: ' + str(row['search_query'] or '') + '. ' + str(row['search_infm_params_text'] or '')
        item_text = ('passage: ' + str(row['item_title_raw'] or '') + '. ' +
                     str(row['item_description_raw'] or '') + '. ' +
                     str(row['item_infm_params_text'] or ''))
        examples.append(InputExample(texts=[query_text, item_text], label=1.0))
    model = SentenceTransformer(MODEL_DIR, device=get_device('auto'))
    model.max_seq_length = MAX_SEQ
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    model.fit(
        train_objectives=[(train_dataloader, losses.MultipleNegativesRankingLoss(model))],
        epochs=10, warmup_steps=100, show_progress_bar=True,
    )
    model.save('models/multilingual-e5-small-finetuned')
    log('Fine-tuning завершён')


if __name__ == '__main__':
    evaluate()
