"""BM25 ретривер."""
import os, sys, time
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__))
from common import load_cached, save_cached, log, norm_text
from valid import recall_at_k

SEED = 42
FIELD_WEIGHTS = {'title': 3.0, 'desc': 1.0, 'params': 2.0}
BM25_K = 1.2
BM25_B = 0.75


class Lemmatizer:
    def __init__(self):
        import pymorphy3
        self._morph = pymorphy3.MorphAnalyzer()
        self._cache = {}

    def doc_tokens(self, text):
        return [self._lemmatize(t) for t in norm_text(text).split() if len(t) > 1]

    def _lemmatize(self, word):
        if word not in self._cache:
            self._cache[word] = self._morph.parse(word)[0].normal_form
        return self._cache[word]


class BM25Index:
    def __init__(self, k1=BM25_K, b=BM25_B):
        self.k1, self.b = k1, b
        self.vocab = {}
        self.idf = None
        self.mat = None
        self.doc_len = None
        self.avgdl = None

    def fit(self, tokenized_docs):
        rows, cols, vals = [], [], []
        doc_len = np.zeros(len(tokenized_docs), dtype=np.float32)
        for d, toks in enumerate(tokenized_docs):
            doc_len[d] = len(toks)
            counts = {}
            for t in toks:
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
                tid = self.vocab[t]
                counts[tid] = counts.get(tid, 0) + 1
            for c, v in counts.items():
                rows.append(d)
                cols.append(c)
                vals.append(v)
        n_docs = len(tokenized_docs)
        self.mat = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_docs, len(self.vocab)),
            dtype=np.float32,
        )
        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean())
        df = np.asarray((self.mat > 0).sum(axis=0)).ravel()
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        return self

    def score(self, query_tokens, topn=None):
        qcols = [self.vocab[t] for t in query_tokens if t in self.vocab]
        if not qcols:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        qcols = list(dict.fromkeys(qcols))
        tf = self.mat[:, qcols].toarray()
        idf_q = self.idf[qcols]
        denom = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        scores = (tf * (self.k1 + 1)) / (tf + denom[:, None])
        scores = scores @ idf_q
        order = np.argsort(scores)[::-1]
        return (order[:topn], scores[order[:topn]]) if topn else (order, scores)


class BM25Retriever:
    def __init__(self, corpus, lem, field_weights=FIELD_WEIGHTS):
        self.corpus = corpus.reset_index(drop=True)
        self.ids = self.corpus['item_id'].values
        self.lem = lem
        self.field_weights = field_weights
        self.indexes = {}
        cache_path = 'bm25_index.pkl'
        cached = load_cached(cache_path)
        if cached is not None:
            self.indexes = cached['indexes']
            self.ids = cached['ids']
            self.corpus = cached['corpus']
            log('Загружен кэш BM25-индекса')
            return
        t0 = time.time()
        fields = {
            'title': self.corpus['item_title_raw'].fillna('').astype(str),
            'desc': self.corpus['item_description_raw'].fillna('').astype(str),
            'params': self.corpus['item_infm_params_text'].fillna('').astype(str),
        }
        for field, series in fields.items():
            toks = [lem.doc_tokens(s) for s in series]
            self.indexes[field] = BM25Index().fit(toks)
            log(f'BM25 "{field}": vocab={len(self.indexes[field].vocab)}, {time.time() - t0:.1f}s')
        save_cached({'indexes': self.indexes, 'ids': self.ids, 'corpus': self.corpus}, 'bm25_index.pkl')

    def retrieve(self, query_text, topn=150, weights=None):
        qt = self.lem.doc_tokens(query_text)
        if not qt:
            return []
        weights = weights or self.field_weights
        combined = None
        for field, w in weights.items():
            idx, sc = self.indexes[field].score(qt, topn=topn * 2)
            if len(idx) == 0:
                continue
            vec = np.zeros(len(self.ids), dtype=np.float32)
            vec[idx] = sc * w
            combined = vec if combined is None else combined + vec
        if combined is None:
            return []
        order = np.argsort(combined)[::-1][:topn]
        return [self.ids[i] for i in order]


def evaluate(sample=500, topn=150, seed=SEED):
    t0 = time.time()
    _, val_queries, corpus, _ = load_cached('split_v1.pkl')
    if sample and sample < len(val_queries):
        rng = np.random.default_rng(seed)
        vq = val_queries.iloc[rng.choice(len(val_queries), size=sample, replace=False)].copy()
        log(f'Подвыборка val: {len(vq)}')
    else:
        vq = val_queries
    log('Построение BM25...')
    retriever = BM25Retriever(corpus, Lemmatizer())
    log(f'Индексы готовы, {time.time() - t0:.1f}s')
    log(f'Ретривал для {len(vq)} запросов...')
    ranked = {row['search_query']: retriever.retrieve(row['search_query'], topn=topn) for _, row in vq.iterrows()}
    log(f'Ретривал завершён, {time.time() - t0:.1f}s')
    for k in (10, 50, topn):
        r, rn = recall_at_k(ranked, vq, k)
        log(f'  bm25 recall@{k}: overall={r:.4f}, new-item={rn:.4f}')


if __name__ == '__main__':
    evaluate()
