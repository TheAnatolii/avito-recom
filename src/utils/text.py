"""Нормализация текста и лемматизация."""
import re

_PUNCT = re.compile(r'[^a-zа-я0-9\s\-]')
_WS = re.compile(r'\s+')


def norm_text(s):
    if not isinstance(s, str):
        return ''
    s = s.lower().replace('ё', 'е')
    s = _PUNCT.sub(' ', s)
    return _WS.sub(' ', s).strip()


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
