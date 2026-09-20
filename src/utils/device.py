"""Выбор вычислительного устройства."""


def get_device(pref='auto'):
    if pref in ('cpu', 'mps'):
        return pref
    try:
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'
