"""Логирование."""
import time


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)
