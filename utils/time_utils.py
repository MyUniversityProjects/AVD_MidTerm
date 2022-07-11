import collections
import time
from typing import *
from contextlib import contextmanager


_TIME_MEMORY = collections.defaultdict(list)


@contextmanager
def count_time(name: Text, hide=False):
    start = time.time()
    yield
    elapsed = int((time.time() - start) * 1000)
    mem = _TIME_MEMORY[name]
    mem.append(elapsed)
    mean = int(sum(mem) / len(mem))
    max_val, min_val = max(mem), min(mem)
    if not hide:
        print(f'{name} --> Elapsed=({elapsed}ms) Mean=({mean}ms) Max=({max_val}ms) Min=({min_val}ms)')


@contextmanager
def fixed_time(amt: float):
    start = time.time()
    yield
    sleep_time = amt - (time.time() - start)
    if sleep_time > 0.01:
        time.sleep(sleep_time)
