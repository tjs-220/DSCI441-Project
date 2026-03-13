# -*- coding: utf-8 -*-
"""
DSCI441 Project

utils

Taylor Schultz
"""

import time
import psutil
import os
import functools


def measure_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start:.2f} seconds")
        return result
    return wrapper


def memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"Memory usage: {mem:.2f} MB")
