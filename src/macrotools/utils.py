import time
from contextlib import contextmanager
from functools import wraps

def timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} execution time: {elapsed_time:.2f} seconds")
        return result
    return wrapper


@contextmanager
def progress(message: str):
    """
    Print `message` (without newline), then on exit append ` done in X.XXs`
    so the start and end land on the same line.
    """
    print(message, end='', flush=True)
    start = time.time()
    try:
        yield
    finally:
        print(f' done in {time.time() - start:.2f}s')
