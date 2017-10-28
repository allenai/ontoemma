import functools
from typing import Callable, Iterator, List, TypeVar

import sys
from emma.utils.config import Config, App, setup_default_logging

T = TypeVar('T')
U = TypeVar('U')


def flatten(lst):
    """Flatten `lst` (return the depth first traversal of `lst`)"""
    out = []
    for v in lst:
        if v is None: continue
        if isinstance(v, list):
            out.extend(flatten(v))
        else:
            out.append(v)
    return out


def once(fn):
    cache = {}

    @functools.wraps(fn)
    def _fn():
        if not 'result' in cache:
            cache['result'] = fn()
        return cache['result']

    return _fn


def display_progress(msg, *args):
    msg = msg % args
    print('\r', msg, file=sys.stdout, end='', flush=True)


def batch_compute(generator: Iterator[T], evaluator: Callable[[List[T]], List[U]], batch_size=256):
    """
    Invoke `evaluator` using batches consumed from `generator`.

    Some functions (e.g. Keras models) are much more efficient when evaluted with large batches of inputs at a time.
    This function simplifies streaming data through these models.

    :param generator:
    :param evaluator:
    :param batch_size:
    :return: generator: results output from `evaluator`
    """
    batch = []

    def _compute():
        if len(batch) > 0:
            resp = evaluator(batch)
            return resp
        else:
            return []

    for item in generator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield from _compute()
            batch = []

    yield from _compute()
