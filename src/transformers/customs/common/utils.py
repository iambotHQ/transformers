import signal
import subprocess as sp
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import AnyStr, Iterable, List, TypeVar, Union


T = TypeVar("T")

PathLike = Union[AnyStr, Path]


def raise_timeout(signum, frame):
    raise TimeoutError


@contextmanager
def timeout(time: int):
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def override(func):
    return func


def run(*args) -> str:
    retval = sp.check_output(list(map(str, args)))
    return str(retval, encoding="utf8")


def wc(filename: PathLike) -> int:
    return int(run("wc", "-l", filename).split()[0])


def get_specific_line(filepath: PathLike, line_no: int) -> str:
    return run("sed", "-n", f"{line_no + 1}p", filepath)


def split_by_key(iterable: Iterable[T], *keys: T, skip_key: bool = True) -> List[List[T]]:
    splits: List[List[T]] = []
    split: List[T] = []

    for item in iterable:
        if item in keys:
            if split:
                splits.append(deepcopy(split))

            if not skip_key:
                splits.append([item])

            split.clear()
        else:
            split.append(item)

    if split:
        splits.append(split)

    return splits


def ids_of_key(iterable: Iterable[T], *keys: T, not_key: bool = False) -> List[int]:
    if not not_key:
        pred = lambda item: item in keys
    else:
        pred = lambda item: item not in keys

    return [idx for idx, item in enumerate(iterable) if pred(item)]
