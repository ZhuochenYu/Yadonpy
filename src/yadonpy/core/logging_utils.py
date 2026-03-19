"""Lightweight printing/logging helpers.

This module is used by many parts of YadonPy very early in the import chain.
Keep it dependency-light and avoid importing heavy modules here.
"""

from __future__ import annotations

from . import const
from .exceptions import YadonPyError

def yadon_print(text, level=0):
    """Unified console logger.

    Levels:
      0 debug, 1 info, 2 warning, 3 error (raise)
    """
    if level == 0:
        text = 'YadonPy debug info: ' + str(text)
    elif level == 1:
        text = 'YadonPy info: ' + str(text)
    elif level == 2:
        text = 'YadonPy warning: ' + str(text)
    elif level == 3:
        raise YadonPyError(str(text))

    if level >= const.print_level or const.debug:
        print(text, flush=True)


def radon_print(text, level=0):
    return yadon_print(text, level=level)


def tqdm_stub(it, **kwargs):
    return it
