"""System-related helpers."""

from __future__ import annotations
import os
import psutil

def cpu_count():
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count is None:
            cpu_count = psutil.cpu_count(logical=True)

    return cpu_count
