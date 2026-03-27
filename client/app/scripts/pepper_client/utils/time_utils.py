import time


def now_ts():
    return time.time()


def sleep_seconds(seconds):
    try:
        seconds = float(seconds)
    except Exception:
        seconds = 0.0
    if seconds > 0:
        time.sleep(seconds)
