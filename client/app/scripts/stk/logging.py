"""Logging helpers."""

import functools
import traceback

import qi


def _safe_convert(x):
    try:
        if isinstance(x, unicode):
            return x.encode("utf-8")
        elif isinstance(x, str):
            return x  # already bytes
        elif isinstance(x, (list, tuple)):
            return type(x)(_safe_convert(i) for i in x)
        elif isinstance(x, dict):
            return {_safe_convert(k): _safe_convert(v) for k, v in x.items()}
        else:
            return str(x)
    except Exception:
        return repr(x)


class SafeLogger(object):

    def __init__(self, logger):
        self._logger = logger

    def _wrap(self, method, msg, *args):
        msg = _safe_convert(msg)
        args = tuple(_safe_convert(a) for a in args)
        return method(msg, *args)

    def info(self, msg, *args):
        return self._wrap(self._logger.info, msg, *args)

    def warning(self, msg, *args):
        return self._wrap(self._logger.warning, msg, *args)

    def error(self, msg, *args):
        return self._wrap(self._logger.error, msg, *args)

    def debug(self, msg, *args):
        return self._wrap(self._logger.debug, msg, *args)

    def __getattr__(self, name):
        return getattr(self._logger, name)


def get_logger(session, app_id):
    base_logger = qi.logging.Logger(app_id)
    try:
        qicore = qi.module("qicore")
        log_manager = session.service("LogManager")
        provider = qicore.createObject("LogProvider", log_manager)
        log_manager.addProvider(provider)
    except Exception:
        pass
    return SafeLogger(base_logger)
