"""Logging helpers."""

import functools
import traceback

import qi



def get_logger(session, app_id):
    logger = qi.logging.Logger(app_id)
    try:
        qicore = qi.module("qicore")
        log_manager = session.service("LogManager")
        provider = qicore.createObject("LogProvider", log_manager)
        log_manager.addProvider(provider)
    except Exception:
        pass
    return logger



def log_exceptions(func):
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception:
            self.logger.error(traceback.format_exc())
            raise

    return wrapped
