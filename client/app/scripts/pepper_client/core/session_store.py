import copy
import threading

from pepper_client.utils import time_utils


class SessionStore(object):
    def __init__(self, logger=None):
        self.logger = logger
        self._lock = threading.RLock()
        self._state = {}
        self.reset_all()

    def reset_all(self):
        with self._lock:
            self._state = {
                "chat_id": None,
                "last_caption": None,
                "last_caption_ts": None,
                "last_detect_ts": None,
                "last_scan_id": None,
                "last_response": None,
                "last_query": None,
                "last_detect_request_id": None,
                "output_language_mode": "default",
                "last_server_base_url": None,
            }

    def reset_conversation(self):
        with self._lock:
            self._state["chat_id"] = None
            self._state["last_response"] = None
            self._state["last_query"] = None
            self._state["last_caption"] = None
            self._state["last_caption_ts"] = None
            self._state["last_detect_request_id"] = None
        if self.logger is not None:
            self.logger.info("Conversation session reset")

    def update_after_caption(self, caption_response):
        with self._lock:
            self.logger.info("Updating SessionStore after caption: %s", caption_response)
            self._state["last_caption"] = caption_response.get("caption")
            self._state["last_caption_ts"] = time_utils.now_ts()
            self._state["last_detect_request_id"] = caption_response.get(
                "detect_request_id"
            )
            self._state["last_response"] = caption_response

    def update_after_detect(self, detect_response, scan_id=None):
        with self._lock:
            self.logger.info("Updating SessionStore after detect: %s", detect_response)
            self._state["last_detect_ts"] = time_utils.now_ts()
            self._state["last_scan_id"] = scan_id
            self._state["last_response"] = detect_response

    def update_after_chat(self, query, chat_response):
        with self._lock:
            self.logger.info("Updating SessionStore after chat: %s", chat_response)
            self._state["chat_id"] = chat_response.get("chat_id")
            self._state["last_query"] = query
            self._state["last_response"] = chat_response

    def set_output_language_mode(self, mode):
        with self._lock:
            self.logger.info("Setting output language mode to %s", mode)
            self._state["output_language_mode"] = mode

    def set_server_base_url(self, value):
        with self._lock:
            self.logger.info("Setting server base url to %s", value)
            self._state["last_server_base_url"] = value

    def get_chat_id(self):
        with self._lock:
            return self._state.get("chat_id")

    def get_output_language_mode(self):
        with self._lock:
            return self._state.get("output_language_mode")

    def needs_visual_refresh(self, ttl_seconds):
        with self._lock:
            last_detect_ts = self._state.get("last_detect_ts")
        if not last_detect_ts:
            return True
        return (time_utils.now_ts() - last_detect_ts) > float(ttl_seconds)

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._state)
