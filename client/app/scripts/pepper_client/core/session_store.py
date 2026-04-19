import copy
import threading

from pepper_client.utils import text as text_utils
from pepper_client.utils import timing as time_utils


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
                "last_server_base_url": None,
                "remembered_labels": [],
                "remembered_attributes": [],
                "remembered_relations": [],
                "last_memory_summary": None,
                "last_memory_summary_ts": None,
                "cached_questions": [],
                "cached_answers": {},
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

    def reset_memory_state(self):
        with self._lock:
            self._state["remembered_labels"] = []
            self._state["remembered_attributes"] = []
            self._state["remembered_relations"] = []
            self._state["last_memory_summary"] = None
            self._state["last_memory_summary_ts"] = None
            self._state["cached_questions"] = []
            self._state["cached_answers"] = {}
        if self.logger is not None:
            self.logger.info("Memory state reset")

    def update_after_caption(self, caption_response):
        with self._lock:
            self.logger.info("Updating SessionStore after caption: %s",
                             caption_response)
            self._state["last_caption"] = caption_response.get("caption")
            self._state["last_caption_ts"] = time_utils.now_ts()
            self._state["last_detect_request_id"] = caption_response.get(
                "detect_request_id")
            self._state["last_response"] = caption_response

    def update_after_detect(self, detect_response, scan_id=None):
        with self._lock:
            self.logger.info("Updating SessionStore after detect: %s",
                             detect_response)
            self._state["last_detect_ts"] = time_utils.now_ts()
            self._state["last_scan_id"] = scan_id
            self._state["last_response"] = detect_response

    def update_after_chat(self, query, chat_response):
        with self._lock:
            self.logger.info("Updating SessionStore after chat: %s",
                             chat_response)
            self._state["chat_id"] = chat_response.get("chat_id")
            self._state["last_query"] = query
            self._state["last_response"] = chat_response

    def set_server_base_url(self, value):
        with self._lock:
            self.logger.info("Setting server base url to %s", value)
            self._state["last_server_base_url"] = value

    def update_after_memory_summary(self, summary):
        summary = summary or {}
        labels = self._sorted_unique(summary.get("labels", []))
        attributes = []
        relations = []
        for edge in summary.get("scene_graph", []) or []:
            if not isinstance(edge, dict):
                continue
            rel = text_utils.clean_text(edge.get("rel"))
            if not rel:
                continue
            sub = text_utils.clean_text(edge.get("sub"))
            obj = text_utils.clean_text(edge.get("obj"))
            if sub and obj and sub == obj:
                attributes.append(rel)
            else:
                relations.append(rel)
        with self._lock:
            self.logger.info("Updating SessionStore after memory summary")
            self._state["remembered_labels"] = labels
            self._state["remembered_attributes"] = self._sorted_unique(
                attributes)
            self._state["remembered_relations"] = self._sorted_unique(
                relations)
            self._state["last_memory_summary"] = summary
            self._state["last_memory_summary_ts"] = time_utils.now_ts()
            self._state["last_response"] = summary

    def update_after_pregenerated_qa(self, qa_response):
        qa_response = qa_response or {}
        pairs = qa_response.get("pregenerated_qa", []) or []
        cached_questions = []
        cached_answers = {}
        for item in pairs:
            if not isinstance(item, dict):
                continue
            question = text_utils.clean_text_unicode(item.get("question"))
            answer = text_utils.clean_text_unicode(item.get("answer"))
            if not question or not answer:
                continue
            cached_questions.append(question)
            cached_answers[question] = answer
        with self._lock:
            self.logger.info(
                "Updating SessionStore after pregenerated QA pairs=%s",
                len(cached_questions),
            )
            self._state["cached_questions"] = self._sorted_unique(
                cached_questions)
            self._state["cached_answers"] = cached_answers

    def get_cached_questions(self):
        with self._lock:
            return list(self._state.get("cached_questions", []))

    def get_cached_answers(self):
        with self._lock:
            return copy.deepcopy(self._state.get("cached_answers", {}))

    def get_memory_labels(self):
        with self._lock:
            return list(self._state.get("remembered_labels", []))

    def get_memory_attributes(self):
        with self._lock:
            return list(self._state.get("remembered_attributes", []))

    def get_memory_relations(self):
        with self._lock:
            return list(self._state.get("remembered_relations", []))

    #REMOVAL never used
    def get_last_memory_summary(self):
        with self._lock:
            return copy.deepcopy(self._state.get("last_memory_summary"))

    def get_chat_id(self):
        with self._lock:
            return self._state.get("chat_id")

    def needs_visual_refresh(self, ttl_seconds):
        with self._lock:
            last_detect_ts = self._state.get("last_detect_ts")
        if not last_detect_ts:
            return True
        return (time_utils.now_ts() - last_detect_ts) > float(ttl_seconds)

    def snapshot(self):
        with self._lock:
            return copy.deepcopy(self._state)

    def _sorted_unique(self, values):
        seen = set()
        output = []
        for value in values or []:
            text = text_utils.clean_text(value)
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            output.append(text)
        output.sort()
        return output
