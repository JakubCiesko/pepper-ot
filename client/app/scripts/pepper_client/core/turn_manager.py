# -*- coding: UTF-8 -*-

import threading

from pepper_client.interaction import speech_policy
from pepper_client.perception import scan_planner
from pepper_client.utils import ids
from pepper_client.utils import text as text_utils
from pepper_client.utils import timing as time_utils
from pepper_client.utils.error_policy import (CameraCaptureError,
                                              MalformedResponseError,
                                              ServerTimeoutError,
                                              ServerUnavailableError,
                                              fallback_message)


class TurnManager(object):
    """
    Coordinates one robot interaction turn at a time. A turn may capture images,
    move the head, call the server, update memory concepts, speak, and update the
    tablet. This class owns concurrency and failure handling.
    """


    def __init__(
        self,
        config,
        session_store,
        camera_adapter,
        pose_adapter,
        robot_context,
        metadata_builder,
        transport,
        speech_adapter,
        tablet_adapter,
        dialog_adapter,
        logger,
    ):
        self.config = config
        self.session_store = session_store
        self.camera_adapter = camera_adapter
        self.pose_adapter = pose_adapter
        self.robot_context = robot_context
        self.metadata_builder = metadata_builder
        self.transport = transport
        self.speech_adapter = speech_adapter
        self.tablet_adapter = tablet_adapter
        self.dialog_adapter = dialog_adapter
        self.logger = logger
        self._lock = threading.RLock()
        self._busy = False
        self._active_turn = None
        self._turn_local = threading.local()

    def start_look(self, lang_code):
        return self._start_async("look", lang_code, self._run_look, lang_code)

    def start_scan(self, lang_code):
        return self._start_async("scan", lang_code, self._run_scan, lang_code)

    def start_ask(self, lang_code, query, force_refresh=False):
        return self._start_async(
            "ask",
            lang_code,
            self._run_ask,
            lang_code,
            query,
            bool(force_refresh),
        )

    def start_object_ask(self, lang_code, object_label, query=None):
        return self._start_async(
            "object_ask",
            lang_code,
            self._run_object_ask,
            lang_code,
            object_label,
            query,
        )

    def start_show_memory(self, lang_code):
        return self._start_async(
            "show_memory",
            lang_code,
            self._run_show_memory,
            lang_code,
        )

    def start_show_memory_and_suggest_questions(self, lang_code):
        return self._start_async(
            "show_memory_and_suggest_questions",
            lang_code,
            self._run_show_memory_and_suggest_questions,
            lang_code,
        )

    def start_cached_answer(self, lang_code, query):
        return self._start_async(
            "cached_answer",
            lang_code,
            self._run_cached_answer,
            lang_code,
            query,
        )

    def start_reset_memory(self, lang_code):
        return self._start_async(
            "reset_memory",
            lang_code,
            self._run_reset_memory,
            lang_code,
        )

    def refresh_memory_concepts(self, lang_code=None):
        try:
            self._refresh_dynamic_concepts_from_server(lang_code)
            return True
        except Exception as exc:
            self.logger.warning("Failed to refresh memory concepts: %s", exc)
            return False

    def status(self):
        snapshot = self.session_store.snapshot()
        with self._lock:
            snapshot["busy"] = self._busy
            snapshot["active_turn"] = self._active_turn
        return snapshot

    def shutdown(self):
        self.logger.info("Turn manager shutdown requested")
        self.speech_adapter.stop()

    def _start_async(self, kind, lang_code, target, *args):
        """
        starts a deamon with safe execution of method with provided arguments.
        if previous turn is still active, emits busy message
        """
        turn_id = ids.new_turn_id(kind)
        started_at = time_utils.now_ts()
        speech_mode = self._speech_request_language(lang_code)
        runtime_lang = self._speech_lang(lang_code)
        with self._lock:
            if self._busy:
                self.logger.info(
                    "Rejecting %s because another turn is active: %s",
                    kind,
                    self._active_turn,
                )
                self._safe_say(fallback_message("busy", runtime_lang),
                               speech_mode,
                               phase="busy")
                return False
            self._busy = True
            self._active_turn = {
                "id": turn_id,
                "kind": kind,
                "started_at": started_at,
            }
        self._log_latency_event(
            "turn_started",
            turn_id=turn_id,
            kind=kind,
            started_at=started_at,
        )
        if self.config["behavior"].get("speak_acknowledgements", True):
            ack = speech_policy.acknowledgement(kind, runtime_lang)
            if ack:
                self._safe_say(
                    ack,
                    speech_mode,
                    phase="ack",
                    turn_id=turn_id,
                    kind=kind,
                    started_at=started_at,
                )
        thread = threading.Thread(
            target=self._run_guarded,
            name="pepper-turn-%s" % turn_id,
            args=(turn_id, kind, lang_code, started_at, target, args),
        )
        thread.setDaemon(True)
        thread.start()
        return True

    def _run_guarded(self, turn_id, kind, lang_code, started_at, target, args):
        """
        records turn-local latency context, maps expected failures to user-friendly speech, and always clears the busy flag.
        """
        self._set_current_turn(turn_id, kind, started_at)
        self.logger.info("Starting turn id=%s kind=%s", turn_id, kind)
        try:
            target(*args)
            self.logger.info("Completed turn id=%s kind=%s", turn_id, kind)
        except CameraCaptureError as exc:
            self.logger.warning("Camera error in turn %s: %s", turn_id, exc)
            self._safe_say(
                fallback_message("camera", self._speech_lang(lang_code)),
                self._speech_request_language(lang_code),
                phase="error",
            )
        except ServerTimeoutError as exc:
            self.logger.warning("Server timeout in turn %s: %s", turn_id, exc)
            self._safe_say(
                fallback_message("server_timeout",
                                 self._speech_lang(lang_code)),
                self._speech_request_language(lang_code),
                phase="error",
            )
        except ServerUnavailableError as exc:
            self.logger.warning("Server unavailable in turn %s: %s", turn_id,
                                exc)
            self._safe_say(
                fallback_message("server_unavailable",
                                 self._speech_lang(lang_code)),
                self._speech_request_language(lang_code),
                phase="error",
            )
        except MalformedResponseError as exc:
            self.logger.warning("Malformed server response in turn %s: %s",
                                turn_id, exc)
            self._safe_say(
                fallback_message("malformed", self._speech_lang(lang_code)),
                self._speech_request_language(lang_code),
                phase="error",
            )
        except Exception as exc:
            self.logger.error(
                "Unexpected turn failure id=%s kind=%s, message=%s",
                turn_id,
                kind,
                exc,
            )
            self._safe_say(
                fallback_message("unexpected", self._speech_lang(lang_code)),
                self._speech_request_language(lang_code),
                phase="error",
            )
        finally:
            self._clear_current_turn()
            with self._lock:
                self._busy = False
                self._active_turn = None

    def _run_look(self, lang_code):
        """
        Quick visual turn: capture one frame, send it to the caption endpoint, cache the response, 
        speak the caption, then optionally refresh dialog concepts.
        """
        self.logger.info("Running look method with lang code: %s", lang_code)
        runtime_lang = self._speech_lang(lang_code)
        speech_mode = self._speech_request_language(lang_code)
        frame_id = ids.new_frame_id(self.config["capture"].get(
            "frame_prefix", "frame"))
        capture, metadata = self._capture_with_metadata(
            frame_id, None, "caption")
        run_detect = bool(self.config["behavior"].get("caption_run_detect",
                                                      True))
        publish = bool(self.config["server"].get("publish", True))
        language = speech_policy.server_language_for_runtime(runtime_lang)
        self._log_latency_event("server_request_start", phase="caption")
        caption_response = self._caption_with_optional_retry(
            capture["image_bytes"],
            metadata,
            run_detect=run_detect,
            publish=publish,
            language=language,
        )
        self._log_latency_event("server_response_received", phase="caption")
        self.session_store.update_after_caption(caption_response)
        self._safe_say(caption_response["caption"],
                       speech_mode,
                       phase="answer")

        if self.config.get("dialog", {}).get("refresh_after_detect", True):
            self._refresh_dynamic_concepts_from_server(lang_code, runtime_lang)

    def _run_scan(self, lang_code):
        """
        Slow multiturn visual look: sweep the head through configured yaw positions, run detection in
        panorama or sequential mode, restore the original head pose, and optionally
        ask the server for a spoken memory summary.
        """
        runtime_lang = self._speech_lang(lang_code)
        speech_mode = self._speech_request_language(lang_code)
        scan_id = ids.new_scan_id(self.config["capture"].get(
            "scan_prefix", "scan"))
        original_pose = self.pose_adapter.snapshot()
        self.logger.info("Starting scan sweep scan_id=%s", scan_id)
        try:
            scan_mode = scan_planner.scan_mode(self.config)
            if scan_mode == "panorama_detect":
                self._run_scan_panorama(scan_id)
            else:
                self._run_scan_sequential(scan_id)
        finally:
            if self.config["behavior"].get("auto_restore_head_pose", True):
                self.pose_adapter.restore_head(
                    original_pose,
                    self.config["capture"].get("head_move_speed", 0.15),
                )

        if self.config.get("dialog", {}).get("refresh_after_scan", True):
            self._refresh_dynamic_concepts_from_server(lang_code, runtime_lang)

        if scan_planner.summary_after_scan(self.config):
            query = self._scan_summary_query(runtime_lang)
            self._log_latency_event("server_request_start", phase="chat")
            chat_response = self.transport.chat_general(
                query,
                self.session_store.get_chat_id(),
                runtime_lang,
            )
            self._log_latency_event("server_response_received", phase="chat")
            self.session_store.update_after_chat(query, chat_response)
            self._safe_say(chat_response["sentence"],
                           speech_mode,
                           phase="answer")
        else:
            self._safe_say(
                speech_policy.generic_message("scan_complete", runtime_lang),
                speech_mode,
                phase="answer",
            )

    def _run_scan_panorama(self, scan_id):
        captures = self._prepare_scan_captures(scan_id)
        stick_together = scan_planner.stick_together(self.config)
        response = self.transport.panorama_detect(
            captures,
            publish=bool(self.config["server"].get("publish", True)),
            resize_image=True,
            stick_together=stick_together,
        )
        self.session_store.update_after_detect(response, scan_id=scan_id)
        self.logger.info(
            "Panorama scan completed objects=%s stick_together=%s",
            len(response.get("objects", [])),
            stick_together,
        )

    def _run_scan_sequential(self, scan_id):
        successes = 0
        last_error = None
        captures = self._prepare_scan_captures(scan_id)
        for capture_item in captures:
            index = capture_item["index"]
            try:
                detect_response = self.transport.detect(
                    capture_item["image_bytes"],
                    capture_item["metadata"],
                    publish=bool(self.config["server"].get("publish", True)),
                )
                self.session_store.update_after_detect(detect_response,
                                                       scan_id=scan_id)
                successes += 1
                self.logger.info(
                    "Scan frame %s completed objects=%s",
                    index,
                    len(detect_response.get("objects", [])),
                )
            except Exception as exc:
                last_error = exc
                self.logger.warning("Scan frame %s failed: %s", index, exc)
        if successes <= 0:
            if last_error is not None:
                raise last_error
            raise ServerUnavailableError("No scan frame completed")

    def _prepare_scan_captures(self, scan_id):
        """
        Move the head through the configured scan angles and collect image+metadata
        payloads. The server uses scan_id and capture_mode to group these captures. Uses pose adapter.
        """
        captures = []
        scan_steps = scan_planner.planned_yaws_radians(self.config)
        pitch = scan_planner.scan_pitch(self.config)
        settle_seconds = self.config["capture"].get("settle_seconds", 0.6)

        for index, yaw in enumerate(scan_steps):
            self.pose_adapter.move_head(
                yaw,
                pitch,
                self.config["capture"].get("head_move_speed", 0.15),
            )
            time_utils.sleep_seconds(settle_seconds)
            frame_id = ids.new_frame_id(self.config["capture"].get(
                "frame_prefix", "frame"))
            capture, metadata = self._capture_with_metadata(
                frame_id, scan_id, "scan")
            captures.append({
                "index": index,
                "image_bytes": capture["image_bytes"],
                "metadata": metadata,
            })
        return captures

    def _run_ask(self, lang_code, query, force_refresh):
        """
        General chat turn. It may refresh visual memory first if the last detection is
        too old, then sends the user query with the current chat_id.
        """
        runtime_lang = self._speech_lang(lang_code)
        speech_mode = self._speech_request_language(lang_code)
        max_chars = int(self.config["behavior"].get("max_query_chars", 320))
        query = text_utils.sanitize_query(query, max_chars)
        self.logger.info("Starting ask with sanitized query %s", query)
        if not query:
            self.logger.info("Ignoring empty query after sanitization")
            return

        refresh_ttl = float(self.config["capture"].get("refresh_ttl_seconds",
                                                       25))
        should_refresh = bool(force_refresh)
        if not should_refresh and self.config["behavior"].get(
                "auto_refresh_before_chat", True):
            should_refresh = self.session_store.needs_visual_refresh(
                refresh_ttl)
        if should_refresh:
            self.logger.info("Refreshing visual context before chat")
            self._refresh_visual_context(lang_code, runtime_lang)
        self._log_latency_event("server_request_start", phase="chat")
        chat_response = self.transport.chat_general(
            query,
            self.session_store.get_chat_id(),
            runtime_lang,
        )
        self._log_latency_event("server_response_received", phase="chat")
        self.session_store.update_after_chat(query, chat_response)
        self._safe_say(chat_response["sentence"], speech_mode, phase="answer")

    def _run_object_ask(self, lang_code, object_label, query):
        """
        Object-grounded chat turn. The object label tells the server which remembered
        object should anchor the answer.
        """
        runtime_lang = self._speech_lang(lang_code)
        speech_mode = self._speech_request_language(lang_code)
        object_label = str(object_label or "").strip()
        if not object_label:
            self.logger.info("Ignoring object ask with empty label")
            return

        max_chars = int(self.config["behavior"].get("max_query_chars", 320))
        query = text_utils.sanitize_query(query, max_chars)
        if not query:
            query = self._object_default_query(runtime_lang, object_label)

        self._log_latency_event("server_request_start", phase="object_chat")
        chat_response = self.transport.chat_object(
            object_label=object_label,
            query=query,
            chat_id=self.session_store.get_chat_id(),
            language=runtime_lang,
        )
        self._log_latency_event("server_response_received",
                                phase="object_chat")
        self.session_store.update_after_chat(query, chat_response)
        self._safe_say(chat_response.get("sentence"),
                       speech_mode,
                       phase="answer")

    def _run_show_memory_and_suggest_questions(self, lang_code):
        runtime_lang = self._speech_lang(lang_code)
        speech_mode = self._speech_request_language(lang_code)
        summary, qa_response = self._load_memory_page(runtime_lang)
        shown = self.tablet_adapter.show_memory_page(
            payload=self._build_memory_page_payload(
                summary,
                qa_response,
                ui_language=runtime_lang,
            ))
        if not shown:
            self._safe_say(
                fallback_message("unexpected", runtime_lang),
                speech_mode,
                phase="error",
            )
            return
        self._safe_say(
            self._suggested_question_text(runtime_lang),
            speech_mode,
            phase="answer",
        )

    def _run_cached_answer(self, lang_code, query):
        """
        Uses dynamic concepts for fast answers on user query which is cached.
        """
        runtime_lang = self._speech_lang(lang_code)
        speech_mode = self._speech_request_language(lang_code)
        max_chars = int(self.config["behavior"].get("max_query_chars", 320))
        query = text_utils.clean_text_unicode(query, max_chars=max_chars)
        if not query:
            self.logger.info("Ignoring cached answer with empty query")
            return
        cached_answers = self.session_store.get_cached_answers()
        answer = cached_answers.get(query)
        if answer is None:
            normalized_query = self._normalize_cached_question(query)
            for question, candidate in cached_answers.items():
                if (self._normalize_cached_question(question) ==
                        normalized_query):
                    answer = candidate
                    break
        if answer:
            self.logger.info("Answering from pregenerated cache for query=%s",
                             query)
            self._safe_say(answer, speech_mode, phase="answer")
            return
        self.logger.info(
            "Cached answer miss, falling back to general chat query=%s", query)
        self._log_latency_event("server_request_start", phase="chat")
        chat_response = self.transport.chat_general(
            query,
            self.session_store.get_chat_id(),
            runtime_lang,
        )
        self._log_latency_event("server_response_received", phase="chat")
        self.session_store.update_after_chat(query, chat_response)
        self._safe_say(chat_response.get("sentence"),
                       speech_mode,
                       phase="answer")

    def _run_show_memory(self, lang_code):
        """
        Displays memory state on tablet, uses tablet adapter.
        """
        runtime_lang = self._speech_lang(lang_code)
        speech_mode = self._speech_request_language(lang_code)
        summary, qa_response = self._load_memory_page(runtime_lang)
        shown = self.tablet_adapter.show_memory_page(
            payload=self._build_memory_page_payload(
                summary,
                qa_response,
                ui_language=runtime_lang,
            ))
        if not shown:
            self._safe_say(
                fallback_message("unexpected", runtime_lang),
                speech_mode,
                phase="error",
            )

    def _load_memory_page(self, runtime_lang):
        """
        Fetch the current visual memory and pregenerated Q/A pairs needed by the tablet UI. 
        Q/A generation is just optional so the memory page can still render without it.
        """
        render_limit = scan_planner.memory_render_limit(self.config)
        summary = self.transport.memory_summary(
            render_limit=render_limit,
            language=runtime_lang,
        )
        self.session_store.update_after_memory_summary(summary)
        qa_response = None
        try:
            qa_response = self.transport.pregenerate_qa(
                requested_number_of_pairs=self._pregenerated_questions_count(),
                language=runtime_lang,
            )
            self.session_store.update_after_pregenerated_qa(qa_response)
        except Exception as exc:
            self.logger.warning("Pregenerated QA request failed: %s", exc)
        self._refresh_dynamic_concepts_from_summary(summary)
        return summary, qa_response

    def _normalize_cached_question(self, value):
        value = text_utils.clean_text_unicode(value)
        return value.strip().lower().rstrip(u" ?.!").strip()

    def _suggested_question_text(self, runtime_lang):
        questions = self.session_store.get_cached_questions()
        if questions:
            question = text_utils.clean_text_unicode(questions[0])
            if runtime_lang == "cs":
                # TODO: u s krouzkem
                return u"Mužeš se zeptat třeba tohle: %s" % question
            return u"You can ask me like this: %s" % question
        if runtime_lang == "cs":
            return u"Zeptej se cokoliv, teď nemám nic přichystané"
        return u"You can ask anything you like. I have nothing prepared"

    def _run_reset_memory(self, lang_code):
        runtime_lang = self._speech_lang(lang_code)
        speech_mode = self._speech_request_language(lang_code)
        chat_id = self.session_store.get_chat_id()
        self.transport.reset_memory()
        self.session_store.reset_memory_state()
        self.session_store.reset_conversation()
        if chat_id:
            try:
                self.transport.reset_conversation(chat_id)
            except Exception as exc:
                self.logger.info(
                    "Server conversation reset failed for %s: %s",
                    chat_id,
                    exc,
                )
        if self.config.get("dialog", {}).get("refresh_after_reset", True):
            self._refresh_dynamic_concepts_from_server(lang_code, runtime_lang)
        self._safe_say(
            speech_policy.acknowledgement("reset", runtime_lang),
            speech_mode,
            phase="answer",
        )

    def _refresh_visual_context(self, lang_code=None, runtime_lang=None):
        if runtime_lang is None:
            runtime_lang = self._speech_lang(lang_code)
        frame_id = ids.new_frame_id(self.config["capture"].get(
            "frame_prefix", "frame"))
        capture, metadata = self._capture_with_metadata(
            frame_id, None, "detect")
        detect_response = self.transport.detect(
            capture["image_bytes"],
            metadata,
            publish=bool(self.config["server"].get("publish", True)),
        )
        self.session_store.update_after_detect(detect_response)
        if self.config.get("dialog", {}).get("refresh_after_detect", True):
            self._refresh_dynamic_concepts_from_server(lang_code, runtime_lang)
        return detect_response

    def _capture_with_metadata(self, frame_id, scan_id, capture_mode):
        capture = self.camera_adapter.capture_frame(frame_id=frame_id)
        context = self.robot_context.snapshot()
        metadata = self.metadata_builder.build(capture, context, frame_id,
                                               scan_id, capture_mode)
        return capture, metadata

    def _caption_with_optional_retry(self, image_bytes, metadata, run_detect,
                                     publish, language):
        try:
            return self.transport.caption(
                image_bytes,
                metadata,
                run_detect=run_detect,
                publish=publish,
                language=language,
            )
        except ServerTimeoutError:
            if not self.config["behavior"].get("caption_retry_on_timeout",
                                               True):
                raise
            self.logger.info("Retrying caption once after timeout")
            return self.transport.caption(
                image_bytes,
                metadata,
                run_detect=run_detect,
                publish=publish,
                language=language,
            )

    def _safe_say(self,
                  text,
                  lang_code,
                  phase="speech",
                  turn_id=None,
                  kind=None,
                  started_at=None):
        if not text:
            return
        try:
            self._log_latency_event(
                "speech_start",
                phase=phase,
                turn_id=turn_id,
                kind=kind,
                started_at=started_at,
            )
            self.speech_adapter.say(text, lang_code)
        except Exception:
            self.logger.error("Speech failed for text: %s", text)

    def _set_current_turn(self, turn_id, kind, started_at):
        self._turn_local.turn_id = turn_id
        self._turn_local.kind = kind
        self._turn_local.started_at = started_at

    def _clear_current_turn(self):
        self._turn_local.turn_id = None
        self._turn_local.kind = None
        self._turn_local.started_at = None

    def _log_latency_event(self,
                           event,
                           phase=None,
                           turn_id=None,
                           kind=None,
                           started_at=None):
        if turn_id is None:
            turn_id = getattr(self._turn_local, "turn_id", None)
        if kind is None:
            kind = getattr(self._turn_local, "kind", None)
        if started_at is None:
            started_at = getattr(self._turn_local, "started_at", None)

        now = time_utils.now_ts()
        elapsed = None
        if started_at is not None:
            try:
                elapsed = max(0.0, now - float(started_at))
            except Exception:
                elapsed = None
        if elapsed is None:
            elapsed = -1.0

        # self.logger.info(
        #     "LATENCY turn_id=%s kind=%s event=%s phase=%s elapsed_s=%.3f wall_ts=%.6f",
        #     turn_id,
        #     kind,
        #     event,
        #     phase,
        #     elapsed,
        #     now,
        # )
        self.logger.info(
    "LATENCY turn_id=%s kind=%s event=%s phase=%s elapsed_s=%.3f wall_ts=%.6f"
    % (turn_id, kind, event, phase, elapsed, now)
)

    def _speech_lang(self, requested_lang):
        _, runtime_lang = speech_policy.resolve_language_state(
            self.config,
            requested=requested_lang,
            tts=self.speech_adapter.tts,
            logger=self.logger,
        )
        return runtime_lang

    def _speech_request_language(self, requested_lang=None):
        if requested_lang is not None:
            return requested_lang
        return speech_policy.normalize_dialog_language(
            self.config.get("dialog", {}).get("language", "auto"))

    def _scan_summary_query(self, lang_code):
        lang_code = self._speech_lang(lang_code)
        if lang_code == "cs":
            return self.config["capture"].get(
                "scan_summary_query_cs",
                "Strucne popis co ted vidis podle aktualni vizualni pameti.",
            )
        return self.config["capture"].get(
            "scan_summary_query_en",
            "Briefly describe what you can see now using the current visual memory.",
        )

    def _object_default_query(self, lang_code, object_label):
        if self._speech_lang(lang_code) == "cs":
            return "Co vis o objektu %s" % object_label
        return "Tell me what you know about %s" % object_label

    def _refresh_dynamic_concepts_from_server(self,
                                              lang_code=None,
                                              runtime_lang=None):
        if self.dialog_adapter is None:
            return False
        if runtime_lang is None:
            runtime_lang = self._speech_lang(lang_code)
        summary = self.transport.memory_summary(
            render_limit=scan_planner.memory_render_limit(self.config),
            language=runtime_lang,
        )
        self.session_store.update_after_memory_summary(summary)
        return self._refresh_dynamic_concepts_from_summary(summary)

    def _refresh_dynamic_concepts_from_summary(self, summary):
        """
        Convert server scene-graph edges into ALDialog dynamic concept lists. Edges
        where subject == object are treated as attributes; other edges are relations.
        """
        if self.dialog_adapter is None:
            return False
        labels = list(summary.get("labels") or [])
        attributes = []
        relations = []
        for edge in summary.get("scene_graph", []) or []:
            if not isinstance(edge, dict):
                continue
            relation = text_utils.clean_text(edge.get("rel"))
            if not relation:
                continue
            sub = text_utils.clean_text(edge.get("sub"))
            obj = text_utils.clean_text(edge.get("obj"))
            if sub and obj and sub == obj:
                attributes.append(relation)
            else:
                relations.append(relation)
        return self.dialog_adapter.refresh_memory_concepts(
            labels,
            attributes,
            relations,
            cached_questions=self.session_store.get_cached_questions(),
        )

    def _pregenerated_questions_count(self):
        tablet_cfg = self.config.get("tablet", {})
        try:
            value = int(tablet_cfg.get("pregenerated_questions_count", 5))
        except Exception:
            value = 5
        return max(1, value)

    def _build_memory_page_payload(self, summary, qa_response, ui_language):
        """Creates payload for displaying at tablet"""
        summary = summary or {}
        scene_graph = list(summary.get("scene_graph") or [])
        attributes = []
        relationships = []
        for edge in scene_graph:
            if not isinstance(edge, dict):
                continue
            relation = text_utils.clean_text(edge.get("rel"))
            if not relation:
                continue
            sub = text_utils.clean_text(edge.get("sub"))
            obj = text_utils.clean_text(edge.get("obj"))
            cleaned = {"sub": sub, "rel": relation, "obj": obj}
            if sub and obj and sub == obj:
                attributes.append(cleaned)
            else:
                relationships.append(cleaned)

        payload = {
            "ui_language": self._speech_lang(ui_language),
            "object_labels": list(summary.get("labels") or []),
            "label_counts": dict(summary.get("label_counts") or {}),
            "attributes": attributes,
            "relationships": relationships,
            "graph_svg": summary.get("graph_svg"),
            "pregenerated_qa": [],
        }
        if isinstance(qa_response, dict):
            payload["pregenerated_qa"] = qa_response.get(
                "pregenerated_qa", []) or []
            payload["qa_metadata"] = qa_response.get("metadata", {}) or {}
        else:
            cached_answers = self.session_store.get_cached_answers()
            payload["pregenerated_qa"] = [{
                "question": q,
                "answer": cached_answers.get(q, "")
            } for q in self.session_store.get_cached_questions()]
        return payload
