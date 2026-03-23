import threading

from pepper_client import ids
from pepper_client import scan_planner
from pepper_client import speech_policy
from pepper_client import time_utils
from pepper_client.error_policy import CameraCaptureError
from pepper_client.error_policy import MalformedResponseError
from pepper_client.error_policy import ServerTimeoutError
from pepper_client.error_policy import ServerUnavailableError
from pepper_client.error_policy import fallback_message


class TurnManager(object):
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
        self.logger = logger
        self._lock = threading.RLock()
        self._busy = False
        self._active_turn = None

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

    def is_busy(self):
        with self._lock:
            return self._busy

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
        turn_id = ids.new_turn_id(kind)
        with self._lock:
            if self._busy:
                self.logger.info("Rejecting %s because another turn is active: %s", kind, self._active_turn)
                self._safe_say(fallback_message("busy", lang_code), lang_code)
                return False
            self._busy = True
            self._active_turn = {"id": turn_id, "kind": kind}
        if self.config["behavior"].get("speak_acknowledgements", True):
            ack = speech_policy.acknowledgement(kind, lang_code)
            if ack:
                self._safe_say(ack, self._speech_lang(lang_code))
        thread = threading.Thread(
            target=self._run_guarded,
            name="pepper-turn-%s" % turn_id,
            args=(turn_id, kind, lang_code, target, args),
        )
        thread.setDaemon(True)
        thread.start()
        return True

    def _run_guarded(self, turn_id, kind, lang_code, target, args):
        self.logger.info("Starting turn id=%s kind=%s", turn_id, kind)
        try:
            target(*args)
            self.logger.info("Completed turn id=%s kind=%s", turn_id, kind)
        except CameraCaptureError as exc:
            self.logger.warning("Camera error in turn %s: %s", turn_id, exc)
            self._safe_say(fallback_message("camera", lang_code), lang_code)
        except ServerTimeoutError as exc:
            self.logger.warning("Server timeout in turn %s: %s", turn_id, exc)
            self._safe_say(fallback_message("server_timeout", lang_code), lang_code)
        except ServerUnavailableError as exc:
            self.logger.warning("Server unavailable in turn %s: %s", turn_id, exc)
            self._safe_say(fallback_message("server_unavailable", lang_code), lang_code)
        except MalformedResponseError as exc:
            self.logger.warning("Malformed server response in turn %s: %s", turn_id, exc)
            self._safe_say(fallback_message("malformed", lang_code), lang_code)
        except Exception as exc:
            self.logger.error("Unexpected turn failure id=%s kind=%s, message=%s", turn_id, kind, exc)
            self._safe_say(fallback_message("unexpected", lang_code), lang_code)
        finally:
            with self._lock:
                self._busy = False
                self._active_turn = None

    def _run_look(self, lang_code):
        self.logger.info("Running look method with lang code: %s", lang_code)
        frame_id = ids.new_frame_id(self.config["capture"].get("frame_prefix", "frame"))
        capture, metadata = self._capture_with_metadata(frame_id, None, "caption")
        run_detect = bool(self.config["behavior"].get("caption_run_detect", True))
        publish = bool(self.config["server"].get("publish", True))
        language = ("english" if lang_code == "en" else "czech") or self.session_store.get_output_language_mode()
        caption_response = self._caption_with_optional_retry(
            capture["image_bytes"],
            metadata,
            run_detect=run_detect,
            publish=publish,
            language=language
        )
        self.session_store.update_after_caption(caption_response)
        self._safe_say(caption_response["caption"], self._speech_lang(lang_code))

    def _run_scan(self, lang_code):
        if self.config["behavior"].get("show_dashboard_during_scan", False):
            self.tablet_adapter.show_dashboard()

        scan_id = ids.new_scan_id(self.config["capture"].get("scan_prefix", "scan"))
        original_pose = self.pose_adapter.snapshot()
        self.logger.info("Starting scan sweep scan_id=%s", scan_id)
        successes = 0
        last_error = None
        try:
            for index, yaw in enumerate(scan_planner.planned_yaws_radians(self.config)):
                self.pose_adapter.move_head(
                    yaw,
                    scan_planner.scan_pitch(self.config),
                    self.config["capture"].get("head_move_speed", 0.15),
                )
                time_utils.sleep_seconds(self.config["capture"].get("settle_seconds", 0.6))
                frame_id = ids.new_frame_id(self.config["capture"].get("frame_prefix", "frame"))
                try:
                    capture, metadata = self._capture_with_metadata(frame_id, scan_id, "scan")
                    detect_response = self.transport.detect(
                        capture["image_bytes"],
                        metadata,
                        publish=bool(self.config["server"].get("publish", True)),
                    )
                    self.session_store.update_after_detect(detect_response, scan_id=scan_id)
                    successes += 1
                    self.logger.info(
                        "Scan frame %s completed objects=%s",
                        index,
                        len(detect_response.get("objects", [])),
                    )
                    self.logger.info("Full detect response %s", detect_response)
                except Exception as exc:
                    last_error = exc
                    self.logger.warning("Scan frame %s failed: %s", index, exc)
            if successes <= 0:
                if last_error is not None:
                    raise last_error
                raise ServerUnavailableError("No scan frame completed")
        finally:
            if self.config["behavior"].get("auto_restore_head_pose", True):
                self.pose_adapter.restore_head(
                    original_pose,
                    self.config["capture"].get("head_move_speed", 0.15),
                )
        language = self._speech_lang(lang_code)
        if self.config["behavior"].get("allow_scan_summary_chat", True):
            query = self._scan_summary_query(lang_code)
            chat_response = self.transport.chat(query, self.session_store.get_chat_id(), language)
            self.session_store.update_after_chat(query, chat_response)
            self._safe_say(chat_response["sentence"], language)
        else:
            self._safe_say(
                speech_policy.generic_message("scan_complete", lang_code),
                language,
            )

    def _run_ask(self, lang_code, query, force_refresh):
        max_chars = int(self.config["behavior"].get("max_query_chars", 320))
        query = speech_policy.sanitize_query(query, max_chars)
        self.logger.info("Starting ask with sanitized query %s", query)
        if not query:
            self.logger.info("Ignoring empty query after sanitization")
            return

        refresh_ttl = float(self.config["capture"].get("refresh_ttl_seconds", 25))
        should_refresh = bool(force_refresh)
        if not should_refresh and self.config["behavior"].get("auto_refresh_before_chat", True):
            should_refresh = self.session_store.needs_visual_refresh(refresh_ttl)
        if should_refresh:
            self.logger.info("Refreshing visual context before chat")
            self._refresh_visual_context()
        language = self._speech_lang(lang_code)
        chat_response = self.transport.chat(query, self.session_store.get_chat_id(), language)
        self.session_store.update_after_chat(query, chat_response)
        self._safe_say(chat_response["sentence"], language)

    def _refresh_visual_context(self):
        frame_id = ids.new_frame_id(self.config["capture"].get("frame_prefix", "frame"))
        capture, metadata = self._capture_with_metadata(frame_id, None, "detect")
        detect_response = self.transport.detect(
            capture["image_bytes"],
            metadata,
            publish=bool(self.config["server"].get("publish", True)),
        )
        self.session_store.update_after_detect(detect_response)
        return detect_response

    def _capture_with_metadata(self, frame_id, scan_id, capture_mode):
        capture = self.camera_adapter.capture_frame(frame_id=frame_id)
        context = self.robot_context.snapshot()
        metadata = self.metadata_builder.build(capture, context, frame_id, scan_id, capture_mode)
        return capture, metadata

    def _caption_with_optional_retry(self, image_bytes, metadata, run_detect, publish, language):
        try:
            return self.transport.caption(image_bytes, metadata, run_detect=run_detect, publish=publish, language=language)
        except ServerTimeoutError:
            if not self.config["behavior"].get("caption_retry_on_timeout", True):
                raise
            self.logger.info("Retrying caption once after timeout")
            return self.transport.caption(image_bytes, metadata, run_detect=run_detect, publish=publish, language=language)

    def _safe_say(self, text, lang_code):
        if not text:
            return
        try:
            self.speech_adapter.say(text, lang_code)
        except Exception:
            self.logger.error("Speech failed for text: %s", text)

    def _speech_lang(self, requested_lang):
        #mode = self.session_store.get_output_language_mode()
        #if mode == "english":
        #    return "en"
        #if mode == "czech":
        #    return "cs"
        return speech_policy.language_code(requested_lang, self.config["language"].get("default_dialog_language", "en"))

    #
    # def _speech_lang(self, requested_lang):
    #     mode = self.session_store.get_output_language_mode()
    #     if mode == "english":
    #         return "en"
    #     if mode == "czech":
    #         return "cs"
    #     return speech_policy.language_code(requested_lang, self.config["language"].get("default_dialog_language", "en"))
    #
    #TODO: BULLSHITS!
    def _scan_summary_query(self, lang_code):
        lang_code = speech_policy.language_code(lang_code)
        if lang_code == "cs":
            return self.config["capture"].get(
                "scan_summary_query_cs",
                "Strucne popis co ted vidis podle aktualni vizualni pameti.",
            )
        return self.config["capture"].get(
            "scan_summary_query_en",
            "Briefly describe what you can see now using the current visual memory.",
        )
