import json

import requests

from pepper_client.utils import logging_utils
from pepper_client.interaction import speech_policy
from pepper_client.utils.error_policy import ConfigUpdateError
from pepper_client.utils.error_policy import MalformedResponseError
from pepper_client.utils.error_policy import ServerTimeoutError
from pepper_client.utils.error_policy import ServerUnavailableError


class PepperServerTransport(object):
    def __init__(self, config, logger):
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self.update_config(config)

    def update_config(self, config):
        self.config = config
        self.base_url = (config["server"].get("base_url") or "").rstrip("/")
        self.verify_tls = bool(config["server"].get("verify_tls", True))
        self.logger.info("Server transport configured for %s", self.base_url)

    def caption(self, image_bytes, metadata, prompt=None, run_detect=True, publish=True, language=None):
        payload = {
            "metadata": json.dumps(metadata),
            "run_detect": self._bool_str(run_detect),
            "publish": self._bool_str(publish),
        }
        if prompt:
            payload["prompt"] = prompt
        if language:
            payload["language"] = language
        data = self._post_multipart(
            self.config["server"]["caption_path"],
            image_bytes,
            payload,
            self.config["server"].get("caption_timeout_seconds", 25),
        )
        if not isinstance(data, dict) or not data.get("caption"):
            raise MalformedResponseError("Caption response missing caption")
        return data

    def detect(self, image_bytes, metadata, publish=True):
        payload = {
            "metadata": json.dumps(metadata),
            "publish": self._bool_str(publish),
        }
        data = self._post_multipart(
            self.config["server"]["detect_path"],
            image_bytes,
            payload,
            self.config["server"].get("detect_timeout_seconds", 120),
        )
        if not isinstance(data, dict) or not isinstance(data.get("objects", []), list):
            raise MalformedResponseError("Detect response missing objects list")
        return data

    def panorama_detect(self, captures, publish=True, resize_image=True, stick_together=True):
        files = []
        form_data = [
            ("publish", self._bool_str(publish)),
            ("resize_image", self._bool_str(resize_image)),
            ("stick_together", self._bool_str(stick_together)),
        ]
        for index, item in enumerate(captures or []):
            image_bytes = item.get("image_bytes")
            metadata = item.get("metadata") or {}
            files.append(
                (
                    "files",
                    ("capture_%s.jpg" % index, image_bytes, "image/jpeg"),
                )
            )
            form_data.append(("metadata", json.dumps(metadata)))
        data = self._request(
            method="post",
            path=self.config["server"].get("detect_panorama_path", "/api/v1/detect/panorama"),
            timeout_seconds=self.config["server"].get("detect_timeout_seconds", 120),
            files=files,
            data=form_data,
        )
        if not isinstance(data, dict) or not isinstance(data.get("objects", []), list):
            raise MalformedResponseError("Panorama detect response missing objects list")
        return data

    def chat(self, query, chat_id=None, language=None, mode=None, object_label=None):
        payload = {"query": query}
        if chat_id:
            payload["chat_id"] = chat_id
        if language:
            language = self._normalize_output_language(language)
            payload["language"] = language
        if mode:
            payload["mode"] = str(mode)
        if object_label:
            payload["object_label"] = str(object_label)
        data = self._post_json(
            self.config["server"]["chat_path"],
            payload,
            self.config["server"].get("chat_timeout_seconds", 45),
        )
        if not isinstance(data, dict) or not data.get("sentence") or not data.get("chat_id"):
            raise MalformedResponseError("Chat response missing sentence or chat_id")
        return data

    def chat_general(self, query, chat_id=None, language=None):
        return self.chat(
            query=query,
            chat_id=chat_id,
            language=language,
            mode="general",
        )

    def chat_object(self, object_label, query, chat_id=None, language=None):
        return self.chat(
            query=query,
            chat_id=chat_id,
            language=language,
            mode="object",
            object_label=object_label,
        )

    def memory_summary(self, render_limit=5):
        path = self.config["server"].get("memory_summary_path", "/api/v1/memory/summary")
        path = "%s?render_limit=%s" % (path, int(render_limit))
        data = self._request(
            method="get",
            path=path,
            timeout_seconds=self.config["server"].get("memory_timeout_seconds", 20),
        )
        if not isinstance(data, dict):
            raise MalformedResponseError("Memory summary response is not an object")
        return data

    def reset_memory(self):
        path = self.config["server"].get("memory_reset_path", "/api/v1/memory/reset")
        if "?" in path:
            request_path = path
        else:
            request_path = "%s?confirm=true" % path
        data = self._post_json(
            request_path,
            {},
            self.config["server"].get("memory_timeout_seconds", 20),
        )
        if not isinstance(data, dict) or not data.get("ok"):
            raise MalformedResponseError("Memory reset failed")
        return data

    def pregenerate_qa(self, requested_number_of_pairs=5, language=None):
        path = self.config["server"].get(
            "pregenerate_qa_path", "/api/v1/chat/pregenerate_qa"
        )
        payload = {"requested_number_of_pairs": int(requested_number_of_pairs)}
        if language:
            payload["output_language"] = self._normalize_output_language(language)
        data = self._post_json(
            path,
            payload,
            self.config["server"].get("chat_timeout_seconds", 45),
        )
        if not isinstance(data, dict) or not isinstance(data.get("pregenerated_qa"), list):
            raise MalformedResponseError("Pregenerated QA response missing pregenerated_qa")
        return data

    def reset_conversation(self, chat_id):
        if not chat_id:
            return {"ok": True, "skipped": True}
        path = "/api/v1/chat/conversations/%s/reset" % chat_id
        return self._post_json(path, {}, self.config["server"].get("config_timeout_seconds", 10))

    def patch_output_language(self, mode):
        payload = {"system": {"output_language": mode}}
        data = self._patch_json(
            self.config["server"]["config_patch_path"],
            payload,
            self.config["server"].get("config_timeout_seconds", 10),
        )
        if not isinstance(data, dict) or not data.get("ok"):
            raise ConfigUpdateError("Output language patch failed")
        return data

    def _post_multipart(self, path, image_bytes, form_data, timeout_seconds):
        files = {"file": ("capture.jpg", image_bytes, "image/jpeg")}
        return self._request(
            method="post",
            path=path,
            timeout_seconds=timeout_seconds,
            files=files,
            data=form_data,
        )

    def _post_json(self, path, payload, timeout_seconds):
        return self._request(
            method="post",
            path=path,
            timeout_seconds=timeout_seconds,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )

    def _patch_json(self, path, payload, timeout_seconds):
        return self._request(
            method="patch",
            path=path,
            timeout_seconds=timeout_seconds,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
        )

    def _request(self, method, path, timeout_seconds, **kwargs):
        url = "%s%s" % (self.base_url, path)
        self.logger.info("HTTP %s %s", method.upper(), url)
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=float(timeout_seconds),
                verify=self.verify_tls,
                **kwargs
            )
        except requests.Timeout as exc:
            raise ServerTimeoutError(str(exc))
        except requests.RequestException as exc:
            raise ServerUnavailableError(str(exc))

        if response.status_code >= 400:
            detail = response.text.strip()
            self.logger.warning(
                "Server returned status=%s body=%s",
                response.status_code,
                speech_policy.clean_text(detail, max_chars=240),
            )
            raise ServerUnavailableError("HTTP %s for %s" % (response.status_code, url))
        try:
            data = response.json()
        except ValueError:
            self.logger.warning("Invalid JSON response body: %s", response.text)
            raise MalformedResponseError("Response was not valid JSON")
        self.logger.info("HTTP response payload: %s", logging_utils.safe_json(data))
        return data

    def _bool_str(self, value):
        return "true" if value else "false"

    def _normalize_output_language(self, language):
        language = str(language or "").strip().lower()
        if language in ("en", "english"):
            return "english"
        if language in ("cs", "cz", "czech", "czc"):
            return "czech"
        return "english"
