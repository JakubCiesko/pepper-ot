import copy
import json
import os
import threading

from pepper_client.utils import timing as time_utils

try:
    from BaseHTTPServer import BaseHTTPRequestHandler
    from BaseHTTPServer import HTTPServer
except ImportError:
    from http.server import BaseHTTPRequestHandler
    from http.server import HTTPServer

try:
    from urlparse import urlparse
except ImportError:
    from urllib.parse import urlparse

try:
    from urllib import unquote
except ImportError:
    from urllib.parse import unquote


class TabletAdapter(object):
    def __init__(self, services, config, logger):
        self.services = services
        self.config = config
        self.logger = logger
        self._app_loaded = False
        self._webview_visible = False

    def local_app_url(self):
        tablet_cfg = self.config.get("tablet", {})
        app_name = str(tablet_cfg.get("local_app_name") or "pepper-grounded-client").strip()
        return "http://198.18.0.1/apps/%s/" % app_name

    def show_memory_page(self, payload=None):
        tablet = self._tablet()
        if tablet is None:
            self.logger.info("ALTabletService unavailable")
            return False
        if not self._ensure_local_app_loaded(tablet):
            return False
        if not self._ensure_webview_visible(tablet):
            return False
        return self.push_memory_payload(payload or {})

    def hide_memory_page(self):
        tablet = self._tablet()
        if tablet is None:
            self.logger.info("ALTabletService unavailable")
            return False
        try:
            tablet.hideWebview()
            self._webview_visible = False
            self.logger.info("Memory webview hidden")
            return True
        except Exception as exc:
            self.logger.info("hideWebview failed (possibly already hidden): %s", exc)
            return True

    def push_memory_payload(self, payload):
        tablet = self._tablet()
        if tablet is None:
            self.logger.info("ALTabletService unavailable, skipping memory payload push")
            return False
        tablet_cfg = self.config.get("tablet", {})
        try:
            attempts = int(tablet_cfg.get("bridge_retry_attempts", 12))
        except Exception:
            attempts = 12
        attempts = max(1, attempts)
        try:
            interval = float(tablet_cfg.get("bridge_retry_interval_seconds", 0.25))
        except Exception:
            interval = 0.25
        interval = max(0.0, interval)
        if not self._wait_page_ready(tablet, attempts, interval):
            self.logger.warning(
                "Memory page did not become ready before payload injection"
            )
            return False
        payload_json = json.dumps(payload or {})
        script = (
            "(function(){"
            "try{"
            "if(window.PepperMemoryPage&&window.PepperMemoryPage.renderFromBridge){"
            "window.PepperMemoryPage.renderFromBridge(%s);"
            "return true;"
            "}"
            "return false;"
            "}catch(e){return false;}"
            "})();"
        ) % payload_json
        for attempt in range(attempts):
            try:
                result = tablet.executeJS(script)
                if self._js_result_is_true(result):
                    self.logger.info(
                        "Injected memory payload into local tablet page on attempt=%s",
                        attempt + 1,
                    )
                    return True
            except Exception as exc:
                self.logger.info(
                    "Memory payload injection attempt=%s failed: %s",
                    attempt + 1,
                    exc,
                )
            if attempt < attempts - 1 and interval > 0:
                time_utils.sleep_seconds(interval)
        self.logger.warning("Failed to inject memory payload into local tablet page")
        return False

    def _js_result_is_true(self, value):
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in ("true", "1", "ok", "success")

    def _wait_page_ready(self, tablet, attempts, interval):
        readiness_script = (
            "(function(){"
            "try{return !!window.PepperMemoryPageReady;}catch(e){return false;}"
            "})();"
        )
        for attempt in range(attempts):
            try:
                result = tablet.executeJS(readiness_script)
                if self._js_result_is_true(result):
                    self.logger.info(
                        "Memory page ready after attempt=%s",
                        attempt + 1,
                    )
                    return True
            except Exception as exc:
                self.logger.info(
                    "Memory page readiness probe attempt=%s failed: %s",
                    attempt + 1,
                    exc,
                )
            if attempt < attempts - 1 and interval > 0:
                time_utils.sleep_seconds(interval)
        return False

    def _ensure_local_app_loaded(self, tablet):
        if self._app_loaded:
            return True
        app_name = str(
            self.config.get("tablet", {}).get("local_app_name")
            or "pepper-grounded-client"
        ).strip()
        url = self.local_app_url()
        try:
            tablet.loadApplication(app_name)
            self._app_loaded = True
            self.logger.info("Loaded local tablet application: %s", app_name)
            return True
        except Exception as exc:
            self.logger.info(
                "loadApplication failed for %s, falling back to loadUrl: %s",
                app_name,
                exc,
            )
        try:
            tablet.loadUrl(url)
            self._app_loaded = True
            self.logger.info("Loaded local tablet URL: %s", url)
            return True
        except Exception as exc:
            self.logger.warning("Failed to load local tablet URL %s: %s", url, exc)
            self._app_loaded = False
            return False

    def _ensure_webview_visible(self, tablet):
        if self._webview_visible:
            return True
        try:
            tablet.showWebview()
            self._webview_visible = True
            self.logger.info("Memory webview shown")
            return True
        except Exception as exc:
            self.logger.info("showWebview() failed, trying showWebview(url): %s", exc)
        url = self.local_app_url()
        try:
            tablet.showWebview(url)
            self._webview_visible = True
            self.logger.info("Memory webview shown via URL: %s", url)
            return True
        except Exception as exc:
            self.logger.warning("Failed to show memory webview for %s: %s", url, exc)
            self._webview_visible = False
            return False

    def _tablet(self):
        return self.services.ALTabletService


class FakeTabletAdapter(object):
    """Desktop mirror of tablet rendering for local development.

    Exposes the same runtime API as TabletAdapter and serves:
    - GET /           -> client/app/html/index.html
    - GET /payload.json -> latest memory payload
    - GET /health     -> ok
    """

    def __init__(self, services, config, logger):
        self.services = services
        self.config = config
        self.logger = logger
        self._lock = threading.RLock()
        self._latest_payload = {}
        self._server = None
        self._server_thread = None
        self._server_started = False
        self._webview_visible = False
        self._url_logged = False

    def show_memory_page(self, payload=None):
        if not self._ensure_server_started():
            return False
        with self._lock:
            self._webview_visible = True
        if not self._url_logged:
            self.logger.info("Fake tablet page available at %s", self.local_fake_url())
            self._url_logged = True
        return self.push_memory_payload(payload or {})

    def hide_memory_page(self):
        with self._lock:
            self._webview_visible = False
        self.logger.info("Fake tablet page marked as hidden")
        return True

    def push_memory_payload(self, payload):
        with self._lock:
            self._latest_payload = copy.deepcopy(payload or {})
        return True

    def local_fake_url(self):
        host = self._fake_host()
        port = self._fake_port()
        poll_ms = self._fake_poll_interval_ms()
        return "http://%s:%s/?fake_tablet=1&poll_ms=%s" % (host, port, poll_ms)

    def _fake_host(self):
        tablet_cfg = self.config.get("tablet", {})
        host = str(tablet_cfg.get("fake_host") or "127.0.0.1").strip()
        if not host:
            host = "127.0.0.1"
        return host

    def _fake_port(self):
        tablet_cfg = self.config.get("tablet", {})
        try:
            port = int(tablet_cfg.get("fake_port", 8766))
        except Exception:
            port = 8766
        if port < 1 or port > 65535:
            port = 8766
        return port

    def _fake_poll_interval_ms(self):
        tablet_cfg = self.config.get("tablet", {})
        try:
            value = int(tablet_cfg.get("fake_poll_interval_ms", 500))
        except Exception:
            value = 500
        if value < 100:
            value = 100
        return value

    def _ensure_server_started(self):
        with self._lock:
            if self._server_started:
                return True

        host = self._fake_host()
        port = self._fake_port()
        adapter = self

        class _RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                path = unquote(parsed.path or "/")
                if path in ("/", "/index.html"):
                    adapter._serve_index(self)
                    return
                if path == "/payload.json":
                    adapter._serve_payload(self)
                    return
                if path == "/health":
                    adapter._serve_health(self)
                    return
                if adapter._is_static_path(path):
                    adapter._serve_static(self, path)
                    return
                adapter._write_response(
                    self,
                    404,
                    "text/plain; charset=utf-8",
                    adapter._to_bytes("not found"),
                )

            def log_message(self, fmt, *args):
                adapter.logger.info("FakeTablet HTTP: " + fmt, *args)

        try:
            server = HTTPServer((host, port), _RequestHandler)
            thread = threading.Thread(
                target=server.serve_forever,
                name="fake-tablet-http",
            )
            thread.setDaemon(True)
            thread.start()
        except Exception as exc:
            self.logger.warning(
                "Failed to start fake tablet server at %s:%s: %s",
                host,
                port,
                exc,
            )
            return False

        with self._lock:
            self._server = server
            self._server_thread = thread
            self._server_started = True

        self.logger.info("Started fake tablet HTTP server at http://%s:%s", host, port)
        return True

    def _serve_index(self, handler):
        index_path = os.path.join(self._html_root(), "index.html")
        try:
            with open(index_path, "rb") as handle:
                body = handle.read()
        except Exception as exc:
            self.logger.warning("Failed to read fake tablet index at %s: %s", index_path, exc)
            self._write_response(
                handler,
                500,
                "text/plain; charset=utf-8",
                self._to_bytes("failed to read index.html"),
            )
            return
        self._write_response(handler, 200, "text/html; charset=utf-8", body)

    def _serve_payload(self, handler):
        with self._lock:
            payload = copy.deepcopy(self._latest_payload or {})
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        self._write_response(
            handler,
            200,
            "application/json; charset=utf-8",
            self._to_bytes(body),
        )

    def _serve_health(self, handler):
        self._write_response(
            handler,
            200,
            "text/plain; charset=utf-8",
            self._to_bytes("ok"),
        )

    def _serve_static(self, handler, path):
        relative = str(path or "/").lstrip("/")
        static_file = self._safe_static_file(relative)
        if not static_file:
            self._write_response(
                handler,
                403,
                "text/plain; charset=utf-8",
                self._to_bytes("forbidden"),
            )
            return
        if not os.path.isfile(static_file):
            self._write_response(
                handler,
                404,
                "text/plain; charset=utf-8",
                self._to_bytes("not found"),
            )
            return
        try:
            with open(static_file, "rb") as handle:
                body = handle.read()
        except Exception as exc:
            self.logger.warning("Failed reading fake tablet static file %s: %s", static_file, exc)
            self._write_response(
                handler,
                500,
                "text/plain; charset=utf-8",
                self._to_bytes("failed to read static file"),
            )
            return
        self._write_response(
            handler,
            200,
            self._guess_content_type(static_file),
            body,
        )

    def _is_static_path(self, path):
        path = str(path or "")
        if path.startswith("/js/"):
            return True
        if path.startswith("/css/"):
            return True
        if "." in os.path.basename(path):
            return True
        return False

    def _html_root(self):
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(
            os.path.join(here, "..", "..", "..", "html")
        )

    def _safe_static_file(self, relative):
        root = self._html_root()
        candidate = os.path.normpath(os.path.join(root, relative))
        if candidate == root:
            return None
        if not candidate.startswith(root + os.sep):
            return None
        return candidate

    def _guess_content_type(self, filename):
        lowered = str(filename or "").lower()
        if lowered.endswith(".html"):
            return "text/html; charset=utf-8"
        if lowered.endswith(".css"):
            return "text/css; charset=utf-8"
        if lowered.endswith(".js"):
            return "application/javascript; charset=utf-8"
        if lowered.endswith(".json"):
            return "application/json; charset=utf-8"
        if lowered.endswith(".svg"):
            return "image/svg+xml"
        if lowered.endswith(".png"):
            return "image/png"
        if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
            return "image/jpeg"
        if lowered.endswith(".gif"):
            return "image/gif"
        return "application/octet-stream"

    def _write_response(self, handler, status, content_type, body_bytes):
        if body_bytes is None:
            body_bytes = b""
        if not isinstance(body_bytes, bytes):
            body_bytes = self._to_bytes(body_bytes)
        handler.send_response(int(status))
        handler.send_header("Content-Type", content_type)
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Content-Length", str(len(body_bytes)))
        handler.end_headers()
        handler.wfile.write(body_bytes)

    def _to_bytes(self, value):
        if value is None:
            return b""
        try:
            unicode_type = unicode
        except NameError:
            unicode_type = str
        if isinstance(value, bytes):
            return value
        if isinstance(value, unicode_type):
            return value.encode("utf-8")
        return str(value).encode("utf-8")
