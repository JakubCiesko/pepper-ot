import json

from pepper_client.utils import timing as time_utils


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
