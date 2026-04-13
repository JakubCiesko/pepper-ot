import json

from pepper_client.utils import time_utils


class TabletAdapter(object):
    def __init__(self, services, config, logger):
        self.tablet = services.ALTabletService
        self.config = config
        self.logger = logger

    def show_dashboard(self):
        url = self.config["server"].get("dashboard_url")
        return self.show_url(url)

    def show_memory_page(self, url):
        return self.show_url(url)

    def local_memory_page_url(self):
        tablet_cfg = self.config.get("tablet", {})
        app_name = str(tablet_cfg.get("local_app_name") or "pepper-grounded-client").strip()
        page_path = str(
            tablet_cfg.get("local_memory_page_path") or "html/memory/index.html"
        ).strip()
        if page_path.startswith("/"):
            page_path = page_path[1:]
        return "http://198.18.0.1/apps/%s/%s" % (app_name, page_path)

    def show_local_memory_page(self, payload=None):
        shown = self.show_url(self.local_memory_page_url())
        if not shown:
            return False
        if payload is None:
            return True
        return self.push_memory_payload(payload)

    def push_memory_payload(self, payload):
        if self.tablet is None:
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
                result = self.tablet.executeJS(script)
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

    def show_url(self, url):
        if not url:
            self.logger.info("No tablet URL configured")
            return False
        if self.tablet is None:
            self.logger.info("ALTabletService unavailable")
            return False
        try:
            self.tablet.showWebview(url)
            self.logger.info("Showing tablet URL: %s", url)
            return True
        except Exception as exc:
            self.logger.warning("Failed to show tablet URL %s: %s", url, exc)
            return False

    def _js_result_is_true(self, value):
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        return text in ("true", "1", "ok", "success")
