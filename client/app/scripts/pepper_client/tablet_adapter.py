class TabletAdapter(object):
    def __init__(self, services, config, logger):
        self.tablet = services.ALTabletService
        self.config = config
        self.logger = logger

    def show_dashboard(self):
        url = self.config["server"].get("dashboard_url")
        return self.show_url(url)

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
