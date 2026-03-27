class SonarAdapter(object):
    LEFT_KEY = "Device/SubDeviceList/US/Left/Sensor/Value"
    RIGHT_KEY = "Device/SubDeviceList/US/Right/Sensor/Value"

    def __init__(self, services, config, logger):
        self.sonar = services.ALSonar
        self.memory = services.ALMemory
        self.config = config
        self.logger = logger
        self.subscription_name = "%s_sonar" % config["app"]["service_name"]
        self._subscribed = False

    def start(self):
        if not self.config["social"].get("enable_sonar", True):
            return
        if self.sonar is None or not hasattr(self.sonar, "subscribe"):
            return
        try:
            self.sonar.subscribe(self.subscription_name)
            self._subscribed = True
            self.logger.info("Subscribed to ALSonar")
        except Exception as exc:
            self.logger.info("ALSonar subscribe skipped: %s", exc)

    def stop(self):
        if self._subscribed and self.sonar is not None and hasattr(self.sonar, "unsubscribe"):
            try:
                self.sonar.unsubscribe(self.subscription_name)
                self.logger.info("Unsubscribed from ALSonar")
            except Exception as exc:
                self.logger.info("ALSonar unsubscribe skipped: %s", exc)
            self._subscribed = False

    def snapshot(self):
        if self.memory is None:
            return None
        left = self._get(self.LEFT_KEY)
        right = self._get(self.RIGHT_KEY)
        if left is None and right is None:
            return None
        payload = {"left": left, "right": right}
        self.logger.info("Sonar snapshot: %s", payload)
        return payload

    def _get(self, key):
        try:
            value = self.memory.getData(key)
            return float(value)
        except Exception:
            return None
