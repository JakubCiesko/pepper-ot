class PeopleAdapter(object):
    """Class wrapper around ALMemory and ALFaceDetection.
    Builds metadata about people position used for attribution on server-side"""

    def __init__(self, services, config, logger):
        self.services = services
        self.config = config
        self.logger = logger
        self.people_perception = services.ALPeoplePerception
        self.memory = services.ALMemory
        self.subscription_name = "%s_people" % config["app"]["service_name"]
        self._subscribed = False

    def start(self):
        if not self.config["social"].get("enable_people_perception", True):
            return
        self._subscribed = self._safe_subscribe(self.people_perception,
                                                self.subscription_name)
        if self._subscribed:
            self.logger.info("Subscribed to ALPeoplePerception")

    def stop(self):
        if self._subscribed:
            self._safe_unsubscribe(self.people_perception,
                                   self.subscription_name)
            self._subscribed = False

    def snapshot_people(self):
        if self.memory is None:
            return []
        people_ids = self._get_memory_value("PeoplePerception/PeopleList",
                                            []) or []
        self.logger.info(
            "Running PeopleAdapter.snapshot_people on people_ids=%s",
            people_ids)
        result = []
        for person_id in people_ids:
            try:
                person_id = int(person_id)
            except Exception:
                continue
            if not self._is_person_visible(person_id):
                continue
            angles = self._get_memory_value(
                "PeoplePerception/Person/%s/AnglesYawPitch" % person_id,
                None,
            )
            distance = self._get_memory_value(
                "PeoplePerception/Person/%s/Distance" % person_id,
                None,
            )
            if not isinstance(angles, (list, tuple)) or len(angles) < 2:
                continue
            if distance is None:
                continue
            try:
                result.append({
                    "id": person_id,
                    "yaw": float(angles[0]),
                    "pitch": float(angles[1]),
                    "distance": float(distance),
                })
            except Exception:
                continue
        self.logger.info("People snapshot contains %s visible people",
                         len(result))
        return result

    def _is_person_visible(self, person_id):
        visible = self._get_memory_value(
            "PeoplePerception/Person/%s/IsVisible" % person_id,
            True,
        )
        return bool(visible)

    def _get_memory_value(self, key, default=None):
        if self.memory is None:
            return default
        try:
            return self.memory.getData(key)
        except Exception:
            return default

    def _safe_subscribe(self, service, name):
        if service is None or not hasattr(service, "subscribe"):
            return False
        try:
            service.subscribe(name)
            return True
        except Exception as exc:
            self.logger.info("PeoplePerception subscribe skipped: %s", exc)
            return False

    def _safe_unsubscribe(self, service, name):
        if service is None or not hasattr(service, "unsubscribe"):
            return False
        try:
            service.unsubscribe(name)
            return True
        except Exception as exc:
            self.logger.info("PeoplePerception unsubscribe skipped: %s", exc)
            return False
