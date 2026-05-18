TEXT_TYPES = (str, )


class FaceAdapter(object):
    """Class wrapper around ALMemory and ALFaceDetection. 
    Builds metadata about people faces used for attribution on server-side"""

    def __init__(self, services, config, logger):
        self.services = services
        self.config = config
        self.logger = logger
        self.memory = services.ALMemory
        self.face_detection = services.ALFaceDetection
        self.subscription_name = "%s_face" % config["app"]["service_name"]
        self._subscribed = False

    def start(self):
        if not self.config["social"].get("enable_face_detection", True):
            return
        self._subscribed = self._safe_subscribe(self.face_detection,
                                                self.subscription_name)
        if self._subscribed:
            self.logger.info("Subscribed to ALFaceDetection")

    def stop(self):
        if self._subscribed:
            self._safe_unsubscribe(self.face_detection, self.subscription_name)
            self._subscribed = False

    def snapshot_faces(self):
        self.logger.info("Running FaceAdapter.snapshot_faces")
        payload = self._get_memory_value("FaceDetected", None)
        if not isinstance(payload, (list, tuple)) or len(payload) < 2:
            return []
        faces_block = payload[1]
        if not isinstance(faces_block, (list, tuple)):
            return []
        result = []
        for face_entry in faces_block:
            if not isinstance(face_entry,
                              (list, tuple)) or len(face_entry) < 2:
                continue
            shape = face_entry[0] if isinstance(face_entry[0],
                                                (list, tuple)) else []
            extra = face_entry[1] if isinstance(face_entry[1],
                                                (list, tuple)) else []
            yaw = self._pick_float(shape, 0)
            pitch = self._pick_float(shape, 1)
            label = self._pick_label(extra)
            confidence = self._pick_confidence(extra)
            if label:
                result.append({
                    "yaw": yaw,
                    "pitch": pitch,
                    "face_label": label,
                    "face_confidence": confidence,
                })
        self.logger.info("Face snapshot contains %s recognized faces",
                         len(result))
        return result

    def match_faces_to_people(self, people):
        self.logger.info("Matching Faces to People")
        face_matches = {}
        max_delta = float(self.config["social"].get("face_match_max_angle_rad",
                                                    0.35))
        faces = self.snapshot_faces()
        for face in faces:
            best_person_id = None
            best_delta = None
            for person in people:
                yaw = face.get("yaw")
                pitch = face.get("pitch")
                if yaw is None or pitch is None:
                    continue
                delta = abs(float(person.get("yaw", 0.0)) - float(yaw)) + abs(
                    float(person.get("pitch", 0.0)) - float(pitch))
                if delta > max_delta:
                    continue
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_person_id = person["id"]
            if best_person_id is None:
                continue
            current = face_matches.get(best_person_id)
            if current is None or (face.get("face_confidence")
                                   or 0.0) > (current.get("face_confidence")
                                              or 0.0):
                face_matches[best_person_id] = face
        self.logger.info("Face Matches: %s", len(face_matches))
        return face_matches

    def _get_memory_value(self, key, default=None):
        if self.memory is None:
            return default
        try:
            return self.memory.getData(key)
        except Exception:
            return default

    def _pick_float(self, values, index):
        try:
            return float(values[index])
        except Exception:
            return None

    def _pick_label(self, values):
        for value in reversed(list(values)):
            if isinstance(value, TEXT_TYPES) and value.strip():
                return value.strip()
        return None

    def _pick_confidence(self, values):
        best = None
        for value in values:
            try:
                number = float(value)
            except Exception:
                continue
            if 0.0 <= number <= 1.0 and (best is None or number > best):
                best = number
        return best

    def _safe_subscribe(self, service, name):
        if service is None or not hasattr(service, "subscribe"):
            return False
        try:
            service.subscribe(name)
            return True
        except Exception as exc:
            self.logger.info("FaceDetection subscribe skipped: %s", exc)
            return False

    def _safe_unsubscribe(self, service, name):
        if service is None or not hasattr(service, "unsubscribe"):
            return False
        try:
            service.unsubscribe(name)
            return True
        except Exception as exc:
            self.logger.info("FaceDetection unsubscribe skipped: %s", exc)
            return False
