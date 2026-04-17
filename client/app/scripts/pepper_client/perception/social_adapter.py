from pepper_client.utils import timing as time_utils


class SocialAdapter(object):
    def __init__(self, services, config, logger, face_adapter):
        self.services = services
        self.config = config
        self.logger = logger
        self.face_adapter = face_adapter
        self.memory = services.ALMemory
        self.subscription_name = "%s_social" % config["app"]["service_name"]
        self._subscribed_services = []
        self._service_map = {
            "face_characteristics": (services.ALFaceCharacteristics, config["social"].get("enable_face_characteristics", True)),
            "gaze_analysis": (services.ALGazeAnalysis, config["social"].get("enable_gaze_analysis", True)),
            "engagement_zones": (services.ALEngagementZones, config["social"].get("enable_engagement_zones", True)),
            "sitting_detection": (services.ALSittingPeopleDetection, config["social"].get("enable_sitting_detection", True)),
            "waving_detection": (services.ALWavingDetection, config["social"].get("enable_waving_detection", True)),
        }

    def start(self):
        for label, (service, enabled) in sorted(self._service_map.items()):
            if not enabled:
                continue
            if self._safe_subscribe(service, self.subscription_name):
                self._subscribed_services.append(service)
                self.logger.info("Subscribed to %s", label)

    def stop(self):
        for service in list(self._subscribed_services):
            self._safe_unsubscribe(service, self.subscription_name)
        del self._subscribed_services[:]

    def snapshot_social_people(self, people):
        if not people:
            return []
        self.logger.info("Running SocailAdapter.snapshot_social_people on (%s) people=%s", len(people), people)
        face_map = self.face_adapter.match_faces_to_people(people)
        result = []
        for person in people:
            social = self._snapshot_person(person["id"])
            face_data = face_map.get(person["id"], {})
            social.update(face_data)
            if len(social.keys()) > 2:
                result.append(social)
        self.logger.info("Social snapshot contains %s enriched people", len(result))
        return result

    #TODO: check this more
    def _snapshot_person(self, person_id):
        payload = {"id": int(person_id), "timestamp": time_utils.now_ts()}

        age_value, age_conf = self._pair("PeoplePerception/Person/%s/AgeProperties" % person_id)
        if age_value is not None:
            payload["age"] = age_value
            payload["age_confidence"] = age_conf
            payload["age_bucket"] = self._age_bucket(age_value)

        gender_value, gender_conf = self._pair("PeoplePerception/Person/%s/GenderProperties" % person_id)
        if gender_value is not None:
            payload["gender_code"] = gender_value
            payload["gender_confidence"] = gender_conf
            payload["gender"] = self._gender_label(gender_value)

        smile_value, smile_conf = self._pair("PeoplePerception/Person/%s/SmileProperties" % person_id)
        if smile_value is not None:
            payload["smile_score"] = smile_value
            payload["smile_confidence"] = smile_conf

        expression_scores = self._get_memory_value(
            "PeoplePerception/Person/%s/ExpressionProperties" % person_id,
            None,
        )
        expression_name, expression_conf = self._expression(expression_scores)
        if expression_name is not None:
            payload["expression"] = expression_name
            payload["expression_confidence"] = expression_conf
            payload["expression_scores"] = list(expression_scores)

        is_looking_at_robot = self._get_memory_value(
            "PeoplePerception/Person/%s/IsLookingAtRobot" % person_id,
            None,
        )
        if is_looking_at_robot is not None:
            payload["is_looking_at_robot"] = bool(is_looking_at_robot)

        looking_score = self._get_memory_value(
            "PeoplePerception/Person/%s/LookingAtRobotScore" % person_id,
            None,
        )
        if looking_score is not None:
            payload["looking_at_robot_score"] = self._float_or_none(looking_score)

        head_angles = self._get_memory_value(
            "PeoplePerception/Person/%s/HeadAngles" % person_id,
            None,
        )
        if isinstance(head_angles, (list, tuple)) and len(head_angles) >= 2:
            payload["head_angles"] = [self._float_or_none(head_angles[0]), self._float_or_none(head_angles[1])]
            payload["gaze_direction"] = [self._gaze_direction_left_right(head_angles), self._gaze_direction_up_down(head_angles)]

        engagement_zone = self._get_memory_value(
            "PeoplePerception/Person/%s/EngagementZone" % person_id,
            None,
        )
        if engagement_zone is not None:
            try:
                payload["engagement_zone"] = int(engagement_zone)
            except Exception:
                pass

        is_sitting = self._get_memory_value(
            "PeoplePerception/Person/%s/IsSitting" % person_id,
            None,
        )
        if is_sitting is not None:
            payload["is_sitting"] = int(is_sitting) == 1

        waving_center = self._get_memory_value(
            "PeoplePerception/Person/%s/IsWavingCenter" % person_id,
            None,
        )
        waving_left = self._get_memory_value(
            "PeoplePerception/Person/%s/IsWavingLeft" % person_id,
            None,
        )
        waving_right = self._get_memory_value(
            "PeoplePerception/Person/%s/IsWavingRight" % person_id,
            None,
        )
        is_waving = self._get_memory_value(
            "PeoplePerception/Person/%s/IsWaving" % person_id,
            None,
        )
        if any(value is not None for value in [waving_center, waving_left, waving_right, is_waving]):
            payload["is_waving"] = bool(is_waving or waving_center or waving_left or waving_right)
            payload["is_waving_center"] = bool(waving_center) if waving_center is not None else False
            payload["is_waving_left"] = bool(waving_left) if waving_left is not None else False
            payload["is_waving_right"] = bool(waving_right) if waving_right is not None else False

        eyes_opened = self._get_memory_value(
            "PeoplePerception/Person/%s/EyeOpeningDegree" % person_id
        )
        if isinstance(eyes_opened, (list, tuple)) and len(eyes_opened) >= 2:
            payload["eyes_opened"] = eyes_opened


        return payload

    def _pair(self, key):
        value = self._get_memory_value(key, None)
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None, None
        return self._float_or_none(value[0]), self._float_or_none(value[1])

    def _expression(self, scores):
        labels = self.config["social"].get("expression_labels") or []
        if not isinstance(scores, (list, tuple)) or not scores or not labels:
            return None, None
        best_index = None
        best_score = None
        for index, value in enumerate(scores):
            try:
                number = float(value)
            except Exception:
                continue
            if best_score is None or number > best_score:
                best_score = number
                best_index = index
        if best_index is None or best_index >= len(labels):
            return None, None
        return labels[best_index], best_score

    def _gaze_direction_left_right(self, head_angles):
        try:
            yaw = float(head_angles[0])
        except Exception:
            return None
        if yaw < -0.2:
            return "right"
        if yaw > 0.2:
            return "left"
        return "center"

    def _gaze_direction_up_down(self, head_angles):
        try:
            pitch = float(head_angles[1])
        except Exception:
            return None
        if pitch < -0.2:
            return "up"
        if pitch > 0.2:
            return "down"
        return "center"

    def _age_bucket(self, age_value):
        try:
            age_value = float(age_value)
        except Exception:
            return None
        if age_value < 13:
            return "child"
        if age_value < 18:
            return "teen"
        if age_value < 65:
            return "adult"
        return "senior"

    def _gender_label(self, value):
        if value is None:
            return None
        if int(round(value)) == 0:
            return "female"
        if int(round(value)) == 1:
            return "male"
        return "unknown"

    def _float_or_none(self, value):
        try:
            return float(value)
        except Exception:
            return None

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
            self.logger.info("Subscribe skipped for service %s: %s", service, exc)
            return False

    def _safe_unsubscribe(self, service, name):
        if service is None or not hasattr(service, "unsubscribe"):
            return False
        try:
            service.unsubscribe(name)
            return True
        except Exception as exc:
            self.logger.info("Unsubscribe skipped for service %s: %s", service, exc)
            return False
