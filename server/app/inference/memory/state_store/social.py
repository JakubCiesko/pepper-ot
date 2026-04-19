from app.schemas.robot import PersonMetadata
from app.schemas.robot import RobotMetadata
from app.schemas.robot import SocialPersonMetadata
from app.schemas.scene import TrackedObjectState


class SceneMemoryStoreSocialMixin:
    # TODO: maybe tinker with this
    LOOKING_AT_ROBOT_THRESHOLD = 0.3
    SMILE_CONFIDENCE_THRESHOLD = 0.3
    SMILE_SCORE_THRESHOLD = 0.3
    DEMOGRAPHIC_CONFIDENCE_THRESHOLD = 0.3
    EXPRESSION_CONFIDENCE_THRESHOLD = 0.3
    ENGAGEMENT_ZONE_TO_ATTRIBUTES_MAPPING = {1: "is_near", 2: "is_not_far", 3: "is_far"}
    EXPRESSION_SCORE_INDEX_ATTRIBUTE_SEMANTICS = [
        "neutral",
        "happy",
        "surprised",
        "angry",
        "sad",
    ]
    EYE_CLOSED_THRESHOLD = 0.7

    @staticmethod
    def people_by_id(
        robot_metadata: RobotMetadata | None,
    ) -> dict[int, PersonMetadata]:
        if robot_metadata is None:
            return {}
        return {person.id: person for person in robot_metadata.people}

    @staticmethod
    def social_people_by_id(
        robot_metadata: RobotMetadata | None,
    ) -> dict[int, SocialPersonMetadata]:
        if robot_metadata is None:
            return {}
        return {person.id: person for person in robot_metadata.social_people}

    @classmethod
    def _is_social_attribute(cls, attribute: str) -> bool:
        return (
            attribute
            in {"is_sitting", "is_waving", "is_looking_at_robot", "is_smiling"}
            or attribute.startswith("gaze_")
            or attribute.startswith("gender_")
            or attribute.startswith("age_")
            or attribute.startswith("expression_")
            or attribute.startswith("engagement_zone_")
        )

    @classmethod
    def extract_binary_social_attributes(
        cls, social_person: SocialPersonMetadata | None
    ) -> set[str]:
        if social_person is None:
            return set()

        attrs: set[str] = set()

        if social_person.is_sitting:
            attrs.add("is_sitting")

        waving = social_person.is_waving or any(
            (
                social_person.is_waving_left,
                social_person.is_waving_center,
                social_person.is_waving_right,
            )
        )
        if waving:
            attrs.add("is_waving")

        if social_person.is_looking_at_robot or (
            social_person.looking_at_robot_score is not None
            and social_person.looking_at_robot_score >= cls.LOOKING_AT_ROBOT_THRESHOLD
        ):
            attrs.add("is_looking_at_robot")

        gaze_direction = social_person.gaze_direction
        if gaze_direction and len(gaze_direction) >= 2:
            if all(
                direction.lower().strip() == "center" for direction in gaze_direction
            ):
                attrs.add("is_looking_forward")
            else:
                left_right, up_down = gaze_direction[0], gaze_direction[1]
                if left_right in {"left", "right"}:
                    attrs.add(f"is_looking_{left_right}")
                if up_down in {"up", "down"}:
                    attrs.add(f"is_looking_{up_down}")

        if social_person.engagement_zone is not None:
            eng_zone = int(social_person.engagement_zone)
            eng_zone_attr = cls.ENGAGEMENT_ZONE_TO_ATTRIBUTES_MAPPING.get(eng_zone, "is_far")
            attrs.add(eng_zone_attr)

        gender = (social_person.gender or "").strip().lower()
        if (
            gender in {"female", "male"}
            and social_person.gender_confidence is not None
            and social_person.gender_confidence >= cls.DEMOGRAPHIC_CONFIDENCE_THRESHOLD
        ):
            attrs.add(f"is_{gender}")

        age_bucket = (social_person.age_bucket or "").strip().lower()
        age_is_believable = (
            social_person.age_confidence is not None
            and social_person.age_confidence >= cls.DEMOGRAPHIC_CONFIDENCE_THRESHOLD
        )
        if age_bucket in {"child", "adult", "senior"} and age_is_believable:
            attrs.add(f"is_{age_bucket}")
        age = social_person.age
        if age is not None and age_is_believable:
            attrs.add(f"is_{int(age)}_years_old")
        expression = (social_person.expression or "").strip().lower()
        if (
            expression
            and social_person.expression_confidence is not None
            and social_person.expression_confidence
            >= cls.EXPRESSION_CONFIDENCE_THRESHOLD
        ):
            attrs.add(f"has_{expression}_expression")

        if (
            len(social_person.expression_scores)
            == 5  # number of expressions detected by the robot
        ):
            max_idx = social_person.expression_scores.index(
                max(social_person.expression_scores)
            )  # this is already has_expression_expression
            for i, score in enumerate(social_person.expression_scores):
                if score < cls.EXPRESSION_CONFIDENCE_THRESHOLD:
                    continue
                if i == max_idx:
                    continue
                expr_name = cls.EXPRESSION_SCORE_INDEX_ATTRIBUTE_SEMANTICS[i]
                attrs.add(f"has_a_bit_{expr_name}_expression")

        if (
            social_person.smile_score is not None
            and social_person.smile_confidence is not None
            and social_person.smile_score >= cls.SMILE_SCORE_THRESHOLD
            and social_person.smile_confidence >= cls.SMILE_CONFIDENCE_THRESHOLD
        ):
            attrs.add("is_smiling")

        if (
            social_person.eyes_opened is not None
            and len(social_person.eyes_opened) == 2
        ):
            eyes_scores = social_person.eyes_opened
            left, right = eyes_scores[0], eyes_scores[1]
            if all(
                eye_closed_score <= cls.EYE_CLOSED_THRESHOLD
                for eye_closed_score in eyes_scores
            ):
                attrs.add("has_open_eyes")
            elif left > cls.EYE_CLOSED_THRESHOLD:
                attrs.add("is_blinking_with_left_eye")
            elif right > cls.EYE_CLOSED_THRESHOLD:
                attrs.add("is_blinking_with_right_eye")

        return attrs

    @classmethod
    def merge_person_social_state(
        cls,
        current_attributes: list[str],
        new_social_attributes: set[str],
    ) -> list[str]:
        preserved = [
            attribute
            for attribute in (current_attributes or [])
            if not cls._is_social_attribute(attribute)
        ]
        merged = {*(preserved or []), *new_social_attributes}
        return sorted(merged)

    def update_person_robot_fields(
        self,
        obj: TrackedObjectState,
        *,
        pepper_person_id: int | None,
        robot_person: PersonMetadata | None,
        social_person: SocialPersonMetadata | None,
        fallback_timestamp: float | None,
    ):
        if pepper_person_id is not None:
            obj.pepper_person_id = pepper_person_id
        if robot_person is not None:
            obj.robot_distance = robot_person.distance
            if obj.attributes:
                obj.attributes.append(f"is_{round(obj.robot_distance, 2)}_meters_away")
        if social_person is not None and social_person.engagement_zone is not None:
            obj.robot_engagement_zone = int(social_person.engagement_zone)

        timestamp = (
            social_person.timestamp
            if social_person is not None and social_person.timestamp is not None
            else fallback_timestamp
        )
        if timestamp is not None:
            obj.robot_last_seen_ts = timestamp

        social_attributes = self.extract_binary_social_attributes(social_person)
        if social_person is not None:
            obj.attributes = self.merge_person_social_state(
                obj.attributes,
                social_attributes,
            )
