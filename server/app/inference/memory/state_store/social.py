from app.schemas.robot import PersonMetadata
from app.schemas.robot import RobotMetadata
from app.schemas.robot import SocialPersonMetadata
from app.schemas.scene import TrackedObjectState


class SceneMemoryStoreSocialMixin:
    LOOKING_AT_ROBOT_THRESHOLD = 0.5
    SMILE_CONFIDENCE_THRESHOLD = 0.5
    SMILE_SCORE_THRESHOLD = 0.5
    DEMOGRAPHIC_CONFIDENCE_THRESHOLD = 0.5
    EXPRESSION_CONFIDENCE_THRESHOLD = 0.5

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

        gaze_direction = (social_person.gaze_direction or "").strip().lower()
        if gaze_direction in {"left", "right", "center"}:
            attrs.add(f"gaze_{gaze_direction}")

        if social_person.engagement_zone is not None:
            attrs.add(f"engagement_zone_{int(social_person.engagement_zone)}")

        gender = (social_person.gender or "").strip().lower()
        if (
            gender in {"female", "male"}
            and social_person.gender_confidence is not None
            and social_person.gender_confidence >= cls.DEMOGRAPHIC_CONFIDENCE_THRESHOLD
        ):
            attrs.add(f"gender_{gender}")

        age_bucket = (social_person.age_bucket or "").strip().lower()
        if (
            age_bucket in {"child", "adult", "senior"}
            and social_person.age_confidence is not None
            and social_person.age_confidence >= cls.DEMOGRAPHIC_CONFIDENCE_THRESHOLD
        ):
            attrs.add(f"age_{age_bucket}")

        expression = (social_person.expression or "").strip().lower()
        if (
            expression
            and social_person.expression_confidence is not None
            and social_person.expression_confidence
            >= cls.EXPRESSION_CONFIDENCE_THRESHOLD
        ):
            attrs.add(f"expression_{expression}")

        if (
            social_person.smile_score is not None
            and social_person.smile_confidence is not None
            and social_person.smile_score >= cls.SMILE_SCORE_THRESHOLD
            and social_person.smile_confidence >= cls.SMILE_CONFIDENCE_THRESHOLD
        ):
            attrs.add("is_smiling")

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
