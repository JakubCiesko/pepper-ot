from pydantic import BaseModel
from pydantic import Field


class PersonMetadata(BaseModel):
    id: int
    yaw: float
    pitch: float
    distance: float


class SocialPersonMetadata(BaseModel):
    id: int
    timestamp: float | None = None
    engagement_zone: int | None = None
    is_waving_left: bool | None = None
    is_waving_center: bool | None = None
    is_waving_right: bool | None = None
    is_waving: bool | None = None
    is_sitting: bool | None = None
    is_looking_at_robot: bool | None = None
    looking_at_robot_score: float | None = None
    head_angles: list[float] = Field(default_factory=list)
    gaze_direction: list[str] = Field(default_factory=list)
    gender_code: float | None = None
    gender: str | None = None
    gender_confidence: float | None = None
    age: float | None = None
    age_bucket: str | None = None
    age_confidence: float | None = None
    expression_scores: list[float] = Field(default_factory=list)
    expression: str | None = None
    expression_confidence: float | None = None
    smile_score: float | None = None
    smile_confidence: float | None = None
    eyes_opened: list[float] = Field(default_factory=list)


class RobotMetadata(BaseModel):
    head_yaw: float = Field(..., description="Current HeadYaw in radians")
    head_pitch: float = Field(..., description="Current HeadPitch in radians")
    body_yaw: float | None = Field(None, description="Body yaw in radians")
    camera_hfov: float | None = Field(None, description="Camera HFOV in radians")
    camera_vfov: float | None = Field(None, description="Camera VFOV in radians")
    image_width: int | None = Field(None, description="Image width in pixels")
    image_height: int | None = Field(None, description="Image height in pixels")
    timestamp: float | None = Field(None, description="Capture timestamp (s)")
    frame_id: str | None = Field(None, description="Frame identifier")
    scan_id: str | None = Field(None, description="Scan session identifier")
    capture_mode: str | None = Field(None, description="Capture mode, e.g. scan/single")
    people: list[PersonMetadata] = Field(default_factory=list)
    social_people: list[SocialPersonMetadata] = Field(default_factory=list)
    battery: int | None = None  # this might be redundant

    @classmethod
    def merge_robot_metadata_for_panorama(
        cls,
        metadata_list: list["RobotMetadata"],
    ) -> "RobotMetadata":
        """
        Merge multiple RobotMetadata objects into one suitable for a stitched panorama.
        Assumes images are placed side by side in the provided order.
        """
        if not metadata_list:
            raise ValueError("No metadata provided for panorama merging.")

        base = metadata_list[0]

        # Aggregate geometry
        total_width = sum(m.image_width or 0 for m in metadata_list)
        max_height = max((m.image_height or 0) for m in metadata_list)

        hfovs = [m.camera_hfov for m in metadata_list if m.camera_hfov is not None]
        vfovs = [m.camera_vfov for m in metadata_list if m.camera_vfov is not None]

        total_hfov = sum(hfovs) if hfovs else None
        avg_vfov = sum(vfovs) / len(vfovs) if vfovs else None

        # Compute yaw offsets for each image
        yaw_offsets = []
        current_offset = 0.0
        for m in metadata_list:
            yaw_offsets.append(current_offset)
            current_offset += m.camera_hfov or 0.0

        # Recenter so that the middle image is yaw = 0
        mid = len(metadata_list) // 2
        center_yaw = sum(metadata_list[i].camera_hfov or 0.0 for i in range(mid))

        yaw_offsets = [offset - center_yaw for offset in yaw_offsets]

        merged_people: list[PersonMetadata] = []
        merged_social_people: list[SocialPersonMetadata] = []

        for idx, meta in enumerate(metadata_list):
            offset = yaw_offsets[idx]

            # Merge detected people
            merged_people.extend(
                [
                    PersonMetadata(
                        id=person.id,
                        yaw=person.yaw + offset,
                        pitch=person.pitch,
                        distance=person.distance,
                    )
                    for person in meta.people
                ]
            )

            merged_social_people.extend(
                [
                    SocialPersonMetadata(**social.model_dump())
                    for social in meta.social_people
                ]
            )

        # Merge timestamps
        timestamps = [m.timestamp for m in metadata_list if m.timestamp is not None]
        timestamp = min(timestamps) if timestamps else None

        # Merge frame IDs
        frame_ids = [m.frame_id for m in metadata_list if m.frame_id is not None]
        frame_id = "+".join(frame_ids) if frame_ids else None

        return RobotMetadata(
            head_yaw=base.head_yaw,
            head_pitch=base.head_pitch,
            body_yaw=base.body_yaw,
            camera_hfov=total_hfov,
            camera_vfov=avg_vfov,
            image_width=total_width,
            image_height=max_height,
            timestamp=timestamp,
            frame_id=frame_id,
            scan_id=base.scan_id,
            capture_mode="panorama",
            people=merged_people,
            social_people=merged_social_people,
            battery=base.battery,
        )
