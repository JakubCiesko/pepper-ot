class MetadataBuilder(object):

    def __init__(self, logger):
        self.logger = logger

    def build(self,
              capture,
              context,
              frame_id,
              scan_id=None,
              capture_mode=None):
        """B
        uild the server RobotMetadata payload from camera capture data and robot-local
        context. The server uses this for geometry fusion, person binding, and memory.
        """
        pose = context.get("pose") or {}
        metadata = {
            "head_yaw": pose.get("head_yaw", 0.0),
            "head_pitch": pose.get("head_pitch", 0.0),
            "body_yaw": pose.get("body_yaw"),
            "camera_hfov": capture.get("camera_hfov"),
            "camera_vfov": capture.get("camera_vfov"),
            "image_width": capture.get("image_width"),
            "image_height": capture.get("image_height"),
            "timestamp": capture.get("timestamp"),
            "frame_id": frame_id,
            "scan_id": scan_id,
            "people": context.get("people") or [],
        }
        social_people = context.get("social_people") or []
        if social_people:
            metadata["social_people"] = social_people
        sonar = context.get("sonar")
        if sonar:
            metadata["sonar"] = sonar
        if capture_mode:
            metadata["capture_mode"] = capture_mode
        self.logger.info(
            "Built metadata payload for frame_id=%s scan_id=%s, metadata=%s",
            frame_id, scan_id, metadata)
        return metadata
