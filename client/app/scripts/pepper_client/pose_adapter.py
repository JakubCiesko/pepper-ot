from pepper_client import time_utils


class PoseAdapter(object):
    def __init__(self, services, logger):
        self.motion = services.ALMotion
        self.logger = logger

    def snapshot(self):
        head_yaw, head_pitch = self.get_head_angles()
        body_yaw = self.get_body_yaw()
        self.logger.info("PoseAdapter.snapshot output: head_yaw=%s head_pitch=%s, body_yaw=%s", head_yaw, head_pitch, body_yaw)
        return {
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
            "body_yaw": body_yaw,
        }

    def get_head_angles(self):
        if self.motion is None:
            return 0.0, 0.0
        try:
            values = self.motion.getAngles(["HeadYaw", "HeadPitch"], True)
            if values and len(values) >= 2:
                return float(values[0]), float(values[1])
        except Exception as exc:
            self.logger.warning("Failed to read head angles: %s", exc)
        return 0.0, 0.0

    def get_body_yaw(self):
        if self.motion is None:
            return None
        try:
            position = self.motion.getRobotPosition(False)
            if position and len(position) >= 3:
                return float(position[2])
        except Exception:
            return None
        return None

    def move_head(self, yaw, pitch, speed):
        if self.motion is None:
            self.logger.info("ALMotion unavailable, skipping head move")
            return False
        try:
            self.motion.setAngles(["HeadYaw", "HeadPitch"], [float(yaw), float(pitch)], float(speed))
            self.logger.info("Moved head to yaw=%s pitch=%s speed=%s", yaw, pitch, speed)
            return True
        except Exception as exc:
            self.logger.warning("Failed to move head: %s", exc)
            return False

    def restore_head(self, pose, speed):
        if not pose:
            return False
        yaw = pose.get("head_yaw", 0.0)
        pitch = pose.get("head_pitch", 0.0)
        ok = self.move_head(yaw, pitch, speed)
        if ok:
            time_utils.sleep_seconds(0.2)
        return ok
