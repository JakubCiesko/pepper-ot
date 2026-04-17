import io
import time
# todo: time utils use instead of time module
from PIL import Image

from pepper_client.utils.error_policy import CameraCaptureError


class CameraAdapter(object):
    def __init__(self, services, config, logger):
        self.services = services
        self.config = config
        self.logger = logger
        self.video = services.ALVideoDevice
        self.app_id = config["app"]["app_id"]

    def capture_frame(self, frame_id=None):
        if self.video is None:
            raise CameraCaptureError("ALVideoDevice is not available")

        capture_cfg = self.config["capture"]
        subscriber_name = "%s_%s" % (self.app_id, time.time())
        handle = None
        try:
            self.logger.info(
                "Subscribing to camera: camera_id=%s resolution=%s colorspace=%s fps=%s",
                capture_cfg["camera_id"],
                capture_cfg["resolution"],
                capture_cfg["color_space"],
                capture_cfg["fps"],
            )
            handle = self.video.subscribeCamera(
                subscriber_name,
                int(capture_cfg["camera_id"]),
                int(capture_cfg["resolution"]),
                int(capture_cfg["color_space"]),
                int(capture_cfg["fps"]),
            )
            nao_image = self.video.getImageRemote(handle)
            if nao_image is None or len(nao_image) < 7:
                raise CameraCaptureError("Camera returned no image")

            width = int(nao_image[0])
            height = int(nao_image[1])
            raw_data = self._coerce_bytes(nao_image[6])
            image = self._build_image(width, height, raw_data)
            jpeg_bytes = self._encode_jpeg(image, capture_cfg.get("jpeg_quality", 90))
            camera_hfov, camera_vfov = self._camera_fov(int(capture_cfg["camera_id"]))
            timestamp = time.time()
            self.logger.info(
                "Captured frame frame_id=%s size=%sx%s hfov=%s vfov=%s",
                frame_id,
                width,
                height,
                camera_hfov,
                camera_vfov,
            )
            return {
                "frame_id": frame_id,
                "image_bytes": jpeg_bytes,
                "image_width": width,
                "image_height": height,
                "timestamp": timestamp,
                "camera_hfov": camera_hfov,
                "camera_vfov": camera_vfov,
            }
        except CameraCaptureError:
            raise
        except Exception as exc:
            raise CameraCaptureError(str(exc))
        finally:
            if handle:
                try:
                    self.video.unsubscribe(handle)
                    self.logger.info("Camera unsubscribed")
                except Exception as exc:
                    self.logger.warning("Failed to unsubscribe camera: %s", exc)

    def _build_image(self, width, height, raw_data):
        try:
            return Image.frombytes("RGB", (width, height), raw_data)
        except AttributeError:
            # TODO: check whether this works
            return Image.fromstring("RGB", (width, height), raw_data)

    def _encode_jpeg(self, image, quality):
        buffer_handle = io.BytesIO()
        image.save(buffer_handle, format="JPEG", quality=int(quality))
        return buffer_handle.getvalue()

    def _camera_fov(self, camera_id):
        hfov = None
        vfov = None
        try:
            hfov = float(self.video.getHorizontalFOV(camera_id))
        except Exception:
            pass
        try:
            vfov = float(self.video.getVerticalFOV(camera_id))
        except Exception:
            pass
        return hfov, vfov

    def _coerce_bytes(self, raw_data):
        if raw_data is None:
            raise CameraCaptureError("Camera returned empty byte buffer")
        if isinstance(raw_data, bytearray):
            return bytes(raw_data)
        if isinstance(raw_data, bytes):
            return raw_data
        try:
            return bytes(raw_data)
        except Exception:
            return str(raw_data)

# TODO: remove this, this is here just for now for use with virtual robot
class FakeCameraAdapter(object):
    def __init__(self, folder, logger):
        import os
        self.folder = folder
        self.logger = logger
        self.image_paths = [
            os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not self.image_paths:
            raise RuntimeError("No images found in folder: %s" % folder)


    def capture_frame(self, frame_id=None):
        import random
        from io import BytesIO
        path = random.choice(self.image_paths)
        nao_image = Image.open(path).convert("RGB")
        buffer = BytesIO()
        nao_image.save(buffer, format="JPEG", quality=90)
        jpeg_bytes = buffer.getvalue()
        width = int(nao_image.width)
        height = int(nao_image.height)
        camera_hfov, camera_vfov = 0.5, 0.5
        timestamp = time.time()
        self.logger.info(
            "Captured frame frame_id=%s size=%sx%s hfov=%s vfov=%s",
            frame_id,
            width,
            height,
            camera_hfov,
            camera_vfov,
        )
        return {
            "frame_id": frame_id,
            "image_bytes": jpeg_bytes,
            "image_width": width,
            "image_height": height,
            "timestamp": timestamp,
            "camera_hfov": camera_hfov,
            "camera_vfov": camera_vfov,
        }
