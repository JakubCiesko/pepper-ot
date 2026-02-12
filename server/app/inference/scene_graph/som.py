import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import supervision as sv

from ..types import DetectionObject


class SoMPainter:
    """Handles Set-of-Mark overlay on pictures for scene graph generation (https://som-gpt4v.github.io/)"""

    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(
            thickness=2, color_lookup=sv.ColorLookup.INDEX
        )
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.CENTER,
            text_color=sv.Color.BLACK,
            color=sv.Color.WHITE,
        )
        self.mask_annotator = sv.MaskAnnotator(
            color=sv.Color.BLACK,
            color_lookup=sv.ColorLookup.INDEX,
            opacity=0.5,
        )
        self.polygon_annotator = sv.PolygonAnnotator(
            color_lookup=sv.ColorLookup.INDEX,
            thickness=2,
        )

    def paint(
        self,
        image: NDArray | Image.Image,
        detections: list[DetectionObject],
        class_names: bool = False,
        bbox: bool = False,
        mask: bool = False,
        polygon: bool = False,
        grab_cut_scale: float = 0.25,
        grab_cut_iter_count: int = 5,
    ) -> np.ndarray:
        """
        Applies SoM visualization to an image.
        """

        # start with 1 because 0 is never displayed somehow
        labels = [
            f"{(det.label + "_" if class_names else "") + str(det.object_id + 1)}"
            for det in detections
        ]  # # was there
        xyxy = np.array([det.bbox for det in detections])
        masks = (
            self.bboxes_to_masks(image, xyxy, grab_cut_scale, grab_cut_iter_count)
            if mask or polygon
            else None
        )
        conf = np.array([det.confidence for det in detections])
        ids = np.array([det.class_id for det in detections])
        detections = sv.Detections(xyxy=xyxy, confidence=conf, class_id=ids, mask=masks)
        annotated_image = (
            self.box_annotator.annotate(scene=image.copy(), detections=detections)
            if bbox
            else image.copy()
        )
        if mask:
            annotated_image = self.mask_annotator.annotate(
                scene=annotated_image, detections=detections
            )
        if polygon:
            annotated_image = self.polygon_annotator.annotate(
                scene=annotated_image, detections=detections
            )
        annotated_image = self.label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )
        return annotated_image

    @staticmethod
    def bboxes_to_masks(
        image: Image.Image, bboxes: np.ndarray, scale: float = 0.25, iter_count: int = 5
    ) -> np.ndarray:
        """
        Convert bounding boxes to approximate masks using GrabCut (scaled image for speed).
        Returns boolean masks of shape (n, H, W).
        """
        orig_img = np.array(image.convert("RGB"))
        H, W = orig_img.shape[:2]

        # Scale image so that it is fast
        img = cv2.resize(orig_img, (0, 0), fx=scale, fy=scale)
        Hs, Ws = img.shape[:2]

        masks = []

        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            # Scale bbox to resized image
            rect = (
                int(x1 * scale),
                int(y1 * scale),
                int((x2 - x1) * scale),
                int((y2 - y1) * scale),
            )

            mask = np.zeros((Hs, Ws), np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)

            # GrabCut on scaled image
            cv2.grabCut(
                img, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT
            )

            # Boolean mask on scaled image
            mask_bool = (mask == 1) | (mask == 3)

            # Resize mask back to original size
            mask_orig_size = cv2.resize(
                mask_bool.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            masks.append(mask_orig_size)

        return np.stack(masks, axis=0)
