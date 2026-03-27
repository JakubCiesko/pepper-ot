from concurrent.futures import ThreadPoolExecutor
import os

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import supervision as sv

from app.inference.types import InferenceDetectionObject


class SoMPainter:
    """Handles Set-of-Mark overlay on pictures for scene graph generation (https://som-gpt4v.github.io/)"""

    def __init__(
        self,
        line_thickness: int = 2,
        color_lookup: str = "index",
        mask_opacity: float = 0.5,
    ):
        lookup = {
            "index": sv.ColorLookup.INDEX,
            "class": sv.ColorLookup.CLASS,
            "track": sv.ColorLookup.TRACK,
        }.get(color_lookup, sv.ColorLookup.INDEX)
        self.box_annotator = sv.BoxAnnotator(
            thickness=line_thickness, color_lookup=lookup
        )
        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.CENTER,
            text_color=sv.Color.BLACK,
            color=sv.Color.WHITE,
        )
        self.mask_annotator = sv.MaskAnnotator(
            color=sv.Color.BLACK,
            color_lookup=lookup,
            opacity=mask_opacity,
        )
        self.polygon_annotator = sv.PolygonAnnotator(
            color_lookup=lookup,
            thickness=line_thickness,
        )
        self._base_line_thickness = line_thickness
        self._base_text_scale = float(self.label_annotator.text_scale)
        self._base_text_thickness = int(self.label_annotator.text_thickness)
        self._base_text_padding = int(self.label_annotator.text_padding)

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(value, max_value))

    def paint(
        self,
        image: NDArray | Image.Image,
        detections: list[InferenceDetectionObject],
        class_names: bool = False,
        bbox: bool = False,
        mask: bool = False,
        polygon: bool = False,
        grab_cut_scale: float = 0.25,
        grab_cut_iter_count: int = 5,
        use_roi_grab_cut: bool = False,
        max_mask_workers: int | None = None,
        auto_style: bool = True,
        style_ref_size: int = 640,
        min_text_scale: float = 0.5,
        max_text_scale: float = 2.5,
        min_text_thickness: int = 1,
        max_text_thickness: int = 6,
        min_line_thickness: int = 1,
        max_line_thickness: int = 8,
        min_text_padding: int = 10,
        max_text_padding: int = 30,
    ) -> NDArray:
        """
        Applies SoM visualization to an image.
        """
        image_h = image.shape[0] if isinstance(image, np.ndarray) else image.height
        image_w = image.shape[1] if isinstance(image, np.ndarray) else image.width

        if auto_style:
            reference = max(1, style_ref_size)
            factor = min(image_h, image_w) / reference
            line_thickness = int(
                round(
                    self._clamp(
                        self._base_line_thickness * factor,
                        float(min_line_thickness),
                        float(max_line_thickness),
                    )
                )
            )
            text_scale = float(
                self._clamp(
                    self._base_text_scale * factor,
                    min_text_scale,
                    max_text_scale,
                )
            )
            text_thickness = int(
                round(
                    self._clamp(
                        self._base_text_thickness * factor,
                        float(min_text_thickness),
                        float(max_text_thickness),
                    )
                )
            )
            # Requested as method arguments; padding is intentionally kept unchanged.
            _ = (min_text_padding, max_text_padding)
        else:
            line_thickness = self._base_line_thickness
            text_scale = self._base_text_scale
            text_thickness = self._base_text_thickness

        self.box_annotator.thickness = line_thickness
        self.polygon_annotator.thickness = line_thickness
        self.label_annotator.text_scale = text_scale
        self.label_annotator.text_thickness = text_thickness

        # start with 1 because 0 is never displayed somehow
        labels = [
            f"{(det.label + "_" if class_names else "") + str(det.object_id)}"  # i sometimes add 1 to det.object_id because the painetr sometimes does not display 0, i dont know why
            for det in detections
        ]  # # was there
        xyxy = np.array([det.bbox for det in detections], dtype=np.float32)
        if xyxy.size == 0:
            xyxy = xyxy.reshape(0, 4)
        masks = (
            self.bboxes_to_masks(
                image=image,
                bboxes=xyxy,
                scale=grab_cut_scale,
                iter_count=grab_cut_iter_count,
                use_roi_grab_cut=use_roi_grab_cut,
                max_mask_workers=max_mask_workers,
            )
            if mask or polygon
            else None
        )
        conf = np.array([det.confidence for det in detections], dtype=np.float32)
        ids = np.array([det.class_id for det in detections], dtype=np.int32)
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
        image: Image.Image | NDArray,
        bboxes: NDArray,
        scale: float = 0.25,
        iter_count: int = 5,
        use_roi_grab_cut: bool = True,
        max_mask_workers: int | None = None,
    ) -> NDArray:
        """
        Convert bounding boxes to approximate masks using GrabCut.
        ROI mode runs GrabCut per bounding box crop over a scaled frame.
        Returns boolean masks of shape (n, H, W).
        """
        orig_img = (
            np.array(image.convert("RGB")) if isinstance(image, Image.Image) else image
        )
        H, W = orig_img.shape[:2]
        if bboxes is None or len(bboxes) == 0:
            return np.empty((0, H, W), dtype=bool)

        scale = max(float(scale), 1e-3)
        img = cv2.resize(orig_img, (0, 0), fx=scale, fy=scale)
        Hs, Ws = img.shape[:2]
        if Hs <= 1 or Ws <= 1:
            return np.zeros((len(bboxes), H, W), dtype=bool)

        def empty_mask() -> np.ndarray:
            return np.zeros((H, W), dtype=bool)

        def full_frame_mask_for_box(
            idx: int, box: np.ndarray
        ) -> tuple[int, np.ndarray]:
            x1, y1, x2, y2 = map(float, box)
            if x2 <= x1 or y2 <= y1:
                return idx, empty_mask()

            sx1 = int(np.floor(x1 * scale))
            sy1 = int(np.floor(y1 * scale))
            sx2 = int(np.ceil(x2 * scale))
            sy2 = int(np.ceil(y2 * scale))

            sx1 = max(0, min(sx1, Ws - 1))
            sy1 = max(0, min(sy1, Hs - 1))
            sx2 = max(sx1 + 1, min(sx2, Ws))
            sy2 = max(sy1 + 1, min(sy2, Hs))
            if sx2 <= sx1 or sy2 <= sy1:
                return idx, empty_mask()

            if use_roi_grab_cut:
                roi = img[sy1:sy2, sx1:sx2]
                if roi.size == 0:
                    return idx, empty_mask()
                rh, rw = roi.shape[:2]
                if rh <= 1 or rw <= 1:
                    return idx, empty_mask()

                gc_mask = np.zeros((rh, rw), np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                rect = (0, 0, rw, rh)
                try:
                    cv2.grabCut(
                        roi,
                        gc_mask,
                        rect,
                        bgd_model,
                        fgd_model,
                        iter_count,
                        cv2.GC_INIT_WITH_RECT,
                    )
                    roi_mask = ((gc_mask == 1) | (gc_mask == 3)).astype(np.uint8)
                except cv2.error:
                    roi_mask = np.ones((rh, rw), dtype=np.uint8)
                mask_scaled = np.zeros((Hs, Ws), dtype=np.uint8)
                mask_scaled[sy1:sy2, sx1:sx2] = roi_mask
            else:
                rect = (sx1, sy1, sx2 - sx1, sy2 - sy1)
                gc_mask = np.zeros((Hs, Ws), np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                try:
                    cv2.grabCut(
                        img,
                        gc_mask,
                        rect,
                        bgd_model,
                        fgd_model,
                        iter_count,
                        cv2.GC_INIT_WITH_RECT,
                    )
                    mask_scaled = ((gc_mask == 1) | (gc_mask == 3)).astype(np.uint8)
                except cv2.error:
                    mask_scaled = np.zeros((Hs, Ws), dtype=np.uint8)
                    mask_scaled[sy1:sy2, sx1:sx2] = 1

            mask_orig_size = cv2.resize(
                mask_scaled,
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            return idx, mask_orig_size

        if max_mask_workers is None:
            max_mask_workers = max(1, min(len(bboxes), os.cpu_count() or 1))
        max_mask_workers = max(1, int(max_mask_workers))

        masks: list[np.ndarray | None] = [None] * len(bboxes)
        if max_mask_workers == 1 or len(bboxes) <= 1:
            for idx, box in enumerate(bboxes):
                out_idx, out_mask = full_frame_mask_for_box(idx, box)
                masks[out_idx] = out_mask
        else:
            try:
                with ThreadPoolExecutor(max_workers=max_mask_workers) as executor:
                    for out_idx, out_mask in executor.map(
                        lambda pair: full_frame_mask_for_box(pair[0], pair[1]),
                        enumerate(bboxes),
                    ):
                        masks[out_idx] = out_mask
            except Exception:
                for idx, box in enumerate(bboxes):
                    out_idx, out_mask = full_frame_mask_for_box(idx, box)
                    masks[out_idx] = out_mask

        resolved = [m if m is not None else empty_mask() for m in masks]
        return np.stack(resolved, axis=0)
