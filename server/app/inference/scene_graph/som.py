from concurrent.futures import ThreadPoolExecutor
import logging
import os

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
import supervision as sv

from app.inference.types import InferenceDetectionObject

logger = logging.getLogger(__name__)


class SoMPainter:
    """Handles Set-of-Mark overlay on pictures for scene graph generation (https://som-gpt4v.github.io/)"""

    def __init__(
        self,
        line_thickness: int = 2,
        color_lookup: str = "index",
        mask_opacity: float = 0.5,
        mask_backend: str = "grabcut",
        device: str = "cuda",
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
        self.mask_backend = (
            mask_backend if mask_backend in {"grabcut", "sam"} else "grabcut"
        )
        self.device = device
        self.processor = None
        self.model = None

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(value, max_value))

    def _ensure_sam_loaded(self) -> bool:
        if self.mask_backend != "sam":
            return False
        if self.processor is not None and self.model is not None:
            return True
        try:
            # Lazy import so startup does not fail when transformers/SAM isn't available.
            from transformers import Sam3Model  # type: ignore
            from transformers import Sam3Processor  # type: ignore

            self.processor = Sam3Processor.from_pretrained("facebook/sam3")
            self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
            logger.info("SAM3 mask backend initialized on device=%s", self.device)
            return True
        except Exception as exc:
            logger.warning(
                "Failed to initialize SAM3 (%s). Falling back to grabcut masks.", exc
            )
            self.mask_backend = "grabcut"
            self.processor, self.model = None, None
            return False

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
        masks = None
        if mask or polygon:
            if self.mask_backend == "sam":
                masks = self.sam_bboxes_to_masks(image=image, bboxes=xyxy)
            if masks is None:
                masks = self.bboxes_to_masks(
                    image=image,
                    bboxes=xyxy,
                    scale=grab_cut_scale,
                    iter_count=grab_cut_iter_count,
                    use_roi_grab_cut=use_roi_grab_cut,
                    max_mask_workers=max_mask_workers,
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

    @staticmethod
    def _bbox_iou(a: NDArray, b: NDArray) -> float:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0
    #TODO: implement batching for smaller GPU load 
    def sam_bboxes_to_masks(
        self,
        image: Image.Image | NDArray,
        bboxes: NDArray,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        batch_size: int = 4,
    ) -> NDArray | None:
        """
        Convert bounding boxes to object masks using SAM3 box prompts.
        Returns bool masks of shape (n, H, W), or None on failure (so caller can fallback).
        """
        if bboxes is None:
            return None
        if len(bboxes) == 0:
            if isinstance(image, Image.Image):
                h, w = image.height, image.width
            else:
                h, w = image.shape[:2]
            return np.empty((0, h, w), dtype=bool)

        if not self._ensure_sam_loaded():
            return None

        try:
            import torch  # type: ignore
        except Exception as exc:
            logger.warning(
                "Torch unavailable for SAM3: %s. Falling back to grabcut.", exc
            )
            self.mask_backend = "grabcut"
            return None

        try:
            if isinstance(image, Image.Image):
                pil_img = image.convert("RGB")
                img_h, img_w = pil_img.height, pil_img.width
            else:
                arr = np.asarray(image)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                pil_img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
                img_h, img_w = arr.shape[:2]

            clipped_boxes: list[list[float]] = []
            for box in bboxes:
                x1, y1, x2, y2 = map(float, box)
                x1 = max(0.0, min(x1, img_w - 1.0))
                y1 = max(0.0, min(y1, img_h - 1.0))
                x2 = max(x1 + 1.0, min(x2, float(img_w)))
                y2 = max(y1 + 1.0, min(y2, float(img_h)))
                clipped_boxes.append([x1, y1, x2, y2])

            if not clipped_boxes:
                return np.empty((0, img_h, img_w), dtype=bool)

            box_labels = [[1 for _ in clipped_boxes]]
            inputs = self.processor(
                images=pil_img,
                input_boxes=[clipped_boxes],
                input_boxes_labels=box_labels,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = inputs.get("original_sizes")
            if target_sizes is None:
                target_sizes = [[img_h, img_w]]
            else:
                target_sizes = target_sizes.tolist()

            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=target_sizes,
            )
            result = results[0] if results else {}
            masks_pred = result.get("masks")
            boxes_pred = result.get("boxes")

            out = np.zeros((len(clipped_boxes), img_h, img_w), dtype=bool)
            if masks_pred is None or boxes_pred is None:
                return out

            if hasattr(masks_pred, "detach"):
                masks_np = masks_pred.detach().cpu().numpy()
            else:
                masks_np = np.asarray(masks_pred)
            if masks_np.ndim == 2:
                masks_np = masks_np[None, ...]
            masks_np = masks_np.astype(bool)

            if hasattr(boxes_pred, "detach"):
                boxes_np = boxes_pred.detach().cpu().numpy()
            else:
                boxes_np = np.asarray(boxes_pred)
            boxes_np = np.asarray(boxes_np, dtype=np.float32).reshape(-1, 4)

            # Assign one best SAM mask per requested box by IoU.
            for i, in_box in enumerate(clipped_boxes):
                if len(boxes_np) == 0 or len(masks_np) == 0:
                    break
                best_idx = -1
                best_iou = 0.0
                in_box_np = np.asarray(in_box, dtype=np.float32)
                for j, pred_box in enumerate(boxes_np):
                    iou = self._bbox_iou(in_box_np, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = j
                if best_idx >= 0 and best_iou >= 0.05:
                    out[i] = masks_np[best_idx]
                else:
                    # Conservative fallback: keep bbox rectangle for this item.
                    x1, y1, x2, y2 = map(int, in_box)
                    out[i, y1:y2, x1:x2] = True
            return out
        except Exception as exc:
            logger.warning(
                "SAM3 mask inference failed, falling back to grabcut: %s", exc
            )
            return None
