import numpy as np
import supervision as sv


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

    def paint(
        self, image: np.ndarray, detections: sv.Detections, bbox: bool = False
    ) -> np.ndarray:
        """
        Applies SoM visualization to an image.
        """

        # start with 1 because 0 is never displayed somehow
        labels = [f"{i + 1}" for i in range(len(detections))]  # # was there

        annotated_image = (
            self.box_annotator.annotate(scene=image.copy(), detections=detections)
            if bbox
            else image.copy()
        )

        annotated_image = self.label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        return annotated_image

    # TODO: multiprocessing or multithreading (is cpu bound?)
    def batch_paint(self):
        pass
