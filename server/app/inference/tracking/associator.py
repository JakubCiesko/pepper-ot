from app.memory.scene_memory import Detection
from app.memory.scene_memory import TrackedObject
import numpy as np
from scipy.optimize import linear_sum_assignment


class Associator:
    """Matches tracks with new detections using the Hungarian Algorithm."""

    def __init__(
        self,
        w_vis: float = 0.8,
        w_geo: float = 0.2,
        max_angle_diff: float = 0.5,
        match_threshold: float = 0.4,
    ):
        self.w_vis = w_vis
        self.w_geo = w_geo
        self.max_angle_diff = max_angle_diff
        self.match_threshold = match_threshold
        self.INF = 1000.0

    def compute_cost_matrix(
        self, tracks: list[TrackedObject], detections: list[Detection]
    ) -> np.ndarray:
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Hard constraint: Labels must match
                if track.label != det.label:
                    cost_matrix[i, j] = self.INF
                    continue

                # Hard constraint: Angle difference
                angle_diff = abs(track.angle - det.angle)
                if angle_diff > self.max_angle_diff:
                    cost_matrix[i, j] = self.INF
                    continue

                # Visual Cost: 1 - Dot Product (since embeddings are normalized)
                dot_product = np.dot(track.embedding, det.embedding)
                vis_cost = 1.0 - dot_product

                # Weighted Fusion
                cost_matrix[i, j] = (self.w_vis * vis_cost) + (self.w_geo * angle_diff)

        return cost_matrix

    def match(
        self, tracks: list[TrackedObject], detections: list[Detection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(tracks))), []

        cost_matrix = self.compute_cost_matrix(tracks, detections)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        matched_track_indices = set()
        matched_det_indices = set()

        for r, c in zip(row_indices, col_indices, strict=True):  # think
            if cost_matrix[r, c] < self.match_threshold:
                matches.append((r, c))
                matched_track_indices.add(r)
                matched_det_indices.add(c)

        unmatched_tracks = [
            i for i in range(len(tracks)) if i not in matched_track_indices
        ]
        unmatched_detections = [
            i for i in range(len(detections)) if i not in matched_det_indices
        ]

        return matches, unmatched_tracks, unmatched_detections
