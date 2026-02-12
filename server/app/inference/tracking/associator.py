import numpy as np
from scipy.optimize import linear_sum_assignment

from ..types import DetectionObject
from ..types import TrackedObject


class Associator:
    """Matches tracks with new detections."""

    def __init__(self, w_vis=0.8, w_geo=0.2, match_threshold=0.4):
        self.w_vis = w_vis
        self.w_geo = w_geo
        self.match_threshold = match_threshold
        self.INF = 1000.0

    def compute_cost(
        self,
        tracks: list[TrackedObject],
        detections: list[DetectionObject],
        embeddings: np.ndarray,
    ):
        """
        Creates a Cost Matrix where rows=tracks, cols=detections.
        Cost is low if they are the same object.
        """
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                # 1. Hard Constraint: Labels must match (Context: "cat" cannot become "dog")
                if track.label != det.label:
                    cost_matrix[t_idx, d_idx] = self.INF
                    continue

                # 2. Visual Cost (Cosine Distance)
                # Dot product of normalized vectors = Cosine Similarity
                # Cost = 1 - Similarity
                det_emb = embeddings[d_idx]
                sim = np.dot(track.embedding, det_emb)
                vis_cost = 1.0 - sim

                # 3. Geometric Cost (Spatial Distance)
                # In Robot production code: Use Angle Difference
                # In Notebook: Use Center Distance Normalized by Image Size (approx 1000px)
                det_center = (
                    (det.bbox[0] + det.bbox[2]) / 2,
                    (det.bbox[1] + det.bbox[3]) / 2,
                )
                dist = np.linalg.norm(np.array(track.center) - np.array(det_center))
                geo_cost = dist / 1000.0  # Normalize roughly

                # Weighted Sum
                cost_matrix[t_idx, d_idx] = (self.w_vis * vis_cost) + (
                    self.w_geo * geo_cost
                )

        return cost_matrix

    def match(
        self,
        tracks: list[TrackedObject],
        detections: list[DetectionObject],
        embeddings: np.ndarray,
    ):
        if not tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(tracks))), []

        # Hungarian Algorithm
        cost_matrix = self.compute_cost(tracks, detections, embeddings)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matches = []
        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_idx, col_idx, strict=False):
            if cost_matrix[r, c] < self.match_threshold:
                matches.append((r, c))
                matched_tracks.add(r)
                matched_dets.add(c)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]

        return matches, unmatched_tracks, unmatched_dets
