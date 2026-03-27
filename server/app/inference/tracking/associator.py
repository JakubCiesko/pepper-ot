import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from app.inference.types import InferenceDetectionObject
from app.inference.types import TrackedObject

logger = logging.getLogger(__name__)


class Associator:
    """Matches tracks with new detections."""

    def __init__(
        self, w_vis: float = 0.8, w_geo: float = 0.2, match_threshold: float = 0.4
    ):
        logger.info(
            "Initializing Associator with w_vis=%.2f, w_geo=%.2f, match_threshold=%.2f",
            w_vis,
            w_geo,
            match_threshold,
        )
        self.w_vis = w_vis
        self.w_geo = w_geo
        self.match_threshold = match_threshold
        self.INF = 1000.0

    def compute_cost(
        self,
        tracks: list[TrackedObject],
        detections: list[InferenceDetectionObject],
        embeddings: np.ndarray,
    ) -> NDArray:
        """
        Creates a Cost Matrix where rows=tracks, cols=detections.
        Cost is low if they are the same object.
        """
        logger.debug(
            "Computing cost matrix for Hungarian Algorithm with geometry weight = %.2f, visual weight = %.2f",
            self.w_geo,
            self.w_vis,
        )
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
        logger.debug("Hungarian Algorithm Cost Matrix computed.")
        return cost_matrix

    def match(
        self,
        tracks: list[TrackedObject],
        detections: list[InferenceDetectionObject],
        embeddings: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks:
            logger.info(
                "No tracked objects, returning all (%d) detections as unmatched.",
                len(detections),
            )
            return [], [], list(range(len(detections)))
        if not detections:
            logger.info(
                "No detected objects, returning all (%d) tracked objects as unmatched tracks.",
                len(tracks),
            )
            return [], list(range(len(tracks))), []

        # Hungarian Algorithm
        cost_matrix = self.compute_cost(tracks, detections, embeddings)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matches: list[tuple[int, int]] = []
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        logger.info(
            "Matching new detections and old tracked objects with distance (1-similarity) "
            "match_threshold=%.2f",
            self.match_threshold,
        )
        for r, c in zip(row_idx, col_idx, strict=False):
            if cost_matrix[r, c] < self.match_threshold:
                matches.append((r, c))
                matched_tracks.add(r)
                matched_dets.add(c)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        logger.info(
            "#matches=%d #unmatched tracked objects=%d #unmatched detected objects=%d",
            len(matches),
            len(unmatched_tracks),
            len(unmatched_dets),
        )
        return matches, unmatched_tracks, unmatched_dets
