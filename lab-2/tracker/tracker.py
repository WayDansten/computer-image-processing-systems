from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from tracker.features import create_backend

LK_PARAMS = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
PATCH_WIDTH = 80
PATCH_OVERSAMPLE = 4


@dataclass
class TrackResult:
    found: bool
    source: str = "none"
    corners: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None
    matches: int = 0
    inliers: int = 0
    score: float = 0.0
    inlier_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), np.float32))
    reason: str = ""


class KeypointTracker:
    def __init__(
        self,
        reference: np.ndarray,
        detector: str = "sift",
        n_features: int = 0,
        ratio: float = 0.75,
        min_inliers: int = 15,
        min_inlier_ratio: float = 0.25,
        ransac_threshold: float = 5.0,
        scales: Sequence[float] = (1.0, 0.5, 0.25),
        use_flow: bool = True,
        use_guided: bool = True,
        flow_threshold: float = 2.0,
        min_score: float = 0.4,
        flow_margin: float = 0.05,
        affine_below: float = 0.25,
        seed_distance: float = 2,
        memory: int = 15,
    ):
        self.backend = create_backend(detector, n_features)
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_threshold = ransac_threshold
        self.use_flow = use_flow
        self.use_guided = use_guided
        self.flow_threshold = flow_threshold
        self.min_score = min_score
        self.flow_margin = flow_margin
        self.affine_below = affine_below
        self.seed_distance = seed_distance
        self.memory = memory
        self.reference_gray = self._gray(reference)
        h, w = self.reference_gray.shape
        self.reference_size = (w, h)
        self.reference_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        self.ref_points, self.ref_descriptors = self.backend.detect_multiscale(self.reference_gray, scales)
        if len(self.ref_points) < self.min_inliers:
            raise ValueError("reference frame has too few keypoints for tracking")
        self._patch_size = (PATCH_WIDTH, max(8, round(PATCH_WIDTH * h / w)))
        self._ref_patch = cv2.resize(self.reference_gray, self._patch_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        self._homography = None
        self._prev_gray = None
        self._points = None
        self._missed = 0

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def track(self, frame: np.ndarray) -> TrackResult:
        gray = self._gray(frame)
        candidates = [self._detect(gray)]
        prior = self._homography
        if prior is not None:
            if self.use_flow and self._points is not None:
                flow = self._flow(gray)
                candidates.append(flow)
                if flow.found:
                    prior = flow.homography
            if self.use_guided:
                candidates.append(self._detect_guided(gray, prior))
        result = self._select(candidates)
        if result.found:
            self._homography = result.homography
            self._points = self._seed(gray, result.corners)
            self._missed = 0
        else:
            self._points = None
            self._missed += 1
            if self._missed > self.memory:
                self._homography = None
        self._prev_gray = gray
        return result

    def _select(self, candidates) -> TrackResult:
        found = [c for c in candidates if c.found]
        if not found:
            return candidates[0]
        best = max(found, key=lambda c: c.score)
        anchored = [c for c in found if c.source != "flow"]
        if anchored:
            anchor = max(anchored, key=lambda c: c.score)
            if anchor.score >= best.score - self.flow_margin:
                return anchor
        return best

    def _detect(self, gray: np.ndarray, rectified: np.ndarray = None, to_frame: np.ndarray = None, threshold: float = None, affine: bool = False) -> TrackResult:
        source = "detect" if rectified is None else "guided"
        keypoints, descriptors = self.backend.detect(gray if rectified is None else rectified)
        matches = self.backend.match(self.ref_descriptors, descriptors, self.ratio)
        if len(matches) < self.min_inliers:
            return TrackResult(False, source, matches=len(matches), reason="few matches")
        src = self.ref_points[[m.queryIdx for m in matches]]
        dst = np.float32([keypoints[m.trainIdx].pt for m in matches])
        homography, mask = self._estimate(src, dst, threshold or self.ransac_threshold, affine)
        if homography is None:
            return TrackResult(False, source, matches=len(matches), reason="no homography")
        inliers = dst[mask.ravel().astype(bool)]
        if to_frame is not None:
            homography = to_frame @ homography
            homography /= homography[2, 2]
            inliers = cv2.perspectiveTransform(inliers.reshape(-1, 1, 2), to_frame).reshape(-1, 2)
        result = self._build(gray, homography, inliers, source)
        result.matches = len(matches)
        if result.found and result.inliers < self.min_inlier_ratio * len(matches):
            result.found, result.reason = False, "few inliers"
        return result

    def _detect_guided(self, gray: np.ndarray, prior: np.ndarray) -> TrackResult:
        w, h = self.reference_size
        size = self._relative_size(prior)
        scale = float(np.clip(2.0 * size, 0.25, 1.0))
        to_rect = np.diag([scale, scale, 1.0]) @ np.linalg.inv(prior)
        rectified = cv2.warpPerspective(gray, to_rect, (int(np.ceil(w * scale)), int(np.ceil(h * scale))), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return self._detect(gray, rectified, np.linalg.inv(to_rect), self.ransac_threshold * max(scale, 0.5), size < self.affine_below)

    def _flow(self, gray: np.ndarray) -> TrackResult:
        prev = self._points.reshape(-1, 1, 2)
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev, None, **LK_PARAMS)
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, self._prev_gray, nxt, None, **LK_PARAMS)
        error = np.linalg.norm((prev - back).reshape(-1, 2), axis=1)
        good = (status.ravel() == 1) & (status_back.ravel() == 1) & (error < 1.0)
        if good.sum() < self.min_inliers:
            return TrackResult(False, "flow", reason="flow lost")
        src, dst = prev.reshape(-1, 2)[good], nxt.reshape(-1, 2)[good]
        step, mask = self._estimate(src, dst, self.flow_threshold, self._relative_size(self._homography) < self.affine_below)
        if step is None:
            return TrackResult(False, "flow", reason="no homography")
        homography = step @ self._homography
        homography /= homography[2, 2]
        return self._build(gray, homography, dst[mask.ravel().astype(bool)], "flow")

    def _relative_size(self, homography: np.ndarray) -> float:
        w, h = self.reference_size
        corners = cv2.perspectiveTransform(self.reference_corners, homography)
        return float(np.sqrt(cv2.contourArea(corners) / (w * h)))

    @staticmethod
    def _estimate(src: np.ndarray, dst: np.ndarray, threshold: float, affine: bool):
        if affine:
            matrix, mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=2000, confidence=0.995)
            if matrix is not None:
                matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
            return matrix, mask
        return cv2.findHomography(src, dst, cv2.RANSAC, threshold, maxIters=2000, confidence=0.995)

    def _seed(self, gray: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(np.round(corners.reshape(-1, 2))), 255)
        points = cv2.goodFeaturesToTrack(gray, 400, 0.01, self.seed_distance, mask=mask)
        if points is None or len(points) < self.min_inliers:
            return None
        return points.reshape(-1, 2).astype(np.float32)

    def _build(self, gray: np.ndarray, homography: np.ndarray, inlier_points: np.ndarray, source: str) -> TrackResult:
        result = TrackResult(False, source, homography=homography, inliers=len(inlier_points), inlier_points=inlier_points)
        if result.inliers < self.min_inliers:
            result.reason = "few inliers"
            return result
        corners = cv2.perspectiveTransform(self.reference_corners, homography)
        result.reason = self._validate(homography, corners, gray.shape)
        if result.reason:
            return result
        result.score = self._similarity(gray, homography)
        if result.score < self.min_score:
            result.reason = "low similarity"
            return result
        result.found, result.corners = True, corners
        return result

    def _similarity(self, gray: np.ndarray, homography: np.ndarray) -> float:
        w, h = self.reference_size
        pw, ph = self._patch_size
        size = (pw * PATCH_OVERSAMPLE, ph * PATCH_OVERSAMPLE)
        to_patch = np.diag([size[0] / w, size[1] / h, 1.0]) @ np.linalg.inv(homography)
        patch = cv2.warpPerspective(gray, to_patch, size, flags=cv2.INTER_LINEAR)
        visible = cv2.warpPerspective(np.full_like(gray, 255), to_patch, size, flags=cv2.INTER_NEAREST)
        patch = cv2.resize(patch, self._patch_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        visible = cv2.resize(visible, self._patch_size, interpolation=cv2.INTER_AREA) == 255
        if visible.mean() < 0.2:
            return 0.0
        a = patch[visible] - patch[visible].mean()
        b = self._ref_patch[visible] - self._ref_patch[visible].mean()
        denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / denominator) if denominator > 1e-6 else 0.0

    def _validate(self, homography: np.ndarray, corners: np.ndarray, shape) -> str:
        reference = self.reference_corners.reshape(-1, 2)
        if np.any(homography[2, 0] * reference[:, 0] + homography[2, 1] * reference[:, 1] + homography[2, 2] <= 0):
            return "behind camera"
        polygon = corners.reshape(-1, 2)
        if self._signed_area(polygon) * self._signed_area(reference) <= 0:
            return "flipped"
        if not cv2.isContourConvex(polygon.astype(np.float32)):
            return "not convex"
        area = cv2.contourArea(polygon)
        frame_area = float(shape[0] * shape[1])
        if area < 0.002 * frame_area or area > 4.0 * frame_area:
            return "bad area"
        sides = np.linalg.norm(polygon - np.roll(polygon, -1, axis=0), axis=1)
        if sides.min() < 1e-3:
            return "degenerate"
        if max(sides[0] / sides[2], sides[2] / sides[0], sides[1] / sides[3], sides[3] / sides[1]) > 4:
            return "distorted"
        return ""

    @staticmethod
    def _signed_area(polygon: np.ndarray) -> float:
        x, y = polygon[:, 0], polygon[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
