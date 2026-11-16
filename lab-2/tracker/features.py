from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

DETECTORS = ("sift", "orb", "akaze")


@dataclass
class FeatureBackend:
    name: str
    detector: cv2.Feature2D
    matcher: cv2.DescriptorMatcher

    def detect(self, gray: np.ndarray, mask: np.ndarray = None):
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)
        if descriptors is None:
            return [], None
        return keypoints, descriptors

    def detect_multiscale(self, gray: np.ndarray, scales: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        points, descriptors = [], []
        for scale in scales:
            image = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            keypoints, desc = self.detect(image)
            if desc is None:
                continue
            points.append(np.float32([kp.pt for kp in keypoints]) / scale)
            descriptors.append(desc)
        if not descriptors:
            return np.empty((0, 2), np.float32), None
        return np.vstack(points), np.vstack(descriptors)

    def match(self, query: np.ndarray, train: np.ndarray, ratio: float) -> List[cv2.DMatch]:
        if query is None or train is None or len(query) < 2 or len(train) < 2:
            return []
        pairs = self.matcher.knnMatch(query, train, k=2)
        return [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def create_backend(name: str, n_features: int = 0) -> FeatureBackend:
    name = name.lower()
    if name == "sift":
        detector = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.01)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    elif name == "orb":
        detector = cv2.ORB_create(nfeatures=n_features or 5000, scaleFactor=1.2, nlevels=8, fastThreshold=7)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    elif name == "akaze":
        detector = cv2.AKAZE_create(threshold=0.0003)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        raise ValueError(f"unknown detector '{name}', expected one of {DETECTORS}")
    return FeatureBackend(name, detector, matcher)
