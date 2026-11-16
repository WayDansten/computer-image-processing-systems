import cv2
import numpy as np

from tracker.tracker import TrackResult

BOX_COLOR = (0, 255, 0)
POINT_COLOR = (0, 200, 255)
TEXT_COLOR = (255, 255, 255)
LOST_COLOR = (0, 0, 255)


def draw_label(image: np.ndarray, text: str, origin, color, scale: float = 0.7) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, 2)
    x = int(np.clip(origin[0], 0, max(0, image.shape[1] - tw - 6)))
    y = int(np.clip(origin[1], th + base + 4, image.shape[0] - 1))
    cv2.rectangle(image, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(image, text, (x + 3, y - base - 1), font, scale, (0, 0, 0), 2, cv2.LINE_AA)


def draw_result(image: np.ndarray, result: TrackResult, label: str, show_points: bool = True, hud: str = "") -> np.ndarray:
    out = image.copy()
    if show_points:
        for x, y in result.inlier_points.astype(int):
            cv2.circle(out, (int(x), int(y)), 2, POINT_COLOR, -1, cv2.LINE_AA)
    if result.found:
        polygon = result.corners.reshape(-1, 2)
        cv2.polylines(out, [np.int32(np.round(polygon))], True, BOX_COLOR, 3, cv2.LINE_AA)
        top = polygon[np.argmin(polygon[:, 1])]
        draw_label(out, label, (top[0], top[1] - 6), BOX_COLOR)
    if hud:
        color = BOX_COLOR if result.found else LOST_COLOR
        cv2.putText(out, hud, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, hud, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
