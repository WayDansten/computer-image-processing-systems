import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tracker import KeypointTracker
from tracker.draw import draw_result
from tracker.features import DETECTORS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Keypoint-based planar object tracker")
    parser.add_argument("input", help="input video; its first frame is the object reference")
    parser.add_argument("-o", "--output", help="path to save the annotated video")
    parser.add_argument("-d", "--detector", default="sift", choices=DETECTORS)
    parser.add_argument("-l", "--label", default="Object")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="object rectangle on the first frame")
    parser.add_argument("--features", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--min-inliers", type=int, default=15)
    parser.add_argument("--ransac", type=float, default=5.0)
    parser.add_argument("--min-score", type=float, default=0.4)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    parser.add_argument("--no-flow", action="store_true")
    parser.add_argument("--no-guided", action="store_true")
    parser.add_argument("--no-points", action="store_true")
    parser.add_argument("--no-hud", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--stats", help="path to save per-run statistics as JSON")
    return parser.parse_args(argv)


def process_video(path, output=None, label="Object", roi=None, show=False, points=True, hud=True, **options) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {path}")
    ok, frame = capture.read()
    if not ok:
        raise SystemExit("video is empty")

    reference = frame
    if roi is not None:
        x, y, w, h = roi
        reference = frame[y:y + h, x:x + w]
    tracker = KeypointTracker(reference, **options)
    detector = options.get("detector", "sift")

    writer = None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames, sources, inliers, scores, elapsed = 0, {}, [], [], 0.0
    while ok:
        start = time.perf_counter()
        result = tracker.track(frame)
        elapsed += time.perf_counter() - start
        if result.found:
            sources[result.source] = sources.get(result.source, 0) + 1
            inliers.append(result.inliers)
            scores.append(result.score)

        if writer is not None or show:
            text = ""
            if hud:
                state = f"{result.source} score={result.score:.2f}" if result.found else f"lost: {result.reason}"
                text = f"#{frames} {detector.upper()} matches={result.matches} inliers={result.inliers} {state}"
            annotated = draw_result(frame, result, label, points, text)
            if writer is not None:
                writer.write(annotated)
            if show:
                cv2.imshow("tracker", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
        frames += 1
        ok, frame = capture.read()

    capture.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    tracked = len(inliers)
    return {
        "video": Path(path).name,
        "detector": detector,
        "frames": frames,
        "tracked": tracked,
        "tracked_ratio": round(tracked / frames, 4),
        "by_detect": sources.get("detect", 0),
        "by_guided": sources.get("guided", 0),
        "by_flow": sources.get("flow", 0),
        "mean_inliers": round(float(np.mean(inliers)), 1) if inliers else 0.0,
        "mean_score": round(float(np.mean(scores)), 3) if scores else 0.0,
        "min_score": round(float(np.min(scores)), 3) if scores else 0.0,
        "reference_keypoints": len(tracker.ref_points),
        "fps": round(frames / elapsed, 1),
    }


def main(argv=None):
    args = parse_args(argv)
    stats = process_video(
        args.input,
        output=args.output,
        label=args.label,
        roi=args.roi,
        show=args.show,
        points=not args.no_points,
        hud=not args.no_hud,
        detector=args.detector,
        n_features=args.features,
        ratio=args.ratio,
        min_inliers=args.min_inliers,
        ransac_threshold=args.ransac,
        min_score=args.min_score,
        scales=args.scales,
        use_flow=not args.no_flow,
        use_guided=not args.no_guided,
    )
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
