import argparse
import csv
import sys
from pathlib import Path

from main import process_video
from tracker.features import DETECTORS

MODES = {
    "detect": dict(use_flow=False, use_guided=False),
    "detect+guided": dict(use_flow=False, use_guided=True),
    "detect+guided+flow": dict(use_flow=True, use_guided=True),
}
DEFAULT_VIDEOS = ("data/mona-lisa.avi", "data/mona-lisa-blur.avi", "data/mona-lisa-blur-extra-credit.avi")
COLUMNS = ("video", "detector", "mode", "frames", "tracked", "tracked_ratio", "by_detect", "by_guided", "by_flow", "mean_inliers", "mean_score", "min_score", "fps")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the tracker over videos, detectors and modes")
    parser.add_argument("videos", nargs="*", default=list(DEFAULT_VIDEOS))
    parser.add_argument("-d", "--detectors", nargs="+", default=list(DETECTORS), choices=DETECTORS)
    parser.add_argument("-m", "--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("-o", "--output", default="results/metrics")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    return parser.parse_args(argv)


def to_markdown(rows) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join(" --- " for _ in COLUMNS) + "|"]
    lines += ["| " + " | ".join(str(row[column]) for column in COLUMNS) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    rows = []
    for video in args.videos:
        for detector in args.detectors:
            for mode in args.modes:
                stats = process_video(video, roi=args.roi, detector=detector, **MODES[mode])
                stats["mode"] = mode
                rows.append(stats)
                print(" ".join(f"{column}={stats[column]}" for column in COLUMNS), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output.with_suffix(".csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".md").write_text(to_markdown(rows))


if __name__ == "__main__":
    sys.exit(main())
