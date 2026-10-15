from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from median_filter import median_filter_library, median_filter_native


FILTERS = {
    "OpenCV": median_filter_library,
    "Native Python": median_filter_native,
}
KERNELS = (3, 5, 7, 9)
BENCHMARK_DIR = Path(__file__).resolve().parent / "results" / "benchmark"


def measure(image: np.ndarray, function, kernel: int, repeats: int) -> list[float]:
    function(image, kernel)
    measurements = []
    for _ in range(repeats):
        started = time.perf_counter()
        function(image, kernel)
        measurements.append((time.perf_counter() - started) * 1000)
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark median filter implementations")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "benchmark.csv")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--native-scale", type=float, default=0.25)
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=tuple(FILTERS),
        default=list(FILTERS),
    )
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.native_scale <= 1:
        raise SystemExit("repeats must be positive and native-scale must be in (0, 1]")

    rows = []
    for path in args.images:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SystemExit(f"cannot read image: {path}")
        scaled_width = max(1, round(image.shape[1] * args.native_scale))
        scaled_height = max(1, round(image.shape[0] * args.native_scale))
        native_image = (
            cv2.resize(image, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
            if image.shape[1] > 1000
            else image
        )
        for kernel in KERNELS:
            for name in args.implementations:
                function = FILTERS[name]
                data = image if name == "OpenCV" else native_image
                measurements = measure(data, function, kernel, args.repeats)
                rows.append(
                    {
                        "image": path.name,
                        "implementation": name,
                        "input_shape": "x".join(map(str, data.shape)),
                        "kernel": kernel,
                        "repeats": args.repeats,
                        "median_ms": f"{np.median(measurements):.3f}",
                        "mean_ms": f"{np.mean(measurements):.3f}",
                        "min_ms": f"{min(measurements):.3f}",
                        "max_ms": f"{max(measurements):.3f}",
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} measurements to {args.output}")
    for row in rows:
        print(
            f"{row['image']}; {row['implementation']}; kernel={row['kernel']}; "
            f"shape={row['input_shape']}; median={row['median_ms']} ms"
        )


if __name__ == "__main__":
    main()
