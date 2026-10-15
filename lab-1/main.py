from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from median_filter import median_filter_library, median_filter_native


FILTERS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "library": median_filter_library,
    "native": median_filter_native,
}
RESULTS_DIR = Path(__file__).resolve().parent / "results"
IMAGES_DIR = RESULTS_DIR / "images"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Median image filter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("library", "native"):
        command = subparsers.add_parser(name, help=f"apply the {name} implementation")
        _add_image_arguments(command)

    compare = subparsers.add_parser("compare", help="compare both implementations")
    compare.add_argument("input", type=Path)
    compare.add_argument("--kernel", type=int, required=True)

    benchmark = subparsers.add_parser("benchmark", help="measure both implementations")
    benchmark.add_argument("input", type=Path)
    benchmark.add_argument("--kernel", type=int, required=True)
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--warmup", type=int, default=1)

    return parser


def _add_image_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--kernel", type=int, required=True)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"cannot write image: {path}")


def _run(args: argparse.Namespace) -> None:
    image = _read_image(args.input)

    if args.command in FILTERS:
        result = FILTERS[args.command](image, args.kernel)
        output = args.output or (
            IMAGES_DIR / f"{args.input.stem}_{args.command}_k{args.kernel}{args.input.suffix}"
        )
        _write_image(output, result)
        print(f"Saved {args.command} result to {output}")
        return

    if args.command == "compare":
        library_result = median_filter_library(image, args.kernel)
        native_result = median_filter_native(image, args.kernel)
        difference = np.abs(
            library_result.astype(np.int16) - native_result.astype(np.int16)
        )
        print(f"shape: {image.shape}")
        print(f"kernel: {args.kernel}")
        print(f"max_difference: {int(difference.max())}")
        print(f"equal: {bool(np.array_equal(library_result, native_result))}")
        return

    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")
    print(f"shape: {image.shape}")
    print(f"kernel: {args.kernel}")
    for name, filter_function in FILTERS.items():
        for _ in range(args.warmup):
            filter_function(image, args.kernel)
        measurements = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            filter_function(image, args.kernel)
            measurements.append(time.perf_counter() - start)
        print(
            f"{name}: median={np.median(measurements):.6f}s "
            f"mean={np.mean(measurements):.6f}s"
        )


def main() -> None:
    args = build_parser().parse_args()
    try:
        _run(args)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
