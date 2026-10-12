import argparse
from typing import Tuple, Annotated, cast
import cv2 as cv
import numpy as np
from numpy.typing import NDArray

Float32Array2D = Annotated[NDArray[np.float32], "Shape: (N, 2)"]
Int32Array2D   = Annotated[NDArray[np.int32],   "Shape: (N, 2)"]
UInt8Array2D   = Annotated[NDArray[np.uint8],   "Shape: (N, 2)"]


def sort_points_clockwise(pts: Float32Array2D) -> Float32Array2D:
    """
    Sorts a set of 2D points in clockwise circular order around their centroid.
    """
    centroid = np.mean(pts, axis=0)

    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)

    return pts[sorted_indices]


def load_polygon_from_file(filepath: str, sort_points: bool) -> Float32Array2D:
    """
    Reads coordinate pairs (x, y) from a file.
    Ignores non-coordinate header lines and blank lines.
    """
    coords = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            coords.append([x, y])

    if len(coords) < 3:
        raise ValueError(f"File '{filepath}' must contain at least 3 points, but only found {len(coords)}.")

    pts = np.array(coords, dtype=np.float32).reshape((-1, 2))

    return sort_points_clockwise(pts) if sort_points else pts


def setup_arguments() -> Tuple[Float32Array2D, Float32Array2D, str]:
    """
    Parses the two polygon arguments and returns the polygon vertices.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='''Calculate polygon pixel metrics from two coordinate files.'''
    )

    parser.add_argument("file1",
                        type=str,
                        help="Path to first polygon coordinate file")

    parser.add_argument("file2",
                        type=str,
                        help="Path to second polygon coordinate file")

    parser.add_argument("-o", "--output",
                        type=str,
                        default="polygons_output.png",
                        help="Filename for the generated result image (default: polygons_output.png)")

    parser.add_argument("-s", "--sort-points",
                        action="store_true",
                        help="Sort polygon vertices clockwise around their centroid (default: False)")

    args = parser.parse_args()

    pts1: Float32Array2D = load_polygon_from_file(args.file1, sort_points=args.sort_points)
    pts2: Float32Array2D = load_polygon_from_file(args.file2, sort_points=args.sort_points)
    output_filename: str = args.output

    return pts1, pts2, output_filename


def prepare_polygons_for_raster(pts1: Float32Array2D, pts2: Float32Array2D) -> Tuple[Int32Array2D, Int32Array2D]:
    """
    Jointly shifts both polygons so all negative coordinates become non-negative.
    """
    all_points = np.vstack([pts1, pts2])
    min_x = np.min(all_points[:, 0])
    min_y = np.min(all_points[:, 1])

    offset_x = min_x if min_x < 0 else 0.0
    offset_y = min_y if min_y < 0 else 0.0

    pts1_shifted = pts1 - np.array([offset_x, offset_y], dtype=np.float32)
    pts2_shifted = pts2 - np.array([offset_x, offset_y], dtype=np.float32)

    pts1: Int32Array2D = np.round(pts1_shifted).astype(np.int32).reshape((-1, 2))
    pts2: Int32Array2D = np.round(pts2_shifted).astype(np.int32).reshape((-1, 2))

    return pts1, pts2


def create_polygon_masks(pts1: Int32Array2D, pts2: Int32Array2D) -> Tuple[UInt8Array2D, UInt8Array2D]:
    """
    Renders two sets of polygon vertices into separate binary raster masks.
    """
    all_points = np.vstack([pts1, pts2])
    canvas_w = int(np.max(all_points[:, 0])) + 1
    canvas_h = int(np.max(all_points[:, 1])) + 1

    mask1: UInt8Array2D = cast(UInt8Array2D, np.zeros((canvas_h, canvas_w), dtype=np.uint8))
    mask2: UInt8Array2D = cast(UInt8Array2D, np.zeros((canvas_h, canvas_w), dtype=np.uint8))

    cv.fillPoly(mask1, [pts1], 255)
    cv.fillPoly(mask2, [pts2], 255)

    return mask1, mask2


def count_pixels(pts1: Int32Array2D, pts2: Int32Array2D) -> Tuple[int, int, int, int]:
    """
    Returns the Area of the polygon described by pts1, the area of the polygon described by pts2,
    the intersection and union of the two polygons.
    """
    mask1, mask2 = create_polygon_masks(pts1, pts2)

    intersection_mask = cv.bitwise_and(mask1, mask2)
    union_mask = cv.bitwise_or(mask1, mask2)

    area1_px = cv.countNonZero(mask1)
    area2_px = cv.countNonZero(mask2)
    intersection_px = cv.countNonZero(intersection_mask)
    union_px = cv.countNonZero(union_mask)

    return area1_px, area2_px, intersection_px, union_px


def save_polygon_image(pts1: Int32Array2D, pts2: Int32Array2D, output_filename: str) -> None:
    """
    Renders Polygon 1, Polygon 2, and their Intersection in distinct colors
    and writes the result to an image file.
    """
    mask1, mask2 = create_polygon_masks(pts1, pts2)

    intersection_mask = cv.bitwise_and(mask1, mask2)
    poly1_only_mask = cv.bitwise_and(mask1, cv.bitwise_not(mask2))
    poly2_only_mask = cv.bitwise_and(mask2, cv.bitwise_not(mask1))

    canvas = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)

    color_poly1 = [255, 100, 0]  # Blue / Cyan
    color_poly2 = [0, 200, 100]  # Green
    color_inter = [0, 0, 255]  # Bright Red

    canvas[poly1_only_mask == 255] = color_poly1
    canvas[poly2_only_mask == 255] = color_poly2
    canvas[intersection_mask == 255] = color_inter

    cv.imwrite(output_filename, canvas)
    print(f"Visualization saved to: {output_filename}")


def print_output(area1: int, area2: int, intersection: int, union: int, width: int = 45) -> None:
    """
    Prints out the Area of the two polygons, the intersection and union of the two polygons.
    """
    metrics = [
        ("Polygon 1 Area:", f"{area1} px"),
        ("Polygon 2 Area:", f"{area2} px"),
        ("Intersection:", f"{intersection} px"),
        ("Union:", f"{union} px")
    ]

    print("=" * width)
    print("PIXEL-LEVEL POLYGON METRICS".center(width))
    print("=" * width)

    for label, val in metrics:
        if label == "Intersection:":
            print("-" * width)
        pad = width - len(label)
        print(f"{label}{val.rjust(pad)}")

    print("=" * width)


if __name__ == '__main__':
    pts1, pts2, output_filename = setup_arguments()
    pts1, pts2 = prepare_polygons_for_raster(pts1, pts2)
    area1, area2, intersection, union = count_pixels(pts1, pts2)

    print_output(area1, area2, intersection, union)

    save_polygon_image(pts1, pts2, output_filename)