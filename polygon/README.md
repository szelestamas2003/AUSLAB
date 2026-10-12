# Polygon Pixel Metrics & Visualization Utility

A lightweight Python CLI tool to parse, rasterize, and analyze pairs of 2D polygons from raw vertex coordinate files. It computes pixel-level surface metrics (individual polygon areas, intersection, and union) using OpenCV binary masks and outputs a color-coded visualization map.

---

## Requirements

Ensure you have Python 3.9+ installed with the following packages:

```bash
pip install numpy opencv-python
```

---

## Input File Format

Coordinate files should contain $(x, y)$ coordinate pairs separated by whitespace. Non-coordinate lines (such as sensor labels or headers) are automatically ignored.

### Example: `polygon1.txt`
```text
camera0
50.0 125.0
300.0 125.0
300.0 375.0
50.0 375.0
```

### Example: `polygon2.txt`
```text
camera1
280.0 250.0
470.0 125.0
470.0 375.0
```

---

## Usage

### Basic Command

Provide two coordinate files as positional arguments:

```bash
python calc_poly.py polygon1.txt polygon2.txt
```

### Specifying Output Image Path

Use the `-o` or `--output` argument to define the output filename (defaults to `polygons_output.png`):

```bash
python polygon_metrics.py polygon1.txt polygon2.txt -o result.png
```

### Enabling Clockwise Sorting

If your vertex coordinates are unordered or shuffled, pass the `-s` / `--sort-points` flag to order them clockwise around their centroid prior to rasterization:

```bash
python polygon_metrics.py polygon1.txt polygon2.txt -s
```

> **Note:** Leave `-s` disabled when using concave shapes (such as stars or saw patterns) that already define their perimeter in sequential order.