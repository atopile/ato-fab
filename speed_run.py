# %%

# 1. Generate a grid of circles
import itertools
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import svg
from PIL import Image


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Helper functions
def only_one[T](glob: Iterable[T]) -> T:
    files = list(glob)
    if len(files) != 1:
        raise ValueError(
            f"Expected exactly one, but found {len(files)}"
        )
    return files[0]

def latest[T](glob: Iterable[T]) -> T:
    return max(glob, key=os.path.getctime)

def save(**kwargs):
    for k, v in kwargs.items():
        img = cv2.cvtColor(v, cv2.COLOR_BGRA2RGBA)
        cv2.imwrite(build_dir / f"{k}.png", img)

def display(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
    # Convert HSV to BGR - Hue cycles from 0 to 180 (in OpenCV), full saturation and value
        hsv_color = np.uint8([[[i * 180 // num_colors, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        # Convert to regular tuple of ints
        colors.append(tuple(map(int, bgr_color)))
    return colors

def draw_grid(image, xs, ys, line_thickness: int = 1):
    if xs is not None:
        for x, color in zip(xs, generate_colors(len(xs))):
            cv2.line(image, (int(x), 0), (int(x), image.shape[0]), color, line_thickness)

    if ys is not None:
        for y, color in zip(ys, generate_colors(len(ys))):
            cv2.line(image, (0, int(y)), (image.shape[1], int(y)), color, line_thickness)

    return image

def display2(
    image,
    title: str = "image",
    key_points: list[cv2.KeyPoint] | None = None,
    points: np.ndarray | None = None,
    point_size: int | None = None,
    grid_xs: list[int] | None = None,
    grid_ys: list[int] | None = None,
    marker_color: tuple[int, int, int] = (0, 0, 255),
    save_image: bool = True,
    open_image: bool = True,
    line_thickness: int = 1,
    downscale: float = 1,
):
    if key_points is None:
        key_points = []

    if points is not None:
        assert point_size is not None, "point_size must be provided if points are provided"
    else:
        points = []

    if len(image.shape) == 2:  # Check if grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image.copy()

    if downscale != 1:
        image = cv2.resize(image, None, fx=downscale, fy=downscale)

    for point in key_points:
        cv2.circle(image, (int(point.pt[0]), int(point.pt[1])), int(point.size/2), marker_color, line_thickness)

    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), int(point_size), marker_color, line_thickness)

    image = draw_grid(image, grid_xs, grid_ys, line_thickness)

    if save_image:
        save(**{title: image})

    if open_image:
        subprocess.run(["open", build_dir / f"{title}.png"])

    return image

def get_mm_to_pixels(image_path: Path) -> float | None:
    Image.MAX_IMAGE_PIXELS = None  # Disable the PIL limit because we own everything
    info = Image.open(image_path).info
    if "dpi" not in info:
        return None
    return info["dpi"][0] / 25.4

# %%
# General properties
drill_radius = 0.25  # radius mm
laser_radius = 1  # radius mm

# We do all the mapping in the 150x150mm space, before final clipping
# This is so we don't have to deal with the border conditions etc... in a smaller space
input_image_dimensions = (150, 150)  # mm
output_image_dimensions = (120, 120)  # mm
drill_hole_count = (14, 15)
drill_pitch = (7.2307692308, 7.714)  # mm

# %%
# 1.1 Calculate the points where the circles will be placed
drill_pattern_size = (drill_hole_count[0] - 1) * drill_pitch[0], (drill_hole_count[1] - 1) * drill_pitch[1]
drill_objpoints = np.mgrid[0:drill_hole_count[0], 0:drill_hole_count[1]].T.reshape(-1, 2)
drill_objpoints = drill_objpoints * np.array(drill_pitch) + ((np.array(input_image_dimensions) - np.array(drill_pattern_size)) / 2)
drill_objpoints = drill_objpoints.astype(np.float32)

plt.scatter(drill_objpoints[:, 0], drill_objpoints[:, 1])
plt.show()

# %%
# 1.2 Generate a synthetic image of the target
canvas = svg.SVG(
    width=f"{output_image_dimensions[0]}mm",
    height=f"{output_image_dimensions[1]}mm",
    elements=[],
)
for x, y in drill_objpoints:
    canvas.elements.append(
        svg.Circle(
            cx=f"{x}mm",
            cy=f"{y}mm",
            r=f"{laser_radius}mm",
            stroke="red",
            stroke_width=f"{0.05}mm",
        )
    )

# %%
# 1.3 Save the SVG file
my_dir = Path(__name__).parent
build_dir = my_dir / "build"
dots_svg = build_dir / "dots.svg"

with dots_svg.open("w") as f:
    f.write(str(canvas))

# %%
latest_scan = latest(build_dir.glob("img*.png"))
print(f"Using {latest_scan}")
image = cv2.imread(latest_scan, cv2.IMREAD_UNCHANGED)
original_scan_size = image.shape[:2]
scan_mm_to_pixels = get_mm_to_pixels(latest_scan)

display(image)
print(f"Image size: {image.shape}")
print(f"DPI: {scan_mm_to_pixels * 25.4}")
print(f"px/mm: {scan_mm_to_pixels}")

# %%
# Print out the default parameters for the blob detector
# See: https://stackoverflow.com/questions/39703407/using-findcirclesgrid-in-large-images
fiber_dots_params = cv2.SimpleBlobDetector.Params()
for attr in dir(fiber_dots_params):
    if attr.startswith("_"):
        continue
    print(f"{attr} = {getattr(fiber_dots_params, attr)}")

# Tweak them as required
def make_params(radius_mm: float, low_tolerance: float = 0.5, high_tolerance: float = 0.1) -> cv2.SimpleBlobDetector.Params:
    params = cv2.SimpleBlobDetector.Params()
    params.filterByArea = True
    area_pixels = math.pi * (radius_mm * scan_mm_to_pixels)**2
    params.minArea = int(area_pixels * (1 - low_tolerance))
    params.maxArea = int(area_pixels * (1 + high_tolerance))
    return params


# %%
def find_blobs(image: np.ndarray, dot_count: tuple[int, int], radius_mm: float) -> tuple[list[cv2.KeyPoint], cv2.SimpleBlobDetector]:
    for (low_tol, high_tol), hole_radius in itertools.product([(0.3, 0.2)], [radius_mm]):
        print(f"Trying {hole_radius=}, {low_tol=}, {high_tol=}")
        params = make_params(hole_radius, low_tol, high_tol)
        # params.filterByCircularity = True
        params.filterByConvexity = False
        params.maxThreshold = 80
        params.minThreshold = 0
        # params.filterByInertia = True
        detector = cv2.SimpleBlobDetector.create(params)
        detected_points = detector.detect(image)

        print(f"Found {len(detected_points)} datums")
        if len(detected_points) >= dot_count[0] * dot_count[1]:
            break

    return detected_points, detector

inverted = cv2.bitwise_not(cv2.blur(image, (8, 8)))
# display2(inverted, "inverted")
detected_drill_points, _ = find_blobs(inverted, drill_hole_count, drill_radius)
display2(
    inverted,
    "detected_points",
    key_points=detected_drill_points,
    line_thickness=10,
    marker_color=(0, 0, 255, 255),
)

# %%
# # Draw circles on the image nearby the detected points
# synthetic_image = image.copy()
# circle_radius = int(laser_radius * scan_mm_to_pixels)
# for point in detected_drill_points:
#     cv2.circle(
#         synthetic_image,
#         (
#             int(point.pt[0] + random.uniform(-1, 1) * scan_mm_to_pixels),
#             int(point.pt[1] + random.uniform(-1, 1) * scan_mm_to_pixels)
#         ),
#         circle_radius,
#         (0, 0, 0, 255),
#         2, # int(40 / 1000 * scan_mm_to_pixels),  # 40 micron line width
#     )
# display2(synthetic_image, "synthetic_image")

# %%
def _find_circles(image: np.ndarray, count: int, radius_mm: float) -> tuple[list[cv2.KeyPoint], None]:
    # Ensure image is grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Convert to 8-bit if necessary
    if gray.dtype != np.uint8:
        gray = cv2.convertScaleAbs(gray)

    min_dist = int(8 * scan_mm_to_pixels)

    # Try different parameter combinations
    param_combinations = [
        # (dp, param1, param2, radius_tolerance)
        (1, 50, 30, 0.1),  # Original parameters
        (1, 30, 20, 0.2),  # More lenient
        (2, 100, 40, 0.15),  # Higher resolution, stricter
        (1.5, 70, 35, 0.25),  # Balanced approach
    ]

    for dp, param1, param2, radius_tolerance in param_combinations:
        print(f"Trying params: {dp=}, {param1=}, {param2=}, {radius_tolerance=}")
        circles = cv2.HoughCircles(
            gray,  # Use the prepared grayscale image
            cv2.HOUGH_GRADIENT,
            dp=dp,            # Use the parameter from combinations
            minDist=min_dist,
            param1=param1,    # Use the parameter from combinations
            param2=param2,    # Use the parameter from combinations
            minRadius=int((radius_mm * (1 - radius_tolerance)) * scan_mm_to_pixels),
            maxRadius=int((radius_mm * (1 + radius_tolerance)) * scan_mm_to_pixels),
        )
        detected_count = circles.shape[1] if circles is not None else 0
        print(f"Found {detected_count} circles.")
        if detected_count >= count:
            break

    if circles is None:
        return [], None

    keypoints = [cv2.KeyPoint(x=c[0], y=c[1], size=c[2]*2) for c in circles[0, :]]
    return keypoints, None

blur = cv2.bilateralFilter(image, 9, 75, 75)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(blur)
# display2(enhanced, "enhanced")

# %%
# Take a 10x10mm slice of image centered around the first point
slice_size = (10 * scan_mm_to_pixels, 10 * scan_mm_to_pixels)
slice_center = detected_drill_points[0].pt
slicer = (
    slice(max(int(slice_center[1] - slice_size[1] / 2), 0), min(int(slice_center[1] + slice_size[1] / 2), enhanced.shape[0])),
    slice(max(int(slice_center[0] - slice_size[0] / 2), 0), min(int(slice_center[0] + slice_size[0] / 2), enhanced.shape[1]))
)
slice_image = enhanced[slicer]
display(slice_image)

# Ensure there's content in the image
assert slice_image.shape[0] and slice_image.shape[1]

detected_laser_points, _ = _find_circles(slice_image, 1, laser_radius)
_ = display2(
    slice_image,
    "detected_laser_points",
    key_points=detected_laser_points,
    line_thickness=5,
    marker_color=(0, 0, 255, 255),
)

# %%
def find_laser_points(
    image: np.ndarray,
    expected_locations: list[cv2.KeyPoint],
    laser_radius_mm: float,
    slice_size: tuple[int, int] = (10, 10),
) -> list[cv2.KeyPoint | None]:
    detected_points = []
    for point in expected_locations:
        slice_start = (
            int(point.pt[0] - slice_size[0] / 2),
            int(point.pt[1] - slice_size[1] / 2),
        )
        slice_image = image[slice_start[1]:slice_start[1] + slice_size[1], slice_start[0]:slice_start[0] + slice_size[0]]
        slice_points, _ = _find_circles(slice_image, 1, laser_radius_mm)

        def _dist(p1: cv2.KeyPoint, p2: cv2.KeyPoint) -> float:
            return math.sqrt((p1.pt[0] - p2.pt[0])**2 + (p1.pt[1] - p2.pt[1])**2)

        if len(slice_points) == 0:
            detected_points.append(None)
        elif len(slice_points) == 1:
            detected_points.append(slice_points[0])
        else:
            closest_point = min(slice_points, key=lambda p: _dist(p, point))
            detected_points.append(closest_point)

    return detected_points

detected_laser_points = find_laser_points(inverted, detected_drill_points, laser_radius)
display2(
    inverted,
    "detected_laser_points",
    key_points=detected_laser_points,
    line_thickness=5,
    marker_color=(0, 0, 255, 255),
)

# %%
