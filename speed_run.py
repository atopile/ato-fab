# %%
# 1. Generate a grid of circles
import itertools
import logging
import math
import os
import subprocess
from pathlib import Path
from typing import Generator, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import svg
from PIL import Image
from scipy.interpolate import Rbf
import cairosvg


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
        cv2.circle(
            image,
            (int(point.pt[0] * downscale), int(point.pt[1] * downscale)),
            int(point.size/2 * downscale),
            marker_color,
            line_thickness,
        )

    for point in points:
        cv2.circle(
            image,
            (int(point[0] * downscale), int(point[1] * downscale)),
            int(point_size * downscale),
            marker_color,
            line_thickness,
        )

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

def zip_closest(
    keypoints1: list[cv2.KeyPoint], keypoints2: list[cv2.KeyPoint]
) -> Generator[tuple[cv2.KeyPoint, cv2.KeyPoint], None, None]:
    def _dist(p1: cv2.KeyPoint, p2: cv2.KeyPoint) -> float:
        return math.sqrt((p1.pt[0] - p2.pt[0])**2 + (p1.pt[1] - p2.pt[1])**2)

    keypoints2 = keypoints2.copy()
    for kp1 in keypoints1:
        closest_kp2 = min(keypoints2, key=lambda kp2: _dist(kp1, kp2))
        yield kp1, closest_kp2
        keypoints2.remove(closest_kp2)

def plot_error_histogram(point_pairs: list[tuple[cv2.KeyPoint, cv2.KeyPoint]], scale: float):
    error = np.array([np.array(p1.pt) - np.array(p2.pt) for p1, p2 in point_pairs]) / scale
    plt.hist2d(
        error[:, 0],
        error[:, 1],
        bins=20,
        cmap='viridis',
        density=True
    )
    plt.colorbar(label='Density')
    plt.title("Error in mm")
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('X Error (mm)')
    plt.ylabel('Y Error (mm)')
    x_range = max(abs(np.min(error[:, 0])), abs(np.max(error[:, 0])))
    y_range = max(abs(np.min(error[:, 1])), abs(np.max(error[:, 1])))

    plt.axis([-x_range, x_range, -y_range, y_range])
    plt.show()

def plot_enumerate_points(points: np.ndarray, image: np.ndarray | None = None):
    if image is not None:
        plt.imshow(image)

    plt.scatter(
        points[:, 0],
        points[:, 1],
        color="red",
        s=5,
    )

    for i, p in enumerate(points):
        plt.annotate(
            f"{i}",
            xy=(p[0], p[1]),
            xytext=(p[0] + 10, p[1] + 10),
            color="red",
            fontsize=6,
        )

    plt.show()

def keypoints_to_array(keypoints: list[cv2.KeyPoint]) -> np.ndarray:
    return np.array([[keypoint.pt[0], keypoint.pt[1]] for keypoint in keypoints])

def order_points_into_grid(points: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    """Orders points into a grid starting from top-left, going right then down.

    Args:
        points: Numpy array of shape (N, 2) containing point coordinates
        grid_size: Tuple of (rows, cols) for the expected grid

    Returns:
        Numpy array of shape (N, 2) with points ordered in row-major order
        (left-to-right, top-to-bottom)
    """
    # Find approximate row and column divisions
    y_coords = points[:, 1]

    # Use percentile-based clustering to find rows
    row_indices = np.argsort(y_coords)
    rows = np.array_split(row_indices, grid_size[0])

    # Sort each row by x coordinate
    ordered_points = []
    for row in rows:
        # Sort points in this row by x coordinate
        row_points = points[row]
        sorted_row = row_points[np.argsort(row_points[:, 0])]
        ordered_points.extend(sorted_row)

    return np.array(ordered_points)

def create_undistortion_mappers(
    detected_points: list[cv2.KeyPoint | None],
    expected_points: list[cv2.KeyPoint],
    invert: bool = False,
    function: str = "cubic",
) -> tuple[Rbf, Rbf]:
    # Filter out None values and create arrays of corresponding points
    valid_detected = []
    valid_expected = []

    for detected, expected in zip(detected_points, expected_points):
        if detected is not None:
            valid_detected.append([detected.pt[0], detected.pt[1]])
            valid_expected.append([expected.pt[0], expected.pt[1]])

    valid_detected = np.array(valid_detected)
    valid_expected = np.array(valid_expected)

    if invert:
        # Map from detected (laser) positions to expected (drill) positions
        rbf_dx = Rbf(valid_detected[:, 0], valid_detected[:, 1], valid_expected[:, 0], function=function)
        rbf_dy = Rbf(valid_detected[:, 0], valid_detected[:, 1], valid_expected[:, 1], function=function)
    else:
        # Map from expected (drill) positions to detected (laser) positions
        rbf_dx = Rbf(valid_expected[:, 0], valid_expected[:, 1], valid_detected[:, 0], function=function)
        rbf_dy = Rbf(valid_expected[:, 0], valid_expected[:, 1], valid_detected[:, 1], function=function)

    return rbf_dx, rbf_dy

def transform_points(points: np.ndarray, homography: np.ndarray | None = None, mappers: tuple[Rbf, Rbf] | None = None) -> np.ndarray:
    if homography is not None:
        return cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),  # Reshape to (N, 1, 2)
            homography,
        ).reshape(-1, 2)

    elif mappers is not None:
        x_mapped = mappers[0](points[:, 0], points[:, 1])
        y_mapped = mappers[1](points[:, 0], points[:, 1])
        return np.column_stack((x_mapped, y_mapped))

    raise ValueError("Either homography or mappers must be provided")

def create_svg_grid(points: np.ndarray, radius: float) -> svg.SVG:
    canvas = svg.SVG(
        width=f"{input_image_dimensions[0]}mm",
        height=f"{input_image_dimensions[1]}mm",
        elements=[
            svg.Rect(
                x=0,
                y=0,
                width=f"{input_image_dimensions[0]}mm",
                height=f"{input_image_dimensions[1]}mm",
                stroke="black",
                stroke_width=f"{0.05}mm",
                fill="none",
            ),
        ],
    )
    for x, y in points:
        canvas.elements.append(
            svg.Circle(
                cx=f"{x}mm",
                cy=f"{y}mm",
                r=f"{radius}mm",
                stroke="red",
                stroke_width=f"{0.05}mm",
                fill="none",
            )
        )
    return canvas


my_dir = Path(__name__).parent
build_dir = my_dir / "build"
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
# Generate a target SVG
canvas = create_svg_grid(drill_objpoints, laser_radius)
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
detected_drill_points, detector = find_blobs(inverted, drill_hole_count, drill_radius)
# display2(
#     inverted,
#     "detected_points",
#     key_points=detected_drill_points,
#     line_thickness=10,
#     marker_color=(0, 0, 255, 255),
# )

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
        (3, 150, 50, 0.1),   # Highest resolution, very strict
        (2, 100, 40, 0.15),  # Higher resolution, stricter
        (1.5, 70, 35, 0.25),  # Balanced approach
        (1, 50, 30, 0.1),  # Original parameters
        (1, 30, 20, 0.2),  # More lenient
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

# # Take a 10x10mm slice of image centered around the first point
# slice_size = (10 * scan_mm_to_pixels, 10 * scan_mm_to_pixels)
# slice_center = detected_drill_points[1].pt
# slicer = (
#     slice(max(int(slice_center[1] - slice_size[1] / 2), 0), min(int(slice_center[1] + slice_size[1] / 2), enhanced.shape[0])),
#     slice(max(int(slice_center[0] - slice_size[0] / 2), 0), min(int(slice_center[0] + slice_size[0] / 2), enhanced.shape[1]))
# )
# slice_image = enhanced[slicer]
# display(slice_image)

# # Ensure there's content in the image
# assert slice_image.shape[0] and slice_image.shape[1]

# detected_laser_points, _ = _find_circles(slice_image, 1, laser_radius)
# display(display2(
#     slice_image,
#     "detected_laser_points",
#     key_points=detected_laser_points,
#     line_thickness=5,
#     marker_color=(255, 0, 0, 255),
#     save_image=False,
#     open_image=False,
# ))

# %%
def find_laser_points(
    image: np.ndarray,
    expected_locations: list[cv2.KeyPoint],
    laser_radius_mm: float,
    slice_size: tuple[int, int] = (10, 10),
    debug: bool = False,
) -> list[cv2.KeyPoint | None]:

    detected_points = []
    slice_size = (slice_size[0] * scan_mm_to_pixels, slice_size[1] * scan_mm_to_pixels)

    for point in expected_locations:
        x_start = max(int(point.pt[0] - slice_size[0] / 2), 0)
        y_start = max(int(point.pt[1] - slice_size[1] / 2), 0)
        slicer = (
            slice(y_start, min(y_start + int(slice_size[1]), image.shape[0])),
            slice(x_start, min(x_start + int(slice_size[0]), image.shape[1])),
        )
        slice_image = image[slicer]
        slice_points, _ = _find_circles(slice_image, 1, laser_radius_mm)
        if debug:
            display(display2(
                slice_image,
                "slice_points",
                key_points=slice_points,
                line_thickness=5,
                marker_color=(255, 0, 0, 255),
                save_image=False,
                open_image=False,
            ))

        def _dist(p1: cv2.KeyPoint, p2: cv2.KeyPoint) -> float:
            return math.sqrt((p1.pt[0] - p2.pt[0])**2 + (p1.pt[1] - p2.pt[1])**2)

        if len(slice_points) == 0:
            detected_points.append(None)
        else:
            if len(slice_points) > 1:
                closest_point = min(slice_points, key=lambda p: _dist(p, point))
            else:
                closest_point = slice_points[0]
            global_point = cv2.KeyPoint(
                closest_point.pt[0] + x_start,
                closest_point.pt[1] + y_start,
                size=closest_point.size,
            )
            detected_points.append(global_point)

    return detected_points

detected_laser_points = find_laser_points(enhanced, detected_drill_points, laser_radius)
# disp = display2(
#     inverted,
#     "detected_laser_points",
#     key_points=list(filter(lambda p: p is not None, detected_laser_points)),
#     line_thickness=5,
#     marker_color=(255, 0, 0, 255),
#     # downscale=0.1,
#     save_image=False,
#     open_image=False,
#     downscale=0.1,
# )
# display(disp)

# %%
disp = display2(
    inverted,
    key_points=list(filter(lambda p: p is not None, detected_laser_points)),
    line_thickness=5,
    marker_color=(255, 0, 0, 255),
    # downscale=0.1,
    save_image=False,
    open_image=False,
)
_ = display2(
    disp,
    "everything",
    key_points=detected_drill_points,
    line_thickness=5,
    marker_color=(0, 255, 0, 255),
)

# %%
ordered_detected_drill_points = order_points_into_grid(keypoints_to_array(detected_drill_points), drill_hole_count)

plot_enumerate_points(ordered_detected_drill_points, image)

# %%
ordered_drill_objpoints = order_points_into_grid(drill_objpoints, drill_hole_count)
obj_to_img_homography, _ = cv2.findHomography(
    ordered_drill_objpoints,
    ordered_detected_drill_points,
)
print(obj_to_img_homography)

plot_enumerate_points(transform_points(drill_objpoints, obj_to_img_homography), image)


img_to_obj_homography, _ = cv2.findHomography(
    ordered_detected_drill_points,
    ordered_drill_objpoints,
)
print(img_to_obj_homography)

# This looks like shit because it's in mm in object space, not pixels in image space, so the image is fucked up
transformed_img = cv2.warpPerspective(image, img_to_obj_homography, input_image_dimensions)
display(display2(
    transformed_img,
    "transformed_img",
    save_image=False,
    open_image=False,
    points=drill_objpoints,
    point_size=5,
    marker_color=(0, 255, 0, 255),
))
# %%
detected_drill_points_mm = transform_points(keypoints_to_array(detected_drill_points), img_to_obj_homography)

detected_laser_points_mm = transform_points(
    keypoints_to_array(detected_laser_points),
    img_to_obj_homography,
)

# zipped_points = list(zip_closest(detected_drill_points_mm, detected_laser_points_mm))

mm_mappers = create_undistortion_mappers(
    [cv2.KeyPoint(p[0], p[1], 1) for p in detected_drill_points_mm],
    [cv2.KeyPoint(p[0], p[1], 1) for p in detected_laser_points_mm],
)

compensated_drill_points = transform_points(drill_objpoints, mappers=mm_mappers)

n_points = None
plt.scatter(
    detected_drill_points_mm[:n_points, 0],
    detected_drill_points_mm[:n_points, 1],
    color="green",
    s=10,
)

for i, p in enumerate(detected_drill_points_mm[:n_points]):
    plt.annotate(
        f"{i}",
        xy=(p[0], p[1]),
        xytext=(p[0] + 1, p[1] + 1),
        color="green",
    )

plt.scatter(
    detected_laser_points_mm[:n_points, 0],
    detected_laser_points_mm[:n_points, 1],
    color="red",
    s=5,
)

for i, p in enumerate(detected_laser_points_mm[:n_points]):
    plt.annotate(
        f"{i}",
        xy=(p[0], p[1]),
        xytext=(p[0] + 1, p[1] + 1),
        color="red",
    )

plt.scatter(
    [p[0] for p in compensated_drill_points[:n_points]],
    [p[1] for p in compensated_drill_points[:n_points]],
    color="blue",
    s=3,
)

for i, p in enumerate(compensated_drill_points[:n_points]):
    plt.annotate(
        f"{i}",
        xy=(p[0], p[1]),
        xytext=(p[0] + 1, p[1] + 1),
        color="blue",
    )

plt.legend(["original", "transformed", "compensated"], loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.gca().invert_yaxis()
# plt.axis([0, input_image_dimensions[0], input_image_dimensions[1], 0])

plt.show()

# %%
def display_compensation(image: np.ndarray, open_image: bool = False):
    drill_color = (0, 255, 0, 255)
    laser_color = (255, 0, 0, 255)
    target_color = (255, 0, 255, 255)
    compensated_color = (0, 0, 255, 255)

    disp = display2(
        image,
        "",
        points=transform_points(compensated_drill_points, obj_to_img_homography),
        point_size=detected_drill_points[0].size / 2,
        marker_color=compensated_color,
        save_image=False,
        open_image=False,
        line_thickness=5,
    )
    lines = [
        ("compensated", compensated_color),
        ("target", target_color),
        ("laser dots", laser_color),
        ("drill dots", drill_color),
    ]

    for i, (text, color) in enumerate(lines):
        cv2.putText(
            disp,
            text,
            (10, 200 + i * 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=10,
            color=color,
            thickness=10,
        )

    disp = display2(
        disp,
        "",
        points=transform_points(detected_drill_points_mm, obj_to_img_homography),
        point_size=detected_drill_points[0].size / 2,
        marker_color=drill_color,
        save_image=False,
        open_image=False,
        line_thickness=5,
    )
    disp = display2(
        disp,
        "",
        points=transform_points(drill_objpoints, obj_to_img_homography),
        point_size=detected_drill_points[0].size / 2,
        marker_color=target_color,
        line_thickness=5,
        save_image=False,
        open_image=False,
    )
    disp = display2(
        disp,
        "compensated_points",
        points=transform_points(detected_laser_points_mm, obj_to_img_homography),
        point_size=detected_drill_points[0].size / 2,
        marker_color=laser_color,
        line_thickness=5,
        save_image=open_image,
        open_image=open_image,
    )

    return disp

display_compensation(image, open_image=True)

# %%
canvas = create_svg_grid(
    compensated_drill_points,
    laser_radius,
)

counter = 0
filename = build_dir / f"compensated_dots{counter}.svg"
while filename.exists():
    counter += 1
    filename = build_dir / f"compensated_dots{counter}.svg"

with filename.open("w") as f:
    f.write(str(canvas))

print(f"Saved to {filename}")


# %%
point_pairs = list(zip_closest(detected_drill_points, detected_laser_points))

def scatter_point_pairs(point_pairs: list[tuple[cv2.KeyPoint, cv2.KeyPoint]]):
    plt.scatter(
        [p[0].pt[0] for p in point_pairs],
        [p[0].pt[1] for p in point_pairs],
        color="blue",
        s=3,
    )

    for i, p in enumerate(point_pairs):
        plt.annotate(
            f"{i}",
            xy=(p[0].pt[0], p[0].pt[1]),
            xytext=(p[0].pt[0] + 1, p[0].pt[1] + 1),
            color="blue",
        )

    plt.scatter(
        [p[1].pt[0] for p in point_pairs],
        [p[1].pt[1] for p in point_pairs],
        color="red",
        s=3,
    )

    for i, p in enumerate(point_pairs):
        plt.annotate(
            f"{i}",
            xy=(p[1].pt[0], p[1].pt[1]),
            xytext=(p[1].pt[0] + 1, p[1].pt[1] + 1),
            color="red",
        )

    plt.gca().invert_yaxis()
    plt.show()

scatter_point_pairs(point_pairs)

# %%
plot_error_histogram(point_pairs, scan_mm_to_pixels)

# %%
# Morph and remap PCB images
# %%
# import gen_svg
# input_pcb_file = only_one(build_dir.glob("*.kicad_pcb"))

# kicad_build_dir = build_dir / "kicad"

# # Process SVG outlines for material to be removed
# edge_cuts_file = gen_svg.export_svg(input_pcb_file, "Edge.Cuts", kicad_build_dir)
# top_copper_file = gen_svg.export_svg(input_pcb_file, "F.Cu", kicad_build_dir)
# bottom_copper_file = gen_svg.export_svg(input_pcb_file, "B.Cu", kicad_build_dir)
# mask_file = gen_svg.export_svg(input_pcb_file, "F.Mask", kicad_build_dir)
# paste_file = gen_svg.export_svg(input_pcb_file, "F.Paste", kicad_build_dir)

# # %%
# dpi = 1000
# def load_svg_as_gray(svg_path: Path, dpi: float) -> np.ndarray:
#     png_data = cairosvg.svg2png(
#         url=str(svg_path),
#         dpi=dpi,
#         scale=1,
#         background_color="none"
#     )

#     image = cv2.imdecode(np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

#     # Return only the alpha channel
#     return image[:, :, 3]

# display(load_svg_as_gray(edge_cuts_file, dpi))

# # %%
# def stroke_mask(image: np.ndarray, stroke_size: int) -> np.ndarray:
#     # Create a kernel for dilation
#     kernel = np.ones((stroke_size, stroke_size), np.uint8)

#     assert image.dtype == np.uint8
#     assert image.ndim == 2
#     # Dilate the mask
#     dilated = cv2.dilate(image, kernel, iterations=1)

#     # Apply stroke color where dilated but not in original
#     return (dilated != 0) & (image == 0)

# display(stroke_mask(load_svg_as_gray(edge_cuts_file, dpi), 20))

# # %%
# # Clip the edge cuts mask to the bounding box
# def clip(mask: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
#     return mask[bounds[1]:bounds[1]+bounds[3], bounds[0]:bounds[0]+bounds[2]]

# # %%
# def flood_fill_center(mask: np.ndarray, color: int) -> np.ndarray:
#     assert mask.dtype == np.uint8
#     assert mask.ndim == 2

#     mask = mask.copy()
#     h, w = mask.shape[:2]

#     # Create a mask that's 2 pixels larger in each dimension
#     flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

#     box = cv2.boundingRect(mask)

#     cv2.floodFill(
#         image=mask,
#         mask=flood_mask,  # Properly sized mask
#         seedPoint=(box[0] + box[2] // 2, box[1] + box[3] // 2),
#         newVal=color,
#     )

#     return mask

# display(flood_fill_center(load_svg_as_gray(edge_cuts_file, dpi), 255))

# # %%
# offset = np.array([-3.931 + 1.3, -0.502 - 3.2]) * dpi / 25.4  # x, y offset
# # offset = np.array([0, 0]) * dpi / 25.4  # x, y offset

# def process(svg_path: Path) -> Path:
#     mask = load_svg_as_mask(svg_path, dpi)
#     clipped = clip(mask, stroked_box)
#     rotated = cv2.rotate(clipped, cv2.ROTATE_90_CLOCKWISE)

#     blank = np.ones((output_size[1], output_size[0]), dtype=np.uint8) * 255
#     start_y = int(blank.shape[0] / 2 - rotated.shape[0] / 2 + offset[1])
#     start_x = int(blank.shape[1] / 2 - rotated.shape[1] / 2 + offset[0])

#     if start_y < 0:
#         start_y = 0
#         print("Warning: offset is too large, clipping y")

#     if start_y + rotated.shape[0] > blank.shape[0]:
#         start_y = blank.shape[0] - rotated.shape[0]
#         print("Warning: offset is too large, clipping y")

#     if start_x < 0:
#         start_x = 0
#         print("Warning: offset is too large, clipping x")

#     if start_x + rotated.shape[1] > blank.shape[1]:
#         start_x = blank.shape[1] - rotated.shape[1]
#         print("Warning: offset is too large, clipping x")

#     blank[
#         start_y:start_y + rotated.shape[0],
#         start_x:start_x + rotated.shape[1],
#     ] = rotated

#     warped = cv2.warpPerspective(
#         blank,
#         scaled_h,
#         output_size,
#         borderValue=(255, 255, 255),
#     )

#     output_dir = build_dir / "shoot"
#     output_dir.mkdir(exist_ok=True)
#     output_path = output_dir / f"{svg_path.stem}.png"
#     cv2.imwrite(output_path, warped)
#     return output_path

# process(top_copper_file)
# # process(bottom_copper_file)
# # process(mask_file)
# # process(paste_file)

# # %%

# %%
# # Morph an image according to the mapping we've found
# target_dpi = 300
# target_px_per_mm = target_dpi / 25.4

# def load_svg_as_gray(svg_path: Path, dpi: float) -> np.ndarray:
#     png_data = cairosvg.svg2png(
#         url=str(svg_path),
#         dpi=dpi,
#         scale=1,
#         background_color="none"
#     )

#     image = cv2.imdecode(np.frombuffer(png_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

#     # Return only the alpha channel
#     return image[:, :, 3]

# target_image = load_svg_as_gray(dots_svg, target_dpi)

# # %%
# disp = display2(
#     target_image,
#     "target_image",
#     points=compensated_drill_points * target_px_per_mm,
#     point_size=drill_radius * target_px_per_mm * 2,
#     marker_color=(255, 0, 0, 255),
# )
# %%
calibration_dir = my_dir / "calibrations"

mm_remap_mappers = create_undistortion_mappers(
    [cv2.KeyPoint(p[0], p[1], 1) for p in detected_drill_points_mm],
    [cv2.KeyPoint(p[0], p[1], 1) for p in detected_laser_points_mm],
    invert=True,
)

import calibration_package

cal_pkg = calibration_package.CalibrationPackage(
    predecessor=None,
    description="Speed run",
    compensate_mm_mapper=mm_mappers,
    remap_mm_mapper=mm_remap_mappers,
    bounds_mm=((0, 0), input_image_dimensions),
    target_drill_points=compensated_drill_points,
    detected_drill_points_mm=detected_drill_points_mm,
    detected_laser_points_mm=detected_laser_points_mm,
)

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
cal_pkg.save(my_dir / "calibrations" / f"{timestamp}-Speed-Run.pkl")

# %%
# cal_pkg = calibration_package.CalibrationPackage.load(my_dir / "calibrations" / "20241103-Top-Calibration.pkl")

target_dpi = 1200
target_px_per_mm = target_dpi / 25.4
input_grid_width, input_grid_height = int(input_image_dimensions[0] * target_px_per_mm), int(input_image_dimensions[1] * target_px_per_mm)

grid_x, grid_y = np.meshgrid(
    np.arange(input_grid_width), np.arange(input_grid_height), indexing='xy'
)

grid_points_x = grid_x.ravel() / target_px_per_mm
grid_points_y = grid_y.ravel() / target_px_per_mm

# Compute the mapping arrays using the forward RBF interpolators
map_inverse_x = cal_pkg.remap_mm_mapper[0](grid_points_x, grid_points_y).reshape(input_grid_height, input_grid_width).astype(np.float32) * target_px_per_mm
map_inverse_y = cal_pkg.remap_mm_mapper[1](grid_points_x, grid_points_y).reshape(input_grid_height, input_grid_width).astype(np.float32) * target_px_per_mm

# Clip the mapping arrays to valid ranges to avoid invalid indices
map_inverse_x = np.clip(map_inverse_x, 0, input_grid_width - 1)
map_inverse_y = np.clip(map_inverse_y, 0, input_grid_height - 1)

# %%
for target_image_path in (build_dir / "targets").glob("*.png"):
    target_image = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)

    target_image = cv2.resize(target_image, (input_grid_height, input_grid_width), interpolation=cv2.INTER_AREA)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGRA2GRAY)
    target_image = cv2.bitwise_not(target_image)

    # Reshape the mappings back to the grid shape
    restored_image = cv2.remap(
        target_image,
        map_inverse_x,
        map_inverse_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    output_image_width = int(output_image_dimensions[0] * target_px_per_mm)
    output_image_height = int(output_image_dimensions[1] * target_px_per_mm)
    width_start = (restored_image.shape[1] - output_image_width) // 2
    height_start = (restored_image.shape[0] - output_image_height) // 2
    restored_image = restored_image[height_start:height_start+output_image_height, width_start:width_start+output_image_width]

    cv2.imwrite(target_image_path.with_suffix(".restored.png"), restored_image)

# %%
# Display both the restored image and target together
# First ensure both images have the same dimensions
target_shape = target_image.shape[:2]
restored_shape = restored_image.shape[:2]

# Resize restored_image to match target_image dimensions
restored_resized = cv2.resize(restored_image, (target_shape[1], target_shape[0]))

both = np.zeros((target_shape[0], target_shape[1], 4), dtype=np.uint8)
both[:, :, 3] = 255
both[:, :, 0] = cv2.cvtColor(restored_resized, cv2.COLOR_BGRA2GRAY)
both[:, :, 1] = cv2.cvtColor(target_image, cv2.COLOR_BGRA2GRAY)
display2(both)

# %%
