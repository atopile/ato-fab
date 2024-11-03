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
def create_svg_grid(points: np.ndarray, radius: float) -> svg.SVG:
    canvas = svg.SVG(
        width=f"{output_image_dimensions[0]}mm",
        height=f"{output_image_dimensions[1]}mm",
        elements=[],
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

canvas = create_svg_grid(drill_objpoints, laser_radius)

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
detected_drill_points, detector = find_blobs(inverted, drill_hole_count, drill_radius)
display2(
    inverted,
    "detected_points",
    key_points=detected_drill_points,
    line_thickness=10,
    marker_color=(0, 0, 255, 255),
)

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
# def create_undistortion_maps(
#     rbf_x: Rbf,
#     rbf_y: Rbf,
#     image_shape: tuple[int, int],
# ) -> tuple[np.ndarray, np.ndarray]:
#     # Generate grid over the distorted image
#     grid_y, grid_x = np.mgrid[0:image_shape[0], 0:image_shape[1]]
#     grid_points_x = grid_x.ravel()
#     grid_points_y = grid_y.ravel()

#     # Compute the source coordinates using the RBF interpolators
#     map_x = rbf_x(grid_points_x, grid_points_y)
#     map_y = rbf_y(grid_points_x, grid_points_y)

#     # Reshape the mappings back to the grid shape
#     map_x = map_x.reshape(image_shape).astype(np.float32)
#     map_y = map_y.reshape(image_shape).astype(np.float32)

#     # Clip the mapping arrays to valid ranges
#     map_x = np.clip(map_x, 0, image_shape[1] - 1)
#     map_y = np.clip(map_y, 0, image_shape[0] - 1)

#     return map_x, map_y

# %%
image_dpi = 300

def scale_keypoints(keypoints: list[cv2.KeyPoint | None], scale: float) -> list[cv2.KeyPoint | None]:
    # TODO: handle None keypoints
    return [
        cv2.KeyPoint(
            x=keypoint.pt[0] * scale,
            y=keypoint.pt[1] * scale,
            size=keypoint.size * scale,
        )
        for keypoint in keypoints
    ]


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

ordered_detected_drill_points = order_points_into_grid(keypoints_to_array(detected_drill_points), drill_hole_count)

def enumerate_points(image, points):
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

enumerate_points(image, ordered_detected_drill_points)

# %%
ordered_drill_objpoints = order_points_into_grid(drill_objpoints, drill_hole_count)
obj_to_img_homography, _ = cv2.findHomography(
    ordered_drill_objpoints,
    ordered_detected_drill_points,
)
print(obj_to_img_homography)

def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2),  # Reshape to (N, 1, 2)
        homography,
    ).reshape(-1, 2)

enumerate_points(image, transform_points(drill_objpoints, obj_to_img_homography))

# %%
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
def create_undistortion_mappers(
    detected_points: list[cv2.KeyPoint | None],
    expected_points: list[cv2.KeyPoint],
    detected_to_expected: bool = False,
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

    if detected_to_expected:
        # Map from detected (laser) positions to expected (drill) positions
        rbf_dx = Rbf(valid_detected[:, 0], valid_detected[:, 1], valid_expected[:, 0], function='cubic')
        rbf_dy = Rbf(valid_detected[:, 0], valid_detected[:, 1], valid_expected[:, 1], function='cubic')
    else:
        # Map from expected (drill) positions to detected (laser) positions
        rbf_dx = Rbf(valid_expected[:, 0], valid_expected[:, 1], valid_detected[:, 0], function='cubic')
        rbf_dy = Rbf(valid_expected[:, 0], valid_expected[:, 1], valid_detected[:, 1], function='cubic')

    return rbf_dx, rbf_dy

detected_drill_points_mm = transform_points(keypoints_to_array(detected_drill_points), img_to_obj_homography)

detected_laser_points_mm = transform_points(
    keypoints_to_array(detected_laser_points),
    img_to_obj_homography,
)

mm_mappers = create_undistortion_mappers(
    [cv2.KeyPoint(p[0], p[1], 1) for p in detected_drill_points_mm],
    [cv2.KeyPoint(p[0], p[1], 1) for p in detected_laser_points_mm],
)

compensated_drill_points = np.array([
    [
        mm_mappers[0](point[0], point[1]),
        mm_mappers[1](point[0], point[1]),
    ]
    for point in drill_objpoints
])

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

# plt.legend(["original", "transformed", "compensated"])
plt.gca().invert_yaxis()
# plt.axis([0, input_image_dimensions[0], input_image_dimensions[1], 0])

plt.show()

# %%
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
_ = display2(
    disp,
    "compensated_points",
    points=transform_points(detected_laser_points_mm, obj_to_img_homography),
    point_size=detected_drill_points[0].size / 2,
    marker_color=laser_color,
    line_thickness=5,
)

# display(disp)


# %%
canvas = create_svg_grid(
    compensated_drill_points,
    laser_radius,
)

counter += 1
with (build_dir / f"compensated_dots{counter}.svg").open("w") as f:
    f.write(str(canvas))


# laser_dots_still_fucked = cv2.cvtColor(
#     cv2.imdecode(
#         np.frombuffer(cairosvg.svg2png(url=str(dots_svg), dpi=image_dpi), dtype=np.uint8),
#         cv2.IMREAD_UNCHANGED
#     ),
#     cv2.COLOR_RGBA2BGRA
# )
# # display2(laser_dots_still_fucked, "laser_dots_still_fucked")

# # %%
# # Apply the undistortion
# laser_dots_unfucked = cv2.remap(
#     laser_dots_still_fucked,
#     *maps,
#     interpolation=cv2.INTER_LINEAR,
#     borderMode=cv2.BORDER_CONSTANT
# )

# cv2.imwrite(build_dir / "laser_dots_unfucked.png", laser_dots_unfucked)

# %%
# Save compensation mappers
import pickle

with (my_dir / "compensation_mappers.pkl").open("wb") as f:
    pickle.dump(mm_mappers, f)


# %%
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

point_pairs = list(zip_closest(detected_drill_points, detected_laser_points))

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

# %%
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

plot_error_histogram(point_pairs, scan_mm_to_pixels)

# %%
