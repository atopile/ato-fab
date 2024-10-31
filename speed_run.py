# %%
# 1. Generate a grid of circles
import itertools
import math
import os
from pathlib import Path
import subprocess

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import svg
from PIL import Image

# Helper functions
def latest[T](glob: list[T]) -> T:
    return max(glob, key=os.path.getctime)

def save(**kwargs):
    for k, v in kwargs.items():
        cv.imwrite(build_dir / f"{k}.png", v)

def display(image):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def generate_colors(num_colors):
    colors = []
    for i in range(num_colors):
    # Convert HSV to BGR - Hue cycles from 0 to 180 (in OpenCV), full saturation and value
        hsv_color = np.uint8([[[i * 180 // num_colors, 255, 255]]])
        bgr_color = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0][0]
        # Convert to regular tuple of ints
        colors.append(tuple(map(int, bgr_color)))
    return colors

def draw_grid(image, xs, ys, line_thickness: int = 1):
    if xs is not None:
        for x, color in zip(xs, generate_colors(len(xs))):
            cv.line(image, (int(x), 0), (int(x), image.shape[0]), color, line_thickness)

    if ys is not None:
        for y, color in zip(ys, generate_colors(len(ys))):
            cv.line(image, (0, int(y)), (image.shape[1], int(y)), color, line_thickness)

    return image

def display2(
    image,
    title: str = "image",
    key_points: list[cv.KeyPoint] | None = None,
    points: np.ndarray | None = None,
    point_size: int | None = None,
    grid_xs: list[int] | None = None,
    grid_ys: list[int] | None = None,
    marker_color: tuple[int, int, int] = (0, 0, 255),
    save_image: bool = True,
    open_image: bool = True,
    line_thickness: int = 1,
):
    if key_points is None:
        key_points = []

    if points is not None:
        assert point_size is not None, "point_size must be provided if points are provided"
    else:
        points = []

    if len(image.shape) == 2:  # Check if grayscale
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        image = image.copy()

    for point in key_points:
        cv.circle(image, (int(point.pt[0]), int(point.pt[1])), int(point.size/2), marker_color, line_thickness)

    for point in points:
        cv.circle(image, (int(point[0]), int(point[1])), int(point_size), marker_color, line_thickness)

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
dot_radius = 0.5  # radius mm
dot_count = (14, 14)
step = (8, 8)
bounding_box = (110, 110)

# 1.1 Calculate the points where the circles will be placed
x_padding = (bounding_box[0] - (dot_count[0] - 1) * step[0]) / 2
y_padding = (bounding_box[1] - (dot_count[1] - 1) * step[1]) / 2

calibration_objpoints = np.zeros((dot_count[0] * dot_count[1], 3), dtype=np.float32)
calibration_objpoints[:, :2] = np.mgrid[0:dot_count[0], 0:dot_count[1]].T.reshape(-1, 2)
calibration_objpoints = calibration_objpoints * np.array([step[0], step[1], 1]) + np.array([x_padding, y_padding, 0])
calibration_objpoints = calibration_objpoints.astype(np.float32)

# 1.2 Generate the SVG file
canvas = svg.SVG(
    width=f"{bounding_box[0]}mm",
    height=f"{bounding_box[1]}mm",
    elements=[
        svg.Rect(
            x=0,
            y=0,
            width=f"{bounding_box[0]}mm",
            height=f"{bounding_box[1]}mm",
            stroke="black",
            fill="none",
        ),
    ],
)
for x, y, _ in calibration_objpoints:
    canvas.elements.append(
        svg.Circle(
            cx=f"{x}mm",
            cy=f"{y}mm",
            r=f"{dot_radius}mm",
            fill="blue",
        )
    )

# 1.3 Save the SVG file
my_dir = Path(__name__).parent
build_dir = my_dir / "build"
dots_svg = build_dir / "dots.svg"

with dots_svg.open("w") as f:
    f.write(str(canvas))

# %%
latest_scan = latest(build_dir.glob("img*.png"))
print(f"Using {latest_scan}")
image = cv.imread(latest_scan)

scan_mm_to_pixels = get_mm_to_pixels(latest_scan)

# Invert the image
image = cv.bitwise_not(image)
# image = cv.rotate(image, cv.ROTATE_180)  # TODO: remove me

display(image)
print(f"Image size: {image.shape}")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# %%
# Print out the default parameters for the blob detector
# See: https://stackoverflow.com/questions/39703407/using-findcirclesgrid-in-large-images
fiber_dots_params = cv.SimpleBlobDetector.Params()
for attr in dir(fiber_dots_params):
    if attr.startswith("_"):
        continue
    print(f"{attr} = {getattr(fiber_dots_params, attr)}")

# Tweak them as required
def make_params(radius_mm: float, tolerance: float = 0.3, tolerance_abs: int = 0) -> cv.SimpleBlobDetector.Params:
    params = cv.SimpleBlobDetector.Params()
    area_pixels = math.pi * (radius_mm * scan_mm_to_pixels)**2
    area_range = tolerance * area_pixels
    params.minArea = int(area_pixels - area_range)
    params.maxArea = int(area_pixels + area_range)
    params.minThreshold = max(0, params.minThreshold - tolerance_abs)
    params.maxThreshold += tolerance_abs
    return params

# %%
# Map the image to our datum coordinate system --------------------------------
# Finding the position and warping of the scan to the datum points
datum_objpoints = np.array(
    [
        [4.344, 6.197, 0],  # Top left
        [104.5, 6.197, 0],  # Top right
        [4.344, 134.072, 0],  # Bottom left
        [104.5, 134.072, 0],  # Bottom right
    ]
)

# Datum -> Target transform is a simple translation, because they're aligned in design
datum_center = np.mean(datum_objpoints, axis=0)
calibration_target_center = np.array([bounding_box[0] / 2, bounding_box[1] / 2, 0])
calibration_objpoints_datum_space = calibration_objpoints - calibration_target_center + datum_center
calibration_objpoints_pixels = (calibration_objpoints_datum_space * scan_mm_to_pixels).astype(np.float32)
datum_objpoints_pixels = (datum_objpoints * scan_mm_to_pixels).astype(np.float32)

# %%
# blurred = cv.blur(gray, (20, 20))
# display2(blurred, "blurred")
for _t, _r in itertools.product([0.05], [2.2/2]):
    print(f"Trying radius={_r}, tolerance={_t}")
    params = make_params(_r, _t)
    params.filterByArea = True
    params.filterByCircularity = True
    params.filterByConvexity = True
    params.filterByInertia = False
    detector = cv.SimpleBlobDetector.create(params)
    detected_points = detector.detect(gray)

    print(f"Found {len(detected_points)} datums")
    if len(detected_points) != 4:
        print("Detected points: " + str([point.pt for point in detected_points]))
    else:

        _img = display2(
            gray,
            "detected_points",
            key_points=detected_points,
            save_image=False,
            open_image=False,
            line_thickness=100,
        )

        display(_img)

        break


# %%
datum_imgpoints = np.array([point.pt for point in detected_points], dtype=np.float32)
# Sort datum points into sectors based on the image quadrants
height, width = gray.shape
center_x = width / 2
center_y = height / 2

sectors = []
for point in datum_imgpoints:
    x, y = point
    if y < center_y:
        if x < center_x:
            sector = 0  # Top left
        else:
            sector = 1  # Top right
    else:
        if x < center_x:
            sector = 2  # Bottom left
        else:
            sector = 3  # Bottom right
    sectors.append((sector, point))

# Sort by sector number and extract just the points
datum_imgpoints = np.array([point for _, point in sorted(sectors)], dtype=np.float32)

# %%
img_to_real_h, _ = cv.findHomography(datum_imgpoints, datum_objpoints_pixels[:, :2])
corrected_image = cv.warpPerspective(gray, img_to_real_h, gray.shape[::-1])

_corrected_image = display2(
    corrected_image,
    "",
    points=datum_objpoints_pixels[:, :2],
    point_size=2.2/2 * scan_mm_to_pixels,
    grid_xs=np.unique(datum_objpoints_pixels[:, 0]),
    grid_ys=np.unique(datum_objpoints_pixels[:, 1]),
    save_image=False,
    open_image=False,
)

# Find all the laser points on the corrected scan ----------------------------
params = make_params(dot_radius, 0.1)  # NOTE: the 5% tolerance actually helps, higher is worse
params.filterByCircularity = False
params.filterByConvexity = False

found, imgpoints = cv.findCirclesGrid(
    corrected_image,
    dot_count,
    cv.CALIB_CB_SYMMETRIC_GRID,
    blobDetector=cv.SimpleBlobDetector.create(params),
)

print(f"Found {len(imgpoints) if imgpoints is not None else 'no'} circles")
assert found, "Circle processing failed"
assert len(imgpoints) == dot_count[0] * dot_count[1], "Incorrect number of circles found"

# %%
# Show the corners we've found
_corrected_image = display2(
    _corrected_image,
    "",
    points=calibration_objpoints_pixels[:, :2],
    point_size=dot_radius * scan_mm_to_pixels,
    marker_color=(255, 0, 0),
    grid_xs=np.unique(calibration_objpoints_pixels[:, 0]),
    grid_ys=np.unique(calibration_objpoints_pixels[:, 1]),
    save_image=False,
    open_image=False,
    line_thickness=10,
)
display2(
    _corrected_image,
    "calibration_dots_detected",
    points=imgpoints.reshape(-1, 2),
    point_size=dot_radius * scan_mm_to_pixels,
    line_thickness=10,
)

# %%
# Find the homography that flattens the board
h, _ = cv.findHomography(imgpoints, calibration_objpoints_pixels[:, :2])
unwarped_image = cv.warpPerspective(corrected_image, h, corrected_image.shape[::-1])

# Display a grid overtop the unwarped image to validate it's correct
circles_xs = np.unique(calibration_objpoints_pixels[:, 0])
circles_ys = np.unique(calibration_objpoints_pixels[:, 1])
display2(
    unwarped_image,
    "unwarped_image",
    points=calibration_objpoints_pixels[:, :2],
    point_size=dot_radius * scan_mm_to_pixels,
    grid_xs=circles_xs,
    grid_ys=circles_ys,
)

# %%
# Warp a new image into the datum coordinate system ----------------------------
latest_target = latest(build_dir.glob("target*.png"))
target_mm_to_pixels = get_mm_to_pixels(latest_target) or 3200

assert target_mm_to_pixels == scan_mm_to_pixels, "Target DPI doesn't match"

target_image = cv.imread(latest_target, cv.IMREAD_UNCHANGED)
display(target_image)

# Warp wrt datum coordinate system
output_size = (int(bounding_box[0] * target_mm_to_pixels), int(bounding_box[1] * target_mm_to_pixels))
target_output = cv.warpPerspective(
    target_image,
    h,
    output_size,
    borderMode=cv.BORDER_CONSTANT,
    borderValue=(0, 0, 0, 0),
)
display(target_output)

cv.imwrite(build_dir / "out-target.png", target_output)

# %%
import pickle

with open(build_dir / "working_homography.pickle", "wb") as f:
    pickle.dump(h, f)

# %%
