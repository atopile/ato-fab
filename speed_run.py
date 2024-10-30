# %%
# 1. Generate a grid of circles
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

def draw_grid(image, xs, ys):
    if xs is not None:
        for x, color in zip(xs, generate_colors(len(xs))):
            cv.line(image, (int(x), 0), (int(x), image.shape[0]), color, 1)

    if ys is not None:
        for y, color in zip(ys, generate_colors(len(ys))):
            cv.line(image, (0, int(y)), (image.shape[1], int(y)), color, 1)

    return image

def display2(
    image,
    title: str = "image",
    key_points: list[cv.KeyPoint] | None = None,
    points: np.ndarray | None = None,
    point_size: int | None = None,
    grid_xs: list[int] | None = None,
    grid_ys: list[int] | None = None,
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
        cv.circle(image, (int(point.pt[0]), int(point.pt[1])), int(point.size/2), (0, 0, 255), 1)

    for point in points:
        cv.circle(image, (int(point[0]), int(point[1])), int(point_size), (0, 0, 255), 1)

    image = draw_grid(image, grid_xs, grid_ys)

    save(**{title: image})

    subprocess.run(["open", build_dir / f"{title}.png"])

    return image

# %%
# General properties
dot_radius = 0.5  # radius mm
dot_count = (14, 14)
step = (8, 8)
bounding_box = (110, 110)

# 1.1 Calculate the points where the circles will be placed
x_padding = (bounding_box[0] - (dot_count[0] - 1) * step[0]) / 2
y_padding = (bounding_box[1] - (dot_count[1] - 1) * step[1]) / 2

objpoints = np.zeros((dot_count[0] * dot_count[1], 3), dtype=np.float32)
objpoints[:, :2] = np.mgrid[0:dot_count[0], 0:dot_count[1]].T.reshape(-1, 2)
objpoints = objpoints * np.array([step[0], step[1], 1]) + np.array([x_padding, y_padding, 0])
objpoints = objpoints.astype(np.float32)

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
for x, y, _ in objpoints:
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

Image.MAX_IMAGE_PIXELS = None  # Disable the PIL limit because we own everything
scan_dpi = Image.open(latest_scan).info["dpi"][0]
mm_to_pixels = scan_dpi / 25.4
objpoints_pixels = (objpoints * mm_to_pixels).astype(np.float32)

# Invert the image
image = cv.bitwise_not(image)
image = cv.rotate(image, cv.ROTATE_180)  # TODO: remove me

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
    area_pixels = math.pi * (radius_mm * mm_to_pixels)**2
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
datum_objpoints_pixels = (datum_objpoints * mm_to_pixels).astype(np.float32)

params = make_params(1)
params.filterByArea = True
params.filterByCircularity = True
params.filterByConvexity = True
params.filterByInertia = False
detector = cv.SimpleBlobDetector.create(params)
detected_points = detector.detect(gray)

print(f"Found {len(detected_points)} datums")
if len(detected_points) != 4:
    print("Detected points: " + str([point.pt for point in detected_points]))
    raise ValueError("Incorrect number of datums found")

display2(gray, "detected_points", key_points=detected_points)

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
corrected_scan = cv.warpPerspective(gray, img_to_real_h, gray.shape[::-1])

display2(
    corrected_scan,
    "corrected_scan_datums",
    points=datum_objpoints_pixels[:, :2],
    point_size=2.2/2 * mm_to_pixels,
    grid_xs=np.unique(datum_objpoints_pixels[:, 0]),
    grid_ys=np.unique(datum_objpoints_pixels[:, 1]),
)

# %%
# Find all the laser points on the corrected scan ----------------------------
params = make_params(dot_radius, 0.1)  # NOTE: the 5% tolerance actually helps, higher is worse
params.filterByCircularity = False
params.filterByConvexity = False

found, imgpoints = cv.findCirclesGrid(
    corrected_scan,
    dot_count,
    cv.CALIB_CB_SYMMETRIC_GRID,
    blobDetector=cv.SimpleBlobDetector.create(params),
)

print(f"Found {len(imgpoints) if imgpoints is not None else 'no'} circles")
assert found, "Circle processing failed"
assert len(imgpoints) == dot_count[0] * dot_count[1], "Incorrect number of circles found"

# Show the corners we've found
display2(corrected_scan, "laser_dots", points=imgpoints.reshape(-1, 2), point_size=dot_radius * mm_to_pixels)

# %%
# Find the homography that flattens the checkerboard
h, _ = cv.findHomography(imgpoints, objpoints_pixels[:, :2])

# Warp the image using the homography
size = image.shape[:2]
unwarped_image = cv.warpPerspective(corrected_scan, h, size)

# Display a grid overtop the unwarped image to validate it's correct
circles_xs = np.unique(objpoints_pixels[:, 0])
circles_ys = np.unique(objpoints_pixels[:, 1])
display2(
    unwarped_image,
    "unwarped_image",
    points=objpoints_pixels[:, :2],
    point_size=dot_radius * mm_to_pixels,
    grid_xs=circles_xs,
    grid_ys=circles_ys,
)
