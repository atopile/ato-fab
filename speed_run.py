# %%
# 1. Generate a grid of circles
from pathlib import Path

import numpy as np
import svg

dot_radius = 0.5  # radius mm
dot_count = (10, 10)
step = (10, 10)
bounding_box = (110, 110)

# 1.1 Calculate the points where the circles will be placed
x_padding = (bounding_box[0] - (dot_count[0]) * step[0] - 1) / 2
y_padding = (bounding_box[1] - (dot_count[1]) * step[1] - 1) / 2

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
import os

import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image


def latest[T](glob: list[T]) -> T:
    return max(glob, key=os.path.getctime)


latest_scan = latest(build_dir.glob("img*.png"))
print(f"Using {latest_scan}")
image = cv.imread(latest_scan)

Image.MAX_IMAGE_PIXELS = None  # Disable the PIL limit because we own everything
scan_dpi = Image.open(latest_scan).info["dpi"][0]
mm_to_pixels = scan_dpi / 25.4
objpoints_pixels = (objpoints * mm_to_pixels).astype(np.float32)

# Invert the image
image = cv.bitwise_not(image)

def display(image):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


display(image)

print(f"Image size: {image.shape}")

# %%
import math

# Find all the circles on a gray-scale image
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

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

params = make_params(dot_radius)

found, imgpoints = cv.findCirclesGrid(
    gray,
    dot_count,
    cv.CALIB_CB_SYMMETRIC_GRID,
    blobDetector=cv.SimpleBlobDetector.create(params),
)

print(f"Found {len(imgpoints) if imgpoints is not None else 'no'} circles")
assert found, "Circle processing failed"
assert len(imgpoints) == dot_count[0] * dot_count[1], "Incorrect number of circles found"

# Show the corners we've found
import copy
centers_image = copy.deepcopy(image)

for center in imgpoints:
    cv.circle(
        centers_image,
        (int(center[0][0]), int(center[0][1])),
        int(dot_radius * mm_to_pixels),
        (0, 0, 255),
        1,
    )

display(centers_image)
cv.imwrite(build_dir / "centers.png", centers_image)

# %%
# Find the homography that flattens the checkerboard
h, _ = cv.findHomography(imgpoints, objpoints_pixels[:, :2])

# Warp the image using the homography
size = image.shape[:2]
unwarped_image = cv.warpPerspective(image, h, size)

display(unwarped_image)
cv.imwrite(build_dir / "unwarped.png", unwarped_image)

# %%
# Display a grid overtop the unwarped image to validate it's correct
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
    for x, color in zip(xs, generate_colors(len(xs))):
        cv.line(image, (int(x), 0), (int(x), size[0]), color, 1)

    for y, color in zip(ys, generate_colors(len(ys))):
        cv.line(image, (0, int(y)), (size[1], int(y)), color, 1)
    return image

circles_xs = np.unique(objpoints_pixels[:, 0])
circles_ys = np.unique(objpoints_pixels[:, 1])
grid_image = copy.deepcopy(unwarped_image)
grid_image = draw_grid(unwarped_image, circles_xs, circles_ys)

for center in objpoints_pixels[:, :2]:
    cv.circle(
        grid_image,
        (int(center[0]), int(center[1])),
        int(dot_radius * mm_to_pixels),
        (0, 0, 255),
    )

display(grid_image)

cv.imwrite(build_dir / "grid.png", grid_image)

# %%
# Create the mapping array (x,y coordinates for each pixel)
map1 = np.full((*image.shape[:2], 2), np.nan, dtype=np.float32)

# Fill in the known correspondences
# imgpoints contains where points are in the source image
# objpoints_pixels contains where we want them in the destination
for src, dst in zip(imgpoints, objpoints_pixels):
    y, x = int(src[0][1]), int(src[0][0])  # src coordinates in image space
    map1[y, x] = [dst[0], dst[1]]  # dst coordinates where we want this point

# Use remap with interpolation to handle the sparse mapping
remapped = cv.remap(image, map1, None, cv.INTER_CUBIC)

assert not np.all(np.isnan(map1)), "No pixels mapped"

for y, x, _ in np.argwhere(~np.isnan(map1)):
    print(f"Mapping {y}, {x} to {map1[y, x]}")

grid_image = copy.deepcopy(remapped)
grid_image = draw_grid(grid_image, circles_xs, circles_ys)

for center in objpoints_pixels[:, :2]:
    cv.circle(
        grid_image,
        (int(center[0]), int(center[1])),
        int(dot_radius * mm_to_pixels),
        (0, 0, 255),
    )

display(grid_image)

cv.imwrite(build_dir / "undistorted.png", grid_image)
# %%
