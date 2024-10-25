import os
from pathlib import Path

import cv2

# Get the latest .png file in the specified directory
image_path = max(Path(__file__).parent.parent.glob("img*.png"), key=os.path.getctime)
print(f"Using image: {image_path}")

output_path = "annotated_checkerboard.png"

# Load the image
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image '{image_path}'. Check the file path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the checkerboard size (number of inner corners per chessboard row and column)
checkerboard_size = (10, 10)  # For an 11x11 checkerboard, there are 10x10 inner corners

# Invert colors in the chessboard
inverted_image = cv2.bitwise_not(image)

# Use the inverted image for further processing
gray = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(
    image=gray,
    patternSize=checkerboard_size,
    flags=(
        cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    ),
)
print(f"Found {len(corners)} corners")

if ret:
    # Define criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Refine corner positions
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Draw and display the corners
    cv2.drawChessboardCorners(image, checkerboard_size, corners_refined, ret)

    # Save the annotated image
    cv2.imwrite(output_path, image)

    # Display the annotated image
    cv2.imshow("Detected Checkerboard", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Checkerboard detected and annotated image saved as '{output_path}'.")
else:
    print("Checkerboard pattern not found in the image.")
