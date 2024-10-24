import numpy as np
import cv2

# Parameters for the checkerboard
board_size_mm = 110  # Total size of the laser area
square_size_mm = 12  # Size of each square in mm
grid_size = (int(board_size_mm // square_size_mm), int(board_size_mm // square_size_mm))  # Number of squares (9x9)

# Set the desired DPI
dpi = 4800

# Calculate pixel dimensions based on physical size and DPI
pixels_per_mm = dpi / 25.4  # Convert DPI to pixels per mm
img_size = (int(board_size_mm * pixels_per_mm), int(board_size_mm * pixels_per_mm))

# Create the checkerboard pattern
checkerboard = np.zeros(img_size, dtype=np.uint8)

for i in range(grid_size[1]):
    for j in range(grid_size[0]):
        if (i + j) % 2 == 0:
            y1 = int(i * square_size_mm * pixels_per_mm)
            y2 = int((i + 1) * square_size_mm * pixels_per_mm)
            x1 = int(j * square_size_mm * pixels_per_mm)
            x2 = int((j + 1) * square_size_mm * pixels_per_mm)
            cv2.rectangle(checkerboard, (x1, y1), (x2, y2), (255, 255, 255), -1)

# Save the checkerboard image
cv2.imwrite('checkerboard_110x110mm_4800dpi.png', checkerboard)

# Display the pattern for verification (resized for screen viewing)
display_size = (800, 800)
display_img = cv2.resize(checkerboard, display_size, interpolation=cv2.INTER_AREA)
cv2.imshow('Checkerboard (Resized for Display)', display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
