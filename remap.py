#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def display_image_with_points(img, points_original=None, points_distorted=None, title='Image'):
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if points_original is not None:
        plt.scatter(points_original[0], points_original[1], 
                   color='red', s=50, label='Original Control Points')
    
    if points_distorted is not None:
        plt.scatter(points_distorted[0], points_distorted[1], 
                   color='blue', s=50, label='Distorted Control Points')
    
    plt.title(title)
    if points_original is not None or points_distorted is not None:
        plt.legend()
    plt.axis('off')
    plt.show()

# Load your image
image = cv2.imread('test_image.png')

# Get image dimensions
height, width = image.shape[:2]

# Parameters for sine wave
amplitude = 20  # Amplitude of the sine wave (in pixels)
period = 300     # Period of the sine wave (in pixels)

# Create the map for remapping
map_x = np.arange(width).astype(np.float32).reshape(1, -1).repeat(height, axis=0)
map_y = np.zeros((height, width), dtype=np.float32)

# Calculate new y-coordinates based on the sine wave
for x in range(width):
    wave_offset = amplitude * np.sin(2 * np.pi * (x / period))
    for y in range(height):
        map_y[y, x] = y + wave_offset

# Ensure map_y is within the image bounds
map_y = np.clip(map_y, 0, height - 1)

# Remap the image
remapped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

display_image(image)
display_image(remapped_image)

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# Define grid size
grid_width = 1000
grid_height = 1000

# Control grid size (3x3)
grid_points_x = 10
grid_points_y = 10

# Original grid control points
x = np.linspace(0, grid_width, grid_points_x)
y = np.linspace(0, grid_height, grid_points_y)
X, Y = np.meshgrid(x, y, indexing='xy')  # Shapes: (3, 3)

# View the grid points
plt.scatter(X, Y, color='red')
plt.title('Control Points')
plt.show()

# %%
# Random displacements between -50 and +50 pixels
max_displacement = 10
np.random.seed(0)  # For reproducibility

dx = np.random.uniform(-max_displacement, max_displacement, X.shape)
dy = np.random.uniform(-max_displacement, max_displacement, Y.shape)

# Apply displacements to control points
X_distorted = X + dx
Y_distorted = Y + dy

# Display the distorted grid points
plt.scatter(X_distorted, Y_distorted, color='blue')
plt.title('Distorted Control Points')
plt.show()

# %%
from scipy.interpolate import Rbf

# Prepare data for inverse mapping
points_distorted = np.column_stack((X_distorted.ravel(), Y_distorted.ravel()))
values_x = X.ravel()
values_y = Y.ravel()

# Create RBF interpolators for inverse mapping using cubic function
rbf_x = Rbf(points_distorted[:, 0], points_distorted[:, 1], values_x, function='cubic')
rbf_y = Rbf(points_distorted[:, 0], points_distorted[:, 1], values_y, function='cubic')

# %%
# Load the input image
input_image = cv2.imread('test_image.png')  # Replace with your image path

# Resize the image to 800x800 pixels
input_image_resized = cv2.resize(input_image, (800, 800), interpolation=cv2.INTER_LINEAR)

# Create an empty image of size 1000x1000 pixels
image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

# Place the resized image in the center of the grid
x_offset = (grid_width - 800) // 2
y_offset = (grid_height - 800) // 2
image[y_offset:y_offset+800, x_offset:x_offset+800] = input_image_resized

# %%
# Generate grid over the distorted image
grid_x_distorted, grid_y_distorted = np.meshgrid(
    np.arange(grid_width), np.arange(grid_height), indexing='xy'
)
grid_points_distorted_x = grid_x_distorted.ravel()
grid_points_distorted_y = grid_y_distorted.ravel()

# Compute the source coordinates using the RBF interpolators
map_x = rbf_x(grid_points_distorted_x, grid_points_distorted_y)
map_y = rbf_y(grid_points_distorted_x, grid_points_distorted_y)

# Reshape the mappings back to the grid shape
map_x = map_x.reshape(grid_height, grid_width).astype(np.float32)
map_y = map_y.reshape(grid_height, grid_width).astype(np.float32)

# Clip the mapping arrays to valid ranges to avoid invalid indices
map_x = np.clip(map_x, 0, grid_width - 1)
map_y = np.clip(map_y, 0, grid_height - 1)

# %%
# Apply the mapping to the image
distorted_image = cv2.remap(
    image,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT
)

# %%
# Display the original and distorted images
def display_image(img, title='Image'):
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

display_image(image, title='Original Image')
display_image(distorted_image, title='Distorted Image')

# Display original image with original control points
display_image_with_points(image, 
                        points_original=(X.ravel(), Y.ravel()),
                        title='Original Image with Control Points')

# Display distorted image with both original and distorted control points
display_image_with_points(distorted_image,
                        points_original=(X.ravel(), Y.ravel()),
                        points_distorted=(X_distorted.ravel(), Y_distorted.ravel()),
                        title='Distorted Image with Control Points')

# %%
# Create RBF interpolators for forward mapping (original to distorted)
rbf_x_forward = Rbf(X.ravel(), Y.ravel(), X_distorted.ravel(), function='cubic')
rbf_y_forward = Rbf(X.ravel(), Y.ravel(), Y_distorted.ravel(), function='cubic')

# %%
# Generate grid over the original image
grid_x, grid_y = np.meshgrid(
    np.arange(grid_width), np.arange(grid_height), indexing='xy'
)
grid_points_x = grid_x.ravel()
grid_points_y = grid_y.ravel()

# Compute the mapping arrays using the forward RBF interpolators
map_inverse_x = rbf_x_forward(grid_points_x, grid_points_y)
map_inverse_y = rbf_y_forward(grid_points_x, grid_points_y)

# Reshape the mappings back to the grid shape
map_inverse_x = map_inverse_x.reshape(grid_height, grid_width).astype(np.float32)
map_inverse_y = map_inverse_y.reshape(grid_height, grid_width).astype(np.float32)

# Clip the mapping arrays to valid ranges to avoid invalid indices
map_inverse_x = np.clip(map_inverse_x, 0, grid_width - 1)
map_inverse_y = np.clip(map_inverse_y, 0, grid_height - 1)

# %%
# Apply the inverse mapping to the distorted image
restored_image = cv2.remap(
    distorted_image,
    map_inverse_x,
    map_inverse_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT
)

# %%

#display original image with control points
display_image_with_points(image, 
                        points_original=(X.ravel(), Y.ravel()),
                        title='Original Image with Control Points')

# Display distorted image with distorted control points
display_image_with_points(distorted_image,
                        points_distorted=(X_distorted.ravel(), Y_distorted.ravel()),
                        title='Distorted Image with Control Points')

# Display restored image with both sets of points
display_image_with_points(restored_image, 
                        points_original=(X.ravel(), Y.ravel()),
                        points_distorted=(X_distorted.ravel(), Y_distorted.ravel()),
                        title='Restored Image with Both Control Points')

# %%
