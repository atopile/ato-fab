# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_match(image, target_color_bgr, threshold=50):
    """
    Create a grayscale "similarity" map for how well each pixel matches the target color.
    """
    # Convert images to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(np.uint8([[target_color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]

    # Calculate color differences
    h1, s1, v1 = cv2.split(hsv)
    h2, s2, v2 = target_hsv

    # Special handling for white (high value, low saturation)
    if v2 > 225 and s2 < 30:
        v_high = v1 > 225
        s_low = s1 < 30
        similarity = np.where(v_high & s_low, 255, 0)
    # Special handling for black (low value)
    elif v2 < 30:
        similarity = np.where(v1 < 30, 255, 0)
    # Special handling for gray (medium value, low saturation)
    elif s2 < 30:
        v_diff = np.abs(v1 - v2) / 255.0
        s_low = s1 < 30
        similarity = np.where(s_low, (1 - v_diff) * 255, 0)
    else:
        # For regular colors, use hue and saturation
        h_diff = np.minimum(np.abs(h1 - h2), 180 - np.abs(h1 - h2)) / 90.0
        s_diff = np.abs(s1 - s2) / 255.0
        similarity = (2 - h_diff - s_diff) * 127.5

        # Only consider hue for highly saturated pixels
        mask = s1 > 30
        similarity[~mask] = 0

    # Ensure output is in uint8 range
    similarity = np.clip(similarity, 0, 255).astype(np.uint8)

    # Apply threshold
    similarity[similarity < threshold] = 0

    return similarity

# Example usage demonstration
if __name__ == "__main__":
    # Create test image
    img = np.zeros((300, 600, 3), dtype=np.uint8)

    # Draw various colored shapes
    cv2.circle(img, (100, 150), 50, (0, 165, 255), -1)    # Orange
    cv2.circle(img, (250, 150), 50, (255, 255, 255), -1)  # White
    cv2.circle(img, (400, 150), 50, (0, 0, 0), -1)        # Black
    cv2.circle(img, (550, 150), 50, (128, 128, 128), -1)  # Gray

    # Test different colors
    colors = {
        'Orange': (0, 165, 255),
        'White': (255, 255, 255),
        'Black': (0, 0, 0),
        'Gray': (128, 128, 128)
    }

    # Show original image
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Show similarity map
    for color_name, bgr in colors.items():
        similarity = color_match(img, bgr, threshold=30)
        plt.subplot(1, 2, 2)
        plt.imshow(similarity, cmap='gray')
        plt.title(f'Similarity Map for {color_name}')
        plt.axis('off')
    plt.show()
# %%
