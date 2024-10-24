import cv2
import numpy as np

# Calibration parameters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
square_size_mm = 12  # Real-world size of each square in millimeters

# Prepare object points based on real-world square size
grid_size = (8, 5)  # Adjust grid size to match your pattern
objp = np.zeros((grid_size[0]*grid_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
objp *= square_size_mm  # Scale object points to actual size in mm

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Open the camera
cap = cv2.VideoCapture(cv2.CAP_ANY)  # Use any available camera

if not cap.isOpened():
    print("Cannot open the camera")
    exit()

# Capture images for calibration
captured_count = 0
total_required = 10  # Number of calibration images to capture

while captured_count < total_required:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, grid_size, corners2, ret)
        captured_count += 1
        print(f"Captured {captured_count}/{total_required} images")

    # Show the captured frame
    cv2.imshow('Calibration Image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Perform camera calibration after collecting enough images
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# Save calibration for future use
np.savez('calibration_data_with_rotation.npz', camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
print("Camera calibration complete and saved.")

# Reopen the camera to show the corrected live feed
cap = cv2.VideoCapture(cv2.CAP_ANY)

# Use the first rotation vector and translation vector
rvec = rvecs[0]
tvec = tvecs[0]

# Compute the rotation matrix from the rotation vector
R, _ = cv2.Rodrigues(rvec)

# Function to convert rotation matrix to Euler angles
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])  # Roll
        y = np.arctan2(-R[2, 0], sy)      # Pitch
        z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])  # Roll
        y = np.arctan2(-R[2, 0], sy)       # Pitch
        z = 0                              # Yaw

    return np.array([x, y, z])

# Function to convert Euler angles back to rotation matrix
def eulerAnglesToRotationMatrix(theta):
    # Rotation matrices around the x, y, and z axes
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]])

    # Combine rotations in order: R = R_z * R_y * R_x
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

# Convert rotation matrix to Euler angles
euler_angles = rotationMatrixToEulerAngles(R)

# Set rotation about z-axis (yaw) to zero
euler_angles[2] = 0  # Zero out yaw

# Reconstruct rotation matrix without z-axis rotation
R_no_z = eulerAnglesToRotationMatrix(euler_angles)

# Compute the inverse of the modified rotation matrix
R_no_z_inv = R_no_z.T  # Transpose is the inverse for rotation matrices

# Compute the new optimal camera matrix
h, w = gray.shape[:2]
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)

# Initialize the undistort rectify map using the inverse of the modified rotation matrix
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, R_no_z_inv, new_camera_mtx, (w, h), cv2.CV_16SC2
)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Apply the correction (lens distortion + rotation without z-axis)
    corrected_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    # Show the original and corrected frames side by side
    combined = np.hstack((frame, corrected_frame))
    cv2.imshow('Original (Left) vs Corrected without Z Rotation (Right)', combined)

    # Press 'q' to exit the live feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
