import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# Load all image paths, change accordingly
left_images = sorted(glob.glob('calib/image_02/data/*.png'))
right_images = sorted(glob.glob('calib/image_03/data/*.png'))

# Lists to store all ROIs
left_rois = []
right_rois = []

# Load the first pair of images for ROI selection
first_left_img = cv2.imread(left_images[0])
first_right_img = cv2.imread(right_images[0])

# Perform 13 ROI selections
for i in range(13):
    print(f"Select ROI for image pair {i + 1}")

    # Select ROI from left image
    left_roi = cv2.selectROI("Select ROI for left img", first_left_img, fromCenter=False, showCrosshair=True)
    left_rois.append(left_roi)

    # Select ROI from right image
    right_roi = cv2.selectROI("Select ROI for right img", first_right_img, fromCenter=False, showCrosshair=True)
    right_rois.append(right_roi)

cv2.destroyAllWindows()

# Define different chessboard configurations
chessboard_configs = [
    (7, 11),  # 7 rows x 11 columns
    (7, 5),  # 7 rows x 5 columns
    (5, 7),  # 5 rows x 7 columns
    (15, 5)  # 15 rows x 5 columns
]

# Iterate over all image pairs
for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
    # Load current image pair
    current_left_img = cv2.imread(left_path)
    current_right_img = cv2.imread(right_path)

    # Convert original images to grayscale
    left_gray = cv2.cvtColor(current_left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(current_right_img, cv2.COLOR_BGR2GRAY)

    # Store 3D points and 2D points for this group of images
    object_points = []  # 3D points in real world space
    left_image_points = []  # 2D points in left image plane
    right_image_points = []  # 2D points in right image plane

    # Process each of the 13 selected ROIs
    for i in range(13):
        x, y, w, h = left_rois[i]
        x1, y1, w1, h1 = right_rois[i]

        # Crop ROI regions
        left_cropped_roi = left_gray[int(y):int(y + h), int(x):int(x + w)]
        right_cropped_roi = right_gray[int(y1):int(y1 + h1), int(x1):int(x1 + w1)]

        # Display cropped regions if desired
        display_images = False
        if display_images:
            plt.subplot(1, 2, 1)
            plt.imshow(left_cropped_roi, cmap='gray')
            plt.title(f"Left Image ROI {i + 1}")

            plt.subplot(1, 2, 2)
            plt.imshow(right_cropped_roi, cmap='gray')
            plt.title(f"Right Image ROI {i + 1}")

            plt.show()

        # Try to find chessboard corners with all configurations
        corners_found = False
        for rows_count, cols_count in chessboard_configs:
            # Create real world 3D points
            corner_spacing_mm = 100
            objp = np.zeros((rows_count * cols_count, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols_count, 0:rows_count].T.reshape(-1, 2) * corner_spacing_mm

            # Find chessboard corners
            ret_left, left_corners = cv2.findChessboardCornersSB(left_cropped_roi, (cols_count, rows_count), None)
            ret_right, right_corners = cv2.findChessboardCornersSB(right_cropped_roi, (cols_count, rows_count), None)

            # If corners are found, save them and the 3D points, break out of config loop
            if ret_left and ret_right:
                object_points.append(objp)

                # Convert corner coordinates from cropped to original image coordinates
                left_corners += np.array([x, y], dtype=np.float32)
                right_corners += np.array([x1, y1], dtype=np.float32)

                left_image_points.append(left_corners)
                right_image_points.append(right_corners)

                # Draw the found corners if desired
                display_images = False
                if display_images:
                    left_img_copy = current_left_img.copy()
                    right_img_copy = current_right_img.copy()
                    cv2.drawChessboardCorners(left_img_copy, (cols_count, rows_count), left_corners, ret_left)
                    cv2.drawChessboardCorners(right_img_copy, (cols_count, rows_count), right_corners, ret_right)

                    cv2.imshow(f"Left Image with Corners {i + 1}", left_img_copy)
                    cv2.imshow(f"Right Image with Corners {i + 1}", right_img_copy)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()

                corners_found = True
                break

        # If no chessboard corners found with any size, print message
        if not corners_found:
            print(
                f"Could not find chessboard corners for image pair {idx + 1}, ROI {i + 1} with any of the given sizes.")

# Close all windows at the end
cv2.destroyAllWindows()

# Perform calibration
ret_val, left_cam_matrix, left_dist, left_rot, left_trans = cv2.calibrateCamera(
    object_points, left_image_points, left_gray.shape[::-1], None, None)
ret_val, right_cam_matrix, right_dist, right_rot, right_trans = cv2.calibrateCamera(
    object_points, right_image_points, right_gray.shape[::-1], None, None)

# Stereo calibration
stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
stereo_flags = cv2.CALIB_FIX_INTRINSIC  # Fix intrinsic matrices from single calibration

# Perform stereo calibration
stereo_ret, left_cam_matrix, left_dist_coeffs, right_cam_matrix, right_dist_coeffs, rotation_mat, translation_vec, essential_mat, fundamental_mat = cv2.stereoCalibrate(
    object_points,
    left_image_points,
    right_image_points,
    left_cam_matrix,
    left_dist,
    right_cam_matrix,
    right_dist,
    left_gray.shape[::-1],
    criteria=stereo_criteria,
    flags=stereo_flags
)

# Output calibration results
print("Stereo Calibration completed.")
print(f"camM1:\n{left_cam_matrix}")
print(f"dist1:\n{left_dist_coeffs}")
print(f"camM2:\n{right_cam_matrix}")
print(f"dist2:\n{right_dist_coeffs}")
print(f"Rotation Matrix (R):\n{rotation_mat}")
print(f"Translation Vector (T):\n{translation_vec}")
print(f"Essential Matrix (E):\n{essential_mat}")
print(f"Fundamental Matrix (F):\n{fundamental_mat}")

# Perform stereo rectification using calibration results
rect_R1, rect_R2, rect_P1, rect_P2, rect_Q, rect_roi1, rect_roi2 = cv2.stereoRectify(
    left_cam_matrix, left_dist_coeffs, right_cam_matrix, right_dist_coeffs, left_gray.shape[::-1], rotation_mat,
    translation_vec, alpha=0.2
)

# Get image dimensions
img_height, img_width = first_left_img.shape[:2]

# Iterate over all image pairs for rectification and display
for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    # Undistort and rectify left and right camera images
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(left_cam_matrix, left_dist_coeffs, rect_R1, rect_P1,
                                                         (img_width, img_height), cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(right_cam_matrix, right_dist_coeffs, rect_R2, rect_P2,
                                                           (img_width, img_height), cv2.CV_32FC1)

    rect_left = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

    # Display rectified images
    cv2.imshow(f'Rectified Left Image {idx + 1}', rect_left)
    cv2.imshow(f'Rectified Right Image {idx + 1}', rect_right)

    # Wait for key press, 'q' to continue to next pair, 'ESC' to exit
    key_press = cv2.waitKey(0)
    if key_press == 27:  # 'ESC' to exit
        break
    elif key_press == ord('q'):  # 'q' to continue
        cv2.destroyAllWindows()

# Close all windows at the end
cv2.destroyAllWindows()

# Assumed intrinsic matrices and distortion coefficients (replace with actual calibration results if needed)
left_K = np.array([[981.2178, 0, 690],
                   [0, 975.8994, 247.1364],
                   [0, 0, 1]])
left_D = np.array([-0.3791375, 0.2148119, 0.001227094, 0.002343833, -0.07910379])

right_K = np.array([[986.3925, 0, 702],
                    [0, 982.1423, 258.8854],
                    [0, 0, 1]])
right_D = np.array([-0.3673556, 0.1862563, 0.00008496128, 0.0001699076, -0.05822524])
right_rot = np.array([[9.993552e-01, 1.830187e-02, -3.089048e-02],
                      [-1.855578e-02, 9.997962e-01, -7.952999e-03],
                      [3.073863e-02, 8.521068e-03, 9.994911e-01]])
right_trans = np.array([[-5.370000e-01], [4.509875e-03], [-1.198621e-02]])

# Generate rectification matrices using stereoRectify
rect_R1, rect_R2, rect_P1, rect_P2, rect_Q, rect_roi1, rect_roi2 = cv2.stereoRectify(left_K, left_D, right_K, right_D,
                                                                                     left_gray.shape[::-1], right_rot,
                                                                                     right_trans, alpha=0)
# Generate mapping matrices
left_map_x, left_map_y = cv2.initUndistortRectifyMap(left_K, left_D, rect_R1, rect_P1, left_gray.shape[::-1],
                                                     cv2.CV_32FC1)
right_map_x, right_map_y = cv2.initUndistortRectifyMap(right_K, right_D, rect_R2, rect_P2, left_gray.shape[::-1],
                                                       cv2.CV_32FC1)
for idx, fname in enumerate(left_images):
    current_left = cv2.imread(left_images[idx])
    current_right = cv2.imread(right_images[idx])

    rect_left = cv2.remap(current_left, left_map_x, left_map_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(current_right, right_map_x, right_map_y, cv2.INTER_LINEAR)

    cv2.imshow('Rectified Left Image', rect_left)
    cv2.imshow('Rectified Right Image', rect_right)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

# Get image dimensions
sample_img = cv2.imread(left_images[0])
(img_height, img_width) = sample_img.shape[:2]

# Get optimal new camera matrix for left camera
left_cam_matrix, left_roi = cv2.getOptimalNewCameraMatrix(left_cam_matrix, left_dist, (img_width, img_height), alpha=0)

# Get optimal new camera matrix for right camera
right_cam_matrix, right_roi = cv2.getOptimalNewCameraMatrix(right_cam_matrix, right_dist, (img_width, img_height),
                                                            alpha=0)

# Iterate over all image pairs for undistortion
for idx, (left_path, right_path) in enumerate(zip(left_images, right_images)):
    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    # Undistort left camera image
    undist_left = cv2.undistort(left_img, left_cam_matrix, left_dist, None, left_cam_matrix)

    # Undistort right camera image
    undist_right = cv2.undistort(right_img, right_cam_matrix, right_dist, None, right_cam_matrix)

    x, y, w, h = left_roi
    undist_left = undist_left[y:y + h, x:x + w]

    x1, y1, w1, h1 = right_roi
    undist_right = undist_right[y1:y1 + h1, x1:x1 + w1]

    # Display undistorted images
    cv2.imshow(f'Undistorted Left Image {idx + 1}', undist_left)
    cv2.imshow(f'Undistorted Right Image {idx + 1}', undist_right)

    # Wait for key press, 'q' to continue to next pair, 'ESC' to exit
    key_press = cv2.waitKey(0)
    if key_press == 27:  # 'ESC' to exit
        break
    elif key_press == ord('q'):  # 'q' to continue
        cv2.destroyAllWindows()

# Close all windows at the end
cv2.destroyAllWindows()