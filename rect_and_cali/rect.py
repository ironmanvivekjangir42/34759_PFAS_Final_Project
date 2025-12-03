import cv2
import numpy as np
import glob
import random
import matplotlib.pyplot as plt


def draw_epilines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    rows, cols = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r_val, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x_start, y_start = map(int, [0, -r_val[2] / r_val[1]])
        x_end, y_end = map(int, [cols, -(r_val[2] + r_val[0] * cols) / r_val[1]])
        img1 = cv2.line(img1, (x_start, y_start), (x_end, y_end), color, 2)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def visualize_epipolar_lines(left_img, right_img):
    '''Draws epipolar lines on the images'''

    # Convert images to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Use SIFT to detect features and compute descriptors
    sift_detector = cv2.SIFT_create()
    left_kp, left_des = sift_detector.detectAndCompute(left_gray, None)
    right_kp, right_des = sift_detector.detectAndCompute(right_gray, None)

    # Perform matching of points
    matcher = cv2.BFMatcher()
    point_matches = matcher.match(left_des, right_des)
    point_matches = sorted(point_matches, key=lambda x: x.distance)
    match_limit = 100  # Limit to 100 best matches
    left_pts = []
    right_pts = []
    for match in point_matches[:match_limit]:
        left_pts.append(left_kp[match.queryIdx].pt)
        right_pts.append(right_kp[match.trainIdx].pt)
    left_pts = np.int32(left_pts)
    right_pts = np.int32(right_pts)

    # Calculate the fundamental matrix
    fund_matrix, inlier_mask = cv2.findFundamentalMat(left_pts, right_pts, method=cv2.FM_RANSAC)

    # Filter to keep only inlier points
    left_pts = left_pts[inlier_mask.ravel() == 1]
    right_pts = right_pts[inlier_mask.ravel() == 1]

    # Compute and draw epipolar lines
    epi_lines1 = cv2.computeCorrespondEpilines(right_pts.reshape(-1, 1, 2), 2, fund_matrix)
    epi_lines1 = epi_lines1.reshape(-1, 3)
    left_with_lines, _ = draw_epilines(left_gray, right_gray, epi_lines1, left_pts, right_pts)
    epi_lines2 = cv2.computeCorrespondEpilines(left_pts.reshape(-1, 1, 2), 1, fund_matrix)
    epi_lines2 = epi_lines2.reshape(-1, 3)
    right_with_lines, _ = draw_epilines(right_gray, left_gray, epi_lines2, right_pts, left_pts)

    # Show images with epipolar lines
    figure, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(cv2.cvtColor(left_with_lines, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Left Image with Epipolar Lines')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(right_with_lines, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Right Image with Epipolar Lines')
    axes[1].axis('off')
    plt.show()


# Define camera parameters
left_K = np.array([[981.2178, 0, 690],
                   [0, 975.8994, 247.1364],
                   [0, 0, 1]])
left_D = np.array([-0.3791375, 0.2148119, 0.001227094, 0.002343833, -0.07910379])

right_K = np.array([[986.3925, 0, 702],
                    [0, 982.1423, 258.8854],
                    [0, 0, 1]])
right_D = np.array([-0.3673556, 0.1862563, 0.00008496128, 0.0001699076, -0.05822524])

right_R = np.array([[9.993552e-01, 1.830187e-02, -3.089048e-02],
                    [-1.855578e-02, 9.997962e-01, -7.952999e-03],
                    [3.073863e-02, 8.521068e-03, 9.994911e-01]])
right_T = np.array([[-5.370000e-01], [4.509875e-03], [-1.198621e-02]])

# Load paths for left and right images, change path accordingly
left_image_paths = sorted(glob.glob('seq_01/image_02/data/*.png'))
right_image_paths = sorted(glob.glob('seq_01/image_03/data/*.png'))

for left_path, right_path in zip(left_image_paths, right_image_paths):
    # Read the images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    # Draw epipolar lines on the original images
    visualize_epipolar_lines(left_img, right_img)

    # Perform rectification on images
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    img_size = left_gray.shape[::-1]

    rect_R1, rect_R2, rect_P1, rect_P2, rect_Q, _, _ = cv2.stereoRectify(left_K, left_D, right_K, right_D, img_size,
                                                                         right_R, right_T, alpha=0)
    left_map_x, left_map_y = cv2.initUndistortRectifyMap(left_K, left_D, rect_R1, rect_P1, img_size, cv2.CV_32FC1)
    right_map_x, right_map_y = cv2.initUndistortRectifyMap(right_K, right_D, rect_R2, rect_P2, img_size, cv2.CV_32FC1)
    rect_left = cv2.remap(left_gray, left_map_x, left_map_y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_gray, right_map_x, right_map_y, cv2.INTER_LINEAR)

    # Draw epipolar lines on the rectified images
    visualize_epipolar_lines(cv2.cvtColor(rect_left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(rect_right, cv2.COLOR_GRAY2BGR))

    # Add horizontal lines to rectified images and display
    img_height, img_width = rect_left.shape
    line_count = 10
    line_step = img_height // (line_count + 1)
    line_color_val = 255
    for idx in range(1, line_count + 1):
        y_pos = idx * line_step
        cv2.line(rect_left, (0, y_pos), (img_width, y_pos), line_color_val, 1)
        cv2.line(rect_right, (0, y_pos), (img_width, y_pos), line_color_val, 1)
    combined_rect = cv2.hconcat([rect_left, rect_right])
    cv2.imshow('Combined Rectified Images with Horizontal Lines', combined_rect)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()