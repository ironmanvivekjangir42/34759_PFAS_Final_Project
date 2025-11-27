import cv2
import numpy as np
import open3d as o3d
import os

# --- Calibration Parameters (Extracted from the class init) ---
# NOTE: These are the original constants derived from the dynamic scaling approach.
W_REF = 1242.0
F_REF = 718.0
BASELINE = 0.54
Z_COMPRESSION_FACTOR = 0.70

def compute_disparity(imgL, imgR):
    """
    Computes a filtered disparity map using StereoSGBM and WLS filtering.
    
    Args:
        imgL (np.array): The left color image (BGR).
        imgR (np.array): The right color image (BGR).
        
    Returns:
        np.array: The filtered disparity map (float32).
    """
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    window_size = 7
    min_disp = 0
    num_disp = 16 * 8
    
    # SGBM Parameters (tuned for smoothing/density)
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp, 
        numDisparities=num_disp, 
        blockSize=window_size,
        P1=8 * 3 * window_size**2, 
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1, 
        uniquenessRatio=5, 
        speckleWindowSize=100, 
        speckleRange=1, 
        preFilterCap=63, 
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    
    # WLS Parameters (tuned for less edge preservation, more smoothing/filling)
    wls_filter.setLambda(4000.0)
    wls_filter.setSigmaColor(1.5)

    dispL = left_matcher.compute(grayL, grayR)
    dispR = right_matcher.compute(grayR, grayL)
    
    filtered_disp = wls_filter.filter(dispL, imgL, disparity_map_right=dispR)
    disp_float = filtered_disp.astype(np.float32) / 16.0
    
    return disp_float

def extract_and_match_features_robust(imgL, imgR):
    """
    Detects SIFT/ORB features, performs ratio test, and applies epipolar constraint.
    
    Args:
        imgL (np.array): The left color image (BGR).
        imgR (np.array): The right color image (BGR).
        
    Returns:
        tuple: (keypoints_left, keypoints_right, final_matches)
    """
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    # Feature Detection (Max 50,000 features)
    try:
        detector = cv2.SIFT_create(nfeatures=50000) 
    except AttributeError:
        detector = cv2.ORB_create(nfeatures=50000)
    
    kp1, des1 = detector.detectAndCompute(grayL, None)
    kp2, des2 = detector.detectAndCompute(grayR, None)
    
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)

    # 1. Ratio Test
    ratio_matches = []
    for m, n in raw_matches:
        if m.distance < 0.9 * n.distance:
            ratio_matches.append(m)
    
    # 2. Epipolar Constraint Check (for rectified images)
    final_matches = []
    EPILINE_TOLERANCE = 1.0 
    
    for m in ratio_matches:
        pt1_y = kp1[m.queryIdx].pt[1]
        pt2_y = kp2[m.trainIdx].pt[1]
        
        # Check if the y-coordinates are nearly equal (within 1.0 pixel)
        if abs(pt1_y - pt2_y) < EPILINE_TOLERANCE:
            final_matches.append(m)
    
    return kp1, kp2, final_matches

def generate_sparse_point_cloud_disparity(imgL, kp1, matches, disparity_map):
    """
    Generates a sparse point cloud by projecting only the matched features, 
    using a median disparity sample for robustness.
    
    Args:
        imgL (np.array): The left color image (BGR).
        kp1 (list): Keypoints from the left image.
        matches (list): Final filtered feature matches.
        disparity_map (np.array): The filtered disparity map (float32).
        
    Returns:
        o3d.geometry.PointCloud: The 3D point cloud.
    """
    h, w = imgL.shape[:2]
    
    # Calibration parameters derived from image size
    cx = w / 2.0
    cy = h / 2.0
    f = F_REF * (w / W_REF)
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    
    final_points = []
    final_colors = []
    
    WINDOW_SIZE = 3 
    W_HALF = WINDOW_SIZE // 2
    
    for p in pts1:
        u, v = int(p[0]), int(p[1])
        
        # Robust Disparity Sampling (Median in 3x3 Window)
        v_min = max(0, v - W_HALF)
        v_max = min(disparity_map.shape[0], v + W_HALF + 1)
        u_min = max(0, u - W_HALF)
        u_max = min(disparity_map.shape[1], u + W_HALF + 1)
        
        roi = disparity_map[v_min:v_max, u_min:u_max]
        
        valid_disparities = roi.flatten()
        valid_disparities = valid_disparities[valid_disparities > 1.0] 
        
        if len(valid_disparities) > 0:
            d = np.median(valid_disparities)
        else:
            continue
        
        # Projection
        if d > 1.0:
            Z = (f * BASELINE * Z_COMPRESSION_FACTOR) / d
            
            if Z > 0.0 and Z < 80.0:
                X = (u - cx) * Z / f
                Y = (v - cy) * Z / f
                
                final_points.append([X, Y, Z])
                
                # Get color (BGR -> RGB)
                color_bgr = imgL[v, u] / 255.0
                final_colors.append(color_bgr[[2, 1, 0]])
    
    final_points = np.array(final_points)
    final_colors = np.array(final_colors)

    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(final_points)
    sparse_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    # Rotation for viewing
    R = sparse_pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0)) 
    sparse_pcd.rotate(R, center=(0,0,0))
    
    return sparse_pcd

def save_point_cloud(pcd, filename):
    """Saves a single point cloud to a file."""
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")

if __name__ == '__main__':
    # --- Example Usage (How you would run a single frame for testing) ---
    import time
    print("Running example for feature-based sparse point cloud generation...")
    
    BASE_DIR = "../Object Detection/34759_final_project_rect/seq_01/"
    FRAME_NUMBER = "000000"

    left_path = os.path.join(BASE_DIR, "image_02/data", f"{FRAME_NUMBER}.png")
    right_path = os.path.join(BASE_DIR, "image_03/data", f"{FRAME_NUMBER}.png")
    
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)
    
    if imgL is None or imgR is None:
        print(f"Error: Could not load images for frame {FRAME_NUMBER}. Check paths.")
    else:
        # 1. Compute Disparity
        disparity_map = compute_disparity(imgL, imgR)
        
        # 2. Extract and Match Features
        kp1, kp2, matches = extract_and_match_features_robust(imgL, imgR)
        print(f"Final matches found: {len(matches)}")
        
        # 3. Generate Sparse Point Cloud
        sparse_pcd = generate_sparse_point_cloud_disparity(imgL, kp1, matches, disparity_map)
        
        print(f"Generated sparse point cloud with {len(sparse_pcd.points)} points.")
        
        # 4. Visualize
        o3d.visualization.draw_geometries([sparse_pcd])