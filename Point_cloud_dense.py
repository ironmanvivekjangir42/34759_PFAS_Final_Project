import cv2
import numpy as np
import open3d as o3d
import os

# --- Calibration Parameters (Extracted from your Kitti calibration data) ---
# NOTE: These values are derived from the rectified projection matrices (P_rect)
# and are used for accurate, real-world metric 3D reconstruction.

# 1. Intrinsic Parameters
FOCAL_LENGTH = 707.0493  # f (Focal length)
CX_OFFSET = 604.0814     # cx (Principal point X)
CY_OFFSET = 180.5066     # cy (Principal point Y)

# 2. Extrinsic Parameter
BASELINE = 0.4725        # Baseline (distance between cameras in meters)

# Z_COMPRESSION_FACTOR is implicitly 1.0 here (no custom scaling)

def compute_disparity(imgL, imgR):
    """
    Computes a dense, filtered disparity map using StereoSGBM and WLS filtering.
    
    Args:
        imgL (np.array): The left color image (BGR).
        imgR (np.array): The right color image (BGR).
        
    Returns:
        np.array: The filtered disparity map (float32).
    """
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    
    # Aggressive Tuning Parameters (using the previously optimized values)
    window_size = 7
    min_disp = 0
    num_disp = 16 * 8
    
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,
        P1=8 * 3 * window_size**2, P2=32 * 3 * window_size**2, disp12MaxDiff=1, 
        uniquenessRatio=5, speckleWindowSize=100, speckleRange=1, preFilterCap=63, 
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    
    wls_filter.setLambda(8000.0) 
    wls_filter.setSigmaColor(1.5)

    dispL = left_matcher.compute(grayL, grayR)
    dispR = right_matcher.compute(grayR, grayL)
    
    filtered_disp = wls_filter.filter(dispL, imgL, disparity_map_right=dispR)
    disp_float = filtered_disp.astype(np.float32) / 16.0
    
    return disp_float

def generate_dense_point_cloud(imgL, disparity):
    """
    Generates the FULL dense Open3D point cloud by projecting all valid disparity pixels.
    
    Args:
        imgL (np.array): The left color image (BGR).
        disparity (np.array): The filtered disparity map (float32).
        
    Returns:
        o3d.geometry.PointCloud: The 3D point cloud.
    """
    h, w = imgL.shape[:2]
    
    # Use Constant Calibration Values
    f = FOCAL_LENGTH
    cx = CX_OFFSET
    cy = CY_OFFSET
    
    # 1. Mask valid disparity points
    mask = disparity > 10.0
    rows, cols = np.indices(disparity.shape)
    r_good, c_good, d_good = rows[mask], cols[mask], disparity[mask]
    
    # 2. Extract colors for valid points
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)[mask] / 255.0
    
    # 3. Compute 3D coordinates (X, Y, Z)
    Z = (f * BASELINE) / d_good
    X = (c_good - cx) * Z / f
    Y = (r_good - cy) * Z / f
    
    points = np.dstack((X, Y, Z)).reshape(-1, 3)
    
    # 4. Create Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 5. Filter outliers and rotate (for visualization alignment)
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    
    R = pcd_clean.get_rotation_matrix_from_xyz((np.pi, 0, 0)) 
    pcd_clean.rotate(R, center=(0,0,0))

    return pcd_clean

# ------------------------------------------------------------------
# NEW FUNCTION: GENERATE SAMPLED POINT CLOUD
# ------------------------------------------------------------------
def generate_sampled_point_cloud(imgL, disparity_map, stride=5, max_depth=80.0):
    """
    Generates a sparse (but uniform) point cloud by sampling the disparity map
    at regular intervals defined by 'stride'.
    
    Args:
        imgL (np.array): The left color image (BGR).
        disparity_map (np.array): The filtered disparity map (float32).
        stride (int): The sampling step (e.g., stride=5 samples every 5th pixel).
        max_depth (float): Maximum depth Z to consider for projection (in meters).
        
    Returns:
        o3d.geometry.PointCloud: The 3D sampled point cloud.
    """
    h, w = imgL.shape[:2]
    
    # Use Constant Calibration Values
    f = FOCAL_LENGTH
    cx = CX_OFFSET
    cy = CY_OFFSET
    
    final_points = []
    final_colors = []
    
    # Iterate over the image with the specified stride
    for v in range(0, h, stride):
        for u in range(0, w, stride):
            
            # Sample Disparity at the current grid point
            d = disparity_map[v, u]
            
            # Check for valid disparity
            if d > 1.0:
                # Z-depth calculation (NO Z_COMPRESSION_FACTOR)
                Z = (f * BASELINE) / d
                
                # Filter by max depth
                if Z > 0.0 and Z < max_depth:
                    X = (u - cx) * Z / f
                    Y = (v - cy) * Z / f
                    
                    final_points.append([X, Y, Z])
                    
                    # Get color (BGR -> RGB)
                    color_bgr = imgL[v, u] / 255.0
                    final_colors.append(color_bgr[[2, 1, 0]])
    
    final_points = np.array(final_points)
    final_colors = np.array(final_colors)

    print(f"    -> Final sampled 3D points generated (Stride={stride}): {len(final_points)}")

    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(final_points)
    sampled_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    # Rotation for viewing
    R = sampled_pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0)) 
    sampled_pcd.rotate(R, center=(0,0,0))
    
    return sampled_pcd


def save_point_cloud(pcd, filename):
    """Saves a single point cloud to a file."""
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")

if __name__ == '__main__':
    # --- Example Usage for BOTH Dense and Sampled Point Clouds ---
    print("Running example for DENSE and SAMPLED point cloud generation with Kitti calibration...")
    
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
        
        # 2. Generate DENSE Point Cloud (all valid pixels)
        dense_pcd = generate_dense_point_cloud(imgL, disparity_map)
        print(f"Generated DENSE point cloud with {len(dense_pcd.points)} points.")
        
        # 3. Generate SAMPLED Point Cloud (using stride=7)
        sampled_pcd = generate_sampled_point_cloud(imgL, disparity_map, stride=7)
        
        # 4. Visualize
        print("\nDisplaying DENSE Point Cloud (close window to view sampled)...")
        o3d.visualization.draw_geometries([dense_pcd])

        print("\nDisplaying SAMPLED Point Cloud...")
        o3d.visualization.draw_geometries([sampled_pcd])