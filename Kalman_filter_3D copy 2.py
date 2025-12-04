import cv2 
import numpy as np
import open3d as o3d
import os
import time
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from ultralytics import YOLO

# =================================================================
# === GLOBAL CONFIGURATION AND PATHS ===
# =================================================================

YOLO_MODEL_PATH = '../Object Detection/best.pt'
BASE_DIR_YOLO = '../Object Detection/34759_final_project_rect/seq_03/' 
IMAGE_DIR_YOLO = os.path.join(BASE_DIR_YOLO, "image_02/data") # Left Image
IMAGE_DIR_STEREO_R = os.path.join(BASE_DIR_YOLO, "image_03/data") # Right Image

CONFIDENCE_THRESHOLD = 0.50

try:
    # Check if the model path exists before loading
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"FATAL ERROR: YOLO model file not found at: {YOLO_MODEL_PATH}")
        exit()
        
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"FATAL ERROR: Failed to load YOLO model globally: {e}")
    exit()

# =================================================================
# === HELPER: 3D NUMBER DRAWER (For IDs) ===
# =================================================================
class NumberDrawer3D:
    def __init__(self):
        self.segments = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 6], [5, 6]]
        self.digits_map = {
            '0': [0, 1, 2, 4, 5, 6], '1': [2, 5], '2': [0, 2, 3, 4, 6],
            '3': [0, 2, 3, 5, 6], '4': [1, 2, 3, 5], '5': [0, 1, 3, 5, 6],
            '6': [0, 1, 3, 4, 5, 6], '7': [0, 2, 5], '8': [0, 1, 2, 3, 4, 5, 6],
            '9': [0, 1, 2, 3, 5, 6]
        }
        
    def get_lineset_for_id(self, track_id, center_pos, scale=0.3, color=[1, 1, 0]):
        str_id = str(track_id)
        all_points = []
        all_lines = []
        offset_x = 0.0
        
        for char in str_id:
            if char not in self.digits_map: continue
            w = 0.5 * scale
            h = 0.5 * scale 
            
            p = [np.array([0, h*2, 0]), np.array([w, h*2, 0]), np.array([0, h, 0]), 
                 np.array([w, h, 0]), np.array([0, 0, 0]), np.array([w, 0, 0])]
            
            active_segments = self.digits_map[char]
            lines_to_add = []
            if 0 in active_segments: lines_to_add.append([p[0], p[1]])
            if 1 in active_segments: lines_to_add.append([p[0], p[2]])
            if 2 in active_segments: lines_to_add.append([p[1], p[3]])
            if 3 in active_segments: lines_to_add.append([p[2], p[3]])
            if 4 in active_segments: lines_to_add.append([p[2], p[4]])
            if 5 in active_segments: lines_to_add.append([p[3], p[5]])
            if 6 in active_segments: lines_to_add.append([p[4], p[5]])
            
            start_idx = len(all_points)
            for i, (pt_a, pt_b) in enumerate(lines_to_add):
                pa = pt_a + np.array([offset_x, 0, 0])
                pb = pt_b + np.array([offset_x, 0, 0])
                all_points.append(pa + center_pos)
                all_points.append(pb + center_pos)
                all_lines.append([start_idx + (i*2), start_idx + (i*2) + 1])
                
            offset_x += (w + 0.1 * scale)
            
        if not all_points: return None
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_points)
        line_set.lines = o3d.utility.Vector2iVector(all_lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(all_lines))])
        return line_set

# =================================================================
# === HELPER: CREATE BOUNDING BOX LINESET ===
# =================================================================
def create_bbox_lineset(center, extent, color):
    """
    Creates a wireframe box (LineSet) given a center and 3D extent (width, height, depth).
    """
    w, h, d = extent[0], extent[1], extent[2]
    
    # Calculate 8 corners relative to center
    x_min, x_max = center[0] - w/2, center[0] + w/2
    y_min, y_max = center[1] - h/2, center[1] + h/2
    z_min, z_max = center[2] - d/2, center[2] + d/2
    
    corners = np.array([
        [x_min, y_min, z_min], # 0
        [x_max, y_min, z_min], # 1
        [x_min, y_max, z_min], # 2
        [x_max, y_max, z_min], # 3
        [x_min, y_min, z_max], # 4
        [x_max, y_min, z_max], # 5
        [x_min, y_max, z_max], # 6
        [x_max, y_max, z_max], # 7
    ])
    
    # 12 lines connecting the corners
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3], # Front face
        [4, 5], [4, 6], [5, 7], [6, 7], # Back face
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connecting lines
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

# =================================================================
# === EXTENDED KALMAN FILTER FUNCTIONS (9D CA Model) ===
# =================================================================

# --- EKF: Non-linear state transition function (Predict) ---
def f_func(x, dt):
    """Predict next state (position, velocity, acceleration - 9D CA)."""
    dt2_half = 0.5 * dt**2
    
    # F is 9x9 for the Constant Acceleration (CA) model
    F = np.array([
        [1, 0, 0, dt, 0, 0, dt2_half, 0, 0],
        [0, 1, 0, 0, dt, 0, 0, dt2_half, 0],
        [0, 0, 1, 0, 0, dt, 0, 0, dt2_half],
        
        [0, 0, 0, 1, 0, 0, dt, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, dt, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, dt],
        
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    return F @ x

# --- EKF: State transition Jacobian (F) ---
def F_func(x, dt):
    """Jacobian of the state transition function (F) (9x9)."""
    dt2_half = 0.5 * dt**2
    
    # For a linear model (CA), the Jacobian is the F matrix itself
    return np.array([
        [1, 0, 0, dt, 0, 0, dt2_half, 0, 0],
        [0, 1, 0, 0, dt, 0, 0, dt2_half, 0],
        [0, 0, 1, 0, 0, dt, 0, 0, dt2_half],
        
        [0, 0, 0, 1, 0, 0, dt, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, dt, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, dt],
        
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])

# --- EKF: Non-linear measurement function (Update) ---
def h_func(x):
    """Converts state vector (9D) to measurement vector (3D position)."""
    # Still only measuring position [x, y, z]
    return x[:3]

# --- EKF: Measurement Jacobian (H) ---
def H_func(x):
    """Jacobian of the measurement function (H) (3x9)."""
    # H is 3x9, selecting only position (x, y, z)
    return np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0, 0, 0, 0]
    ])

# =================================================================
# === EXTENDED KALMAN FILTER CLASS (9-DOF) ===
# =================================================================
class Extended3DEKF:
    """
    Extended Kalman Filter for 3D tracking using a 9D state: 
    [x, y, z, vx, vy, vz, ax, ay, az] (Constant Acceleration model).
    """
    def __init__(self, initial_3d_corner, initial_dims, cls_id): # Added cls_id
        # State space dimension (pos, vel, acc for x, y, z)
        dim_x = 9
        # Measurement dimension (only pos: x, y, z)
        dim_z = 3
        
        self.kf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        # Initial State: [x, y, z, vx, vy, vz, ax, ay, az]
        # Initial state is the measured left-bottom corner position
        self.kf.x = np.array([
            initial_3d_corner[0], initial_3d_corner[1], initial_3d_corner[2], 
            0.0, 0.0, 0.0, 
            0.0, 0.0, 0.0  
        ])
        
        # Initial Covariance Matrix P (9x9) - Higher values allow faster initial convergence
        self.kf.P = np.diag([
            0.5, 0.5, 5.0,  # Position Px, Py, Pz (Z is less certain)
            0.1, 0.1, 0.1,  # Velocity Pvx, Pvy, Pvz
            1.0, 1.0, 1.0   # Acceleration Pax, Pay, Paz (High uncertainty)
        ]) * 10.0
        
        # Measurement Noise Covariance Matrix R (3x3)
        self.kf.R = np.diag([5, 10, 150]) * 1.0 # Z measurement is less accurate
        
        self.lost_frames = 0 
        self.last_dims = initial_dims 
        self.dt = 0.0 
        self.cls_id = cls_id # Store the object class ID

    def predict(self, dt):
        """Performs the prediction step of the EKF."""
        self.dt = dt
        
        # Increase process noise as tracking confidence decreases
        q_std = 1.0 + (self.lost_frames * 0.2) 
        
        # Q is 9x9 for 9D CA model (dim=3: pos, vel, acc; block_size=3: x, y, z)
        self.kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=q_std, block_size=3)
        
        # Use robust prediction method calls for filterpy compatibility
        try:
            self.kf.predict_nonlinear(f_func, F_func, (dt,))
        except AttributeError:
            try:
                self.kf.predict_nl(f_func, F_func, (dt,))
            except AttributeError:
                 self.kf.F = F_func(self.kf.x, dt)
                 self.kf.predict()
        
        # Return the predicted position vector [x, y, z]
        return self.kf.x[:3]

    def update(self, z, dims=None):
        """Performs the update step of the EKF."""
        # Use robust update method calls for filterpy compatibility
        try:
            self.kf.update(z, H_func, h_func) 
        except TypeError: 
             try:
                self.kf.update_nl(z, H_func, h_func)
             except AttributeError as e:
                raise AttributeError(f"Could not find a working EKF update method. Error: {e}")

        if dims is not None:
            self.last_dims = dims 
            
        # Return the corrected position vector [x, y, z]
        return self.kf.x[:3]

# =================================================================
# === STEREO POINT CLOUD GENERATOR ===
# =================================================================
class StereoPointCloudGen:
    def __init__(self, frame_number):
        left_path_full = os.path.join(IMAGE_DIR_YOLO, f"{frame_number}.png")
        right_path_full = os.path.join(IMAGE_DIR_STEREO_R, f"{frame_number}.png")
        
        self.imgL = cv2.imread(left_path_full)
        self.imgR = cv2.imread(right_path_full)

        if self.imgL is None or self.imgR is None:
            raise ValueError(f"Images not found for frame {frame_number}! Check paths.")

        self.grayL = cv2.cvtColor(self.imgL, cv2.COLOR_BGR2GRAY)
        self.grayR = cv2.cvtColor(self.imgR, cv2.COLOR_BGR2GRAY)

        self.h, self.w = self.grayL.shape[:2]
        self.cx = self.w / 2.0
        self.cy = self.h / 2.0
        # Focal length needs scaling based on image size relative to KITTI's original size (1242)
        self.f = 718.0 * (self.w / 1242.0) 
        self.baseline = 0.54
        self.Z_COMPRESSION_FACTOR = 0.70 # Empirical adjustment for better depth
        
        self.disparity_map = None 

    def compute_disparity(self):
        if self.disparity_map is not None:
            return self.disparity_map
            
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
        wls_filter.setLambda(16000.0) 
        wls_filter.setSigmaColor(3.0)   

        dispL = left_matcher.compute(self.grayL, self.grayR)
        dispR = right_matcher.compute(self.grayR, self.grayL)
        
        filtered_disp = wls_filter.filter(dispL, self.imgL, disparity_map_right=dispR)
        disp_float = filtered_disp.astype(np.float32) / 16.0
        self.disparity_map = disp_float
        return disp_float

    def get_3d_point_from_uv(self, u, v, disparity_map, center_depth_z=None):
        """
        Converts pixel coordinates (u, v) to 3D point (X, Y, Z).
        If center_depth_z is provided, it uses that for Z instead of calculating Z from (u, v).
        """
        if u < 0 or v < 0 or v >= self.h or u >= self.w:
            return None, 0.0
            
        d = disparity_map[v, u]
        
        if d > 0.1:
            Z_raw = (self.f * self.baseline * self.Z_COMPRESSION_FACTOR) / d
            
            if center_depth_z is not None:
                # Use the Z from the bounding box center, as requested
                Z = center_depth_z
            else:
                # Calculate Z from the given (u,v) point disparity
                Z = Z_raw
                
            if Z > 0.0 and Z < 100.0: 
                X = (u - self.cx) * Z / self.f
                Y = (v - self.cy) * Z / self.f
                # Return tuple: (Coordinate Vector, Raw Depth Z)
                # Note: Y is inverted (-Y) and Z is camera distance, so it's pointing away (-Z)
                return np.array([X, -Y, -Z]), Z_raw 
        return None, 0.0

# =================================================================
# === 3D TRACKER CORE CLASS ===
# =================================================================
class Yolo3DTracker:
    def __init__(self, yolo_model, conf_threshold, stereo_processor_class):
        self.model = yolo_model
        self.conf_threshold = conf_threshold
        self.stereo_class = stereo_processor_class
        self.active_tracks = {} 
        self.next_track_id = 0
        self.prev_time = time.time()
        self.stereo_processor = None
        self.max_association_dist_3d = 5.0 
        self.max_lost_frames = 40
        self.track_history = {}         

    def _get_detections(self, imgL_rgb):
        results = self.model.predict(source=imgL_rgb, save=False, conf=self.conf_threshold, verbose=False)[0]
        detections = []
        
        if self.stereo_processor is None: return []
        disparity_map = self.stereo_processor.compute_disparity()
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # 1. Get the 3D depth (Z) from the BBox center (x_c, y_c)
            x_c = int((x1 + x2) / 2.0)
            y_c = int((y1 + y2) / 2.0)
            
            xyz_center, depth_Z = self.stereo_processor.get_3d_point_from_uv(x_c, y_c, disparity_map)
            
            if xyz_center is not None:
                
                # 2. Get the 3D X and Y from the LEFT BOTTOM corner (x1, y2)
                x_corner = int(x1) # Left pixel coordinate
                y_corner = int(y2) # Bottom pixel coordinate
                
                # Use the corner (x_corner, y_corner) for X and Y, but reuse the Z from the center (depth_Z)
                xyz_measurement, _ = self.stereo_processor.get_3d_point_from_uv(
                    x_corner, y_corner, disparity_map, center_depth_z=depth_Z
                )
                
                # 3. Calculate 3D dimensions (based on center Z)
                w_pixels = x2 - x1
                h_pixels = y2 - y1
                
                w_3d = (w_pixels * depth_Z) / self.stereo_processor.f
                h_3d = (h_pixels * depth_Z) / self.stereo_processor.f
                d_3d = w_3d # Use width for depth (a common approximation)
                
                dimensions = np.array([w_3d, h_3d, d_3d])

                if xyz_measurement is not None:
                    # Store: [x1, y1, x2, y2, conf, cls_id, x_c, y_c, X_o3d, Y_o3d, Z_o3d, W_3d, H_3d, D_3d]
                    # xyz_measurement is the Left-Bottom Corner + Center Z
                    detections.append([x1, y1, x2, y2, conf, cls_id, x_c, y_c] + xyz_measurement.tolist() + dimensions.tolist())

        return detections

    def run_tracking_step(self, frame_number):
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time
        dt = min(dt, 0.5) 
        
        try:
            self.stereo_processor = self.stereo_class(frame_number) 
        except ValueError as e:
            print(f"!!! Error initializing stereo processor: {e}")
            return []
            
        imgL_rgb = self.stereo_processor.imgL
        
        tracks_to_delete = []
        current_track_ids = list(self.active_tracks.keys())
        
        # --- Prediction ---
        for track_id, tracker in self.active_tracks.items():
            tracker.predict(dt) 
            if tracker.lost_frames >= self.max_lost_frames:
                tracks_to_delete.append(track_id)
            
        for track_id in tracks_to_delete:
            del self.active_tracks[track_id]
            current_track_ids.remove(track_id)
            if track_id in self.track_history:
                del self.track_history[track_id] 

        detections = self._get_detections(imgL_rgb)
        
        # Prepare measurements and track states
        measurements_3d = np.array([det[8:11] for det in detections])
        measurements_dims = np.array([det[11:14] for det in detections])
        measurements_cls = np.array([det[5] for det in detections]) # Get class IDs

        predicted_pos_3d = np.array([self.active_tracks[tid].kf.x[:3] for tid in current_track_ids])
        predicted_cls = np.array([self.active_tracks[tid].cls_id for tid in current_track_ids]) # Get predicted class IDs
        
        # --- Initialization of New Tracks (No current tracks) ---
        if not current_track_ids and measurements_3d.size > 0:
            for i, meas_3d in enumerate(measurements_3d):
                new_id = self.next_track_id
                # PASS CLASS ID to EKF constructor
                self.active_tracks[new_id] = Extended3DEKF(meas_3d, measurements_dims[i], measurements_cls[i])
                self.track_history[new_id] = [meas_3d] 
                self.next_track_id += 1
            current_track_ids = list(self.active_tracks.keys())
            
        # --- Handle prediction-only or no measurements case ---
        if predicted_pos_3d.size == 0 or measurements_3d.size == 0:
            output_3d_points = []
            for tid in current_track_ids:
                tracker = self.active_tracks.get(tid)
                if tracker and tracker.lost_frames < self.max_lost_frames:
                    # Update history for visualization (even if predicted)
                    self.track_history.setdefault(tid, []).append(tracker.kf.x[:3].copy()) 
                    output_3d_points.append({
                        "track_id": tid, 
                        "kf_pos_3d": tracker.kf.x[:3], 
                        "raw_meas_3d": None, 
                        "dims": tracker.last_dims
                    })
            return output_3d_points

        # --- Association (Class-Aware) ---
        
        # Cost Matrix initialization: Large value if no match is possible
        cost_matrix = np.full((len(predicted_pos_3d), len(measurements_3d)), self.max_association_dist_3d + 1.0)
        
        # Calculate Cost Matrix (Distance + Class Match)
        for i, pred_3d in enumerate(predicted_pos_3d):
            pred_cls = predicted_cls[i]
            for j, meas_3d in enumerate(measurements_3d):
                meas_cls = measurements_cls[j]
                
                # CRITICAL: Only calculate cost if classes match
                if pred_cls == meas_cls:
                    dist = np.linalg.norm(pred_3d - meas_3d)
                    if dist < self.max_association_dist_3d:
                        cost_matrix[i, j] = dist

        tracked_indices, detection_indices = linear_sum_assignment(cost_matrix)
        assigned_measurement_indices = set()
        
        # Increment lost frames for all active tracks before update
        for tracker in self.active_tracks.values(): tracker.lost_frames += 1

        output_3d_points = []
        
        # --- Update Assigned Tracks ---
        for i, j in zip(tracked_indices, detection_indices):
            track_id = current_track_ids[i]
            
            if cost_matrix[i, j] < self.max_association_dist_3d:
                z_measurement_3d = measurements_3d[j]
                dims_measurement = measurements_dims[j]
                
                self.active_tracks[track_id].update(z_measurement_3d, dims_measurement)
                self.active_tracks[track_id].lost_frames = 0
                assigned_measurement_indices.add(j)
                
                updated_pos = self.active_tracks[track_id].kf.x[:3].copy()
                self.track_history.setdefault(track_id, []).append(updated_pos)
                
                output_3d_points.append({
                    "track_id": track_id, 
                    "kf_pos_3d": updated_pos, 
                    "raw_meas_3d": z_measurement_3d,
                    "dims": dims_measurement
                })

        # --- Handle Unassigned Detections (NEW TRACKS) ---
        for j, meas_3d in enumerate(measurements_3d):
            if j not in assigned_measurement_indices:
                new_dims = measurements_dims[j]
                new_cls = measurements_cls[j] # Get class ID for new track
                
                # PASS CLASS ID to EKF constructor
                new_kf = Extended3DEKF(meas_3d, new_dims, new_cls) 
                new_id = self.next_track_id
                self.active_tracks[new_id] = new_kf
                self.track_history[new_id] = [meas_3d.copy()] 
                self.next_track_id += 1
                output_3d_points.append({
                    "track_id": new_id, 
                    "kf_pos_3d": new_kf.kf.x[:3], 
                    "raw_meas_3d": meas_3d,
                    "dims": new_dims
                })
                
        # --- Handle Unassigned Tracks (PREDICTED TRACKS) ---
        for track_id in current_track_ids:
            tracker = self.active_tracks.get(track_id)
            if tracker and tracker.lost_frames > 0 and tracker.lost_frames < self.max_lost_frames:
                 predicted_pos = tracker.kf.x[:3].copy()
                 self.track_history.setdefault(track_id, []).append(predicted_pos)
                 
                 output_3d_points.append({
                     "track_id": track_id, 
                     "kf_pos_3d": predicted_pos, 
                     "raw_meas_3d": None,
                     "dims": tracker.last_dims 
                 })
                
        return output_3d_points

# =================================================================
# === HELPER: CREATE PATH LINESET ===
# =================================================================
def create_path_lineset(path_points, color):
    """
    Creates a LineSet representing the path taken by the tracked object.
    """
    if len(path_points) < 2:
        return None
        
    points = np.array(path_points)
    num_points = len(points)
    
    # Lines are [0, 1], [1, 2], [2, 3], ...
    lines = np.array([[i, i + 1] for i in range(num_points - 1)])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    return line_set

# =================================================================
# === MAIN EXECUTION BLOCK ===
# =================================================================

if __name__ == "__main__":
    
    DENSITY_STRIDE = 5
    
    if not os.path.exists(IMAGE_DIR_YOLO):
        print(f"FATAL ERROR: Image directory not found at: {IMAGE_DIR_YOLO}")
        exit()
        
    tracker = Yolo3DTracker(yolo_model, CONFIDENCE_THRESHOLD, StereoPointCloudGen)
    all_frames = sorted([f for f in os.listdir(IMAGE_DIR_YOLO) if f.endswith('.png')]) 
    
    if not all_frames:
        print("ERROR: No images found in the sequence directory.")
        exit()

    print(f"Starting 3D Extended Kalman Filter Tracking (Boxes + IDs + Path) on {len(all_frames)} frames...")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"3D EKF Tracking (Bounding Boxes + IDs + Path)", width=1280, height=720)
    
    render_option = vis.get_render_option()
    render_option.point_size = 2.0 
    render_option.background_color = np.asarray([0, 0, 0]) 

    pcd_ref = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_ref)
    
    tracked_bbox_geometries = {} 
    tracked_text_geometries = {}
    tracked_path_geometries = {}    
    
    number_drawer = NumberDrawer3D() 
    first_frame_points_added = False 
    
    TRACK_COLORS = [(1, 0.5, 0), (0, 1, 0.5), (0.5, 0, 1), (1, 0, 0.5), (0.5, 1, 0)]
    def get_track_color(track_id):
        return TRACK_COLORS[track_id % len(TRACK_COLORS)]

    try:
        # --- MAIN LOOP ---
        for frame_file in all_frames: 
            frame_number = frame_file[:-4]
            print(f"\n--- Processing Frame: {frame_number} ---")

            tracked_3d_points = tracker.run_tracking_step(frame_number)

            if tracker.stereo_processor:
                p = tracker.stereo_processor 
                disparity_map = p.compute_disparity()
                
                final_points = []
                final_colors = []
                
                for v in range(0, p.h, DENSITY_STRIDE):
                    for u in range(0, p.w, DENSITY_STRIDE):
                        d = disparity_map[v, u]
                        if d > 0.1: 
                            Z = (p.f * p.baseline * p.Z_COMPRESSION_FACTOR) / d
                            if Z > 0.0 and Z < 100.0:
                                X = (u - p.cx) * Z / p.f
                                Y = (v - p.cy) * Z / p.f
                                final_points.append([X, Y, Z])
                                color_bgr = p.imgL[v, u] / 255.0
                                final_colors.append(color_bgr[[2, 1, 0]])
                                
                final_points = np.array(final_points)
                final_colors = np.array(final_colors)
                print(f"    -> Generated Background Points: {len(final_points)}")

                if final_points.size > 0:
                    rotated_points = final_points.copy()
                    rotated_points[:, 1] = -rotated_points[:, 1]
                    rotated_points[:, 2] = -rotated_points[:, 2]
                else:
                    rotated_points = final_points

                pcd_ref.points = o3d.utility.Vector3dVector(rotated_points)
                pcd_ref.colors = o3d.utility.Vector3dVector(final_colors)
                vis.update_geometry(pcd_ref)
                
                if not first_frame_points_added and len(final_points) > 0:
                    vis.reset_view_point(True)
                    first_frame_points_added = True
            
            # --- Draw Tracked BOXES, IDs, and PATHS ---
            current_active_ids = {data["track_id"] for data in tracked_3d_points if data["kf_pos_3d"] is not None}
            
            # CLEANUP for BBox, Text, and Path
            ids_to_remove = [tid for tid in tracked_bbox_geometries.keys() if tid not in current_active_ids]
            for tid in ids_to_remove:
                vis.remove_geometry(tracked_bbox_geometries[tid], reset_bounding_box=False) 
                del tracked_bbox_geometries[tid]
                
                if tid in tracked_text_geometries:
                    vis.remove_geometry(tracked_text_geometries[tid], reset_bounding_box=False)
                    del tracked_text_geometries[tid]
                    
                if tid in tracked_path_geometries:
                    vis.remove_geometry(tracked_path_geometries[tid], reset_bounding_box=False)
                    del tracked_path_geometries[tid]


            # UPDATE
            for box_data in tracked_3d_points:
                track_id = box_data["track_id"]
                raw_meas_3d = box_data["raw_meas_3d"]
                kf_pos_3d = box_data["kf_pos_3d"]
                dims = box_data["dims"] 
                
                if kf_pos_3d is None:
                    continue 
                
                # The KF tracks the Left Bottom Corner (x, y) and Center Depth (z)
                w, h, d = dims[0], dims[1], dims[2]
                
                # Use the KF-predicted corner or the raw corner for the corner position
                tracked_corner = kf_pos_3d
                if raw_meas_3d is not None:
                    tracked_corner = raw_meas_3d 
                
                # Calculate the 3D center of the bounding box for visualization
                # Center X = Corner_X + (W/2)
                # Center Y = Corner_Y - (H/2) (Y is negative in the camera frame, so subtract H/2)
                # Center Z = Corner_Z (which is already the center depth)
                visual_center = np.array([
                    tracked_corner[0] + w/2, 
                    tracked_corner[1] - h/2, 
                    tracked_corner[2] 
                ])

                color = [0, 0, 1] # Blue (Predicted)
                if raw_meas_3d is not None:
                    color = [1, 0, 0] # Red (Measured)


                # A. Handle Bounding Box (LineSet)
                if track_id in tracked_bbox_geometries:
                    vis.remove_geometry(tracked_bbox_geometries[track_id], reset_bounding_box=False)
                
                bbox_lineset = create_bbox_lineset(visual_center, dims, color)
                tracked_bbox_geometries[track_id] = bbox_lineset
                vis.add_geometry(bbox_lineset, reset_bounding_box=False)

                # B. Handle Text (ID)
                text_pos = visual_center + np.array([0.0, dims[1]/2.0 + 0.3, 0.0]) 
                
                if track_id in tracked_text_geometries:
                    vis.remove_geometry(tracked_text_geometries[track_id], reset_bounding_box=False)
                
                lineset_text = number_drawer.get_lineset_for_id(track_id, text_pos, scale=0.3, color=[1, 1, 0]) 
                if lineset_text:
                    tracked_text_geometries[track_id] = lineset_text
                    vis.add_geometry(lineset_text, reset_bounding_box=False)
                    
                # C. Handle Path Trace (LineSet)
                # Path still tracks the corner point
                if track_id in tracker.track_history and len(tracker.track_history[track_id]) > 1:
                    path_color = get_track_color(track_id)
                    path_lineset = create_path_lineset(tracker.track_history[track_id], path_color)
                    
                    if track_id in tracked_path_geometries:
                        vis.remove_geometry(tracked_path_geometries[track_id], reset_bounding_box=False)
                        
                    if path_lineset:
                         tracked_path_geometries[track_id] = path_lineset
                         vis.add_geometry(path_lineset, reset_bounding_box=False)


            # --- RENDER ---
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01) 
        
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("\n3D Extended Kalman Filter tracking loop finished successfully.")

    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected exception occurred: {e}")
        cv2.destroyAllWindows()
        if vis.get_render_option() is not None:
             vis.destroy_window()