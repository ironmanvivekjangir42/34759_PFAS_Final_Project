import cv2
import numpy as np
import os
import time
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter 
from typing import Dict, Any, List, Optional, Tuple
from ultralytics import YOLO 

# =================================================================
# === CONFIGURATION AND PATHS ===
# =================================================================

IMAGE_DIR = '../Object Detection/34759_final_project_rect/seq_03/image_02/data' 

# --- YOLO Configuration ---
MODEL_PATH = '../Object Detection/best.pt' 
CONFIDENCE_THRESHOLD = 0.50              

# --- Tracking Parameters ---
MAX_MAHALANOBIS_DIST = 20.0 
MAX_LOST_FRAMES = 40       

# --- Visualization ---
PREDICTION_COLOR = (128, 128, 128) 

# Global model initialization
try:
    yolo_model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"FATAL ERROR: Failed to load YOLO model globally from {MODEL_PATH}: {e}")
    yolo_model = None 


# =================================================================
# === KALMAN TRACK CLASS (ORIGINAL CONSTANT ACCELERATION MODEL) ===
# =================================================================
class KalmanTrack:
    """
    Implements a 2D Constant Acceleration (CA) Kalman Filter.
    State Vector (x): [x, y, vx, vy, ax, ay] (dim_x=6)
    Measurement (z): [x, y] (dim_z=2)
    """
    def __init__(self, initial_box_center: np.ndarray, track_id: int):
        
        self.track_id = track_id
        self.kf = KalmanFilter(dim_x=6, dim_z=2) 
        self.kf.x = np.array([initial_box_center[0], initial_box_center[1], 0.0, 0.0, 0.0, 0.0])
        self.kf.P = np.diag([50.0, 50.0, 100.0, 100.0, 100.0, 100.0])
        self.kf.R = np.diag([15.0, 70.0]) 
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]) 
        self.base_Q = np.diag([1.0, 1.0, 1.0, 1.0, 5.0, 5.0]) * 7.5
        self.kf.Q = self.base_Q 
        
        self.lost_frames = 0
        self.S = None 
        self.class_id: Optional[int] = None 
        self.history: List[Dict[str, Any]] = [{'pos': initial_box_center, 'type': 'corrected'}] 
        self.track_color: Tuple[int, int, int] = tuple(np.random.randint(100, 256, 3).tolist())

    def predict(self, dt: float):
        dt_sq = dt**2
        self.kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt_sq, 0],
            [0, 1, 0, dt, 0, 0.5*dt_sq],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        inflation_factor = 1.0 + (self.lost_frames * 0.2) if self.lost_frames > 0 else 1.0
        self.kf.Q = self.base_Q * inflation_factor
        self.kf.predict() 
        self.history.append({'pos': self.kf.x[:2].copy(), 'type': 'predicted'})
        self.S = self.kf.H @ self.kf.P @ self.kf.H.T + self.kf.R

    def update(self, z: np.ndarray):
        self.kf.update(z)
        if self.history and self.history[-1]['type'] == 'predicted':
            self.history[-1] = {'pos': self.kf.x[:2].copy(), 'type': 'corrected'}
        else:
             self.history.append({'pos': self.kf.x[:2].copy(), 'type': 'corrected'})
        self.lost_frames = 0
        return self.kf.x[:2] 
    
    def mahalanobis_distance(self, z: np.ndarray) -> float:
        try:
            y = z - (self.kf.H @ self.kf.x) 
            dist_sq = y.T @ np.linalg.inv(self.S) @ y
            return np.sqrt(dist_sq)
        except np.linalg.LinAlgError:
            return np.inf

# =================================================================
# === TRACKER CORE CLASS (SORT Principle) ===
# =================================================================

class SimpleTracker:
    def __init__(self, yolo_model, yolo_names):
        self.model = yolo_model
        self.yolo_names = yolo_names 
        self.active_tracks: Dict[int, KalmanTrack] = {} 
        self.next_track_id = 0 
        self.prev_time = time.time()
        self.max_lost_frames = MAX_LOST_FRAMES 
        
    def get_detections(self, imgL: np.ndarray) -> List[Dict[str, Any]]:
        if self.model is None: return [] 
        results = self.model.predict(source=imgL, save=False, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist() 
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append({"box": [x1, y1, x2, y2], "conf": conf, "cls_id": cls_id})
        return detections

    def run_tracking_step(self, imgL: np.ndarray) -> List[Dict[str, Any]]:
        dt = 1.0 / 30.0 

        # --- 1. Predict and Delete Lost Tracks ---
        tracks_to_delete = []
        for track_id, tracker in list(self.active_tracks.items()): 
            tracker.predict(dt) 
            tracker.lost_frames += 1 
            if tracker.lost_frames >= self.max_lost_frames:
                tracks_to_delete.append(track_id)
        for track_id in tracks_to_delete:
            del self.active_tracks[track_id]
            
        # --- 2. Measurement (YOLO Detections) ---
        detections = self.get_detections(imgL)
        if not detections and not self.active_tracks: return []
            
        # --- 3. Association (Mahalanobis Cost with Class Constraint) ---
        N_tracks = len(self.active_tracks)
        N_dets = len(detections)
        track_id_map = list(self.active_tracks.keys())
        cost_matrix = np.full((N_tracks, N_dets), MAX_MAHALANOBIS_DIST + 1.0)
        
        for i, track_id in enumerate(track_id_map):
            track = self.active_tracks[track_id]
            track_cls_id = track.class_id
            if track_cls_id is None: continue

            for j, det in enumerate(detections):
                det_cls_id = det["cls_id"]
                if track_cls_id != det_cls_id: continue 

                center_x = (det["box"][0] + det["box"][2]) / 2.0
                center_y = (det["box"][1] + det["box"][3]) / 2.0
                measurement = np.array([center_x, center_y])
                
                dist_maha = track.mahalanobis_distance(measurement)
                if dist_maha < MAX_MAHALANOBIS_DIST:
                    cost_matrix[i, j] = dist_maha
        
        tracked_indices, detection_indices = linear_sum_assignment(cost_matrix)
        
        # --- 4. Update Matched Tracks ---
        assigned_measurement_indices = set()
        output_tracks = []
        
        for i, j in zip(tracked_indices, detection_indices):
            track_id = track_id_map[i]
            if cost_matrix[i, j] < MAX_MAHALANOBIS_DIST: 
                det = detections[j]
                center_x = (det["box"][0] + det["box"][2]) / 2.0
                center_y = (det["box"][1] + det["box"][3]) / 2.0
                measurement = np.array([center_x, center_y])
                self.active_tracks[track_id].class_id = det["cls_id"]
                self.active_tracks[track_id].update(measurement)
                assigned_measurement_indices.add(j)

                output_tracks.append({
                    "track_id": track_id, "box": det["box"], "status": "detected",
                    "class_id": det["cls_id"], "confidence": det["conf"], 
                    "track_obj": self.active_tracks[track_id]
                })

        # --- 5. Initialize New Tracks ---
        for j, det in enumerate(detections):
            if j not in assigned_measurement_indices:
                center_x = (det["box"][0] + det["box"][2]) / 2.0
                center_y = (det["box"][1] + det["box"][3]) / 2.0
                measurement = np.array([center_x, center_y])
                
                new_kf = KalmanTrack(measurement, self.next_track_id)
                new_kf.class_id = det["cls_id"]
                self.active_tracks[self.next_track_id] = new_kf
                
                output_tracks.append({
                    "track_id": self.next_track_id, "box": det["box"], "status": "detected",
                    "class_id": det["cls_id"], "confidence": det["conf"],
                    "track_obj": new_kf
                })
                self.next_track_id += 1
                
        # --- 6. Return All Active Tracks (Matched and Predicted) ---
        for track_id, track in self.active_tracks.items():
            is_matched = any(d['track_id'] == track_id for d in output_tracks)
            if not is_matched:
                output_tracks.append({
                    "track_id": track_id, "box": [0, 0, 0, 0], "status": "predicted",
                    "class_id": track.class_id, "confidence": 0, "track_obj": track
                })
                
        return output_tracks

# =================================================================
# === VISUALIZATION AND MAIN LOOP ===
# =================================================================

def draw_track_info(image_frame: np.ndarray, track_data: Dict[str, Any], image_dims: Tuple[int, int]):
    """
    Draws track information directly onto the image_frame.
    No offset is used. The image coordinates (x, y) are the canvas coordinates.
    """
    
    track_obj: KalmanTrack = track_data["track_obj"]
    W_img, H_img = image_dims
    unique_track_color = track_obj.track_color
    
    # --- 1. Draw Full Path History (Only points inside image bounds) ---
    all_points = track_obj.history
    
    for i in range(1, len(all_points)):
        pt1_orig = all_points[i-1]['pos']
        pt2_orig = all_points[i]['pos']

        # Check if BOTH points of the segment are within the original image boundary (0 to W/H)
        pt1_in_bounds = (0 <= pt1_orig[0] <= W_img) and (0 <= pt1_orig[1] <= H_img)
        pt2_in_bounds = (0 <= pt2_orig[0] <= W_img) and (0 <= pt2_orig[1] <= H_img)
        
        if pt1_in_bounds and pt2_in_bounds:
            
            # Since we are drawing directly on the image, the point coordinates ARE the canvas coordinates
            pt1 = (int(pt1_orig[0]), int(pt1_orig[1]))
            pt2 = (int(pt2_orig[0]), int(pt2_orig[1]))

            if all_points[i]['type'] == 'corrected':
                segment_color = unique_track_color
                thickness = 2
            else:
                segment_color = PREDICTION_COLOR
                thickness = 1
            
            cv2.line(image_frame, pt1, pt2, segment_color, thickness)
        
    # --- 2. Draw Current Estimated Position and Bounding Box and Label ---
    current_pos_orig = track_obj.kf.x[:2]
    
    # Only draw the final position, box, and label if the estimated position is within bounds
    if (0 <= current_pos_orig[0] <= W_img) and (0 <= current_pos_orig[1] <= H_img):
        
        current_pos = (int(current_pos_orig[0]), int(current_pos_orig[1]))

        if track_data["status"] == "detected":
            class_name = yolo_names.get(track_obj.class_id, "Unknown")
            label = f'ID:{track_obj.track_id} {class_name}'
            
            x1, y1, x2, y2 = map(int, track_data["box"])
            cv2.rectangle(image_frame, (x1, y1), (x2, y2), unique_track_color, 2)
            
        else: # status == "predicted"
            label = f'ID:{track_obj.track_id} (P: {track_obj.lost_frames})'

        cv2.circle(image_frame, current_pos, 5, unique_track_color, -1)
        
        cv2.putText(image_frame, label, (current_pos[0] + 10, current_pos[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, unique_track_color, 2)


if __name__ == "__main__":
    
    if yolo_model is None: exit()
    if not os.path.exists(IMAGE_DIR):
        print(f"FATAL ERROR: Image directory not found at: {IMAGE_DIR}"); exit()
        
    yolo_names = yolo_model.names 
    all_frames = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]) 
    if not all_frames: exit()

    # Get dimensions from the first frame for boundary checks
    first_frame_path = os.path.join(IMAGE_DIR, all_frames[0])
    first_img = cv2.imread(first_frame_path)
    if first_img is None:
        print("Error reading first image."); exit()
        
    W_img, H_img = first_img.shape[1], first_img.shape[0]
    image_dim = (W_img, H_img)

    # --- MAIN LOOP ---
    print("\n--- Running tracker and plotting directly on image... ---")

    tracker = SimpleTracker(yolo_model, yolo_names)
    
    cv2.namedWindow("Tracking Visualization", cv2.WINDOW_AUTOSIZE) 

    # Create VideoWriter
    h, w, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10  # adjust as needed
    out = cv2.VideoWriter('2D tracking.mp4', fourcc, fps, (w, h))
    
    try:
        for frame_idx, frame_file in enumerate(all_frames): 
            frame_path = os.path.join(IMAGE_DIR, frame_file)
            image_original = cv2.imread(frame_path)

            if image_original is None: continue

            # 1. Run the Tracking Step.
            tracked_data = tracker.run_tracking_step(image_original)

            # 2. Draw tracks directly onto a COPY of the original image
            image_with_tracks = image_original.copy()

            for data in tracked_data:
                # Pass the image copy and image dimensions (for bounds checking)
                draw_track_info(image_with_tracks, data, image_dim)

            # 3. Display Frame Index and Info (with Total Count)
            active_count = len(tracker.active_tracks)
            total_count = tracker.next_track_id 
            
            info_text = f'Frame: {frame_idx} | Active: {active_count} | Total: {total_count}'
            cv2.putText(image_with_tracks, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


            # Write into video
            out.write(image_with_tracks)

            # 4. DISPLAY IMAGE
            cv2.imshow("Tracking Visualization", image_with_tracks)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cv2.destroyAllWindows()
        out.release()
        print("\nProgram finished successfully.")

    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected exception occurred: {e}")