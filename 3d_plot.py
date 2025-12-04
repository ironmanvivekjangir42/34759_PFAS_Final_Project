import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.linalg import inv
from typing import Dict, Any, List

# =================================================================
# === CONFIGURATION (UPDATED) ===
# =================================================================

# Define the path to the 3D tracking data CSV
INPUT_CSV_PATH = 'tracking_results_3d.csv'
# Define the path for the NEW Kalman-filtered output CSV
OUTPUT_CSV_PATH = 'tracking_results_kalman-temp.csv'

# Define the fixed limits for the plot
MAX_Z_M = 40.0 
MAX_X_M = 15.0 

# --- Kalman Filter Parameters (Tuned for noisy data) ---
Q_SCALE = 0.5e-3 
R_SCALE = 150.0 # Increased for maximum smoothness

# --- NEW Measurement Filter Parameter ---
# Ignore the Z measurement (depth) if the difference between the 
# Kalman prediction and the raw measurement is greater than this threshold.
MAX_Z_JUMP_M = 6.0 # Meters 

# =================================================================
# === HELPER FUNCTIONS ===
# =================================================================

def get_track_color(track_id: int) -> str:
    """
    Generates a consistent, unique, and diverse color for each track ID.
    """
    cmap = plt.cm.get_cmap('tab20', 20) 
    return cmap(track_id % 20)

# =================================================================
# === KALMAN FILTER IMPLEMENTATION (WITH Z-MEASUREMENT GATE) ===
# =================================================================

def kalman_filter_trajectory(df: pd.DataFrame, max_z_jump: float) -> pd.DataFrame:
    """
    Applies a 4-state (X, Z, Vx, Vz) Constant Velocity Kalman Filter to each track,
    with an added gate to ignore outlier Z-measurements.
    """
    
    final_filtered_data = []

    # State Vector: [x, z, vx, vz] (4x1)
    # Measurement Vector: [x, z] (2x1)
    
    # Measurement Matrix (H): Maps the state to the measurement
    H = np.array([
        [1, 0, 0, 0],  # x measurement
        [0, 1, 0, 0]   # z measurement
    ])

    # Measurement Noise Covariance (R): Trustworthiness of the input measurements
    R = np.eye(2) * R_SCALE
    
    # R_Z_REJECT: A very large R matrix used when Z is an outlier. 
    R_Z_REJECT = np.array([
        [R_SCALE, 0.0],
        [0.0, 1e9] # Extreme noise for Z dimension
    ])
    
    # Identity matrix (4x4)
    I = np.eye(4)

    for track_id, group in df.groupby('track_id'):
        
        raw_x = group['center_x_3d'].values
        raw_z = group['center_z_3d'].values
        raw_t = group['frame_idx'].values 

        if len(raw_x) < 2:
            continue

        # Initialize state and covariance
        x = np.array([raw_x[0], raw_z[0], 0.0, 0.0])
        P = np.eye(4) * 10.0
        
        filtered_x = [raw_x[0]]
        filtered_z = [raw_z[0]]
        
        # Also store velocity estimates for potential later use
        filtered_vx = [0.0] 
        filtered_vz = [0.0] 

        for i in range(1, len(raw_x)):
            dt = raw_t[i] - raw_t[i-1] 
            
            if dt <= 0: continue 

            # State Transition Matrix (F) and Process Noise (Q)
            dt_sq = dt**2
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            Q = np.diag([dt_sq / 2, dt_sq / 2, dt, dt]) * Q_SCALE

            # --- PREDICT STEP ---
            x = F @ x
            P = F @ P @ F.T + Q

            # --- UPDATE STEP ---
            z = np.array([raw_x[i], raw_z[i]]) # Current noisy measurement
            
            predicted_z = x[1]
            
            # --- Z-Measurement Gating Check ---
            use_r = R # Default to normal R
            if np.abs(z[1] - predicted_z) > max_z_jump:
                use_r = R_Z_REJECT 
            
            y = z - H @ x # Residual/Innovation
            S = H @ P @ H.T + use_r # Use conditional R matrix
            K = P @ H.T @ inv(S) # Kalman Gain
            
            x = x + K @ y # Corrected state estimate
            P = (I - K @ H) @ P # Corrected covariance

            # Store the resulting clean position and velocity
            filtered_x.append(x[0])
            filtered_z.append(x[1])
            filtered_vx.append(x[2])
            filtered_vz.append(x[3])
            
        # --- 4. Package results ---
        # The list slicing ensures we handle any skipped frames by matching the length of the filtered data.
        temp_df = pd.DataFrame({
            'frame_idx': raw_t[:len(filtered_x)], 
            'track_id': track_id,
            'kalman_x': filtered_x,
            'kalman_z': filtered_z,
            'kalman_vx': filtered_vx,
            'kalman_vz': filtered_vz
        })
        final_filtered_data.append(temp_df)
        
    return pd.concat(final_filtered_data, ignore_index=True)


# =================================================================
# === MAIN VISUALIZATION SCRIPT (UNCHANGED) ===
# =================================================================

def visualize_full_paths(raw_df: pd.DataFrame, filtered_df: pd.DataFrame):
    """
    Creates a single, static 2D top-down view showing both the raw, noisy
    data and the clean, Kalman-filtered path for comparison.
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"3D Trajectories: Kalman Filter (Z-Gate: {MAX_Z_JUMP_M}m)", fontsize=14)
    ax.set_xlabel("X coordinate (meters, Right (+))", fontsize=12)
    ax.set_ylabel("Z coordinate (meters, Forward (+))", fontsize=12)
    
    ax.set_xlim([-MAX_X_M, MAX_X_M]) 
    ax.set_ylim([0, MAX_Z_M])
    ax.set_aspect('equal', adjustable='box') 
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add a point for the vehicle/camera origin
    ax.plot(0, 0, 'k^', markersize=12, label='Camera Origin', zorder=5)

    # --- Plotting Loop ---
    legend_handles = []
    legend_labels = []
    
    unique_track_ids = raw_df['track_id'].unique()
    
    for track_id in unique_track_ids:
        
        raw_group = raw_df[raw_df['track_id'] == track_id]
        filtered_group = filtered_df[filtered_df['track_id'] == track_id]
        
        if raw_group.empty or filtered_group.empty: continue
            
        color = get_track_color(track_id)
        
        # --- 1. Plot the RAW points (faint, noisy line) ---
        raw_handle, = ax.plot(
            raw_group['center_x_3d'].values, 
            raw_group['center_z_3d'].values, 
            color=color, 
            linestyle='--', 
            linewidth=1.0, 
            alpha=0.3, 
            zorder=1
        )
        
        # --- 2. Plot the FILTERED Path (Bold, smooth curve) ---
        final_x = filtered_group['kalman_x'].values
        final_z = filtered_group['kalman_z'].values
        
        filtered_handle, = ax.plot(
            final_x, 
            final_z, 
            color=color, 
            linestyle='-', 
            linewidth=3.5, 
            alpha=1.0, 
            zorder=3
        )
        
        legend_handles.append(filtered_handle)
        legend_labels.append(f'ID {track_id} (Kalman)')

        # --- 3. Plot the final ending point ---
        ax.plot(
            final_x[-1], 
            final_z[-1], 
            marker='o', 
            markersize=8, 
            color=color, 
            markeredgecolor='k', 
            zorder=4
        )
        
    # --- 4. Finalize Plot ---
    origin_handle = ax.lines[0]
    legend_handles.append(origin_handle)
    legend_labels.append('Camera Origin')
    
    # Add a generic handle for the raw data to the legend
    legend_handles.append(plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.0))
    legend_labels.append('Raw Input (Noisy)')

    ax.legend(legend_handles, legend_labels, loc='upper right', title="Trajectories")
    
    plt.show()

# =================================================================
# === EXECUTION (MODIFIED TO SAVE CSV) ===
# =================================================================

if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"FATAL ERROR: Tracking data CSV not found at: {INPUT_CSV_PATH}")
        exit()

    try:
        raw_df = pd.read_csv(INPUT_CSV_PATH)
        # Filter for valid depth readings
        df_for_processing = raw_df[(raw_df['center_z_3d'] > 0.1) & (raw_df['center_z_3d'] < MAX_Z_M)] 
        
    except Exception as e:
        print(f"Error reading or processing CSV: {e}")
        exit()

    if df_for_processing.empty:
        print("Processed data is empty or contains no valid tracks.")
        exit()

    print(f"Loaded {df_for_processing['track_id'].nunique()} unique tracks.")
    print(f"Applying Kalman Filter (R={R_SCALE}, Q={Q_SCALE}, Z-Gate: {MAX_Z_JUMP_M}m)...")

    # --- Step 1: Filter the data using Kalman ---
    filtered_df = kalman_filter_trajectory(df_for_processing, MAX_Z_JUMP_M)
    
    # --- Step 2: Save the filtered data to a new CSV ---
    # Merge the filtered data (position, velocity) back with the original raw data
    # (keeping only unique columns from the filtered data)
    merged_df = pd.merge(
        raw_df, 
        filtered_df[['frame_idx', 'track_id', 'kalman_x', 'kalman_z', 'kalman_vx', 'kalman_vz']], 
        on=['frame_idx', 'track_id'], 
        how='left'
    )
    
    merged_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ Filtered data saved successfully to '{OUTPUT_CSV_PATH}'")
    
    # --- Step 3: Visualize the paths ---
    visualize_full_paths(df_for_processing, filtered_df)