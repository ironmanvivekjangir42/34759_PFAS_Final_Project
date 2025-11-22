# visualize.py

import os
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Configuration ---
# Path to the YOLO config file created by your converter script
YOLO_YAML_PATH = 'kitti_yolo.yaml'

def normalize_yolo_to_abs(x_center_norm, y_center_norm, w_norm, h_norm, img_width, img_height):
    """
    Converts normalized YOLO coordinates (x_center, y_center, w, h) to
    absolute pixel coordinates (x_min, y_min, x_max, y_max).
    """
    # Denormalize
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    w = w_norm * img_width
    h = h_norm * img_height

    # Calculate min/max
    x_min = int(x_center - w / 2)
    y_min = int(y_center - h / 2)
    x_max = int(x_center + w / 2)
    y_max = int(y_center + h / 2)
    
    return x_min, y_min, x_max, y_max

def visualize_random_image(data_root, image_dir_rel, label_dir_rel, class_names, dataset_type):
    """
    Selects a random image, reads its labels, plots the boxes, and saves the figure.
    """
    image_dir = os.path.join(data_root, image_dir_rel)
    label_dir = os.path.join(data_root, label_dir_rel)

    # 1. Get list of available labels
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    if not label_files:
        print(f"Skipping {dataset_type}: No label files found in {label_dir}")
        return

    # 2. Select a random label file
    random_label_file = random.choice(label_files)
    base_name = random_label_file.split('.')[0]
    
    # YOLO label files are typically 6-digit padded, but the image might be 10-digit padded
    # Let's assume the corresponding image will have the same base name (but with .png)
    # Since your converter script makes the YOLO label file names 6-digit padded, 
    # we'll look for the corresponding 6-digit image file for now, 
    # as the script copies the images to the YOLO structure.

    # 3. Try to find the image file (both 6-digit and 10-digit names, just in case)
    image_file_6d = f'{base_name}.png'
    image_file_10d = f'{int(base_name):010d}.png'
    
    image_path = os.path.join(image_dir, image_file_6d)
    
    if not os.path.exists(image_path):
        # Check for 10-digit padded name if the 6-digit is not found (useful for test set)
        image_path = os.path.join(image_dir, image_file_10d)
        if not os.path.exists(image_path):
            print(f"Skipping {dataset_type}: Image file not found for {base_name}.")
            return

    label_path = os.path.join(label_dir, random_label_file)
    
    # 4. Load Image
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # 5. Load Labels
    boxes = [] # List of [class_id, xmin, ymin, xmax, ymax]
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    yolo_coords = [float(x) for x in parts[1:]]
                    
                    # Convert normalized YOLO to absolute pixel coordinates
                    x_min, y_min, x_max, y_max = normalize_yolo_to_abs(
                        yolo_coords[0], yolo_coords[1], yolo_coords[2], yolo_coords[3],
                        img_width, img_height
                    )
                    boxes.append([class_id, x_min, y_min, x_max, y_max])
    except Exception as e:
        print(f"Error reading labels from {label_path}: {e}")
        return
        
    # 6. Plotting
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    title = f"Dataset: {dataset_type.upper()} | File: {os.path.basename(image_path)}"
    
    # Draw Bounding Boxes
    for class_id, xmin, ymin, xmax, ymax in boxes:
        # Create a Rectangle patch
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)
        
        # Add class label
        class_name = class_names[class_id]
        ax.text(xmin, ymin - 10, class_name, color='red', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    ax.set_title(title, fontsize=12)
    ax.axis('off') # Hide axes ticks and labels
    
    # 7. Save the figure
    output_filename = f'test_{dataset_type}.png'
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to {output_filename}")


if __name__ == '__main__':
    # Load YAML Configuration
    if not os.path.exists(YOLO_YAML_PATH):
        print(f"Error: YAML file not found at {YOLO_YAML_PATH}. Run kitti_to_yolo_converter.py first.")
        exit()
        
    with open(YOLO_YAML_PATH, 'r') as f:
        config = yaml.safe_load(f)

    DATA_ROOT = config['path']
    CLASS_NAMES = config['names']
    
    # Define the datasets to visualize
    datasets = [
        ('train', config['train'], 'labels/train'),
        ('val', config['val'], 'labels/val'),
        ('test', config['test'], 'labels/test')
    ]

    for dataset_type, image_dir_rel, label_dir_rel in datasets:
        visualize_random_image(DATA_ROOT, image_dir_rel, label_dir_rel, CLASS_NAMES, dataset_type)
        
    print("\nVisualization process complete.")