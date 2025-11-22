# kitti_to_yolo_converter.py
import os
import shutil
from PIL import Image

# --- Configuration ---
# 1. Original KITTI data path: Must point to the root directory where the script is run.
KITTI_BASE_DIR = '.'
KITTI_IMAGES_DIR = os.path.join(KITTI_BASE_DIR, 'data_object_image_2/training/image_2')
KITTI_LABELS_DIR = os.path.join(KITTI_BASE_DIR, 'training/label_2')

# 2. Custom Dataset Path (for Validation/Testing)
CUSTOM_DATA_BASE_DIR = os.path.join(KITTI_BASE_DIR, '34759_final_project_rect')

# 3. Output directory for YOLO format
YOLO_OUTPUT_DIR = 'kitti_yolo_dataset'
YOLO_TRAIN_IMAGES_DIR = os.path.join(YOLO_OUTPUT_DIR, 'images/train')
YOLO_TRAIN_LABELS_DIR = os.path.join(YOLO_OUTPUT_DIR, 'labels/train')
YOLO_VAL_IMAGES_DIR = os.path.join(YOLO_OUTPUT_DIR, 'images/val')
YOLO_VAL_LABELS_DIR = os.path.join(YOLO_OUTPUT_DIR, 'labels/val')
YOLO_TEST_IMAGES_DIR = os.path.join(YOLO_OUTPUT_DIR, 'images/test')
YOLO_TEST_LABELS_DIR = os.path.join(YOLO_OUTPUT_DIR, 'labels/test')
YOLO_YAML_PATH = 'kitti_yolo.yaml'

# 4. Class mapping
# Use the KITTI classes, but ensure the new data's classes ('Car', 'Pedestrian', 'Cyclist') are covered.
# KITTI_CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
KITTI_CLASSES = ['Car', 'Pedestrian', 'Cyclist']
CLASS_MAP = {name: i for i, name in enumerate(KITTI_CLASSES)}

# 5. Filter classes
FILTER_CLASS = 'DontCare'


def kitti_to_yolo_bbox(img_width, img_height, xmin, ymin, xmax, ymax):
    """
    Converts KITTI's absolute (xmin, ymin, xmax, ymax) to YOLO's normalized
    (x_center, y_center, w, h).
    """
    # Calculate box dimensions
    box_w = xmax - xmin
    box_h = ymax - ymin
    
    # Calculate center coordinates
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    
    # Normalize (divide by image dimensions)
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = box_w / img_width
    h_norm = box_h / img_height
    
    # Return as a space-separated string with 6 decimal places
    return f"{x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

# --- UPDATED YAML FUNCTION ---
def create_yolo_yaml():
    """Creates the data.yaml file required by Ultralytics YOLO."""
    yaml_content = f"""
# kitti_yolo.yaml
path: {os.path.abspath(YOLO_OUTPUT_DIR)}
train: images/train
val: images/val   # Updated to point to validation data
test: images/test # Updated to point to testing data

# Number of classes
nc: {len(KITTI_CLASSES)}

# Class names
names: {KITTI_CLASSES}
"""
    with open(YOLO_YAML_PATH, 'w') as f:
        f.write(yaml_content)
    print(f"Created YOLO YAML config at: {YOLO_YAML_PATH}")

# --- NEW CONVERSION FUNCTION FOR CUSTOM DATA (FINAL PADDING AND OFFSET FIX) ---
def convert_custom_sequence_to_yolo(sequence_name, target_images_dir, target_labels_dir, frame_offset=0):
    """
    Converts labels and copies images for a single custom sequence (seq_01, seq_02, or seq_03).
    Includes a frame_offset to ensure unique filenames when combining sequences.
    Assumes ALL custom images are now 10-digit padded.
    """
    
    print(f"\n--- Starting conversion for sequence: {sequence_name} (Offset: {frame_offset}) ---")
    
    seq_dir = os.path.join(CUSTOM_DATA_BASE_DIR, sequence_name)
    
    # The directory containing the .png image files
    source_images_folder = os.path.join(seq_dir, 'image_02', 'data') 
    source_labels_path = os.path.join(seq_dir, 'labels.txt')
    
    if not os.path.exists(source_labels_path):
        # This handles seq_03 where labels.txt is not provided (as per README)
        if sequence_name == 'seq_03':
            print(f"Note: No labels.txt found for {sequence_name}. Only copying images.")
            
            # Copy all images (assuming they are named <image_seq_no>.png)
            if os.path.exists(source_images_folder):
                image_files = [f for f in os.listdir(source_images_folder) if f.endswith('.png')]
                
                # NOTE: We copy the original 10-digit image files to the test folder.
                for img_file in image_files:
                    # Rename the image file to 6-digit padding for the YOLO structure
                    # Assuming the original file name is 0000000XXX.png
                    original_frame_no = int(img_file.split('.')[0])
                    target_img_name = f'{original_frame_no:06d}.png'
                    
                    shutil.copy(os.path.join(source_images_folder, img_file), os.path.join(target_images_dir, target_img_name))
                    
                print(f"Copied {len(image_files)} images to {target_images_dir}")
            else:
                 print(f"Error: Image folder not found at {source_images_folder}. Skipping image copy.")
            return 0 # Return 0 count for no labels
        
        print(f"Error: Label file not found at {source_labels_path}. Skipping conversion.")
        return 0

    # 1. Read all labels from the single labels.txt file
    frame_annotations = {}
    
    print(f"Reading labels from {source_labels_path}...")

    with open(source_labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            
            # Extract necessary parts
            try:
                frame_no = int(parts[0])      # Index 0
                kitti_class = parts[2]        # Index 2
                
                # Bounding box is at indices 6 to 9 (left, top, right, bottom)
                xmin, ymin, xmax, ymax = map(float, parts[6:10])
            except (IndexError, ValueError):
                print(f"Warning: Skipping malformed line in {sequence_name} labels.txt: {line.strip()}")
                continue
            
            # Check for valid class
            if kitti_class not in CLASS_MAP:
                # Only keep classes in the defined map
                continue 
            
            class_id = CLASS_MAP[kitti_class]
            
            # 2. Get image dimensions
            image_file = f'{frame_no:010d}.png' # Images are zero-padded to 10 digits
            image_path = os.path.join(source_images_folder, image_file) # Correct path: source_images_folder is image_02/data
            
            # If we haven't processed this frame yet, open the image to get dimensions
            if frame_no not in frame_annotations:
                try:
                    with Image.open(image_path) as img:
                        img_width, img_height = img.size
                except FileNotFoundError:
                    print(f"Error: Image {image_file} not found at {image_path}. Skipping frame.")
                    continue
                # Initialize list for this frame
                frame_annotations[frame_no] = {'yolo_lines': [], 'dims': (img_width, img_height), 'image_file': image_file}

            # 3. Convert to YOLO format
            img_width, img_height = frame_annotations[frame_no]['dims']
            yolo_coords = kitti_to_yolo_bbox(img_width, img_height, xmin, ymin, xmax, ymax)
            
            # YOLO format: <class_id> <x_center> <y_center> <w> <h>
            frame_annotations[frame_no]['yolo_lines'].append(f"{class_id} {yolo_coords}\n")

    # 4. Save YOLO annotations and copy images
    saved_count = 0
    for frame_no, data in frame_annotations.items():
        yolo_lines = data['yolo_lines']
        image_file = data['image_file'] # This is the 10-digit filename from the source data
        
        # --- APPLY OFFSET AND DETERMINE FINAL NAME ---
        final_frame_no = frame_no + frame_offset
        
        # Create corresponding YOLO label file name (e.g., '000145.txt')
        yolo_label_file = f'{final_frame_no:06d}.txt'
        yolo_label_path = os.path.join(target_labels_dir, yolo_label_file)

        # Save YOLO labels
        with open(yolo_label_path, 'w') as yf:
            yf.writelines(yolo_lines)
        
        # Copy the image file
        source_image_path = os.path.join(source_images_folder, image_file)
        
        # Create the new 6-digit padded target image name
        target_image_name = f'{final_frame_no:06d}.png'
        target_image_path = os.path.join(target_images_dir, target_image_name)

        # Copy and rename the image file
        shutil.copy(source_image_path, target_image_path)
        saved_count += 1

    print(f"Conversion complete for {sequence_name}.")
    print(f"Saved {saved_count} images/labels to {target_images_dir} and {target_labels_dir}")
    
    # Return the number of files saved for offset calculation
    return saved_count


# --- ORIGINAL CONVERSION FUNCTION ---
def convert_kitti_training_data():
    """Converts the entire KITTI dataset to the YOLO training format."""
    
    print("\n--- Starting KITTI Training Data Conversion ---")
    
    # --- Conversion Loop ---
    # Ensure directory exists before listing
    if not os.path.exists(KITTI_LABELS_DIR):
        print(f"Error: KITTI label directory not found at {KITTI_LABELS_DIR}. Skipping training data conversion.")
        return
        
    label_files = [f for f in os.listdir(KITTI_LABELS_DIR) if f.endswith('.txt')]
    
    for label_file in label_files:
        # File names are e.g., '000000.txt'. Image is '000000.png'.
        base_name = label_file.split('.')[0]
        image_file = base_name + '.png' 
        kitti_label_path = os.path.join(KITTI_LABELS_DIR, label_file)
        kitti_image_path = os.path.join(KITTI_IMAGES_DIR, image_file)
        yolo_label_path = os.path.join(YOLO_TRAIN_LABELS_DIR, label_file)
        
        # 1. Get image dimensions
        try:
            with Image.open(kitti_image_path) as img:
                img_width, img_height = img.size
        except FileNotFoundError:
            print(f"Warning: Image file not found for {base_name} at {kitti_image_path}. Skipping.")
            continue
        
        yolo_annotations = []
        
        # 2. Read KITTI annotations
        with open(kitti_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                kitti_class = parts[0]
                
                if kitti_class == FILTER_CLASS or kitti_class not in CLASS_MAP:
                    continue # Skip 'DontCare' or unrecognized classes
                
                class_id = CLASS_MAP[kitti_class]
                
                # Bounding box is at indices 4 to 7: [4: xmin, 5: ymin, 6: xmax, 7: ymax]
                try:
                    xmin, ymin, xmax, ymax = map(float, parts[4:8])
                except ValueError:
                    print(f"Warning: Malformed bbox in {label_file}. Skipping line.")
                    continue
                
                # 3. Convert to YOLO format
                yolo_coords = kitti_to_yolo_bbox(img_width, img_height, xmin, ymin, xmax, ymax)
                
                # YOLO format: <class_id> <x_center> <y_center> <w> <h>
                yolo_annotations.append(f"{class_id} {yolo_coords}\n")

        # 4. Save YOLO annotations if any valid boxes were found
        if yolo_annotations:
            with open(yolo_label_path, 'w') as yf:
                yf.writelines(yolo_annotations)
            
            # 5. Copy the image file to the YOLO structure
            shutil.copy(kitti_image_path, YOLO_TRAIN_IMAGES_DIR)

    print("\n--- KITTI Training Conversion Complete ---")
    print(f"Total processed files: {len(label_files)}")
    print(f"YOLO training images saved to: {YOLO_TRAIN_IMAGES_DIR}")
    print(f"YOLO training labels saved to: {YOLO_TRAIN_LABELS_DIR}")
    
# --- MAIN EXECUTION BLOCK (MODIFIED TO USE OFFSET) ---
def convert_all_data():
    """Sets up directories and runs all conversion functions with sequence offsetting."""
    
    # --- Setup Directories ---
    if os.path.exists(YOLO_OUTPUT_DIR):
        print(f"Warning: Deleting existing directory: {YOLO_OUTPUT_DIR}")
        # Only delete if it's the target folder
        if YOLO_OUTPUT_DIR != '.' and YOLO_OUTPUT_DIR != '/':
            shutil.rmtree(YOLO_OUTPUT_DIR)
    
    # Create all necessary YOLO directories
    os.makedirs(YOLO_TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(YOLO_TRAIN_LABELS_DIR, exist_ok=True)
    os.makedirs(YOLO_VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(YOLO_VAL_LABELS_DIR, exist_ok=True)
    os.makedirs(YOLO_TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(YOLO_TEST_LABELS_DIR, exist_ok=True) 
    
    print(f"Created output directory structure: {YOLO_OUTPUT_DIR}")

    # 1. Convert KITTI Training Data
    convert_kitti_training_data()

    # 2. Convert Custom Sequences for Validation (seq_01 and seq_02)
    # Start offset at 0
    validation_offset = 0
    
    # Process seq_01 (Offset = 0)
    seq01_count = convert_custom_sequence_to_yolo('seq_01', YOLO_VAL_IMAGES_DIR, YOLO_VAL_LABELS_DIR, validation_offset)
    
    # Update offset for the next sequence
    validation_offset += seq01_count
    
    # Process seq_02 (Starts where seq_01 left off)
    convert_custom_sequence_to_yolo('seq_02', YOLO_VAL_IMAGES_DIR, YOLO_VAL_LABELS_DIR, validation_offset) 

    # 3. Convert Custom Sequence for Testing (seq_03)
    # The test set doesn't need offsetting unless you plan to add more test sequences later.
    convert_custom_sequence_to_yolo('seq_03', YOLO_TEST_IMAGES_DIR, YOLO_TEST_LABELS_DIR)

    # 4. Create Final YAML Config
    create_yolo_yaml()
    
    print("\n✅ ALL DATASET CONVERSIONS COMPLETE!")


if __name__ == '__main__':
    convert_all_data()