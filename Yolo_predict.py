from ultralytics import YOLO

MODEL_PATH = '../Object Detection/best.pt'
IMAGE_PATH = '../Object Detection/34759_final_project_rect/seq_03/image_02/data/0000000000.png'

def YOLO_predictio(IMAGE_PATH):
    # Load a model
    model = YOLO(MODEL_PATH)
    results = model.predict(source=IMAGE_PATH, save=False, conf=0.25)
    result = results[0]


model = YOLO(MODEL_PATH) 

# run the model and save the image with bounding boxes inside the runs folder if save is true
results = model.predict(source=IMAGE_PATH, save=False, conf=0.25)

# --- Process Results ---

# The 'results' object is a list, where each element corresponds to one image.
result = results[0]
print(result)
# Get the original image (as a NumPy array) for processing
original_image_array = result.orig_img 

print(f"Original image shape: {original_image_array.shape}")
print("\n--- Bounding Box Details ---")

# Iterate over all detected boxes
for box in result.boxes:
    # xyxy format: [x_min, y_min, x_max, y_max]
    coords = box.xyxy[0].tolist() 
    
    # Confidence and Class
    confidence = float(box.conf[0])
    class_id = int(box.cls[0])
    class_name = model.names[class_id]# fetct the class name from model metadata

    print(f"Found: {class_name}")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  Coordinates (xyxy): {[int(c) for c in coords]}")