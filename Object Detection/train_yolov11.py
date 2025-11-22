# train_yolov11.py
import os
from ultralytics import YOLO

# the configuration yaml file contains images, labels and class names for training
DATA_YAML_PATH = 'kitti_yolo.yaml'

# MODEL_WEIGHTS = 'yolo11s.pt' # load model with pretrained weights, using small model, its is a good balance for accuracy and speed
MODEL_WEIGHTS = 'yolo11s.yaml' #initialize a model with random weights and train from scratch

# hyperparameters for training
HYPERPARAMETERS = {
    'epochs': 100,          # Number of epochs
    'batch': 16,            # Batch size
    'imgsz': 1280,          # Input image size
    'workers': 8,           # Number of Dataloader workers
    'patience': 50,         # Stop training if mAP doesn't improve after 50 epochs
    'name': 'yolov11_kitti_run_3classes_scratch', # Name for the results directory
    'project': 'YOLOv11_Training_scratch' # Project name
}

def train_yolov11():
    """
    Loads a YOLOv11 model and starts training on the dataset defined in DATA_YAML_PATH.
    """
    print(f"--- Starting YOLOv11 Training ---")
    print(f"Loading model: {MODEL_WEIGHTS}")
    
    try:
        #loading the model
        model = YOLO(MODEL_WEIGHTS)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    if not os.path.exists(DATA_YAML_PATH):
        #loading the config file .yaml
        print(f"Error: Data config file not found at '{DATA_YAML_PATH}'")
        return

    print(f"Training with parameters: {HYPERPARAMETERS}")

    #training the model
    results = model.train(
        data=DATA_YAML_PATH,
        **HYPERPARAMETERS
    )

    print("--- Training Complete ---")
    print(f"Best model weights saved in: {model.trainer.best}")
    print(f"results saved in: {model.trainer.save_dir}")

if __name__ == '__main__':
    train_yolov11()

"""
/path/to/Object Detection/
├── runs/
│   └── detect/
│       └── YOLOv11_Training/
│           └── yolov11s_kitti_3class_640/
│               └── weights/
│                   ├── best.pt    <--TRAINED MODEL WITH BEST ACCURACY
│                   └── last.pt
└── train_yolov11.py
"""