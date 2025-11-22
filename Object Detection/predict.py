from ultralytics import YOLO

model = YOLO('./YOLOv11_Training/yolov11_kitti_run_3classes_transfer_learning/weights/best.pt') 

results = model.predict(source='./34759_final_project_rect/seq_03/image_02/data/0000000000.png', save=True, conf=0.25)

print(f"Result saved at: {results[0].save_dir}")