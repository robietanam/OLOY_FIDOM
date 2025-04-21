from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/modified/yolo11-Starnet-C3k2-Star.yaml")  
# Train the model
results = model.info(verbose=True, detailed=False)
