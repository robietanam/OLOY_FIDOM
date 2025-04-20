from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/modified/yolo11-Starnet-C2f-Star-LSCD.yaml")  

# Train the model
results = model.info(verbose=True, detailed=True)
