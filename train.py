from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")  
# Train the model
results = model.info(verbose=True, detailed=False)



model = YOLO("ultralytics/cfg/models/11/modified/yolo11-C3k2-Star-LSCD.yaml")  
# Train the model
results = model.info(verbose=True, detailed=False)

