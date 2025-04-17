from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-starnet-C2f-Star-LADH.yaml")  

# Train the model
results = model.info(verbose=True, detailed=False)
