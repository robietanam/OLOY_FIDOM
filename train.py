from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")  
# Train the model
results = model.info(verbose=True, detailed=False)



model = YOLO("ultralytics/cfg/models/11/modified/yolo11-Starnet.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/11/modified/yolo11-C3k2-Star.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/11/modified/yolo11-LSCD.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/11/modified/yolo11-LADH.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/11/modified/yolo11-Starnet-C3k2-Star-LSCD.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/11/modified/yolo11-C3k2-Star-LSCD.yaml") 
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/11/modified/yolo11-C3k2-Star-LADH.yaml") 
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  
results = model.info(verbose=True, detailed=False)


model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-starnet.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-C2f-star.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-LSCD.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-LADH.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-starnet-C2f-star-LSCD.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-C2f-star-LSCD.yaml")  
results = model.info(verbose=True, detailed=False)

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-C2f-star-LADH.yaml")  
results = model.info(verbose=True, detailed=False)