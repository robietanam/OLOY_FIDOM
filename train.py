from ultralytics import YOLO


model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn-C2f-Star.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn-C2f-Star-LADH.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn-C2f-Star-LSCD.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()