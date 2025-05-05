from ultralytics import YOLO



model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-C2f-Star-LADH.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn-LADH.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()
model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn-C2f-Star-LADH.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()


model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn-BC2f-Star-LADH.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-starnet-LADH.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-starnet-bifpn.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-starnet-bifpn-LADH.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()


model = YOLO("ultralytics/cfg/models/11/modified/yolo11-bifpn.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()


model = YOLO("ultralytics/cfg/models/11/modified/yolo11-bifpn-LADH.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()