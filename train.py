from ultralytics import YOLO


model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-LADH.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-LSCD.yaml")  
results = model.info(verbose=True, detailed=False)
results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-C2f-Star-LADH.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn-LADH.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn-C2f-Star-LADH.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn-BC2f-Star-LADH.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/ablation/yolov8-AFPN-P345.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/ablation/yolov8-FDPN.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/ablation/yolov8-GDFPN.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/ablation/yolov8-GFPN.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/ablation/yolov8-HSFPN.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/ablation/yolov8-HSPAN.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/ablation/yolov8-HSPAN-DySample.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/ablation/yolov8-HSFPN-LADH.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-bifpn-LADH.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()

# model = YOLO("ultralytics/cfg/models/v9/yolov9t.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()


# model = YOLO("ultralytics/cfg/models/v9/modified/yolov9t-bifpn.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()


# model = YOLO("ultralytics/cfg/models/v9/modified/yolov9t-bifpn-LADH.yaml")  
# results = model.info(verbose=True, detailed=False)
# results = model.fuse()