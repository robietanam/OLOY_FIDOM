from ultralytics import YOLO

dir = "D:/YOLO/RUNS2/"
tipe = "tflite"

folders = [
    'yolov11',
    'yolov8',
    'yolov8-bifpn',
    'yolov8-C2f-Star',
    'yolov8-LADH',
    'yolov8-Focaler-MPDIou',
    'yolov8-TSTAR-CIou2',
    'yolov8-C2f-Star-LADH-Focaler',
    'yolov8-TSTAR-Focaler-MPDIou',
    'yolov8-bifpn-C2f-Star-LADH3',
    'yolov8-bifpn-LADH',
    'yolov8-bifpn-LADH-Focaler',
    'yolov8-bifpn-LADH-MPDIoU',
    'yolov8-bifpn-LADH-FocalerMPDIoU',
    'yolov8-bifpn-LADH-Wise-Focaler-MPDIoU3',
]

for name in folders:
    model = YOLO(dir + name + '/best.pt')
    model.export(format=tipe)
