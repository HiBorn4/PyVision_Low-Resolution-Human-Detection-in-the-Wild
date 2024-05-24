from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('/media/iiitdwd/Elements/Vinayak/DAV/yolov8n.pt')

# Run validation on a set specified as 'val' argument
metrics = model.val(data='./data.yaml')

print(metrics.results_dict)