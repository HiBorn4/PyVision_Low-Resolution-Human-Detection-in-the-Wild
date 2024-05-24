import cv2
from pathlib import Path
from ultralytics import YOLO
import torch

# Define paths
model_path = Path("D:\YOLOV8_Hazardous\yolo\yolov8n.pt")  
image_folder = Path("../uav0000158_00000_v")  
output_video_path = Path("./output.avi") 

# Load YOLOv8 model
model = YOLO(model_path)


image_paths = [str(p) for p in image_folder.glob("*.jpg")][::1]  


fourcc = cv2.VideoWriter_fourcc(*'MJPG')  
fps = 10  

# Get frame width and height from the first image (assuming consistent size)
frame = cv2.imread(str(image_paths[0]))
width, height = frame.shape[1], frame.shape[0]
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

for image_path in image_paths:
    # Read image
    frame = cv2.imread(image_path)

    # Make predictions
    results = model(frame)

    
    for detection in results:
        boxes = detection.boxes  # Bounding box coordinates as Boxes instance
        cls = detection.boxes.cls  # Class index
        conf = detection.boxes.conf  # Confidence score

        
        for bbox, c in zip(boxes.xyxy, cls):
            class_name = detection.names[int(c)]

            
            if class_name == 'traffic light':
                continue

            
            x1, y1, x2, y2 = [int(coord) for coord in bbox]

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
            cv2.putText(frame, f"{class_name} {conf[int(c)]:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Write frame with detections to output video
    out.write(frame)


out.release()
cv2.destroyAllWindows()
print("Video processing complete. Output video saved to:", output_video_path)