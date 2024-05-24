from ultralytics import YOLO
import torch
import cv2
from pathlib import Path

# # Load the trained model
# model = YOLO("/media/iiitdwd/Elements/Vinayak/DAV/yolov8s.pt")  # Replace with your trained model path
# nc = 12  # Number of classes (replace with your actual value)

# # Define paths for test data and output folder
# test_images_dir = Path("/media/iiitdwd/Elements/Vinayak/DAV/Test/images")  # Replace with your test image directory
# output_dir = Path("./test_results")  # Output folder for images with bounding boxes

# # Create output folder if it doesn't exist
# output_dir.mkdir(parents=True, exist_ok=True)

# # Function to draw bounding boxes and save the image
# def save_image_with_bbox(image, predictions, output_path):
#   image = cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2RGB)  # Convert to RGB for OpenCV
#   for pred in predictions:
#     x1, y1, x2, y2, conf, cls = pred
#     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green for bounding box
#     cv2.putText(image, f"{model.names[int(cls)]} {conf:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Label with confidence score
#   cv2.imwrite(str(output_path), image)

# # Initialize variables for metrics
# total_images = 0
# correct_predictions = 0
# class_wise_tp = {cls: 0 for cls in range(nc)}  # True Positives for each class
# class_wise_gt = {cls: 0 for cls in range(nc)}  # Ground Truth for each class

# # Loop through test images
# for image_path in test_images_dir.glob("*.jpg"):
#   total_images += 1
#   image = cv2.imread(str(image_path))

#   # Perform inference
#   results = model(image)

#   # Get predictions and ground truth (if available)
#   predictions = df(results).xyxy[0]  # Assuming predictions are in xyxy format
#   ground_truth = None  # Modify this to access ground truth if available

#   # Update metrics
#   if ground_truth is not None:
#     for gt in ground_truth:
#       class_wise_gt[gt[4]] += 1  # Update ground truth count for each class
#     for pred in predictions.itertuples():
#       if pred.confidence > 0.5:  # Adjust confidence threshold as needed
#         class_wise_tp[pred.name] += 1  # Update true positives for each class with predicted class and confidence > threshold
#         if pred.name == gt[4]:  # Check if predicted class matches ground truth class
#           correct_predictions += 1

#   # Save image with bounding boxes
#   output_path = output_dir / image_path.name
#   save_image_with_bbox(image, predictions.to_numpy(), output_path)

# # Calculate accuracy, precision, and recall
# accuracy = correct_predictions / total_images
# precision = {cls: tp / (tp + sum([other for o, other in class_wise_tp.items() if o != cls])) for cls, tp in class_wise_tp.items()}  # Handle division by zero
# recall = {cls: tp / gt_count if gt_count > 0 else 0 for cls, tp, gt_count in zip(class_wise_tp.keys(), class_wise_tp.values(), class_wise_gt.values())}

# # Print results
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)

# print("Images with bounding boxes saved to:", output_dir)
from pathlib import Path
from ultralytics import YOLO
model=YOLO("D:\YOLOV8_Hazardous\yolo\yolov8n.pt")
model.predict(source="../YST_Bing_667_final.jpg",save=True)
