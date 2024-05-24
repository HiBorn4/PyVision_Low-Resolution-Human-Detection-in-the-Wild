from ultralytics import YOLO
from ultralytics.utils.loss import CustomLoss
model_cfg = "./modified_yolov8s.yaml"  # Replace with your actual path
model = YOLO(model_cfg)
import yaml
yolo_format=dict(
                 train="/media/iiitdwd/Elements/Vinayak/DAV/Train/images",
                 val= "/media/iiitdwd/Elements/Vinayak/DAV/Val/images",
                 test= "/media/iiitdwd/Elements/Vinayak/DAV/Test/images",
                 nc=12,
                 names={  0:"ignored region",
                          1:"pedestrian",
                          2: "people",
                          3:"bicycle",
                          4:"car",
                          5:"van",
                          6:"truck",
                          7:"tricycle",
                          8:"awning tricycle",
                          9:"bus",
                          10:"bike",
                          11:"others"})

with open('./data.yaml', 'w') as outfile:
    yaml.dump(yolo_format, outfile, default_flow_style=False)
# Set your custom loss function as the loss function for the YOLOv8 model
model.model.loss = CustomLoss(lambda1=1.5, lambda2=0.5, lambda3=2.0, alpha=1.0, beta=2.0, delta=1.0, gamma=1.5) 


# Train the model with your custom loss function
model.train(data='data.yaml', epochs=130)

model.export(f="best_model.pt")