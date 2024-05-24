# Image Dehazing, Super-Resolution, and Object Detection with YOLOv8

This project focuses on enhancing the quality of images captured in challenging conditions and accurately detecting objects within those images. The methodology includes three main stages: Image Dehazing, Super-Resolution, and Object Detection using YOLOv8.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Image Dehazing](#image-dehazing)
  - [Super-Resolution](#super-resolution)
  - [Object Detection with YOLOv8](#object-detection-with-yolov8)
- [Results and Discussion](#results-and-discussion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
In this project, we aim to enhance the visibility and resolution of images captured under adverse conditions, such as fog or haze, and accurately detect objects within these images. The workflow consists of three stages:

1. **Image Dehazing**: Removing haze to improve image clarity.
2. **Super-Resolution**: Enhancing the resolution of dehazed images.
3. **Object Detection**: Identifying and localizing objects using YOLOv8.

## Methodology

### Image Dehazing
The first stage involves removing haze and improving visibility in low-resolution images captured in wild conditions. We employ advanced image dehazing techniques, leveraging deep learning algorithms to estimate and remove the effects of haze from the input images. By enhancing contrast and clarity, image dehazing lays the foundation for subsequent stages, enabling more accurate analysis and detection.

![Image Dehazing](path/to/dehazing_image.png)
*Image dehazing results by our method. Top: input haze images. Middle: the dehazing results. Bottom: the recovered transmission functions. The recovered transmission gives an estimation of the density map of hazes in the input image. (Best viewed in color)*

### Super-Resolution
In the second stage, we address the issue of low-resolution imagery by employing super-resolution techniques. Super-resolution algorithms leverage advanced image processing methods, including generative adversarial networks (GANs) and convolutional neural networks (CNNs), to reconstruct high-resolution images from low-resolution inputs.

![Super Resolution](path/to/super_resolution.png)
*Real-ESRGAN adopts the same generator network as that in ESRGAN. For the scale factor of ×2 and ×1, it first employs a pixel-unshuffle operation to reduce spatial size and re-arrange information to the channel dimension.*

### Object Detection with YOLOv8
The final stage involves detecting objects within the high-resolution, dehazed images using the YOLOv8 architecture. YOLOv8 is known for its speed and accuracy in object detection tasks. The network is trained to identify various objects within the enhanced images, enabling effective analysis and decision-making.

![YOLOv8 Architecture](path/to/yolov8_architecture.png)
*Diagram illustrating the architecture of YOLOv8, highlighting its backbone, neck, and head components.*

## Results and Discussion
The results demonstrate significant improvements in image clarity, resolution, and object detection accuracy. The dehazing process effectively removes haze, revealing more details and contrast. The super-resolution stage enhances the spatial resolution, making the images more suitable for detailed analysis. Finally, the YOLOv8-based object detection accurately identifies and localizes objects within the enhanced images.

### Qualitative Comparisons
![Qualitative Comparisons](path/to/qualitative_comparisons.png)
*Qualitative comparisons on several representative real-world samples with upsampling scale factor of 4. Our Real-ESRGAN outperforms previous approaches in both removing artifacts and restoring texture details.*

## Installation
To run this project, you need to have Python installed. Follow the steps below to set up the environment:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
To use this project, follow these steps:

1. **Preprocess the images**: Apply the dehazing algorithm to your images.
2. **Enhance the images**: Use the super-resolution model to increase the resolution of the dehazed images.
3. **Detect objects**: Run the YOLOv8 model to detect objects within the enhanced images.

Example commands:
```sh
python dehaze.py --input input_image.jpg --output dehazed_image.jpg
python super_resolve.py --input dehazed_image.jpg --output high_res_image.jpg
python detect.py --input high_res_image.jpg --output detection_result.jpg
```

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.