# Sitting Posture Detection

## Introduction
Poor sitting posture is a common issue that can lead to various health problems, including back pain and musculoskeletal disorders. Many people are unaware of their sitting habits, which greatly affect their health. This project aims to accurately identify sitting postures as either good or bad, based on ergonomic guidelines, and provide real-time feedback to users to encourage proper posture.

## Objectives
The objectives of this project are:
1. Detect and classify sitting postures as either good or bad based on predefined ergonomic criteria.
2. Provide real-time feedback to users to encourage maintaining proper postures and prevent health issues associated with poor sitting habits.
3. Leverage advanced image recognition techniques and pretrained models to ensure high accuracy and efficiency in posture detection.
4. Integrate seamlessly with various devices and environments, making it accessible for use.

## Design and Methods
The Sitting Posture Detection system is implemented using a DNN-based model built on the NVIDIA Jetson-Inference library using TensorRT from Hello AI World. This framework supports various DNN vision primitives, including:
- `imageNet` for image classification
- `detectNet` for object detection
- `segNet` for semantic segmentation
- `poseNet` for pose estimation
- `actionNet` for action recognition

### Pose Estimation
Pose estimation involves locating various body parts (keypoints) that form a skeletal topology (links). It has applications in gestures, AI/VR, HMI (human/machine interface), and posture/gait correction. This project implements the `poseNet` model, which uses the ResNet18 network architecture.

### ResNet18 Network Architecture
The ResNet18 network architecture is composed of 18 layers structured into four stages. Each stage contains multiple residual blocks:
1. **Initial Convolution and Max Pooling**
   - 7x7 convolutional layer with 64 filters, stride of 2, followed by batch normalization and ReLU activation.
   - 3x3 max pooling layer with a stride of 2.
2. **Four Stages of Residual Blocks**
   - Stage 1: 2 residual blocks, each with two 3x3 convolutional layers with 64 filters.
   - Stage 2: 2 residual blocks, each with two 3x3 convolutional layers with 128 filters.
   - Stage 3: 2 residual blocks, each with two 3x3 convolutional layers with 256 filters.
   - Stage 4: 2 residual blocks, each with two 3x3 convolutional layers with 512 filters.
3. **Global Average Pooling and Fully Connected Layer**
   - Global average pooling layer to reduce spatial dimensions to 1x1.
   - Fully connected layer with softmax activation for final classification output.

## Experimental Setup
The system runs on the Jetson Nano, which is equipped with a quad-core ARM Cortex-A57 CPU and a 128-core Maxwell GPU. It supports the NVIDIA JetPack SDK, which includes a comprehensive set of tools, libraries, and APIs.

### Experiment
Using the pretrained model, the system detects the angle between the hips and ears to determine if the person is leaning forwards, backwards, or sitting straight. It also checks the vertical alignment of the shoulders and hips to determine if the person is slanted. The script overlays text on video frames to provide real-time feedback about the detected posture:
- "Good sitting posture XD"
- "Bad sitting posture!!!"
- "Leaning forwards!!"
- "Leaning backwards!!"
- "Not leaning!"
- "Cannot detect"

### Results
The model accurately detects whether the person is sitting slanted. However, detection accuracy for leaning forwards and backwards needs improvement due to unstable camera setup and interference from slanted detection.

## Conclusion
The experiment demonstrates the efficacy and practicality of deploying the application using the NVIDIA Jetson Nano and the pretrained pose estimation model. The Jetson Nano shows robust computational capabilities and successfully runs complex neural networks to detect and classify sitting postures in real-time. The model utilizing the ResNet18 backbone is adept at identifying key body points and assessing posture quality, providing immediate feedback on good and bad sitting postures.

## References
1. Dusty-nv. "Jetson Inference." GitHub, https://github.com/dusty-nv/jetson-inference?tab=readme-ov-file#hello-ai-world.
2. "ResNet-18 Architecture." ResearchGate, https://www.researchgate.net/figure/ResNet-18-architecture-20-The-numbers-added-to-the-end-of-ResNet-represent-the_fig2_349241995.
3. "Sitting Posture Recognition." GitHub, https://github.com/nvinayvarma189/Sitting-Posture-Recognition.

## Instructions to Run the Project

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ryantang247/jetson-inference
   cd jetson-inference

2. **Go to project file**
      ```bash
   cd python
   cd examples
   
3. **Run the project**
      ```bash
   python3 posenet.py
