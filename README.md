# KrishiMitra
### An Edge-AI Based Precision Agriculture Rover for Early Crop Disease Detection and Targeted Treatment

**Smart. Precise. Affordable.**

KrishiMitra is an AI-powered agricultural rover designed to help farmers identify crop diseases at an early stage and enable targeted treatment. By combining computer vision, embedded systems, robotics and a digital agricultural platform, KrishiMitra aims to reduce unnecessary pesticide usage and make precision agriculture more accessible.

---

## Project Overview

Traditional crop monitoring often requires farmers to manually inspect large agricultural fields. Diseases may go unnoticed until they spread, potentially resulting in crop losses and unnecessary pesticide application.

KrishiMitra addresses this challenge through a six-wheel rocker-bogie agricultural rover equipped with a camera-based disease detection system.

The rover captures crop images, processes them using a custom-trained YOLOv11n model and supports targeted treatment of identified diseased plants.

The accompanying **AgriSetu** platform provides digital agricultural support and crop-related information.

## Key Features

- **AI-Based Disease Detection:** Custom-trained YOLOv11n model for identifying tomato plant diseases.
- **Edge-AI Processing:** Local image inference capability.
- **Six-Wheel Mobility:** Rocker-bogie suspension for navigating uneven agricultural terrain.
- **Targeted Treatment:** Spraying mechanism designed to apply treatment to identified plants.
- **AgriSetu Platform:** Web-based agricultural assistance and monitoring.
- **Affordable Design:** Built using accessible hardware and open-source technologies.

## AI Model

KrishiMitra uses a custom-trained YOLOv11n object detection model.

The model supports 10 tomato plant health and disease categories:

1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Spider Mites
7. Target Spot
8. Yellow Leaf Curl Virus
9. Healthy
10. Mosaic Virus

**Model information:**
- Architecture: YOLOv11n
- Training images: Approximately 2,200
- Input resolution during training: 224 × 224
- Inference deployment: ONNX Runtime
- Inference input: 640 × 640

See `ai-model/model-information.md` for further details.

## Technology Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv11n |
| Model Deployment | ONNX Runtime |
| Programming | Python |
| Embedded Control | Arduino |
| Motor Drivers | BTS7960 |
| Communication | Bluetooth |
| Web Application | Flask |
| Frontend | HTML, CSS |
| Digital Platform | AgriSetu |

## System Architecture

KrishiMitra consists of three major subsystems:

**1. Agricultural Rover**
- Six-wheel rocker-bogie chassis
- Camera-based crop observation
- Embedded motor control
- Targeted spraying mechanism

**2. AI Disease Detection**
- Image acquisition
- Image preprocessing
- YOLOv11n inference
- Disease classification and detection output

**3. AgriSetu**
- Agricultural information
- Crop recommendations
- Disease-related assistance
- Digital farmer support

## Impact and Objectives

KrishiMitra aims to:

- Encourage early crop disease identification.
- Reduce unnecessary chemical application.
- Improve the efficiency of crop monitoring.
- Make precision agriculture more accessible to small and marginal farmers.
- Support sustainable agricultural practices.

## Future Scope

- Autonomous field navigation.
- Support for additional crop species.
- Improved environmental sensing.
- Enhanced offline operation.
- GPS-based field mapping.
- Advanced agricultural analytics.


*KrishiMitra — Bringing intelligent technology closer to every farmer.*
