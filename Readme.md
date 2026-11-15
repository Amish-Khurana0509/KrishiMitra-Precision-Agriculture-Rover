# KrishiMitra
### An Edge-AI Based Precision Agriculture Rover for Early Crop Disease Detection and Targeted Treatment

**Smart. Precise. Affordable.**

KrishiMitra is an AI-powered agricultural rover designed to help farmers identify crop diseases at an early stage and enable targeted treatment. By combining computer vision, embedded systems, robotics, and the **AgriSetu** digital agricultural platform, KrishiMitra aims to reduce unnecessary pesticide usage and make precision agriculture more accessible.

---

## Project Overview

Traditional crop monitoring often requires farmers to manually inspect large agricultural fields. Diseases may go unnoticed until they spread, potentially resulting in crop losses and unnecessary pesticide application.

KrishiMitra addresses this challenge through a six-wheel rocker-bogie agricultural rover equipped with a camera-based disease detection system. The rover captures crop images, processes them using a custom-trained YOLOv11n model, and supports targeted treatment of identified diseased plants.

The accompanying **AgriSetu** web platform brings leaf-health screening, daily mandi prices, multilingual crop advisory, MSP scheme guides, and live rover telemetry together for farmers.

## Key Features

- **AI-Based Disease Detection:** Custom-trained YOLOv11n model for identifying tomato plant diseases.
- **Edge & Lightweight Inference:** Fast local and cloud inference powered by ONNX Runtime.
- **Six-Wheel Mobility:** Rocker-bogie suspension for navigating uneven agricultural terrain.
- **Targeted Treatment:** Multi-tank spraying mechanism designed to apply treatment only to affected plants.
- **AgriSetu Web Platform:** Leaf health scanner, live mandi rates, multilingual crop advisor, MSP guides, and real-time rover console.
- **Affordable Design:** Built using accessible hardware and open-source technologies.

## AI Model

KrishiMitra uses a custom-trained YOLOv11n object detection model exported to ONNX for lightweight deployment.

The model supports 10 tomato plant health and disease categories:

1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Spider Mites (Two-spotted spider mite)
7. Target Spot
8. Yellow Leaf Curl Virus
9. Healthy
10. Mosaic Virus

**Model Information:**
- **Architecture:** YOLOv11n
- **Training Images:** Approximately 2,200
- **Input Resolution (Training):** 224 × 224
- **Inference Deployment:** ONNX Runtime
- **Inference Input Resolution:** 640 × 640

## Technology Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv11n |
| Model Deployment | ONNX Runtime |
| Backend & Programming | Python, Flask |
| Embedded Control | Arduino |
| Motor Drivers | BTS7960 |
| Communication | Bluetooth & HTTP Telemetry API |
| Frontend | HTML, CSS, JavaScript |
| Digital Platform | AgriSetu |

## System Architecture

KrishiMitra consists of three major subsystems:

**1. Agricultural Rover**
- Six-wheel rocker-bogie chassis
- Camera-based crop observation
- Embedded motor control (Arduino + BTS7960)
- Targeted multi-tank spraying mechanism

**2. AI Disease Detection**
- Image acquisition and preprocessing
- Lightweight ONNX Runtime inference (`640 × 640`)
- Bounding-box localization and disease classification output

**3. AgriSetu Web Platform**
- **Leaf Health Scanner:** Upload or capture leaf photos for instant screening
- **Mandi Rates:** Daily wholesale market prices and arrival trends
- **Crop Advisor:** Stage-wise guidance in multiple Indian languages
- **Schemes & MSP Desk:** Plain-language summaries and Q&A on official MSP guides
- **Field Rover Console:** Live battery, spray tank levels, GPS location, and coverage tracking

## (Web Platform)

Access the Website on Vercel: **https://agrisetu-psi.vercel.app/**

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
