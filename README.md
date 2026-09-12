# Drone Face Recognition Greeting System

A lightweight computer vision project that combines robotics, face detection, face recognition, and speech to create an interactive drone greeting system.

## Project Overview

The goal of this project is to demonstrate how a drone can interact with people using computer vision and artificial intelligence.

The drone's camera will provide a video stream to a computer. The system will detect a person's face, recognize previously registered participants, and respond with a personalized greeting such as:

> "Hello Bruce, nice to meet you!"

The project is being developed and tested on a laptop webcam first. Once the drone is available, the camera input can be replaced with the drone's video stream while keeping the main AI pipeline unchanged.

## Current Architecture

```text
Camera
   ↓
Face Detection
   ↓
Face Recognition
   ↓
Identity
   ↓
Greeting
   ↓
Speech
```

## Technologies

* Python
* OpenCV
* YuNet — face detection
* SFace — face recognition
* Computer Vision
* Speech synthesis
* Git / GitHub

## Project Structure

```text
Drone_face_Reco/
│
├── dataset/
│   ├── Bruce/
│   ├── Student_A/
│   └── Student_B/
│
├── encodings/
│
├── models/
│
├── src/
│   ├── camera.py
│   ├── detection.py
│   ├── encoding.py
│   ├── recognition.py
│   ├── speech.py
│   └── main.py
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Development Approach

The system is being developed incrementally.

### Phase 1 — Camera Input

Test the webcam and establish a camera interface that can later be replaced by the drone's video stream.

### Phase 2 — Face Detection

Use YuNet to locate faces in each video frame.

### Phase 3 — Face Recognition

Use a lightweight pretrained face-recognition model to generate facial embeddings and compare them with registered participants.

### Phase 4 — Speech

Generate a personalized greeting after a person has been recognized.

### Phase 5 — Drone Integration

Replace the laptop webcam input with the drone's camera stream and test the complete system.

## Current Progress

* [x] Project structure created
* [x] Python environment configured
* [x] OpenCV installed
* [x] Webcam input working
* [x] YuNet face detection working
* [ ] Dataset collection
* [ ] Face embeddings
* [ ] Face recognition
* [ ] Speech output
* [ ] Drone camera integration
* [ ] Final exhibition testing

## Privacy

This project is intended as an educational demonstration.

Face images should only be collected from participants who have agreed to take part in the demonstration. Face datasets and generated face embeddings should remain local and should not be committed to the public repository.

## Project Goal

The final demonstration should show how robotics and AI can work together:

**Drone → Camera → Computer Vision → AI Recognition → Human Interaction**
