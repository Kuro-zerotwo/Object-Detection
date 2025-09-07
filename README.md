# Object-Detection with YOLOv8

A **deep learning–based real-time object detection system** using Python, OpenCV, and YOLOv8. The system detects objects from the webcam, displays bounding boxes, confidence scores, and detailed object information, and provides **real-time audio feedback** using Google Text-to-Speech (gTTS) and Playsound.

---

## Features

- Real-time object detection using YOLOv8n pre-trained model.
- Bounding boxes drawn with random colors for each class.
- Displays a table of detection values for each object:
  - Confidence (Pc)
  - Center coordinates (Bx, By)
  - Width and Height (Bw, Bh)
  - Class indicators (C1, C2, …)
- Audio feedback for detected objects: "I see a [object]".
- User-friendly exit by pressing `q` or closing the window.

---

## Key Skills & Technologies

- **Programming:** Python  
- **Computer Vision:** OpenCV  
- **Deep Learning:** YOLOv8, Convolutional Neural Networks (CNNs)  
- **Audio Integration:** gTTS, Playsound  
- **Version Control:** Git, GitHub  
- **Other Concepts:** Real-time video processing, bounding boxes, object confidence calculation  

---

## Installation

1. **Clone the repository**  
```bash
git clone https://github.com/Kuro-zerotwo/Object-Detection.git
cd Object-Detection
```

2. **Install dependencies**  
```bash
pip install numpy opencv-python ultralytics gTTS playsound

```
3. Download YOLOv8 pretrained weights
   Place yolov8n.pt in the weights/ folder.

4. Ensure coco.names exists in the project root.

5. Usage
Run the detection script:
bash
Copy code
python yolo_model.py.

 The webcam opens and detects objects in real-time.

 Bounding boxes, confidence scores, and object details are displayed.

 Detected objects are announced via audio.

 Press q or close the window to exit.


## How It Works
1. Loads YOLOv8n pretrained weights.

2. Reads class names from coco.names.

3. Generates random colors for each class.

4. Captures frames from the webcam.

5. Performs object detection per frame.

6. Draws bounding boxes, confidence values, and a table of values.

7. Converts detected labels to speech using gTTS & Playsound.

## Experience & Learning
1. Implemented real-time deep learning object detection using YOLOv8 and OpenCV.

2. Applied CNNs to enhance detection accuracy.

3. Integrated text-to-speech functionality for accessibility.

4. Gained practical experience with Python libraries, audio integration, and GitHub version control.




