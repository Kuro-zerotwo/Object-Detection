# Import required libraries
import numpy as np
import cv2
from ultralytics import YOLO
import random
import os
from gtts import gTTS
from playsound import playsound

# Load class names from coco.names file
coco_file_path = "coco.names"

# Check if the coco.names file exists
if not os.path.exists(coco_file_path):
    print(f"Error: {coco_file_path} not found.")
    exit()

# Open the coco.names file and read class names
with open(coco_file_path, "r") as my_file:
    class_list = my_file.read().strip().split("\n")

# Generate random colors for each class
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt")

# Set up camera capture (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Define a function to convert detected label to speech with a sentence format
def speak_label(label):
    sentence = f"I see a {label}"
    tts = gTTS(text=sentence, lang='en')
    tts.save("label.mp3")
    playsound("label.mp3")
    os.remove("label.mp3")

# Define a function to draw the table of values on the frame
def draw_values_table(frame, values, start_x, start_y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 1
    line_height = 30

    # Draw each value in the table format
    for i, value in enumerate(values):
        y = start_y + i * line_height
        text = f"{value[0]}: {value[1]}"
        cv2.putText(frame, text, (start_x, y), font, font_scale, (255, 255, 255), font_thickness)

while True:
    # Capture frame-by-frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_wid, frame_hyt = 640, 480
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on the current frame using the YOLO model
    results = model(frame, conf=0.45, save=False)

    # Initialize C1 and C2 as 0 (No Dog or Person detected by default)
    C1, C2 = 0, 0

    if results:
        boxes = results[0].boxes
        for box in boxes:
            bb = box.xyxy.numpy().astype(int)[0]
            conf = box.conf.numpy()[0]
            clsID = int(box.cls.numpy()[0])

            Bx = (bb[0] + bb[2]) // 2
            By = (bb[1] + bb[3]) // 2
            Bw = bb[2] - bb[0]
            Bh = bb[3] - bb[1]

            # If the detected class is a Dog (class ID for Dog is usually 16 in COCO)
            if class_list[clsID] == "dog":
                C1 = 1  # Set C1 = 1 for Dog
                C2 = 0  # Ensure C2 is 0
            # If the detected class is a Person (class ID for Person is usually 0 in COCO)
            elif class_list[clsID] == "person":
                C2 = 1  # Set C2 = 1 for Person
                C1 = 0  # Ensure C1 is 0

            # Prepare values for display in the required format
            table_values = [
                ("Pc", round(conf * 100, 2)),    # Confidence score
                ("Bx", Bx),                      # Center X
                ("By", By),                      # Center Y
                ("Bw", Bw),                      # Width
                ("Bh", Bh),                      # Height
                ("C1", C1),                      # Dog Class Indicator
                ("C2", C2),                      # Person Class Indicator
            ]

            # Draw bounding boxes and labels on the frame
            cv2.rectangle(
                frame,
                (bb[0], bb[1]),
                (bb[2], bb[3]),
                detection_colors[clsID],
                3,
            )

            label = f"{class_list[clsID]} {round(conf * 100, 2)}%"
            cv2.putText(
                frame,
                label,
                (bb[0], bb[1] - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 0),
                2,
            )

            # Speak the detected label
            speak_label(class_list[clsID])

            # Draw the values table on the frame (on the right side of the frame)
            draw_values_table(frame, table_values, start_x=frame_wid - 150, start_y=50)

    # Display the resulting frame with bounding boxes and labels
    cv2.imshow('YOLOv8 Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('YOLOv8 Object Detection', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
