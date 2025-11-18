import cv2
from ultralytics import YOLO
import numpy as np

class FaceDetector:
    def __init__(self):
        self.model = YOLO("yolov8n-face.pt", task="detect")

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        faces = []
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()
            faces.append([int(x1), int(y1), int(x2), int(y2)])
        return faces

    def get_direction(self, frame, face):
        x1, y1, x2, y2 = face
        cx = x1 + (x2 - x1) // 2
        frame_center = frame.shape[1] // 2

        if cx < frame_center - 80:
            return "looking_left"
        elif cx > frame_center + 80:
            return "looking_right"
        return "center"

        
