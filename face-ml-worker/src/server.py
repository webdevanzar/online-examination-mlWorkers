import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import cv2

from src.detector import FaceDetector
from src.object_detector import ObjectDetector
from src.fraud_rules import check_fraud
from src.object_rules import check_objects_for_fraud


app = FastAPI()

face_detector = FaceDetector()
object_detector = ObjectDetector()

class FrameData(BaseModel):
    image: str


def decode_image(base64_str):
    base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


@app.post("/analyze-frame")
def analyze_frame(data: FrameData):
    frame = decode_image(data.image)

    # face detecting
    faces = face_detector.detect(frame)
    direction = "none"
    if len(faces) > 0:
        direction = face_detector.get_direction(frame, faces[0])

    fraud_face = check_fraud(faces, direction, frame)

    # object detecting
    objects = object_detector.detect_objects(frame)
    fraud_obj = check_objects_for_fraud(objects)

    fraud = fraud_face + fraud_obj

    return {
        "faces": faces,
        "direction": direction,
        "objects": objects,
        "fraud": fraud
    }
