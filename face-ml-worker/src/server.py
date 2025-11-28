import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import cv2

from src.detector import FaceDetector
from src.object_detector import ObjectDetector
from src.fraud_rules import check_fraud, classify_fraud_severity
from src.object_rules import check_objects_for_fraud
from src.face_verifier import FaceVerifier
import requests
import tempfile


app = FastAPI()

face_detector = FaceDetector()
object_detector = ObjectDetector()
face_verifier = FaceVerifier()


class FrameData(BaseModel):
    image: str


class EnrollmentData(BaseModel):
    user_id: str
    video_url: str


class VerificationData(BaseModel):
    user_id: str
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

    # Classify fraud by severity
    fraud_severity = classify_fraud_severity(fraud)

    return {
        "faces": faces,
        "direction": direction,
        "objects": objects,
        "fraud": fraud,
        "fraud_severity": fraud_severity,
    }


@app.post("/enroll-face")
def enroll_face(data: EnrollmentData):
    """Enroll user's face from selfie video."""
    try:
        # Download video from URL
        response = requests.get(data.video_url, timeout=30)
        response.raise_for_status()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Enroll user
        success, message = face_verifier.enroll_user(data.user_id, tmp_path)

        # Clean up temp file
        os.unlink(tmp_path)

        return {"success": success, "message": message}
    except Exception as e:
        return {"success": False, "message": f"Enrollment failed: {str(e)}"}


@app.post("/verify-face")
def verify_face(data: VerificationData):
    """Verify face in frame matches enrolled user."""
    try:
        frame = decode_image(data.image)
        is_match, confidence, message = face_verifier.verify_face(data.user_id, frame)

        return {
            "verified": is_match,
            "confidence": float(confidence),
            "message": message,
        }
    except Exception as e:
        return {
            "verified": False,
            "confidence": 0.0,
            "message": f"Verification failed: {str(e)}",
        }
