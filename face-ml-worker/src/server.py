import sys
import os
import base64
import tempfile
import requests
import numpy as np
import cv2

# Optimize TensorFlow before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------
# Path setup (keep as you had)
# --------------------------------------------------

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --------------------------------------------------
# Internal imports
# --------------------------------------------------

from src.detector import FaceDetector
from src.object_detector import ObjectDetector
from src.fraud_rules import check_fraud, classify_fraud_severity
from src.object_rules import check_objects_for_fraud
from src.face_verifier import FaceVerifier

# --------------------------------------------------
# App & singletons
# --------------------------------------------------

app = FastAPI(title="Online Examination ML Worker")

# Initialize lightweight models at startup
print("Initializing face detector...")
face_detector = FaceDetector()
print("✓ Face detector ready")

print("Initializing object detector...")
object_detector = ObjectDetector()
print("✓ Object detector ready")

# Lazy load FaceVerifier (heavy model - only load when needed)
face_verifier = None

def get_face_verifier():
    """Lazy load face verifier on first use."""
    global face_verifier
    if face_verifier is None:
        print("First face verification request - loading FaceNet model (this may take 1-2 minutes)...")
        face_verifier = FaceVerifier()
        print("✓ Face verifier ready")
    return face_verifier

# --------------------------------------------------
# Schemas
# --------------------------------------------------

class FrameData(BaseModel):
    image: str  # base64 image


class EnrollmentData(BaseModel):
    user_id: str
    video_url: str


class VerificationData(BaseModel):
    user_id: str
    image: str  # base64 image

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def decode_image(base64_str: str) -> np.ndarray:
    """
    Decode base64 image to OpenCV BGR frame.
    """
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        img_bytes = base64.b64decode(base64_str)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            raise ValueError("Decoded image is empty")

        return frame

    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/health")
def health_check():
    """Quick health check - doesn't load heavy models."""
    return {
        "status": "healthy",
        "face_detector": "ready",
        "object_detector": "ready",
        "face_verifier": "ready" if face_verifier is not None else "not_loaded"
    }

@app.post("/analyze-frame")
def analyze_frame(data: FrameData):
    frame = decode_image(data.image)

    # ---------- Face analysis ----------
    faces = face_detector.detect(frame)
    direction = "none"

    if faces:
        direction = face_detector.get_direction(frame, faces[0])

    fraud_face = check_fraud(faces, direction, frame)

    # ---------- Object analysis ----------
    objects = object_detector.detect_objects(frame)
    fraud_object = check_objects_for_fraud(objects)

    fraud = fraud_face + fraud_object
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
    """
    Enroll user's face using a selfie video URL.
    """
    tmp_path = None

    try:
        print(f"\n[ENROLL] User: {data.user_id}")
        print(f"[ENROLL] Video URL: {data.video_url}")

        # ---------- Download video safely ----------
        response = requests.get(
            data.video_url,
            stream=True,
            timeout=30,
        )
        response.raise_for_status()

        # Detect extension
        content_type = response.headers.get("content-type", "").lower()
        ext = ".mp4"

        if "webm" in content_type or data.video_url.endswith(".webm"):
            ext = ".webm"
        elif "quicktime" in content_type or data.video_url.endswith(".mov"):
            ext = ".mov"
        elif "avi" in content_type or data.video_url.endswith(".avi"):
            ext = ".avi"

        print(f"[ENROLL] Detected format: {ext}")

        # ---------- Save temp file ----------
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise ValueError("Downloaded video file is empty")

        print(f"[ENROLL] Saved to: {tmp_path}")

        # ---------- Enroll ----------
        verifier = get_face_verifier()
        success, message = verifier.enroll_user(
            user_id=data.user_id,
            video_path=tmp_path
        )

        print(f"[ENROLL] Result: {success}, {message}")
        return {"success": success, "message": message}

    except requests.RequestException as e:
        return {"success": False, "message": f"Video download failed: {str(e)}"}

    except ValueError as e:
        return {"success": False, "message": str(e)}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Enrollment failed: {str(e)}"}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print(f"[ENROLL] Temp file removed")
            except Exception as e:
                print(f"[WARN] Temp cleanup failed: {e}")


@app.post("/verify-face")
def verify_face(data: VerificationData):
    """
    Verify if a live frame matches the enrolled user.
    """
    try:
        frame = decode_image(data.image)

        verifier = get_face_verifier()
        is_match, confidence, message = verifier.verify_face(
            user_id=data.user_id,
            frame=frame
        )

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
