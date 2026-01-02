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
from src.face_verifier_deepface import FaceVerifier

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


class VerifyWithVideoData(BaseModel):
    user_id: str
    video_url: str  # Cloudinary video URL
    image: str  # base64 image of current frame to verify


class DebugCompareData(BaseModel):
    image1: str  # base64 image 1
    image2: str  # base64 image 2

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def decode_image(base64_str: str) -> np.ndarray:
    """
    Decode base64 image to OpenCV BGR frame with robust error handling.
    """
    try:
        # Validate input
        if not base64_str or not isinstance(base64_str, str):
            raise ValueError("Image data is empty or not a string")

        # Remove data URL prefix if present
        if "," in base64_str:
            header, base64_str = base64_str.split(",", 1)
            print(f"[DECODE] Removed header: {header[:50]}...")

        # Remove whitespace
        base64_str = base64_str.strip()

        if len(base64_str) < 100:
            raise ValueError(f"Image data too short ({len(base64_str)} chars) - likely corrupted")

        # Decode base64
        try:
            img_bytes = base64.b64decode(base64_str, validate=True)
        except Exception as e:
            raise ValueError(f"Base64 decode failed: {str(e)}")

        if len(img_bytes) == 0:
            raise ValueError("Decoded bytes are empty")

        print(f"[DECODE] Decoded {len(img_bytes)} bytes")

        # Convert to numpy array
        arr = np.frombuffer(img_bytes, dtype=np.uint8)

        if arr.size == 0:
            raise ValueError("Numpy array is empty")

        # Decode image
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("cv2.imdecode returned None - invalid image format")

        if frame.size == 0:
            raise ValueError("Decoded image has zero size")

        print(f"[DECODE] ✓ Image decoded successfully: {frame.shape}")

        return frame

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Image decoding failed: {type(e).__name__}: {str(e)}")

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/health")
def health_check():
    """Quick health check - doesn't load heavy models."""
    enrolled_count = 0
    if face_verifier is not None:
        enrolled_count = len(face_verifier.reference_encodings)

    return {
        "status": "healthy",
        "face_detector": "ready",
        "object_detector": "ready",
        "face_verifier": "ready" if face_verifier is not None else "not_loaded",
        "enrolled_users": enrolled_count
    }

@app.get("/enrollment-status/{user_id}")
def check_enrollment(user_id: str):
    """Check if a user is enrolled."""
    if face_verifier is None:
        return {
            "enrolled": False,
            "message": "Face verifier not initialized"
        }

    is_enrolled = face_verifier.is_enrolled(user_id)
    embeddings_count = 0

    if is_enrolled:
        embeddings_count = len(face_verifier.reference_encodings[user_id])

    return {
        "enrolled": is_enrolled,
        "user_id": user_id,
        "embeddings_count": embeddings_count
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

    # Debug logging
    print(f"[DEBUG] Faces: {len(faces)}, Objects: {len(objects)}")
    print(f"[DEBUG] Fraud issues: {fraud}")
    print(f"[DEBUG] Fraud severity: {fraud_severity}")
    if faces:
        face = faces[0]
        face_width = face[2] - face[0]
        frame_width = frame.shape[1]
        print(f"[DEBUG] Face width: {face_width}, Frame width: {frame_width}, Ratio: {face_width/frame_width:.2%}")

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
        print(f"\n[VERIFY] User: {data.user_id}")

        frame = decode_image(data.image)
        print(f"[VERIFY] Frame decoded: {frame.shape}")

        verifier = get_face_verifier()

        # Check if user is enrolled
        is_enrolled = verifier.is_enrolled(data.user_id)
        print(f"[VERIFY] User enrolled: {is_enrolled}")

        if is_enrolled:
            num_embeddings = len(verifier.reference_encodings[data.user_id])
            print(f"[VERIFY] Stored embeddings: {num_embeddings}")

        is_match, confidence, message = verifier.verify_face(
            user_id=data.user_id,
            frame=frame
        )

        print(f"[VERIFY] Result: is_match={is_match}, confidence={confidence:.3f}, message={message}")

        return {
            "verified": is_match,
            "confidence": float(confidence),
            "message": message,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[VERIFY] Error: {str(e)}")
        return {
            "verified": False,
            "confidence": 0.0,
            "message": f"Verification failed: {str(e)}",
        }


@app.post("/verify-with-video")
def verify_with_video(data: VerifyWithVideoData):
    """
    One-shot verification: Extract face from video and compare with live frame.
    No enrollment needed - does both in one call!
    """
    tmp_path = None

    try:
        print(f"\n[VERIFY-VIDEO] User: {data.user_id}")
        print(f"[VERIFY-VIDEO] Video URL: {data.video_url}")

        # ---------- Step 1: Download video ----------
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

        print(f"[VERIFY-VIDEO] Detected format: {ext}")

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
            tmp_path = tmp.name

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise ValueError("Downloaded video file is empty")

        print(f"[VERIFY-VIDEO] Saved to: {tmp_path}")

        # ---------- Step 2: Extract embeddings from video ----------
        verifier = get_face_verifier()
        # Reduced from 30 to 10 for faster processing
        video_embeddings = verifier.extract_encodings_from_video(tmp_path, max_frames=10)

        if len(video_embeddings) == 0:
            return {
                "verified": False,
                "confidence": 0.0,
                "message": "No valid face detected in enrollment video. Please ensure face is clearly visible and well-lit.",
            }

        print(f"[VERIFY-VIDEO] ✓ Extracted {len(video_embeddings)} valid embeddings from video")

        # ---------- Step 3: Extract embedding from current frame ----------
        frame = decode_image(data.image)

        current_embedding = verifier.extract_face_embedding(frame)

        if current_embedding is None:
            return {
                "verified": False,
                "confidence": 0.0,
                "message": "No valid face detected in current frame. Please ensure face is clearly visible and well-lit.",
            }

        print(f"[VERIFY-VIDEO] ✓ Extracted valid embedding from current frame")
        print(f"[VERIFY-VIDEO] Current embedding - Shape: {current_embedding.shape}, Norm: {np.linalg.norm(current_embedding):.4f}")

        # ---------- Step 4: Compare embeddings ----------
        # Calculate distances using cosine distance (more reliable than euclidean for face embeddings)
        distances = []

        for i, video_emb in enumerate(video_embeddings):
            # Cosine distance (1 - cosine similarity)
            dot_product = np.dot(video_emb, current_embedding)
            norm_product = np.linalg.norm(video_emb) * np.linalg.norm(current_embedding)
            cosine_sim = dot_product / (norm_product + 1e-8)
            cosine_distance = 1.0 - cosine_sim

            distances.append(cosine_distance)

            if i < 5:  # Only log first 5 samples to reduce clutter
                print(f"[VERIFY-VIDEO] Sample {i+1}: Cosine Distance={cosine_distance:.4f}, Similarity={cosine_sim:.4f}")

        min_distance = float(np.min(distances))
        avg_distance = float(np.mean(distances))

        # Facenet512 with cosine distance thresholds:
        # Same person: 0.0 - 0.40
        # Different people: 0.50+
        threshold = 0.40

        # Calculate confidence score
        if min_distance <= threshold:
            confidence = 1.0 - (min_distance / threshold)
        else:
            confidence = 0.0

        is_match = min_distance <= threshold

        print(f"\n[VERIFY-VIDEO] ========== RESULTS ==========")
        print(f"[VERIFY-VIDEO] Min Distance: {min_distance:.4f}")
        print(f"[VERIFY-VIDEO] Avg Distance: {avg_distance:.4f}")
        print(f"[VERIFY-VIDEO] Threshold: {threshold}")
        print(f"[VERIFY-VIDEO] Match: {is_match}")
        print(f"[VERIFY-VIDEO] Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
        print(f"[VERIFY-VIDEO] Video Samples: {len(video_embeddings)}")
        print(f"[VERIFY-VIDEO] =====================================\n")

        message = "✓ Face verified - match confirmed" if is_match else \
                  f"✗ Face does not match (distance: {min_distance:.3f} > threshold: {threshold})"

        return {
            "verified": is_match,
            "confidence": float(confidence),
            "message": message,
            "distance": float(min_distance),
            "threshold": threshold,
            "video_samples": len(video_embeddings),
        }

    except requests.RequestException as e:
        return {
            "verified": False,
            "confidence": 0.0,
            "message": f"Video download failed: {str(e)}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "verified": False,
            "confidence": 0.0,
            "message": f"Verification failed: {str(e)}"
        }

    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print(f"[VERIFY-VIDEO] Temp file removed")
            except Exception as e:
                print(f"[WARN] Temp cleanup failed: {e}")


@app.post("/debug-compare-faces")
def debug_compare_faces(data: DebugCompareData):
    """
    DEBUG ENDPOINT: Compare two face images and return detailed metrics.
    Use this to test if face recognition is working correctly.
    """
    try:
        print("\n[DEBUG] ========== FACE COMPARISON DEBUG ==========")

        # Decode both images
        frame1 = decode_image(data.image1)
        frame2 = decode_image(data.image2)

        print(f"[DEBUG] Image 1 shape: {frame1.shape}")
        print(f"[DEBUG] Image 2 shape: {frame2.shape}")

        # Extract embeddings
        verifier = get_face_verifier()

        emb1 = verifier.extract_face_embedding(frame1)
        if emb1 is None:
            return {
                "success": False,
                "message": "No face detected in image 1"
            }

        emb2 = verifier.extract_face_embedding(frame2)
        if emb2 is None:
            return {
                "success": False,
                "message": "No face detected in image 2"
            }

        print(f"[DEBUG] Embedding 1 - shape: {emb1.shape}, norm: {np.linalg.norm(emb1):.4f}")
        print(f"[DEBUG] Embedding 1 - first 10 values: {emb1[:10]}")
        print(f"[DEBUG] Embedding 2 - shape: {emb2.shape}, norm: {np.linalg.norm(emb2):.4f}")
        print(f"[DEBUG] Embedding 2 - first 10 values: {emb2[:10]}")

        # Calculate distance
        distance = np.linalg.norm(emb1 - emb2)

        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        cosine_sim = dot_product / (norm_product + 1e-8)

        # Calculate cosine distance (same as used in verify-with-video)
        cosine_dist = 1.0 - cosine_sim

        # Check if embeddings are identical
        are_identical = np.allclose(emb1, emb2, rtol=1e-5, atol=1e-8)

        threshold = 0.40  # Facenet512 cosine distance threshold
        is_match = cosine_dist <= threshold

        if cosine_dist <= threshold:
            confidence = 1.0 - (cosine_dist / threshold)
        else:
            confidence = 0.0

        print(f"[DEBUG] ========== RESULTS ==========")
        print(f"[DEBUG] Euclidean Distance: {distance:.4f}")
        print(f"[DEBUG] Cosine Similarity: {cosine_sim:.4f}")
        print(f"[DEBUG] Cosine Distance: {cosine_dist:.4f}")
        print(f"[DEBUG] Embeddings Identical: {are_identical}")
        print(f"[DEBUG] Threshold: {threshold}")
        print(f"[DEBUG] Is Match: {is_match}")
        print(f"[DEBUG] Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
        print(f"[DEBUG] ===================================")

        return {
            "success": True,
            "euclidean_distance": float(distance),
            "cosine_distance": float(cosine_dist),
            "cosine_similarity": float(cosine_sim),
            "embeddings_identical": bool(are_identical),
            "threshold": threshold,
            "is_match": is_match,
            "confidence": float(confidence),
            "embedding1_norm": float(np.linalg.norm(emb1)),
            "embedding2_norm": float(np.linalg.norm(emb2)),
            "interpretation": {
                "same_person": cosine_dist < 0.35,
                "maybe_same": 0.35 <= cosine_dist < 0.45,
                "different_people": cosine_dist >= 0.45,
                "model_working": np.linalg.norm(emb1) > 5.0 and np.linalg.norm(emb2) > 5.0  # Facenet512 norms
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Debug comparison failed: {str(e)}"
        }
