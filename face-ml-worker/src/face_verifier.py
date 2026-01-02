import cv2
import numpy as np
from keras_facenet import FaceNet
from typing import Optional, List, Tuple


class FaceVerifier:
    """
    Face enrollment & verification using OpenCV DNN + FaceNet.
    No dlib or MediaPipe required - works reliably on Windows!
    """

    def __init__(self):
        self.reference_encodings: dict[str, List[np.ndarray]] = {}

        # Initialize OpenCV DNN Face Detector (built-in, no extra downloads needed)
        # Using Caffe model - lightweight and fast
        try:
            # Try to use DNN face detector (more accurate)
            model_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(model_file)
            print("✓ OpenCV Haar Cascade face detector loaded")
        except Exception as e:
            print(f"Warning: Could not load face detector: {e}")
            self.face_cascade = None

        # Initialize FaceNet for embeddings
        print("Loading FaceNet model...")
        self.facenet = FaceNet()
        print("✓ FaceNet model loaded")

    def _auto_rotate(self, frame: np.ndarray) -> np.ndarray:
        """Rotate portrait frames to landscape."""
        h, w = frame.shape[:2]
        if h > w:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def _ensure_uint8(self, frame: np.ndarray) -> np.ndarray:
        """Ensure frame is uint8 type."""
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        return frame

    def extract_face_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from a single frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            128-dimensional face embedding or None if no face detected
        """
        if frame is None or frame.size == 0:
            print("[EXTRACT] Frame is None or empty")
            return None

        # Auto-rotate if needed
        frame = self._auto_rotate(frame)
        frame = self._ensure_uint8(frame)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces with OpenCV Haar Cascade
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            print(f"[EXTRACT] No faces detected in frame of shape {frame.shape}")
            return None

        print(f"[EXTRACT] Detected {len(faces)} face(s) in frame")

        # Get first detected face (x, y, w, h)
        x, y, w, h = faces[0]
        print(f"[EXTRACT] Face bbox: x={x}, y={y}, w={w}, h={h}")

        # Add padding
        padding = int(max(w, h) * 0.2)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        # Crop face from original BGR frame
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            print("[EXTRACT] Face crop is empty after padding")
            return None

        print(f"[EXTRACT] Face crop shape: {face_crop.shape}")

        # Convert to RGB for FaceNet
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Resize to 160x160 (FaceNet input size)
        face_resized = cv2.resize(face_rgb, (160, 160))

        # Normalize pixel values
        face_normalized = face_resized.astype(np.float32) / 255.0

        # Get embedding from FaceNet
        # FaceNet expects batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        embedding = self.facenet.embeddings(face_batch)

        emb = embedding[0]  # Return first (and only) embedding
        print(f"[EXTRACT] Embedding extracted: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
        print(f"[EXTRACT] Embedding sample (first 5 values): {emb[:5]}")

        return emb

    def extract_encodings_from_video(
        self,
        video_path: str,
        max_frames: int = 30
    ) -> List[np.ndarray]:
        """
        Extract face embeddings from video.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process

        Returns:
            List of face embeddings
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, total_frames // max_frames)

        encodings: List[np.ndarray] = []
        frame_idx = 0
        faces_detected = 0

        print(f"Video info - Frames: {total_frames}, FPS: {fps}")
        print(f"Processing every {interval} frame(s)")

        while cap.isOpened() and len(encodings) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval != 0:
                frame_idx += 1
                continue

            # Try to extract embedding
            embedding = self.extract_face_embedding(frame)

            if embedding is not None:
                encodings.append(embedding)
                faces_detected += 1
                print(f"Frame {frame_idx}: ✓ Face detected, embedding extracted ({len(encodings)} total)")
            else:
                print(f"Frame {frame_idx}: ✗ No face detected")

            frame_idx += 1

        cap.release()

        print(f"\n=== Summary ===")
        print(f"Frames processed: {frame_idx}")
        print(f"Faces detected: {faces_detected}")
        print(f"Embeddings extracted: {len(encodings)}")

        return encodings

    def enroll_user(
        self,
        user_id: str,
        video_path: str,
        min_samples: int = 2
    ) -> Tuple[bool, str]:
        """
        Enroll user by extracting face embeddings from selfie video.

        Args:
            user_id: Unique user identifier
            video_path: Path to selfie video
            min_samples: Minimum number of face samples required

        Returns:
            Tuple of (success, message)
        """
        print(f"\n=== Enrolling user: {user_id} ===")

        try:
            embeddings = self.extract_encodings_from_video(video_path)

            if len(embeddings) < min_samples:
                return False, (
                    f"Enrollment failed. Found {len(embeddings)} face samples, need at least {min_samples}.\n"
                    "Tips:\n"
                    "- Ensure face is clearly visible and well-lit\n"
                    "- Face should be looking at camera\n"
                    "- Video should be 3-5 seconds long\n"
                    "- Avoid motion blur"
                )

            self.reference_encodings[user_id] = embeddings
            print(f"✓ User {user_id} enrolled successfully with {len(embeddings)} samples\n")
            return True, f"Successfully enrolled with {len(embeddings)} face samples"

        except Exception as e:
            print(f"✗ Enrollment failed: {str(e)}")
            return False, f"Enrollment error: {str(e)}"

    def verify_face(
        self,
        user_id: str,
        frame: np.ndarray,
        threshold: float = 0.4
    ) -> Tuple[bool, float, str]:
        """
        Verify if face in frame matches enrolled user.

        Args:
            user_id: User ID to verify
            frame: BGR frame from OpenCV
            threshold: Distance threshold (lower = stricter, default 0.4)
                      FaceNet typical: same person 0.0-0.4, different 0.6+

        Returns:
            Tuple of (is_match, confidence, message)
        """
        if user_id not in self.reference_encodings:
            return False, 0.0, "User not enrolled"

        # Extract embedding from current frame
        current_embedding = self.extract_face_embedding(frame)

        if current_embedding is None:
            return False, 0.0, "No face detected in frame"

        # Compare with stored embeddings
        reference_embeddings = self.reference_encodings[user_id]

        # Calculate Euclidean distances
        distances = []
        for i, ref_embedding in enumerate(reference_embeddings):
            distance = np.linalg.norm(ref_embedding - current_embedding)
            distances.append(distance)
            print(f"Distance to enrolled sample {i+1}: {distance:.4f}")

        min_distance = float(np.min(distances))
        avg_distance = float(np.mean(distances))

        # Correct confidence calculation
        # If distance is 0.0 = 100% confidence
        # If distance is at threshold = 0% confidence
        if min_distance <= threshold:
            confidence = 1.0 - (min_distance / threshold)
        else:
            confidence = 0.0

        is_match = min_distance <= threshold

        print(f"[VERIFY] Min distance: {min_distance:.4f}, Avg: {avg_distance:.4f}, Threshold: {threshold}")
        print(f"[VERIFY] Is Match: {is_match}, Confidence: {confidence:.4f} ({confidence*100:.1f}%)")

        message = "Face verified" if is_match else f"Face does not match (distance: {min_distance:.3f})"

        return is_match, confidence, message

    def is_enrolled(self, user_id: str) -> bool:
        """Check if user is enrolled."""
        return user_id in self.reference_encodings

    def unenroll_user(self, user_id: str) -> bool:
        """Remove user's enrollment."""
        if user_id in self.reference_encodings:
            del self.reference_encodings[user_id]
            return True
        return False

    def __del__(self):
        """Cleanup resources."""
        pass  # OpenCV Haar Cascade doesn't require explicit cleanup
