from deepface import DeepFace
import numpy as np
import cv2
from typing import Optional, List, Tuple
import os


class FaceVerifier:
    """
    Face verification using DeepFace (easier to install on Windows).
    Uses VGG-Face model for face recognition.
    """

    def __init__(self, model_name="VGG-Face"):
        self.reference_embeddings = {}  # {user_id: [embedding1, embedding2, ...]}
        self.model_name = model_name
        # Preload model
        DeepFace.build_model(model_name)

    def extract_encodings_from_video(self, video_path: str, max_frames: int = 10) -> List[np.ndarray]:
        """
        Extract face embeddings from video.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to sample

        Returns:
            List of face embeddings
        """
        cap = cv2.VideoCapture(video_path)
        embeddings = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return embeddings

        interval = max(1, total_frames // max_frames)

        while len(embeddings) < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                try:
                    # DeepFace expects RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Get embedding
                    embedding_objs = DeepFace.represent(
                        img_path=rgb_frame,
                        model_name=self.model_name,
                        enforce_detection=True,
                        detector_backend="opencv"
                    )

                    if len(embedding_objs) > 0:
                        embedding = np.array(embedding_objs[0]["embedding"])
                        embeddings.append(embedding)
                except Exception:
                    # Skip frames where face detection fails
                    pass

            frame_count += 1

        cap.release()
        return embeddings

    def extract_encoding_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from single frame.

        Args:
            frame: OpenCV image in BGR format

        Returns:
            Face embedding or None if no face found
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embedding_objs = DeepFace.represent(
                img_path=rgb_frame,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend="opencv"
            )

            if len(embedding_objs) > 0:
                return np.array(embedding_objs[0]["embedding"])
        except Exception:
            return None

        return None

    def enroll_user(self, user_id: str, video_path: str) -> Tuple[bool, str]:
        """
        Enroll user by extracting face embeddings from selfie video.

        Args:
            user_id: Unique identifier for the user
            video_path: Path to selfie video file

        Returns:
            Tuple of (success: bool, message: str)
        """
        embeddings = self.extract_encodings_from_video(video_path)

        if len(embeddings) < 3:
            return False, f"Could not extract enough face samples. Found {len(embeddings)}, need at least 3."

        self.reference_embeddings[user_id] = embeddings
        return True, f"Successfully enrolled with {len(embeddings)} face samples"

    def verify_face(self, user_id: str, frame: np.ndarray, threshold: float = 0.4) -> Tuple[bool, float, str]:
        """
        Verify if face in frame matches enrolled user.

        Args:
            user_id: User ID to verify against
            frame: OpenCV image in BGR format
            threshold: Distance threshold (lower = stricter). For cosine: 0.4 is standard

        Returns:
            Tuple of (is_match: bool, confidence: float, message: str)
        """
        if user_id not in self.reference_embeddings:
            return False, 0.0, "User not enrolled"

        current_embedding = self.extract_encoding_from_frame(frame)
        if current_embedding is None:
            return False, 0.0, "No face detected in frame"

        reference_embeddings = self.reference_embeddings[user_id]

        # Calculate cosine distances
        distances = []
        for ref_embedding in reference_embeddings:
            # Cosine similarity
            cosine_sim = np.dot(current_embedding, ref_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(ref_embedding)
            )
            # Convert to distance (0 = identical, 1 = completely different)
            distance = 1 - cosine_sim
            distances.append(distance)

        min_distance = float(np.min(distances))

        # Convert distance to confidence
        confidence = max(0.0, 1.0 - min_distance)
        is_match = min_distance <= threshold

        message = "Face verified" if is_match else f"Face does not match (distance: {min_distance:.3f})"

        return is_match, confidence, message

    def is_enrolled(self, user_id: str) -> bool:
        """Check if user is enrolled."""
        return user_id in self.reference_embeddings

    def unenroll_user(self, user_id: str) -> bool:
        """Remove user's enrollment."""
        if user_id in self.reference_embeddings:
            del self.reference_embeddings[user_id]
            return True
        return False
