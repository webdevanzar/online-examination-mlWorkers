import face_recognition
import numpy as np
import cv2
from typing import Optional, List, Tuple
# import tempfile
# import os


class FaceVerifier:
    """
    Handles face encoding and verification using face_recognition library.
    Compares live frames against reference video for identity verification.
    """

    def __init__(self):
        self.reference_encodings = {}  # {user_id: [encoding1, encoding2, ...]}

    def extract_encodings_from_video(self, video_path: str, max_frames: int = 10) -> List[np.ndarray]:
        """
        Extract face encodings from video - samples max_frames frames evenly distributed.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to sample (default 10)

        Returns:
            List of face encodings (128-dimensional vectors)
        """
        cap = cv2.VideoCapture(video_path)
        encodings = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return encodings

        interval = max(1, total_frames // max_frames)

        while len(encodings) < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                # Convert BGR to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame)
                if len(face_encodings) > 0:
                    encodings.append(face_encodings[0])

            frame_count += 1

        cap.release()
        return encodings

    def extract_encoding_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face encoding from single frame (BGR format).

        Args:
            frame: OpenCV image in BGR format

        Returns:
            Face encoding (128-dimensional vector) or None if no face found
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame)
        return face_encodings[0] if len(face_encodings) > 0 else None

    def enroll_user(self, user_id: str, video_path: str) -> Tuple[bool, str]:
        """
        Enroll user by extracting face encodings from selfie video.

        Args:
            user_id: Unique identifier for the user
            video_path: Path to selfie video file

        Returns:
            Tuple of (success: bool, message: str)
        """
        encodings = self.extract_encodings_from_video(video_path)

        if len(encodings) < 3:  # Need at least 3 good encodings
            return False, f"Could not extract enough face samples. Found {len(encodings)}, need at least 3."

        self.reference_encodings[user_id] = encodings
        return True, f"Successfully enrolled with {len(encodings)} face samples"

    def verify_face(self, user_id: str, frame: np.ndarray, threshold: float = 0.6) -> Tuple[bool, float, str]:
        """
        Verify if face in frame matches enrolled user.

        Args:
            user_id: User ID to verify against
            frame: OpenCV image in BGR format
            threshold: Distance threshold (lower = stricter). 0.6 is standard, 0.5 is strict

        Returns:
            Tuple of (is_match: bool, confidence: float, message: str)
        """
        if user_id not in self.reference_encodings:
            return False, 0.0, "User not enrolled"

        current_encoding = self.extract_encoding_from_frame(frame)
        if current_encoding is None:
            return False, 0.0, "No face detected in frame"

        reference_encodings = self.reference_encodings[user_id]
        distances = face_recognition.face_distance(reference_encodings, current_encoding)
        min_distance = float(np.min(distances))

        # Convert distance to confidence (0-1 scale)
        # Lower distance = higher confidence
        confidence = max(0.0, 1.0 - min_distance)
        is_match = min_distance <= threshold

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
