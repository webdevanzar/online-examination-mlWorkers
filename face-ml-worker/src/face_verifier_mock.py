import numpy as np
import cv2
from typing import Optional, List, Tuple


class FaceVerifier:
    """
    Mock FaceVerifier - placeholder for when face_recognition is not available
    """

    def __init__(self):
        self.reference_encodings = {}  # {user_id: [encoding1, encoding2, ...]}

    def extract_encodings_from_video(self, video_path: str, max_frames: int = 10) -> List[np.ndarray]:
        """Mock implementation - returns dummy encodings"""
        return [np.random.rand(128) for _ in range(min(5, max_frames))]

    def extract_encoding_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Mock implementation - returns dummy encoding"""
        return np.random.rand(128)

    def enroll_user(self, user_id: str, video_path: str) -> Tuple[bool, str]:
        """Mock enrollment"""
        encodings = self.extract_encodings_from_video(video_path)
        self.reference_encodings[user_id] = encodings
        return True, f"Mock enrolled with {len(encodings)} face samples"

    def verify_face(self, user_id: str, frame: np.ndarray, threshold: float = 0.6) -> Tuple[bool, float, str]:
        """Mock verification"""
        if user_id not in self.reference_encodings:
            return False, 0.0, "User not enrolled"

        # Mock verification - always returns true with high confidence
        confidence = 0.95
        is_match = True
        message = "Mock face verified"

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
