from typing import List, Optional, Tuple
import numpy as np
from ..models.multi_class_svm import MultiUserKeystrokeSVM


class KeystrokeAuthenticator:
    """
    Handles user authentication using keystroke dynamics (SVM only).
    Wraps the MultiUserKeystrokeSVM model to provide a simpler interface.
    """
    
    def __init__(self, models_dir: str = "data/models"):
        self.model = MultiUserKeystrokeSVM(models_dir=models_dir)
    
    def enroll_user(self, user_id: str, features: List[List[float]]) -> bool:
        """
        Enroll a user with multiple keystroke feature vectors.
        """
        return self.model.train_user(user_id, features)
    
    def verify_user(self, user_id: str, features: List[float], threshold: float = 0.0) -> Tuple[bool, float]:
        """
        Verify if typing belongs to a specific user.
        Returns (is_match, confidence)
        """
        return self.model.verify(user_id, features, threshold)
    
    def delete_user(self, user_id: str) -> bool:
        """
        Delete trained model for a specific user.
        """
        return self.model.delete_user_model(user_id)
    
    def get_user_status(self, user_id: str) -> dict:
        """
        Get training and model existence status for a user.
        """
        return self.model.get_user_status(user_id)

    def identify_user(self, features: List[float], min_confidence: float = 0.0) -> Tuple[Optional[str], float]:
        """
        Identify which enrolled user the typing most likely belongs to.
        If confidence < min_confidence → return (None, 0.0)
        """

        if not hasattr(self.model, "models") or not self.model.models:
            return None, 0.0

        best_user: Optional[str] = None
        best_raw_conf: float = float("-inf")

        # Loop across users and keep the highest raw confidence
        for user_id, model_data in self.model.models.items():
            if not model_data.get("is_trained", False):
                continue

            _, raw_conf = self.model.verify(user_id, features)
            if raw_conf > best_raw_conf:
                best_raw_conf = raw_conf
                best_user = user_id

        if best_user is None:
            return None, 0.0

        # Convert raw SVM distance into normalized sigmoid confidence
        normalized_conf = float(1 / (1 + np.exp(-best_raw_conf)))

        # If confidence too low → treat as unknown user
        if normalized_conf < min_confidence:
            return None, 0.0

        return best_user, normalized_conf

    def list_users(self) -> List[str]:
        """
        List all registered user IDs in memory.
        """
        return list(self.model.models.keys())

    def identify_topk(self, features: List[float], k: int = 3) -> List[Tuple[str, float]]:
        """
        Return top-K most likely users ranked by confidence.
        Each confidence is normalized to [0,1].
        """
        candidates: List[Tuple[str, float]] = []

        for user_id, model_data in self.model.models.items():
            if not model_data.get("is_trained", False):
                continue

            _, raw_conf = self.model.verify(user_id, features)
            norm_conf = float(1 / (1 + np.exp(-raw_conf)))   # sigmoid normalized
            candidates.append((user_id, norm_conf))

        if not candidates:
            return []

        # Sort highest → lowest confidence
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:max(0, k)]
