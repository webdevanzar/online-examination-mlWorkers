import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


class MultiUserKeystrokeSVM:
    """
    Multi-user keystroke authentication using One-Class SVM.
    Each user has their own model and scaler.
    """

    def __init__(self, models_dir: str = "data/models", nu: float = 0.05, gamma: float = 0.5):
        """
        Initialize the multi-user SVM system.

        Args:
            models_dir: directory to store/load user models
            nu: expected proportion of outliers (lower = stricter)
            gamma: kernel coefficient (higher = tighter fit)
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.nu = nu
        self.gamma = gamma
        self.models: Dict[str, dict] = {}  # {user_id: {"model": ..., "scaler": ..., "is_trained": bool}}

    def _get_model_path(self, user_id: str) -> Path:
        return self.models_dir / f"{user_id}_keystroke_svm.joblib"

    def add_user(self, user_id: str) -> bool:
        if user_id in self.models:
            return False

        self.models[user_id] = {
            "model": OneClassSVM(nu=self.nu, gamma=self.gamma),
            "scaler": StandardScaler(),
            "is_trained": False
        }
        return True

    def train_user(self, user_id: str, samples: List[List[float]], save: bool = True) -> bool:
        """
        Train a user-specific One-Class SVM using multiple feature vectors.
        """
        if user_id not in self.models:
            self.add_user(user_id)

        if not samples or len(samples) < 1:
            # Require at least one sample to fit scaler/model
            raise ValueError("At least 1 enrollment sample is required")

        try:
            X = np.array(samples)
            model_data = self.models[user_id]

            X_scaled = model_data["scaler"].fit_transform(X)
            model_data["model"].fit(X_scaled)
            model_data["is_trained"] = True

            if save:
                self.save_user_model(user_id)

            return True
        except Exception as e:
            print(f"Error training user {user_id}: {e}")
            return False

    def verify(self, user_id: str, sample: List[float], threshold: float = 0.0) -> Tuple[bool, float]:
        """
        Verify if the given typing sample belongs to the given user.
        Returns (is_match, confidence)
        """
        if user_id not in self.models or not self.models[user_id]["is_trained"]:
            return False, 0.0

        model_data = self.models[user_id]
        X = np.array([sample])

        try:
            X_scaled = model_data["scaler"].transform(X)
            score = float(model_data["model"].decision_function(X_scaled)[0])

            # Combine SVM decision with distance-to-mean (after scaling, mean≈0)
            # This yields better separation for synthetic tests and noisy data.
            z = X_scaled[0]
            dist = float(np.linalg.norm(z))  # distance from training center in z-space
            dist_weight = 1.0  # tunable
            confidence = score - dist_weight * dist

            # Classification based on combined confidence for better rejection
            match = confidence >= (threshold if threshold != 0 else 0.0)

            return match, confidence
        except Exception as e:
            print(f"Error verifying sample for {user_id}: {e}")
            return False, 0.0

    def save_user_model(self, user_id: str) -> bool:
        try:
            joblib.dump(self.models[user_id], self._get_model_path(user_id))
            return True
        except Exception as e:
            print(f"Error saving model for {user_id}: {e}")
            return False

    def load_user_model(self, user_id: str) -> bool:
        path = self._get_model_path(user_id)
        if not path.exists():
            return False
        try:
            self.models[user_id] = joblib.load(path)
            return True
        except Exception as e:
            print(f"Error loading model for {user_id}: {e}")
            return False

    def delete_user_model(self, user_id: str) -> bool:
        path = self._get_model_path(user_id)

        if user_id in self.models:
            del self.models[user_id]

        try:
            if path.exists():
                path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting model for {user_id}: {e}")
            return False

    def get_user_status(self, user_id: str) -> dict:
        if user_id not in self.models:
            return {"exists": False, "trained": False}

        return {
            "exists": True,
            "trained": self.models[user_id]["is_trained"],
            "model_type": "OneClassSVM",
            "parameters": {"nu": self.nu, "gamma": self.gamma}
        }

    def list_users(self) -> List[str]:
        """
        Return all known user_ids (trained or untrained).
        """
        return list(self.models.keys())

    def rank_users_by_confidence(self, sample: List[float]) -> List[Tuple[str, float]]:
        """
        Compute confidence for the sample against all trained users and return a
        descending list of (user_id, confidence).
        """
        results: List[Tuple[str, float]] = []
        if not self.models:
            return results

        for user_id, data in self.models.items():
            if not data.get("is_trained", False):
                continue
            try:
                _, conf = self.verify(user_id, sample)
                results.append((user_id, conf))
            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results
 