# ml-workers/keystroke-ml-worker/src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import os

# Import the storage and feature extractor
from src.storage.keystroke_storage import KeystrokeStorage
from src.preprocessing.feature_extractor import FeatureExtractor

app = FastAPI()
storage = KeystrokeStorage(
    base_dir=os.path.join(os.path.dirname(__file__), "..", "models", "keystroke_models")
)
feature_extractor = FeatureExtractor()


class KeystrokeEvent(BaseModel):
    key: str
    event: str  # 'keydown' or 'keyup'
    timestamp: float


class KeystrokeRequest(BaseModel):
    user_id: str
    keystrokes: List[KeystrokeEvent]
    is_enrollment: bool = False
    metadata: Dict[str, Any] = {}


@app.post("/enroll")
async def enroll_keystroke(data: KeystrokeRequest):
    try:
        if len(data.keystrokes) < 150:
            raise HTTPException(
                status_code=400,
                detail="At least 150 keystrokes required for enrollment",
            )

        # Extract features
        raw_features = feature_extractor.extract_features(
            [k.dict() for k in data.keystrokes]
        )

        raw_features = np.array(raw_features, dtype=float)

        # ✅ Z-score normalization
        mean = raw_features.mean()
        std = raw_features.std() or 1.0
        normalized_features = (raw_features - mean) / std

        model_data = {
            "user_id": data.user_id,
            "features": normalized_features.tolist(),
            "scaler": {"mean": mean, "std": std},
            "threshold": 0.85,  # ✅ cosine similarity baseline
            "metadata": {
                "keystroke_count": len(data.keystrokes),
                "feature_vector_length": len(raw_features),
            },
        }

        storage.save_model(data.user_id, model_data)

        return {
            "success": True,
            "message": "Keystroke enrollment completed",
            "features": len(raw_features),
        }

    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/verify")
async def verify_keystroke(data: KeystrokeRequest):
    try:
        model_data = storage.load_model(data.user_id)
        if not model_data:
            raise HTTPException(404, "User not enrolled")

        # Extract features
        raw_features = feature_extractor.extract_features(
            [k.dict() for k in data.keystrokes]
        )

        raw_features = np.array(raw_features, dtype=float)
        stored_features = np.array(model_data["features"], dtype=float)

        # ✅ Feature length check (CRITICAL)
        if len(raw_features) != len(stored_features):
            raise HTTPException(
                400, "Feature length mismatch – inconsistent typing sample"
            )

        # ✅ Normalize using stored scaler
        mean = model_data["scaler"]["mean"]
        std = model_data["scaler"]["std"] or 1.0
        normalized_features = (raw_features - mean) / std

        # ✅ Cosine similarity
        similarity = float(
            np.dot(stored_features, normalized_features)
            / (
                np.linalg.norm(stored_features) * np.linalg.norm(normalized_features)
                + 1e-8
            )
        )

        threshold = model_data.get("threshold", 0.85)
        authenticated = similarity >= threshold

        confidence = max(0.0, min(1.0, (similarity - threshold) / (1 - threshold)))

        return {
            "authenticated": authenticated,
            "similarity": similarity,
            "confidence": confidence,
            "threshold": threshold,
            "user_id": data.user_id,
        }

    except Exception as e:
        raise HTTPException(400, str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "storage_path": str(storage.base_dir.absolute())}
