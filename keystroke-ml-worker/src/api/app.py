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


def _chunk_keystrokes(keystrokes: List[Dict[str, Any]], window_size: int, step: int) -> List[List[Dict[str, Any]]]:
    if window_size <= 0:
        return []
    if step <= 0:
        step = window_size
    if len(keystrokes) < window_size:
        return [keystrokes]
    chunks: List[List[Dict[str, Any]]] = []
    for start in range(0, len(keystrokes) - window_size + 1, step):
        chunks.append(keystrokes[start : start + window_size])
    if not chunks:
        chunks.append(keystrokes)
    return chunks


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _extract_feature_matrix(keystrokes: List[Dict[str, Any]], window_size: int, step: int) -> np.ndarray:
    chunks = _chunk_keystrokes(keystrokes, window_size=window_size, step=step)
    vectors: List[np.ndarray] = []
    for c in chunks:
        raw = feature_extractor.extract_features(c)
        vectors.append(np.array(raw, dtype=float))
    if not vectors:
        raise ValueError("No keystroke features extracted")
    return np.vstack(vectors)


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

        keystrokes = [k.dict() for k in data.keystrokes]
        X = _extract_feature_matrix(keystrokes, window_size=80, step=40)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        Xn = (X - mu) / sigma
        centroid = Xn.mean(axis=0)
        sims = np.array([_cosine_similarity(v, centroid) for v in Xn], dtype=float)
        thr = float(np.mean(sims) - 2.0 * np.std(sims))
        thr = float(max(0.55, min(0.98, thr)))

        model_data = {
            "user_id": data.user_id,
            "version": 2,
            "centroid": centroid.tolist(),
            "scaler": {"mean": mu.tolist(), "std": sigma.tolist()},
            "threshold": thr,
            "metadata": {
                "keystroke_count": len(data.keystrokes),
                "feature_vector_length": int(X.shape[1]),
                "windows": int(X.shape[0]),
                "window_size": 80,
                "step": 40,
            },
        }

        storage.save_model(data.user_id, model_data)

        return {
            "success": True,
            "message": "Keystroke enrollment completed",
            "features": int(X.shape[1]),
        }

    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/verify")
async def verify_keystroke(data: KeystrokeRequest):
    try:
        model_data = storage.load_model(data.user_id)
        if not model_data:
            raise HTTPException(404, "User not enrolled")

        keystrokes = [k.dict() for k in data.keystrokes]

        if "centroid" in model_data and isinstance(model_data.get("scaler"), dict):
            X = _extract_feature_matrix(
                keystrokes,
                window_size=int(model_data.get("metadata", {}).get("window_size", 80)),
                step=int(model_data.get("metadata", {}).get("step", 40)),
            )

            mu = np.array(model_data["scaler"]["mean"], dtype=float)
            sigma = np.array(model_data["scaler"]["std"], dtype=float)
            sigma = np.where(sigma == 0, 1.0, sigma)

            if X.shape[1] != mu.shape[0]:
                raise HTTPException(
                    400,
                    f"Feature length mismatch: current={int(X.shape[1])}, enrolled={int(mu.shape[0])}. "
                    f"Please re-enroll your typing profile to fix this issue."
                )

            Xn = (X - mu) / sigma
            centroid = np.array(model_data["centroid"], dtype=float)
            sims = np.array([_cosine_similarity(v, centroid) for v in Xn], dtype=float)
            similarity = float(np.median(sims))
            threshold = float(model_data.get("threshold", 0.75))
            authenticated = similarity >= threshold
            confidence = max(0.0, min(1.0, (similarity - threshold) / (1 - threshold)))

            return {
                "authenticated": authenticated,
                "similarity": similarity,
                "confidence": confidence,
                "threshold": threshold,
                "user_id": data.user_id,
            }

        raw_features = feature_extractor.extract_features(keystrokes)
        raw_features = np.array(raw_features, dtype=float)
        stored_features = np.array(model_data["features"], dtype=float)

        if len(raw_features) != len(stored_features):
            raise HTTPException(
                400,
                f"Feature length mismatch: current={len(raw_features)}, enrolled={len(stored_features)}. "
                f"Please re-enroll your typing profile to fix this issue."
            )

        mean = model_data["scaler"]["mean"]
        std = model_data["scaler"]["std"] or 1.0
        normalized_features = (raw_features - mean) / std

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


@app.get("/check-model/{user_id}")
async def check_model(user_id: str):
    """Check if a keystroke model exists for a user"""
    exists = storage.model_exists(user_id)
    return {"exists": exists, "user_id": user_id}
