from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
from datetime import datetime

from authentication.authenticator import KeystrokeAuthenticator
from preprocessing.feature_extractor import FeatureExtractor

app = FastAPI(title="Keystroke Authentication API")
auth = KeystrokeAuthenticator(models_dir="data/models")

class KeystrokeSample(BaseModel):
    user_id: str | None = None   # user_id optional for identification
    keystrokes: List[Dict]       # {key, event, timestamp}
    is_enrollment: bool = False

class AuthResponse(BaseModel):
    authenticated: bool
    confidence: float
    session_id: str
    timestamp: str


@app.post("/authenticate", response_model=AuthResponse)
async def authenticate(sample: KeystrokeSample):
    """
    SVM authentication: verifies if typing pattern matches the given user_id.
    """
    try:
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample.keystrokes)

        # Enrollment call
        if sample.is_enrollment:
            auth.enroll_user(sample.user_id, [features])
            return AuthResponse(
                authenticated=True,
                confidence=1.0,
                session_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat()
            )

        # Verification call
        is_match, confidence = auth.verify_user(sample.user_id, features)
        return AuthResponse(
            authenticated=is_match,
            confidence=confidence,
            session_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class IdentifyTopKRequest(BaseModel):
    keystrokes: List[Dict]
    k: Optional[int] = 3


@app.post("/identify/topk")
async def identify_topk(req: IdentifyTopKRequest):
    """
    Return top-k users ranked by confidence.
    """
    try:
        extractor = FeatureExtractor()
        features = extractor.extract_features(req.keystrokes)
        ranked = auth.identify_topk(features, k=req.k or 3)
        return {
            "topk": [
                {"user_id": user_id, "confidence": conf}
                for user_id, conf in ranked
            ],
            "k": req.k or 3,
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/users")
async def list_users():
    """
    List all known users (trained or untrained) and their status.
    """
    try:
        users = auth.list_users()
        return {
            "users": [
                {"user_id": u, **auth.get_user_status(u)} for u in users
            ],
            "count": len(users),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/identify")
async def identify(sample: KeystrokeSample):
    """
    Identify the user based purely on typing (no user_id needed).
    """
    try:
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample.keystrokes)
        user_id, confidence = auth.identify_user(features)

        return {
            "identified_user": user_id,
            "confidence": confidence,
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/users/{user_id}/stats")
async def get_user_stats(user_id: str):
    """
    Check whether a user model exists and is trained.
    """
    return auth.get_user_status(user_id)
