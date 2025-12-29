# server.py
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from .audio_detector import listen_and_detect, vad
import uvicorn
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(title="Voice ML Worker")

# allow your front-end (change origins to restrict)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production lock this to your front-end origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared storage for latest detection
_latest_lock = threading.Lock()
_latest_result: Dict[str, Any] = {
    "speech_probability": 0.0,
    "issues": [],
    "timestamp": 0.0,
    "status": "stopped"
}

# background thread control
_stop_event = threading.Event()
_detector_thread = None

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def detector_thread_fn():
    global _latest_result
    gen = listen_and_detect()
    _latest_result["status"] = "running"
    for res in gen:
        with _latest_lock:
            _latest_result.update(res)
        if _stop_event.is_set():
            break
    _latest_result["status"] = "stopped"

# --- API models ---
class ThresholdsModel(BaseModel):
    whisper: float | None = Field(None, ge=0.0, le=1.0)
    low: float | None = Field(None, ge=0.0, le=1.0)
    normal: float | None = Field(None, ge=0.0, le=1.0)
    continuous: float | None = Field(None, ge=0.0, le=1.0)

class ConfigUpdateModel(BaseModel):
    thresholds: ThresholdsModel | None = None
    sustain_ms: int | None = Field(None, ge=0)
    history_window: int | None = Field(None, ge=1)

@app.on_event("startup")
def startup_event():
    global _detector_thread, _stop_event
    _stop_event.clear()
    # start detection thread
    _detector_thread = threading.Thread(target=detector_thread_fn, daemon=True)
    _detector_thread.start()

@app.on_event("shutdown")
def shutdown_event():
    global _stop_event, _detector_thread
    _stop_event.set()
    if _detector_thread:
        _detector_thread.join(timeout=2.0)

@app.get("/voice-status")
def voice_status():
    """
    Returns the latest detection as JSON.
    Poll this endpoint periodically from front-end (e.g. every 300-500 ms).
    """
    with _latest_lock:
        # Convert numpy types to Python native types for JSON serialization
        return convert_numpy_types(_latest_result)

@app.get("/voice/health")
def voice_health():
    """Simple health and model readiness check."""
    try:
        cfg = vad.get_config()
        with _latest_lock:
            status = _latest_result.get("status", "unknown")
        return convert_numpy_types({"status": status, "model": "silero_vad", "config": cfg})
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/voice/config")
def get_config():
    return convert_numpy_types(vad.get_config())

@app.post("/voice/config")
def update_config(payload: ConfigUpdateModel):
    data = payload.dict(exclude_none=True)
    thresholds = data.get("thresholds")
    if thresholds is not None:
        thresholds = {k: v for k, v in thresholds.items() if v is not None}
    vad.update_config(
        thresholds=thresholds,
        sustain_ms=data.get("sustain_ms"),
        history_window=data.get("history_window"),
    )
    return convert_numpy_types({"ok": True, "config": vad.get_config()})

@app.post("/voice/start")
def start_detection():
    global _detector_thread, _stop_event
    if _detector_thread and _detector_thread.is_alive():
        return {"status": "already_running"}
    _stop_event.clear()
    _detector_thread = threading.Thread(target=detector_thread_fn, daemon=True)
    _detector_thread.start()
    return {"status": "started"}

@app.post("/voice/stop")
def stop_detection():
    global _stop_event
    _stop_event.set()
    return {"status": "stopping"}

if __name__ == "__main__":
    # optional: run standalone for debugging
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=False)
