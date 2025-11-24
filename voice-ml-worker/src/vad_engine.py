import torch
import numpy as np
from collections import deque
import time

class VoiceVAD:
    def __init__(self, device: str = "cpu"):
        self.device = device
        # configuration (can be updated at runtime)
        self.thresholds = {
            "whisper": 0.10,
            "low": 0.25,
            "normal": 0.45,
            "continuous": 0.70,
        }
        self.sustain_ms = 2000  # speech must be above 'normal' for this long to count as sustained
        self.history_window = 200  # last N chunks history (approx N*32ms)
        self._prob_history = deque(maxlen=self.history_window)
        self._ts_history = deque(maxlen=self.history_window)
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            (self.get_speech_ts,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils

            self.model.to(self.device)
            self.model.eval()
            print("✅ Silero VAD model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading VAD model: {e}")
            raise

    def detect_cheating(self, audio_chunk, sample_rate=16000):
        try:
            # Convert to numpy array and ensure float32
            if isinstance(audio_chunk, torch.Tensor):
                audio_chunk = audio_chunk.cpu().numpy()
            
            audio_chunk = audio_chunk.astype(np.float32).flatten()
            
            # CRITICAL: Must be exactly 512 samples
            target_samples = 512
            if len(audio_chunk) != target_samples:
                if len(audio_chunk) > target_samples:
                    audio_chunk = audio_chunk[:target_samples]
                else:
                    padding = np.zeros(target_samples - len(audio_chunk), dtype=np.float32)
                    audio_chunk = np.concatenate((audio_chunk, padding))

            # Enhanced preprocessing for ultra-low volume signals
            max_val = np.max(np.abs(audio_chunk))
            rms = np.sqrt(np.mean(audio_chunk**2))
            
            # Apply additional normalization if still too quiet
            if rms < 0.01:
                # Boost to reasonable level while avoiding clipping
                target_rms = 0.1
                if rms > 0:
                    additional_gain = min(1000.0, target_rms / rms)
                    audio_chunk = audio_chunk * additional_gain
            
            # Final clipping to safe range
            audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
            
            # Final RMS after all processing
            final_rms = np.sqrt(np.mean(audio_chunk**2))
            
            # Convert to tensor and process
            tensor = torch.from_numpy(audio_chunk).to(self.device)
            
            with torch.no_grad():
                prob = self.model(tensor, sample_rate).item()

            # Append history for sustained detection
            now = time.time() * 1000.0  # ms
            self._prob_history.append(prob)
            self._ts_history.append(now)

            # Fraud rule signals and issues
            t = self.thresholds
            issues = []
            flags = []
            sustained_ms = 0.0

            # Clipping detection
            clipping = bool(max_val >= 0.99)
            if clipping:
                issues.append("Clipping or very loud signal")
                flags.append("clipping")

            # Whisper/low/normal/continuous classification
            if prob > t["continuous"]:
                issues.append("Very high speech probability")
            elif prob > t["normal"]:
                issues.append("Normal speech detected")
            elif prob > t["low"]:
                issues.append("Low voice detected")
            elif prob > t["whisper"]:
                issues.append("Whisper detected")

            # Sustained speech: how long have we stayed above normal
            # walk backward until below normal
            for p, ts in zip(reversed(self._prob_history), reversed(self._ts_history)):
                if p >= t["normal"]:
                    if sustained_ms == 0.0:
                        last_ts = ts
                        sustained_ms = 0.0
                    else:
                        sustained_ms = max(sustained_ms, last_ts - ts)
                        last_ts = ts
                else:
                    break
            if sustained_ms >= self.sustain_ms:
                issues.append("Sustained talking detected")
                flags.append("sustained_speech")

            # Background noise suspicion: loud but low probability
            if final_rms > 0.1 and prob < t["whisper"]:
                issues.append("Loud background noise")
                flags.append("noise")

            # Risk score (0-1): weighted combination
            risk = 0.0
            risk += min(1.0, max(0.0, (prob - t["low"]) / (1.0 - t["low"])) * 0.6)
            risk += 0.2 if sustained_ms >= self.sustain_ms else 0.0
            risk += 0.2 if clipping else 0.0
            risk = float(min(1.0, risk))

            return {
                "speech_probability": prob, 
                "issues": issues,
                "audio_max_val": float(max_val),
                "audio_final_rms": float(final_rms),
                "samples_processed": len(audio_chunk),
                "risk_score": risk,
                "flags": flags,
                "clipping": clipping,
                "sustained_speech_ms": float(sustained_ms),
                "history_len": len(self._prob_history),
            }

        except Exception as e:
            return {
                "speech_probability": 0.0, 
                "issues": [f"VAD processing error: {str(e)}"],
                "audio_max_val": 0.0,
                "audio_final_rms": 0.0,
                "samples_processed": 0,
                "risk_score": 0.0,
                "flags": ["error"],
                "clipping": False,
                "sustained_speech_ms": 0.0,
                "history_len": 0,
            }

    # --- Config management ---
    def get_config(self):
        return {
            "device": self.device,
            "thresholds": dict(self.thresholds),
            "sustain_ms": self.sustain_ms,
            "history_window": self.history_window,
        }

    def update_config(self, thresholds: dict | None = None, sustain_ms: int | None = None, history_window: int | None = None):
        if thresholds:
            self.thresholds.update({k: float(v) for k, v in thresholds.items() if k in self.thresholds})
        if sustain_ms is not None:
            self.sustain_ms = int(sustain_ms)
        if history_window is not None and history_window > 0:
            self.history_window = int(history_window)
            # reinit deques to new size while preserving latest data
            old_probs = list(self._prob_history)[-self.history_window:]
            old_ts = list(self._ts_history)[-self.history_window:]
            self._prob_history = deque(old_probs, maxlen=self.history_window)
            self._ts_history = deque(old_ts, maxlen=self.history_window)