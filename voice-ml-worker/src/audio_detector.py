import sounddevice as sd
from .vad_engine import VoiceVAD
from typing import Generator, Dict, Any
import time
import numpy as np
import warnings

warnings.filterwarnings("ignore")

vad = VoiceVAD()
SAMPLE_RATE = 16000
CHUNK_SIZE = 512


def list_audio_devices():
    """List all available audio devices with detailed info"""
    print("\n📋 Available Audio Input Devices:")
    devices = sd.query_devices()
    input_devices = []

    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            input_devices.append((i, device))
            print(f"  {i}: {device['name']}")
            print(
                f"     Channels: {device['max_input_channels']}, Sample Rate: {device['default_samplerate']}"
            )

    return input_devices


def test_microphone_sensitivity(device_id):
    """Test microphone with different gain levels"""
    print("\n🎯 Testing microphone sensitivity...")

    test_gains = [1, 10, 50, 100, 200]
    best_gain = 1
    max_rms = 0

    for gain in test_gains:
        # Record test audio
        audio = sd.rec(
            frames=CHUNK_SIZE * 4,  # Longer recording for better test
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device_id,
        )
        sd.wait()
        audio = audio.flatten()

        # Apply gain
        audio_gained = audio * gain
        rms = np.sqrt(np.mean(audio_gained**2))

        print(f"   Gain {gain:3d}x: RMS = {rms:.6f}")

        if rms > max_rms and rms < 10.0:  # Avoid clipping
            max_rms = rms
            best_gain = gain

    print(f"✅ Recommended gain: {best_gain}x")
    return best_gain


def listen_and_detect() -> Generator[Dict[str, Any], None, None]:
    print("🎤 Voice Fraud Detection - MAXIMUM SENSITIVITY MODE")

    # List devices
    input_devices = list_audio_devices()

    if not input_devices:
        print("❌ No input devices found!")
        return

    # Try to find the best device
    device_id = None
    for i, device in input_devices:
        if any(
            keyword in device["name"].lower()
            for keyword in ["microphone", "mic", "array", "input"]
        ):
            device_id = i
            break

    if device_id is None:
        device_id = input_devices[0][0]

    device_info = sd.query_devices(device_id, "input")
    print(f"\n✅ Using device: {device_info['name']} (ID: {device_id})")

    # Test microphone and determine optimal gain
    optimal_gain = test_microphone_sensitivity(device_id)

    # Apply even more aggressive gain for very weak signals
    if optimal_gain < 100:
        optimal_gain = 200  # Maximum reasonable gain
        print(f"🚀 Applying maximum gain: {optimal_gain}x")

    print(
        f"\n🎹 Final Config: {SAMPLE_RATE}Hz, {CHUNK_SIZE} samples, Gain: {optimal_gain}x"
    )
    print("💡 SPEAK LOUDLY into your microphone to test...\n")

    # Use stream for continuous recording
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            device=device_id,
            channels=1,
            dtype="float32",
            latency="low",
        ) as stream:
            while True:
                try:
                    # Read audio from stream
                    audio, overflowed = stream.read(CHUNK_SIZE)
                    audio = audio.flatten()

                    if overflowed:
                        print("⚠️  Audio overflow detected")

                    # Apply maximum gain
                    audio_boosted = audio * optimal_gain

                    # Calculate volume
                    rms = np.sqrt(np.mean(audio_boosted**2))

                    # Process with VAD
                    result = vad.detect_cheating(audio_boosted, sample_rate=SAMPLE_RATE)
                    result["timestamp"] = time.time()
                    result["audio_rms"] = rms
                    result["gain_applied"] = optimal_gain
                    result["samples_captured"] = len(audio)

                    yield result

                except Exception as e:
                    yield {
                        "speech_probability": 0.0,
                        "issues": [f"stream_error: {str(e)}"],
                        "timestamp": time.time(),
                        "audio_rms": 0.0,
                        "gain_applied": optimal_gain,
                    }

    except Exception as e:
        print(f"❌ Stream error: {e}")
        print("🔄 Falling back to single recording mode...")

        # Fallback to single recording mode
        while True:
            try:
                audio = sd.rec(
                    frames=CHUNK_SIZE,
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=device_id,
                )
                sd.wait()
                audio = audio.flatten()

                # Apply maximum gain
                audio_boosted = audio * optimal_gain
                rms = np.sqrt(np.mean(audio_boosted**2))

                result = vad.detect_cheating(audio_boosted, sample_rate=SAMPLE_RATE)
                result["timestamp"] = time.time()
                result["audio_rms"] = rms
                result["gain_applied"] = optimal_gain
                result["samples_captured"] = len(audio)

                yield result

            except Exception as e:
                yield {
                    "speech_probability": 0.0,
                    "issues": [f"recording_error: {str(e)}"],
                    "timestamp": time.time(),
                    "audio_rms": 0.0,
                    "gain_applied": optimal_gain,
                }
