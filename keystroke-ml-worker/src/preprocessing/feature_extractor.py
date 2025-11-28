import numpy as np
from collections import defaultdict


class FeatureExtractor:
    def extract_features(self, keystroke_data):
        # ✅ Reset per call
        features = {
            "hold_times": defaultdict(list),
            "inter_key_press": defaultdict(list),
            "inter_key_release": defaultdict(list),
            "release_press": defaultdict(list),
        }

        # Group events by key
        key_events = defaultdict(list)
        for event in keystroke_data:
            key_events[event["key"]].append(event)

        # Hold time: keydown → keyup
        for key, events in key_events.items():
            presses = [e for e in events if e["event"] == "keydown"]
            releases = [e for e in events if e["event"] == "keyup"]

            for p, r in zip(presses, releases):
                if p["timestamp"] < r["timestamp"]:
                    features["hold_times"][key].append(r["timestamp"] - p["timestamp"])

        # Inter-key timings
        for i in range(len(keystroke_data) - 1):
            curr = keystroke_data[i]
            next_evt = keystroke_data[i + 1]

            delta = next_evt["timestamp"] - curr["timestamp"]
            pair = f"{curr['key']}-{next_evt['key']}"

            if curr["event"] == "keydown" and next_evt["event"] == "keydown":
                features["inter_key_press"][pair].append(delta)

            elif curr["event"] == "keyup" and next_evt["event"] == "keyup":
                features["inter_key_release"][pair].append(delta)

            elif curr["event"] == "keyup" and next_evt["event"] == "keydown":
                features["release_press"][pair].append(delta)

        return self._aggregate_features(features)

    def _aggregate_features(self, features):
        feature_vector = []

        for key in sorted(features["hold_times"]):
            times = features["hold_times"][key]
            if times:
                feature_vector.extend(
                    [
                        np.mean(times),
                        np.std(times),
                        np.min(times),
                        np.max(times),
                    ]
                )

        for timing_type in ["inter_key_press", "inter_key_release", "release_press"]:
            times = []
            for v in features[timing_type].values():
                times.extend(v)

            if times:
                feature_vector.extend(
                    [
                        np.mean(times),
                        np.std(times),
                        np.min(times),
                        np.max(times),
                    ]
                )

        if not feature_vector:
            raise ValueError("No keystroke features extracted")

        return np.array(feature_vector)
