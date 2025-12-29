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
        """
        Create fixed-length feature vector using aggregate statistics only.
        This ensures consistent feature vector length regardless of which keys are typed.
        """
        feature_vector = []

        # ✅ Aggregate hold times across ALL keys (not per-key)
        all_hold_times = []
        for key, times in features["hold_times"].items():
            all_hold_times.extend(times)

        if all_hold_times:
            feature_vector.extend([
                np.mean(all_hold_times),
                np.std(all_hold_times),
                np.min(all_hold_times),
                np.max(all_hold_times),
                np.median(all_hold_times),
            ])
        else:
            # Add zeros if no hold times
            feature_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # ✅ Aggregate timing statistics for each timing type
        for timing_type in ["inter_key_press", "inter_key_release", "release_press"]:
            times = []
            for v in features[timing_type].values():
                times.extend(v)

            if times:
                feature_vector.extend([
                    np.mean(times),
                    np.std(times),
                    np.min(times),
                    np.max(times),
                    np.median(times),
                ])
            else:
                # Add zeros if no timing data for this type
                feature_vector.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        if not feature_vector or len(feature_vector) == 0:
            raise ValueError("No keystroke features extracted")

        # ✅ Always returns 20 features (4 types × 5 stats each)
        return np.array(feature_vector)
