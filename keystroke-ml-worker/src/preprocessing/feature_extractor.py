# feature_extractor.py
import numpy as np
from collections import defaultdict

class FeatureExtractor:
    def __init__(self):
        self.features = {
            'hold_times': defaultdict(list),  # PR (Press-Release)
            'inter_key_press': defaultdict(list),  # PP
            'inter_key_release': defaultdict(list),  # RR
            'release_press': defaultdict(list),  # RP
        }
        
    def extract_features(self, keystroke_data):
        # Group by key
        key_events = defaultdict(list)
        for event in keystroke_data:
            key_events[event['key']].append(event)
            
        # Calculate timing features
        for key, events in key_events.items():
            presses = [e for e in events if e['event'] == 'press']
            releases = [e for e in events if e['event'] == 'release']
            
            # Hold time (PR)
            for p, r in zip(presses, releases):
                if p['timestamp'] < r['timestamp']:  # Ensure press before release
                    self.features['hold_times'][key].append(r['timestamp'] - p['timestamp'])
                    
        # Calculate inter-key timings
        for i in range(len(keystroke_data)-1):
            curr = keystroke_data[i]
            next_evt = keystroke_data[i+1]
            
            if curr['event'] == 'press' and next_evt['event'] == 'press':
                self.features['inter_key_press'][f"{curr['key']}-{next_evt['key']}"] = (
                    next_evt['timestamp'] - curr['timestamp']
                )
            elif curr['event'] == 'release' and next_evt['event'] == 'release':
                self.features['inter_key_release'][f"{curr['key']}-{next_evt['key']}"] = (
                    next_evt['timestamp'] - curr['timestamp']
                )
            elif curr['event'] == 'release' and next_evt['event'] == 'press':
                self.features['release_press'][f"{curr['key']}-{next_evt['key']}"] = (
                    next_evt['timestamp'] - curr['timestamp']
                )
                
        return self._aggregate_features()
        
    def _aggregate_features(self):
        """Convert features to fixed-length vector"""
        feature_vector = []
        
        # Add hold time statistics
        for key in sorted(self.features['hold_times'].keys()):
            times = self.features['hold_times'][key]
            if times:
                feature_vector.extend([
                    np.mean(times),
                    np.std(times) if len(times) > 1 else 0,
                    np.min(times),
                    np.max(times)
                ])
                
        # Add inter-key timing statistics
        for timing_type in ['inter_key_press', 'inter_key_release', 'release_press']:
            times = list(self.features[timing_type].values())
            if times:
                feature_vector.extend([
                    np.mean(times),
                    np.std(times) if len(times) > 1 else 0,
                    np.min(times),
                    np.max(times)
                ])
                
        return np.array(feature_vector)