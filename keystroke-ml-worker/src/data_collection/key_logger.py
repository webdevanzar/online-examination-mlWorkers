# key_logger.py
from pynput import keyboard
import time
import json
from datetime import datetime

class KeystrokeLogger:
    def __init__(self):
        self.keystrokes = []
        self.start_time = None
        self.last_release = None
        
    def on_press(self, key):
        try:
            current_time = time.time()
            if not self.start_time:
                self.start_time = current_time
            
            # Record key press
            self.keystrokes.append({
                'event': 'press',
                'key': key.char if hasattr(key, 'char') else str(key),
                'timestamp': current_time - self.start_time
            })
            
        except Exception as e:
            print(f"Error in key press: {e}")

    def on_release(self, key):
        try:
            current_time = time.time()
            if not self.start_time:
                self.start_time = current_time
                
            # Record key release
            self.keystrokes.append({
                'event': 'release',
                'key': key.char if hasattr(key, 'char') else str(key),
                'timestamp': current_time - self.start_time
            })
            self.last_release = current_time
            
        except Exception as e:
            print(f"Error in key release: {e}")

    def start_logging(self):
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()
            
    def save_session(self, user_id, session_id):
        session_data = {
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'keystrokes': self.keystrokes
        }
        # Save to file
        filename = f"data/raw/{user_id}_{session_id}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f)
        return filename