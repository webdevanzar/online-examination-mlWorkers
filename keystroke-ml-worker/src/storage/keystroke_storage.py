# ml-workers/keystroke-ml-worker/src/storage/keystroke_storage.py
import json
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class KeystrokeStorage:
    def __init__(self, base_dir: str = "src/models/keystroke_models"):
        """
        Initialize keystroke model storage.
        
        Args:
            base_dir: Base directory to store keystroke models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized keystroke storage at {self.base_dir.absolute()}")

    def _get_model_path(self, user_id: str) -> Path:
        """Get the file path for a user's keystroke model"""
        # Sanitize user_id to prevent directory traversal
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_'))
        if not safe_user_id:
            raise ValueError("Invalid user ID")
        return self.base_dir / f"{safe_user_id}.json"

    def save_model(self, user_id: str, model_data: Dict[str, Any]) -> None:
        """
        Save keystroke model to disk.
        
        Args:
            user_id: Unique user identifier
            model_data: Dictionary containing model data (features, metadata, etc.)
        """
        model_path = self._get_model_path(user_id)
        
        # Create a serializable copy of the data
        serializable_data = {}
        for key, value in model_data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif hasattr(value, 'tolist'):  # For numpy scalars
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
        
        try:
            # Write to temporary file first, then rename (atomic operation)
            temp_path = model_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            # On Windows, we need to remove the destination file first if it exists
            if model_path.exists():
                model_path.unlink()
            temp_path.rename(model_path)
            
            logger.info(f"Saved keystroke model for user {user_id} to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model for user {user_id}: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_model(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Load keystroke model from disk.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            Dictionary containing model data or None if not found
        """
        model_path = self._get_model_path(user_id)
        if not model_path.exists():
            logger.debug(f"No model found for user {user_id}")
            return None
            
        try:
            with open(model_path, 'r') as f:
                data = json.load(f)
            logger.debug(f"Loaded keystroke model for user {user_id}")
            return data
        except Exception as e:
            logger.error(f"Failed to load model for user {user_id}: {str(e)}")
            return None

    def delete_model(self, user_id: str) -> bool:
        """
        Delete a user's keystroke model.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            True if model was deleted, False if it didn't exist
        """
        model_path = self._get_model_path(user_id)
        try:
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Deleted keystroke model for user {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete model for user {user_id}: {str(e)}")
            return False

    def model_exists(self, user_id: str) -> bool:
        """
        Check if a keystroke model exists for a user.

        Args:
            user_id: Unique user identifier

        Returns:
            True if model exists, False otherwise
        """
        model_path = self._get_model_path(user_id)
        return model_path.exists()

    def list_models(self) -> list[str]:
        """List all stored model user IDs"""
        try:
            return [
                f.stem for f in self.base_dir.glob("*.json")
                if f.is_file() and not f.name.startswith('.')
            ]
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []