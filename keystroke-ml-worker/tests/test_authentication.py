import unittest
import numpy as np
from src.authentication.authenticator import KeystrokeAuthenticator


class TestKeystrokeAuth(unittest.TestCase):
    def setUp(self):
        self.auth = KeystrokeAuthenticator(models_dir="data/models_test")
        self.user_id = "test_user"

        # Generate more distinct enrollment samples with tighter variance
        self.enroll_samples = [
            self._generate_feature_vector(mean=100, std=5) for _ in range(10)  # More samples for better training
        ]
        
        # Valid sample: similar to enrollment
        self.valid_sample = self._generate_feature_vector(mean=105, std=8)
        
        # Invalid sample: very different from enrollment
        self.invalid_sample = self._generate_feature_vector(mean=500, std=100)

    def _generate_feature_vector(self, mean, std, length=20):
        """Generate a fake keystroke feature vector"""
        return list(np.random.normal(mean, std, length))

    def test_enrollment(self):
        """Ensure enrollment succeeds"""
        success = self.auth.enroll_user(self.user_id, self.enroll_samples)
        self.assertTrue(success)

    def test_verification(self):
        """Verify a valid and invalid typing attempt"""
        self.auth.enroll_user(self.user_id, self.enroll_samples)

        # Test valid sample
        is_match, conf_valid = self.auth.verify_user(self.user_id, self.valid_sample)
        print(f"Valid sample - Match: {is_match}, Confidence: {conf_valid}")
        
        # Test invalid sample
        is_match_invalid, conf_invalid = self.auth.verify_user(
            self.user_id, self.invalid_sample
        )
        print(f"Invalid sample - Match: {is_match_invalid}, Confidence: {conf_invalid}")

        # Check that confidence for valid sample is higher than for invalid
        self.assertGreater(conf_valid, conf_invalid, 
                         f"Valid confidence ({conf_valid:.3f}) should be > invalid confidence ({conf_invalid:.3f})")

    def test_identification(self):
        """Check automatic user identification"""
        # Test identification before enrollment (should return None)
        user, conf = self.auth.identify_user(self.valid_sample)
        self.assertIsNone(user)
        self.assertEqual(conf, 0.0)
        
        # Enroll the test user
        self.assertTrue(self.auth.enroll_user(self.user_id, self.enroll_samples))
        
        # Test identification with valid sample
        predicted_user, confidence = self.auth.identify_user(self.valid_sample)
        print(f"Identified user: {predicted_user} with confidence: {confidence:.3f}")
        self.assertEqual(predicted_user, self.user_id)
        self.assertGreater(confidence, 0.0)
        
        # Test identification with invalid sample (should return None)
        invalid_user, invalid_conf = self.auth.identify_user(
            self.invalid_sample,
            min_confidence=0.1  # Set a minimum confidence threshold
        )
        print(f"Invalid sample - Identified as: {invalid_user} with confidence: {invalid_conf:.3f}")
        self.assertIsNone(invalid_user)
        
        # Verify the valid sample has higher confidence than invalid
        self.assertGreater(confidence, invalid_conf)


if __name__ == "__main__":
    unittest.main()
