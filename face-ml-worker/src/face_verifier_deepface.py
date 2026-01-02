from deepface import DeepFace
import numpy as np
import cv2
from typing import Optional, List, Tuple
import os


class FaceVerifier:
    """
    Face verification using DeepFace with Facenet512 model.
    Facenet512 produces reliable 512-dimensional embeddings.
    """

    def __init__(self, model_name="Facenet512"):
        self.reference_embeddings = {}  # {user_id: [embedding1, embedding2, ...]}
        self.model_name = model_name
        print(f"[FACE-VERIFIER] Loading model: {model_name}")
        # Preload model
        DeepFace.build_model(model_name)
        print(f"[FACE-VERIFIER] ✓ Model {model_name} loaded successfully")

        # Set expected embedding properties based on model
        if model_name == "Facenet512":
            self.expected_dim = 512
            self.min_norm = 5.0  # Facenet embeddings typically have norm 10-20
            self.max_norm = 50.0
        elif model_name == "VGG-Face":
            self.expected_dim = 4096
            self.min_norm = 10.0  # VGG-Face should NOT be normalized to 1.0
            self.max_norm = 100.0
        elif model_name == "ArcFace":
            self.expected_dim = 512
            self.min_norm = 0.8
            self.max_norm = 1.2
        else:
            self.expected_dim = None
            self.min_norm = 0.1
            self.max_norm = 1000.0

    def extract_encodings_from_video(self, video_path: str, max_frames: int = 10) -> List[np.ndarray]:
        """
        Extract face embeddings from video - OPTIMIZED to process fewer frames.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to sample (default: 10)

        Returns:
            List of valid face embeddings
        """
        cap = cv2.VideoCapture(video_path)
        embeddings = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"[VIDEO-EXTRACT] Total frames: {total_frames}, FPS: {fps}")

        if total_frames == 0:
            cap.release()
            print("[VIDEO-EXTRACT] ❌ Video has no frames")
            return embeddings

        # Calculate which frame indices to extract
        interval = max(1, total_frames // max_frames)
        target_frames = [i * interval for i in range(max_frames)]
        print(f"[VIDEO-EXTRACT] Will extract frames at indices: {target_frames[:5]}... (every {interval} frames)")

        frame_idx = 0
        frames_read = 0

        while cap.isOpened() and len(embeddings) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames at target indices
            if frame_idx in target_frames:
                frames_read += 1
                try:
                    # Validate frame before processing
                    if frame is None or frame.size == 0:
                        print(f"[VIDEO-EXTRACT] Frame {frame_idx}: ✗ Empty frame")
                        frame_idx += 1
                        continue

                    # DeepFace expects RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Get embedding with strict detection
                    embedding_objs = DeepFace.represent(
                        img_path=rgb_frame,
                        model_name=self.model_name,
                        enforce_detection=True,
                        detector_backend="opencv",
                        align=True  # Align face for better results
                    )

                    if len(embedding_objs) > 0:
                        embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)

                        # CRITICAL: Validate embedding quality
                        emb_norm = np.linalg.norm(embedding)
                        non_zero_count = np.count_nonzero(embedding)
                        non_zero_ratio = non_zero_count / len(embedding)

                        # Check dimension
                        if self.expected_dim and len(embedding) != self.expected_dim:
                            print(f"[VIDEO-EXTRACT] Frame {frame_idx}: ✗ Wrong dimension (got {len(embedding)}, expected {self.expected_dim})")
                            frame_idx += 1
                            continue

                        # Check if embedding has realistic norm
                        if emb_norm < self.min_norm or emb_norm > self.max_norm:
                            print(f"[VIDEO-EXTRACT] Frame {frame_idx}: ✗ Invalid norm {emb_norm:.4f} (expected {self.min_norm:.1f}-{self.max_norm:.1f})")
                            frame_idx += 1
                            continue

                        # For dense embeddings (Facenet), most values should be non-zero
                        if self.model_name == "Facenet512" and non_zero_ratio < 0.95:
                            print(f"[VIDEO-EXTRACT] Frame {frame_idx}: ✗ Too sparse ({non_zero_ratio:.1%} non-zero, expected >95%)")
                            frame_idx += 1
                            continue

                        # Check for NaN/Inf
                        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                            print(f"[VIDEO-EXTRACT] Frame {frame_idx}: ✗ Contains NaN or Inf")
                            frame_idx += 1
                            continue

                        embeddings.append(embedding)
                        print(f"[VIDEO-EXTRACT] Frame {frame_idx}: ✓ Valid embedding ({len(embeddings)}/{max_frames})")
                        print(f"  Dim: {len(embedding)}, Norm: {emb_norm:.4f}, Non-zero: {non_zero_ratio:.1%}")
                        print(f"  Sample: {embedding[:5]}")

                except Exception as e:
                    print(f"[VIDEO-EXTRACT] Frame {frame_idx}: ✗ Failed - {str(e)[:100]}")

            frame_idx += 1

        cap.release()

        print(f"\n[VIDEO-EXTRACT] === Summary ===")
        print(f"[VIDEO-EXTRACT] Total frames in video: {total_frames}")
        print(f"[VIDEO-EXTRACT] Frames checked: {frames_read}")
        print(f"[VIDEO-EXTRACT] Valid embeddings: {len(embeddings)}")

        return embeddings

    def extract_face_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from single frame.
        (Alias for compatibility with other code)

        Args:
            frame: OpenCV image in BGR format

        Returns:
            Face embedding or None if no face found
        """
        return self.extract_encoding_from_frame(frame)

    def extract_encoding_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from single frame with validation.

        Args:
            frame: OpenCV image in BGR format

        Returns:
            Face embedding or None if no face found or invalid embedding
        """
        try:
            # Validate input frame
            if frame is None or frame.size == 0:
                print(f"[EXTRACT-DF] ❌ Invalid frame - empty or None")
                return None

            print(f"[EXTRACT-DF] Frame shape: {frame.shape}")

            # Convert to RGB for DeepFace
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract embedding with face alignment
            embedding_objs = DeepFace.represent(
                img_path=rgb_frame,
                model_name=self.model_name,
                enforce_detection=True,
                detector_backend="opencv",
                align=True  # Align face for better results
            )

            if len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]["embedding"], dtype=np.float32)

                # CRITICAL: Validate embedding quality
                emb_norm = np.linalg.norm(embedding)
                non_zero_count = np.count_nonzero(embedding)
                non_zero_ratio = non_zero_count / len(embedding)

                print(f"[EXTRACT-DF] Embedding extracted:")
                print(f"  Dim: {len(embedding)} (expected: {self.expected_dim})")
                print(f"  Norm: {emb_norm:.4f} (expected: {self.min_norm:.1f}-{self.max_norm:.1f})")
                print(f"  Non-zero: {non_zero_ratio:.1%} ({non_zero_count}/{len(embedding)})")
                print(f"  Sample (first 10): {embedding[:10]}")

                # Validate dimension
                if self.expected_dim and len(embedding) != self.expected_dim:
                    print(f"[EXTRACT-DF] ❌ REJECTED: Wrong dimension (got {len(embedding)}, expected {self.expected_dim})")
                    return None

                # Validate norm is in expected range
                if emb_norm < self.min_norm or emb_norm > self.max_norm:
                    print(f"[EXTRACT-DF] ❌ REJECTED: Norm {emb_norm:.4f} out of range ({self.min_norm:.1f}-{self.max_norm:.1f})")
                    print(f"[EXTRACT-DF] ⚠️ This indicates the model is not working correctly!")
                    return None

                # For dense embeddings (Facenet), check sparsity
                if self.model_name == "Facenet512" and non_zero_ratio < 0.95:
                    print(f"[EXTRACT-DF] ❌ REJECTED: Too sparse ({non_zero_ratio:.1%}, expected >95%)")
                    return None

                # Check for NaN/Inf
                if np.any(np.isnan(embedding)):
                    print(f"[EXTRACT-DF] ❌ REJECTED: Contains NaN values")
                    return None

                if np.any(np.isinf(embedding)):
                    print(f"[EXTRACT-DF] ❌ REJECTED: Contains infinite values")
                    return None

                print(f"[EXTRACT-DF] ✓ Valid embedding - all checks passed")
                return embedding
            else:
                print(f"[EXTRACT-DF] ❌ No face detected in frame")
                return None

        except Exception as e:
            print(f"[EXTRACT-DF] ❌ Failed to extract embedding: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            return None

    def enroll_user(self, user_id: str, video_path: str) -> Tuple[bool, str]:
        """
        Enroll user by extracting face embeddings from selfie video.

        Args:
            user_id: Unique identifier for the user
            video_path: Path to selfie video file

        Returns:
            Tuple of (success: bool, message: str)
        """
        embeddings = self.extract_encodings_from_video(video_path)

        if len(embeddings) < 3:
            return False, f"Could not extract enough face samples. Found {len(embeddings)}, need at least 3."

        self.reference_embeddings[user_id] = embeddings
        return True, f"Successfully enrolled with {len(embeddings)} face samples"

    def verify_face(self, user_id: str, frame: np.ndarray, threshold: float = 0.40) -> Tuple[bool, float, str]:
        """
        Verify if face in frame matches enrolled user.

        Args:
            user_id: User ID to verify against
            frame: OpenCV image in BGR format
            threshold: Cosine distance threshold. For Facenet512: 0.40 (same person < 0.40)

        Returns:
            Tuple of (is_match: bool, confidence: float, message: str)
        """
        if user_id not in self.reference_embeddings:
            return False, 0.0, "User not enrolled"

        current_embedding = self.extract_encoding_from_frame(frame)
        if current_embedding is None:
            return False, 0.0, "No face detected in frame"

        reference_embeddings = self.reference_embeddings[user_id]

        # Calculate cosine distances
        distances = []
        similarities = []
        for i, ref_embedding in enumerate(reference_embeddings):
            # Cosine similarity
            cosine_sim = np.dot(current_embedding, ref_embedding) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(ref_embedding) + 1e-8
            )
            # Convert to distance (0 = identical, 1 = completely different)
            distance = 1 - cosine_sim
            distances.append(distance)
            similarities.append(cosine_sim)
            print(f"[VERIFY-DF] Sample {i+1}: Distance={distance:.4f}, Cosine Similarity={cosine_sim:.4f}")

        min_distance = float(np.min(distances))
        max_similarity = float(np.max(similarities))
        avg_distance = float(np.mean(distances))

        # Convert distance to confidence
        # For cosine distance: distance of 0 = 100% confidence
        if min_distance <= threshold:
            confidence = 1.0 - (min_distance / threshold)
        else:
            confidence = 0.0

        is_match = min_distance <= threshold

        print(f"[VERIFY-DF] Min distance: {min_distance:.4f}, Avg: {avg_distance:.4f}, Threshold: {threshold}")
        print(f"[VERIFY-DF] Max similarity: {max_similarity:.4f}")
        print(f"[VERIFY-DF] Is Match: {is_match}, Confidence: {confidence:.4f} ({confidence*100:.1f}%)")

        message = "Face verified" if is_match else f"Face does not match (distance: {min_distance:.3f})"

        return is_match, confidence, message

    def is_enrolled(self, user_id: str) -> bool:
        """Check if user is enrolled."""
        return user_id in self.reference_embeddings

    def unenroll_user(self, user_id: str) -> bool:
        """Remove user's enrollment."""
        if user_id in self.reference_embeddings:
            del self.reference_embeddings[user_id]
            return True
        return False
