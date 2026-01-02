# ALLOWED OBJECTS - Only these objects are permitted during exam
# Any other detected object will be flagged as suspicious
ALLOWED_OBJECTS = [
    "person",  # The student themselves
    "chair",   # Sitting furniture
    "couch",   # Alternative seating
    "bench",   # Alternative seating
    "desk",    # Table/desk (not in standard YOLO classes, but keeping for completeness)
    "dining table",  # Table/desk equivalent in YOLO
]

# Confidence threshold - only flag objects detected with high confidence
CONFIDENCE_THRESHOLD = 0.5

def check_objects_for_fraud(objects):
    """
    Flag ANY object detection as potential fraud except allowed items.
    Only allows: person, chair, desk, and similar furniture.
    """
    issues = []

    for obj in objects:
        cls = obj["class"].lower()
        confidence = obj.get("confidence", 1.0)  # Default to 1.0 if not provided

        # Skip low-confidence detections to avoid false positives
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Flag any object NOT in the allowed list
        if cls not in ALLOWED_OBJECTS:
            issues.append(f"Suspicious object detected: {cls} (confidence: {confidence:.2f})")

    return issues
