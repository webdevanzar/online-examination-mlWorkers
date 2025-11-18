import time

prev_face_center = None
prev_time = time.time()
still_frames = 0
missing_frames = 0
brightness_history = []

def check_fraud(faces, face_direction, frame):
    global prev_face_center, prev_time, still_frames, missing_frames, brightness_history

    issues = []

    # Always initialize movement
    movement = 0

    # 1. No face
    if len(faces) == 0:
        missing_frames += 1
        if missing_frames > 10:
            issues.append("Face missing repeatedly")

        issues.append("No face detected")
        return issues
    else:
        missing_frames = 0

    # Main face
    x1, y1, x2, y2 = faces[0]
    face_width = x2 - x1
    face_height = y2 - y1
    face_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    # 2. Multiple faces
    if len(faces) > 1:
        issues.append(f"Multiple faces detected ({len(faces)})")

    # 3. Looking away
    if face_direction in ["looking_left", "looking_right"]:
        issues.append("Looking away")

    # 4. Face too close
    if face_width > frame.shape[1] * 0.6:
        issues.append("Face too close to camera")

    # 5. Face too far
    if face_width < frame.shape[1] * 0.1:
        issues.append("Face too far from camera")

    # 6. Face not centered
    frame_center_x = frame.shape[1] // 2
    offset = abs(face_center[0] - frame_center_x)

    if offset > frame.shape[1] * 0.25:
        issues.append("Face not centered")

    # 7. Rapid movement detection
    if prev_face_center is not None:
        movement = abs(face_center[0] - prev_face_center[0]) + abs(face_center[1] - prev_face_center[1])
        if movement > 120:
            issues.append("Rapid suspicious movement")

    prev_face_center = face_center

    # 8. No movement for long time (possible spoofing)
    if movement < 5:  
        still_frames += 1
        if still_frames > 120:  
            issues.append("Possible frozen screen or static image")
    else:
        still_frames = 0

    # 9. Sudden brightness change
    gray = frame.mean()
    brightness_history.append(gray)

    if len(brightness_history) > 5:
        brightness_history.pop(0)

    if len(brightness_history) == 5:
        if max(brightness_history) - min(brightness_history) > 40:
            issues.append("Sudden brightness change detected")

    return issues

