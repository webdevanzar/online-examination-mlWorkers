import cv2
from detector import FaceDetector
from fraud_rules import check_fraud
from utils import draw_boxes, draw_objects
from object_detector import ObjectDetector
from object_rules import check_objects_for_fraud   # new


# Initialize detectors
face_detector = FaceDetector()
object_detector = ObjectDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------
    # 1️⃣ Object Detection
    # -----------------------
    objects = object_detector.detect_objects(frame)

    # Draw object boxes on screen
    frame = draw_objects(frame, objects)

    # -----------------------
    # 2️⃣ Face Detection
    # -----------------------
    faces = face_detector.detect(frame)

    direction = "none"
    if len(faces) > 0:
        direction = face_detector.get_direction(frame, faces[0])

    # Draw face boxes
    frame = draw_boxes(frame, faces, direction)

    # -----------------------
    # 3️⃣ Apply Fraud Rules (Face + Objects)
    # -----------------------
    fraud_face = check_fraud(faces, direction, frame)
    fraud_object = check_objects_for_fraud(objects)

    issues = fraud_face + fraud_object

    print("Fraud:", issues)

    # -----------------------
    # 4️⃣ Display Output
    # -----------------------
    cv2.imshow("Exam Proctoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
