from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")   # general object detection model

    def detect_objects(self, frame):
        results = self.model(frame, verbose=False)[0]

        detected_objects = []

        for box, cls_id, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = results.names[int(cls_id)]
            confidence = float(conf)

            detected_objects.append({
                "box": [x1, y1, x2, y2],
                "class": class_name,
                "confidence": confidence
            })

        return detected_objects
