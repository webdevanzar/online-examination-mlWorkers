import cv2

def draw_boxes(frame, faces, direction):
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"Face", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(frame, f"Direction: {direction}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return frame

def draw_objects(frame, objects):
    for obj in objects:
        x1, y1, x2, y2 = obj["box"]
        cls = obj["class"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, cls, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame


