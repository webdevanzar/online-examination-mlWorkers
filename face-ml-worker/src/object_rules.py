DANGEROUS_CLASSES = [
    "cell phone",
    "laptop",
    "keyboard",
    "mouse",
    "remote",
    "tv",
    "book",
    "backpack",
    "handbag",
    "suitcase",
]

def check_objects_for_fraud(objects):
    issues = []

    for obj in objects:
        cls = obj["class"].lower()

        # Match only YOLO's real trained classes
        if cls in DANGEROUS_CLASSES:
            issues.append(f"Cheating object detected: {cls}")


    return issues
