import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLOv8 from the Ultralytics library

# Load YOLOv8 model
model_path = "yolov8n.pt"  # Replace with the path to your YOLOv8 model (e.g., yolov8n, yolov8s, etc.)
model = YOLO(model_path)

# Load the COCO class names (auto-loaded by YOLOv8, but you can customize this if needed)
classes = model.names

# Replace with your video file path
video_path = "output.mp4"

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get total frames and FPS
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a resizable window
cv2.namedWindow("Vehicle Detection", cv2.WINDOW_NORMAL)

# Variables for playback
current_frame = 0
speed = 1
paused = False

# Variables for adjustable points (line for crossing detection)
point1 = (100, 100)
point2 = (300, 300)
point_radius = 5
dragging_point1 = False
dragging_point2 = False

# Vehicle counting variables
vehicle_count = 0
tracked_vehicles = {}  # {vehicle_id: (x, y, w, h, crossed, label)}
vehicle_id_counter = 0

# Mouse callback function to drag points
def mouse_callback(event, x, y, flags, param):
    global point1, point2, dragging_point1, dragging_point2
    if event == cv2.EVENT_LBUTTONDOWN:
        if (x - point1[0]) ** 2 + (y - point1[1]) ** 2 < point_radius**2:
            dragging_point1 = True
        elif (x - point2[0]) ** 2 + (y - point2[1]) ** 2 < point_radius**2:
            dragging_point2 = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point1:
            point1 = (x, y)
        elif dragging_point2:
            point2 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point1 = False
        dragging_point2 = False


cv2.setMouseCallback("Vehicle Detection", mouse_callback)


# Trackbar callback for seeking
def on_trackbar(val):
    global current_frame
    current_frame = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)


cv2.createTrackbar("Seek", "Vehicle Detection", 0, total_frames - 1, on_trackbar)


def is_crossing_line(center, prev_center):
    """
    Check if a vehicle crosses the line defined by point1 and point2.
    Uses line intersection logic for accurate detection.
    """
    # Ensure there is a previous center to check the trajectory
    if prev_center is None:
        return False

    # Line segment defined by the vehicle's trajectory
    line_vehicle = (prev_center, center)

    # Line segment defined by the crossing detection line
    line_crossing = (point1, point2)

    # Helper function to check orientation
    def orientation(p, q, r):
        """
        Find the orientation of the triplet (p, q, r).
        0 -> Collinear, 1 -> Clockwise, 2 -> Counterclockwise
        """
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    # Helper function to check if two line segments intersect
    def do_intersect(p1, q1, p2, q2):
        """
        Check if line segments (p1, q1) and (p2, q2) intersect.
        """
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        # General case: segments intersect
        if o1 != o2 and o3 != o4:
            return True

        # Special cases: segments are collinear and overlap
        def on_segment(p, q, r):
            return (
                min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
            )

        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    # Check if the trajectory line intersects the crossing line
    return do_intersect(line_vehicle[0], line_vehicle[1], line_crossing[0], line_crossing[1])



while True:
    if not paused:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            print("End of video reached.")
            break

        # Run YOLOv8 inference
        results = model(frame)

        # Extract bounding boxes, confidences, and class IDs
        detections = results[0].boxes  # YOLOv8 stores detection boxes in `results[0].boxes`

        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            x1, y1, x2, y2 = detection.xyxy[0].tolist()  # Bounding box coordinates
            confidence = detection.conf[0]  # Confidence score
            class_id = int(detection.cls[0])  # Class ID

            label = classes[class_id]
            if label in ["car", "truck", "bus", "motorcycle"]:  # Focus on vehicles
                boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        current_detections = []
        for i, box in enumerate(boxes):
            x, y, w, h = box
            current_detections.append((x, y, w, h))

        # Update tracked vehicles
        updated_tracked_vehicles = {}
        for i, (x, y, w, h) in enumerate(current_detections):
            center = (x + w // 2, y + h // 2)
            matched = False

            for vid, (vx, vy, vw, vh, crossed, vlabel) in tracked_vehicles.items():
                prev_center = (vx + vw // 2, vy + vh // 2)
                if (
                    abs(center[0] - prev_center[0]) < 50
                    and abs(center[1] - prev_center[1]) < 50
                ):
                    matched = True
                    updated_tracked_vehicles[vid] = (x, y, w, h, crossed, vlabel)

                    # Check for crossing
                    if not crossed and is_crossing_line(center, prev_center):
                        updated_tracked_vehicles[vid] = (x, y, w, h, True, vlabel)
                        vehicle_count += 1
                    break

            if not matched:
                vehicle_id_counter += 1
                label = classes[class_ids[i]]  # Use the correct index `i` for the label
                updated_tracked_vehicles[vehicle_id_counter] = (
                    x,
                    y,
                    w,
                    h,
                    False,
                    label,
                )

        tracked_vehicles = updated_tracked_vehicles

        # Draw bounding boxes and labels
        for vid, (x, y, w, h, crossed, label) in tracked_vehicles.items():
            color = (0, 255, 0) if crossed else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"ID {label}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Draw the adjustable line
        cv2.circle(frame, point1, point_radius, (0, 0, 255), -1)
        cv2.circle(frame, point2, point_radius, (255, 0, 0), -1)
        cv2.line(frame, point1, point2, (0, 255, 255), 2)

        # Display vehicle count
        cv2.putText(
            frame,
            f"Vehicles Crossed: {vehicle_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Playback speed
        cv2.putText(
            frame,
            f"Speed: {speed}x",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.setTrackbarPos("Seek", "Vehicle Detection", current_frame)

        cv2.imshow("Vehicle Detection", frame)
        current_frame += speed
        if current_frame >= total_frames:
            current_frame = total_frames - 1

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("p"):
        paused = not paused
    elif key == ord("+"):
        speed = min(speed + 1, 5)
    elif key == ord("-"):
        speed = max(speed - 1, 1)
    elif key == ord("r"):
        current_frame = 0
    elif key == ord("c"):
        vehicle_count = 0
        tracked_vehicles.clear()

cap.release()
cv2.destroyAllWindows()
