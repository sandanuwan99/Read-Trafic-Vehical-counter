import cv2
import pandas as pd
from ultralytics import YOLO

# ---------------- LOAD YOLO MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- LOAD VIDEO ----------------
cap = cv2.VideoCapture(r"E:/Reserch/Road1.mp4")

# ---------------- VEHICLE CLASS IDS (COCO) ----------------
# 2 = car, 3 = motorbike, 5 = bus, 7 = truck
vehicle_classes = [2, 3, 5, 7]

# ---------------- COUNT VARIABLES ----------------
counted_ids = set()

counted_car = set()
counted_bike = set()
counted_bus = set()
counted_truck = set()

total_count = 0
car_count = 0
bike_count = 0
bus_count = 0
truck_count = 0

# ---------------- COUNTING LINE ----------------
line_y = 780  # adjust based on your video height

# ---------------- PROCESS VIDEO ----------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.4, iou=0.5)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id
        classes = results[0].boxes.cls

        for box, track_id, cls in zip(boxes, ids, classes):
            cls_id = int(cls.item())
            track_id = int(track_id.item())

            if cls_id in vehicle_classes:
                x1, y1, x2, y2 = map(int, box)
                cy = (y1 + y2) // 2

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # -------- COUNT WHEN VEHICLE CROSSES LINE --------
                if cy > line_y and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_count += 1

                    if cls_id == 2 and track_id not in counted_car:
                        car_count += 1
                        counted_car.add(track_id)

                    elif cls_id == 3 and track_id not in counted_bike:
                        bike_count += 1
                        counted_bike.add(track_id)

                    elif cls_id == 5 and track_id not in counted_bus:
                        bus_count += 1
                        counted_bus.add(track_id)

                    elif cls_id == 7 and track_id not in counted_truck:
                        truck_count += 1
                        counted_truck.add(track_id)

    # ---------------- DRAW COUNTING LINE ----------------
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

    # ---------------- DISPLAY COUNTS ----------------
    cv2.putText(frame, f"Total: {total_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Car: {car_count}  Bike: {bike_count}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"Bus: {bus_count}  Truck: {truck_count}",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Vehicle Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- RELEASE VIDEO ----------------
cap.release()
cv2.destroyAllWindows()

# ---------------- SAVE RESULTS TO EXCEL ----------------
data = {
    "Video Name": ["Road1.mp4"],
    "Total Vehicles": [total_count],
    "Cars": [car_count],
    "Bikes": [bike_count],
    "Buses": [bus_count],
    "Trucks": [truck_count]
}

df = pd.DataFrame(data)
df.to_excel("vehicle_count.xlsx", index=False)

print("Vehicle count saved to vehicle_count.xlsx")
print("Total:", total_count)
print("Cars:", car_count)
print("Bikes:", bike_count)
print("Buses:", bus_count)
print("Trucks:", truck_count)
