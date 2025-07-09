import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from math import tan, radians
import ssl
import certifi
import numpy
import torch

print(torch.__version__)
print(torch.cuda.is_available())  # Checks if CUDA is available (for GPU installations)

# Configure SSL to use certifi certificates
ssl._create_default_https_context = ssl.create_default_context
ssl._create_default_https_context().load_verify_locations(certifi.where())

# Now load the model
import torch

# Load the YOLO model
model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path="./yolov5(1)/content/yolov5/runs/train/exp/weights/best.pt",
    force_reload=True,
)


# Function to calculate the field of view (FoV) based on focal length and sensor size
def calculate_fov(focal_length, sensor_width):
    fov = 2 * atan(sensor_width / (2 * focal_length)) * (180 / np.pi)
    return fov


# Function to calculate distance based on object size in the image
def estimate_distance(object_height_px, real_object_height, fov, image_height):
    # Calculate the distance based on FoV and object size
    distance = (real_object_height * image_height) / (
        2 * object_height_px * tan(radians(fov / 2))
    )
    return distance


# Function to process video stream and detect objects
def process_frame(frame):
    results = model(frame)
    return results


# Streamlit App
st.title("Wall Measurement and Cost Estimation App")

# Input for camera parameters
st.sidebar.header("Camera Parameters")
focal_length = st.sidebar.number_input("Focal Length (mm)", value=4.5)
sensor_width = st.sidebar.number_input("Sensor Width (mm)", value=4.8)
cost_per_sqft = st.sidebar.number_input("Cost per Square Foot", value=2.0)

# Video capture
cap = cv2.VideoCapture(0)

if st.button("Start Detection"):
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in the frame
        results = process_frame(frame)
        detected_data = results.pandas().xyxy[0].to_dict()

        # Draw detections and measurements
        for index, row in detected_data.items():
            class_name = row["name"]
            x1, y1, x2, y2 = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            )

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {x2 - x1}x{y2 - y1}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # Calculate distance for walls
            if class_name == "wall":
                height_px = y2 - y1
                real_wall_height = 2.5  # Assume real wall height in meters
                fov = calculate_fov(focal_length, sensor_width)

                distance = estimate_distance(
                    height_px, real_wall_height, fov, frame.shape[0]
                )

                # Display the distance and estimated area
                area = real_wall_height * (
                    distance * tan(radians(fov / 2))
                )  # Simplified area calculation
                cost = area * cost_per_sqft

                cv2.putText(
                    frame,
                    f"Distance: {distance:.2f} m, Area: {area:.2f} sq.ft, Cost: ${cost:.2f}",
                    (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

        # Display frame
        stframe.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
