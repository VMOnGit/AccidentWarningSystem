import torch
import cv2
import requests
import numpy as np
import time
import socket
import streamlit as st

# Set page title and configuration
st.set_page_config(page_title="Car Detection System", layout="wide")
st.title("Car Detection and Distance Estimation")

# Constants
REAL_CAR_WIDTH = 1.8  # Average car width in meters
FOCAL_LENGTH = 480  # Adjust based on calibration
ESP32_IP = "192.168.10.16"
BUZZER_COOLDOWN = 3  # Cooldown time in seconds
last_buzzer_time = 0

# URL input field
url = st.text_input("Camera URL", "#####")

# UDP Configuration
udp_ip = st.sidebar.text_input("UDP IP Address", "192.168.29.89")
udp_port = st.sidebar.number_input("UDP Port", value=12345, min_value=1, max_value=65535)

# Settings sidebar
st.sidebar.header("Detection Settings")
distance_threshold = st.sidebar.slider("Warning Distance (meters)", 1, 20, 5)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Create columns for controls
col1, col2 = st.columns([1, 1])

with col1:
    start_button_pressed = st.button("Start")
    if start_button_pressed:
        st.session_state.running = True
        st.write("Starting Stream")

with col2:
    stop_button_pressed = st.button("Stop")
    if stop_button_pressed:
        st.session_state.running = False
        st.write("Ending Stream")

# Initialize session state for running status if not exists
if 'running' not in st.session_state:
    st.session_state.running = False

# Create placeholder for the video frame
frame_placeholder = st.empty()

def send_udp_signal():
    message = "BUZZ"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(message.encode(), (udp_ip, int(udp_port)))
        return True
    except Exception as e:
        st.error(f"Failed to send UDP signal: {e}")
        return False

# Load YOLO model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.classes = [2]  # Detect only 'car' class
    return model

model = load_model()

# Main loop - runs only when session state is running
if st.session_state.running:
    try:
        # Status indicator
        status_placeholder = st.empty()
        status_placeholder.info("Connecting to camera...")
        
        while st.session_state.running:
            try:
                # Fetch the latest frame using MJPEG
                response = requests.get(url, timeout=1)
                img_arr = np.array(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(img_arr, -1)

                if frame is None:
                    status_placeholder.warning("Empty frame received. Retrying...")
                    time.sleep(0.5)
                    continue

                # Resize to improve YOLO speed
                frame = cv2.resize(frame, (640, 360))  # Larger for display in Streamlit

                # Run YOLO detection
                results = model(frame)
                
                # Set status to connected after getting first valid frame
                status_placeholder.success("Connected to camera feed")

                # Process results
                detection_count = 0
                for *box, conf, cls in results.xyxy[0]:
                    if conf < confidence_threshold:
                        continue
                        
                    detection_count += 1
                    x1, y1, x2, y2 = map(int, box)
                    box_width = x2 - x1

                    # Estimate distance
                    distance = (REAL_CAR_WIDTH * FOCAL_LENGTH) / box_width

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Car: {conf:.2f}, {distance:.2f}m"

                    if distance < distance_threshold and (time.time() - last_buzzer_time) > BUZZER_COOLDOWN:
                        try:
                            if send_udp_signal():
                                cv2.putText(frame, "WARNING: Close Distance!", (x1, y1 - 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                last_buzzer_time = time.time()
                        except Exception as e:
                            st.error(f"Failed to send UDP signal: {e}")

                    color = (0, 0, 255) if distance < distance_threshold else (0, 255, 0)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Determine car position
                    frame_center = frame.shape[1] // 2
                    car_center = x1 + box_width // 2
                    position = "Left" if car_center < frame_center - 50 else "Right" if car_center > frame_center + 50 else "Center"
                    cv2.putText(frame, f"Position: {position}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Display the frame in Streamlit
                frame_placeholder.image(frame, channels="BGR", caption=f"Detected Cars: {detection_count}")

                # Add small delay to prevent overwhelming the browser
                time.sleep(0.05)

            except requests.exceptions.RequestException:
                status_placeholder.warning("Failed to fetch frame. Retrying...")
                time.sleep(1)
                
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.running = False

else:
    # Display instructions when not running
    st.info("Click 'Start' to begin car detection. Make sure you've entered a valid camera URL.")
    
    # Show sample image or placeholder
    placeholder_image = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder_image, "Camera Feed Will Appear Here", (120, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    frame_placeholder.image(placeholder_image, channels="BGR")

# Additional info section
with st.expander("About This Application"):
    st.markdown("""
    ### Car Detection and Distance Estimation
    
    This application uses YOLOv5 to detect cars in a video stream and estimate their distance from the camera.
    When cars are detected too close (below the warning threshold), it can send a UDP signal to trigger an external device.
    
    **Features:**
    - Real-time car detection
    - Distance estimation
    - Position tracking (left/center/right)
    - Warning signals via UDP
    """)
