import os
import requests
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from PIL import Image  # Import PIL for better image handling
import cv2
import numpy as np
from roboflow import Roboflow

# ✅ Load environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")

# ✅ Function to send Telegram alerts
def send_telegram_alert(message):
    """Sends an alert message to Telegram when violence is detected."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "MarkdownV2",  # Escape special characters in MarkdownV2
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print("✅ Telegram alert sent successfully!")
    else:
        print("❌ Failed to send alert:", response.text)


# ✅ Function to predict from a local image
def image_pred(path):
    """Perform inference on a local image using Roboflow."""
    
    # Initialize the Roboflow client
    CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY)

    # Perform inference
    result = CLIENT.infer(path, model_id="violence-detection-up5bw/1")

    # Load image using Matplotlib
    image = plt.imread(path)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)  # Show the image

    detected = False  # Flag to check if violence is detected
    alert_message = "🚨 *Violence Detected\\!* 🚨\n\n"

    # Draw bounding boxes if predictions exist
    for pred in result["predictions"]:
        detected = True  # Set flag if violence is detected

        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        confidence = pred["confidence"]
        label = f"{pred['class']} ({confidence:.2f})"

        # Convert from center-based coordinates to top-left corner
        x1, y1 = x - w / 2, y - h / 2

        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Add text label
        ax.text(x1, y1 - 10, label, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='red'))

        # Append details to the alert message
        alert_message += f"🔥 *Class:* `{pred['class']}`\n🎯 *Confidence:* `{confidence:.2f}%`\n📍 *Location:* `(x={x}, y={y})`\n\n"

    # Hide axes and save the image
    ax.axis("off")
    plt.savefig("static/output.png")
    plt.close()

    # Send Telegram alert if violence is detected
    if detected:
        send_telegram_alert(alert_message)


# ✅ Function to predict from an image URL
def global_url(path):
    """Perform inference on an image URL using Roboflow."""
    
    # Initialize the Roboflow client
    CLIENT = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=ROBOFLOW_API_KEY)

    # Infer using the image URL
    result = CLIENT.infer(path, model_id="violence-detection-up5bw/1")

    # Load the image from URL
    response = requests.get(path)
    image = Image.open(BytesIO(response.content))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)

    detected = False
    alert_message = "🚨 *Violence Detected\\!* 🚨\n\n"

    for pred in result["predictions"]:
        detected = True

        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        confidence = pred["confidence"]
        label = f"{pred['class']} ({confidence:.2f})"

        x1, y1 = x - w / 2, y - h / 2
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 10, label, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='red'))

        alert_message += f"🔥 *Class:* `{pred['class']}`\n🎯 *Confidence:* `{confidence:.2f}%`\n📍 *Location:* `(x={x}, y={y})`\n\n"

    ax.axis("off")
    plt.savefig("static/output.png")
    plt.close()

    if detected:
        send_telegram_alert(alert_message)


# ✅ Function for real-time video detection
def live_video():
    """Perform real-time inference on live video."""
    
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project("violence-detection-up5bw")
    model = project.version("1").model

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, confidence=40, overlap=30).json()

        violence_detected = False
        alert_message = "🚨 *Violence Detected\\!* 🚨\n\n"

        for pred in result.get("predictions", []):
            violence_detected = True

            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            confidence = pred["confidence"]
            label = f"{pred['class']} ({confidence:.2f})"

            x1, y1 = int(x - w / 2), int(y - h / 2)
            w, h = int(w), int(h)

            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            alert_message += f"🔥 *Class:* `{pred['class']}`\n🎯 *Confidence:* `{confidence:.2f}%`\n📍 *Location:* `(x={x}, y={y})`\n\n"

        cv2.imshow("Live Detection", frame)

        if violence_detected:
            send_telegram_alert(alert_message)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
# ✅ Function for video file prediction
def video_pred(path):
    """Perform inference on a video file using Roboflow."""

    # 🔹 Initialize Roboflow
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project("violence-detection-up5bw")
    model = project.version("1").model

    # 🔹 Load video file
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("❌ Error: Unable to open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop when no frame is captured

        # 🔹 Perform inference (Convert OpenCV frame to NumPy array)
        result = model.predict(frame, confidence=40, overlap=30).json()

        violence_detected = False  # Flag to track violence detection
        alert_message = "🚨 *Violence Detected\\!* 🚨\n\n"

        # 🔹 Process detections
        for pred in result.get("predictions", []):
            violence_detected = True  # Set flag if violence is detected

            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            confidence = pred["confidence"]
            label = f"{pred['class']} ({confidence:.2f})"

            # Convert center (x, y) to top-left (x1, y1)
            x1, y1 = int(x - w / 2), int(y - h / 2)
            w, h = int(w), int(h)

            # 🔹 Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

            # 🔹 Display label
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 255), 2)

            # 🔹 Append details to the alert message
            alert_message += (
                f"🔥 *Class:* `{pred['class']}`\n"
                f"🎯 *Confidence:* `{confidence:.2f}%`\n"
                f"📍 *Location:* `(x={x}, y={y})`\n\n"
            )

        # 🔹 Show the processed video
        cv2.imshow("Violence Detection", frame)
         # 🔹 Send Telegram alert *only once per frame* if violence is detected
        if violence_detected:
            send_telegram_alert(alert_message)

        # 🔹 Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 🔹 Release resources
    cap.release()
    cv2.destroyAllWindows()
