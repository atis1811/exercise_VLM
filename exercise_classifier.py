import os
import requests
from PIL import Image
import base64
from collections import Counter
import cv2

def extract_frames(video_path, output_folder, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_rate == 0:
            cv2.imwrite(f"{output_folder}/frame_{count}.jpg", frame)
            count += 1
    cap.release()
extract_frames("C:/Users/atish/OneDrive/Desktop/VLM/Images/push ups.mp4", "C:/Users/atish/OneDrive/Desktop/VLM/frames output", frame_rate=30)
# NVIDIA Kosmos-2 Inference API endpoint
api_key = " Replace with your API key"
KOSMOS_API_URL = "https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2" # https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2
API_KEY = api_key  

# Define exercise types
exercise_types = ["squats", "push-ups", "pull-ups", "lunges", "jumping jacks"]

# Path to the folder containing frames
frames_folder = "C:/Users/atish/OneDrive/Desktop/VLM/frames output"

# List to store predictions
predictions = []

# Function to encode image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Iterate through all frames in the folder
for frame_file in sorted(os.listdir(frames_folder)):
    if frame_file.endswith((".jpg", ".png")):  # Check for image files
        frame_path = os.path.join(frames_folder, frame_file)
        
        # Encode the image
        encoded_image = encode_image(frame_path)

        # Prepare the corrected payload for Kosmos-2
        payload = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": f"Classify this image into one of these exercises: {', '.join(exercise_types)}."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
                ]}
            ]
        }
        
        # API headers
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Send the request to Kosmos-2 API
        response = requests.post(KOSMOS_API_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            # Extract the predicted exercise from the response
            predicted_exercise = result.get("choices", [{}])[0].get("message", {}).get("content", "unknown")
            predictions.append(predicted_exercise)
            print(f"Frame: {frame_file}, Predicted Exercise: {predicted_exercise}")
        else:
            print(f"Error with frame {frame_file}: {response.status_code} - {response.text}")

# Majority voting to determine the final predicted exercise
if predictions:
    final_exercise = Counter(predictions).most_common(1)[0][0]
    print("\nFinal Predicted Exercise:", final_exercise)
else:
    print("No predictions were made.")
