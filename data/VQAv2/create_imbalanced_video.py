import json
import cv2
import os
import numpy as np
import random

# Define paths
output_video_path = '/home/adamchun/cs231n-project/data/VQAv2/imbalanced_output_video.mp4'
combined_mapping_file_path = '/home/adamchun/cs231n-project/data/VQAv2/combined_mapping.json'
timestamp_file_path = '/home/adamchun/cs231n-project/data/VQAv2/imbalanced_timestamps.json'

# Load the combined mapping
with open(combined_mapping_file_path, 'r') as f:
    combined_mapping = json.load(f)

# Create the video from the selected images
image_paths = [entry['image_path'] for entry in combined_mapping.values()]

# Check if there are any images
if not image_paths:
    print("No images found to create video")
    exit()

# Read the first image to get the dimensions
first_image_path = next(iter(combined_mapping.values()))['image_path']
first_image = cv2.imread(first_image_path)
height, width, layers = first_image.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
fps = 1
video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Mapping of image to timestamp
key_frame_mapping = {}

# Initialize current timestamp in seconds
current_timestamp = 0

# Function to resize and add padding to an image
def resize_and_pad(image, target_height, target_width):
    h, w, _ = image.shape

    # Calculate the scaling factor to fit the image within the target dimensions
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_w, new_h))

    # Calculate padding to center the resized image
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left

    color = [0, 0, 0]  # Black color
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image

current_frame = 0
for key, entry in combined_mapping.items():
    image_path = entry['image_path']
    duration = random.randint(1, 5)  # random duration between 1 to 5 seconds

    if not os.path.exists(image_path):
        print(f"Image path does not exist: {image_path}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        continue

    img = cv2.resize(img, (width, height))  # Ensure all images are the same size

    start_frame = current_frame
    for _ in range(duration):
        video.write(img)  # Write the image for 'duration' seconds
        current_frame += 1
    end_frame = current_frame - 1  # Last frame where the image was shown

    # Record the start and end frame for this key
    key_frame_mapping[key] = f"{start_frame}-{end_frame}"

# Release the VideoWriter object
video.release()
cv2.destroyAllWindows()

with open(timestamp_file_path, 'w') as json_file:
    json.dump(key_frame_mapping, json_file, indent=4)

print("Timestamps saved to:", timestamp_file_path)
print(f"Video created successfully at {output_video_path}")

# Verify the number of frames in the output video
cap = cv2.VideoCapture(output_video_path)
frame_count_verified = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()


print(f"Total number of frames verified in the video: {frame_count_verified}")