import json
import cv2
import os
import numpy as np

# Define paths
output_images_dir = '/home/adamchun/cs231n-project/data/VQAv2/selected_images'
output_video_path = '/home/adamchun/cs231n-project/data/VQAv2/output_video.mp4'
combined_mapping_file_path = '/home/adamchun/cs231n-project/data/VQAv2/combined_mapping.json'

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
first_image = cv2.imread(image_paths[0])
if first_image is None:
    print("Failed to read the first image")
    exit()
height, width, layers = first_image.shape

# Define the video codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
video = cv2.VideoWriter(output_video_path, fourcc, 1, (width, height))

# Initialize frame counter
frame_count = 0

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

# Add each image to the video
for image_path in image_paths:
    if not os.path.exists(image_path):
        print(f"Image path does not exist: {image_path}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        continue

    # Resize and add padding if dimensions do not match
    if img.shape != (height, width, layers):
        print(f"Resizing and adding padding to image: {image_path}")
        img = resize_and_pad(img, height, width)

    video.write(img)
    frame_count += 1

# Release the VideoWriter object
video.release()
cv2.destroyAllWindows()

print(f"Video created successfully at {output_video_path}")
print(f"Total number of frames added to the video: {frame_count}")

# Verify the number of frames in the output video
cap = cv2.VideoCapture(output_video_path)
frame_count_verified = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"Total number of frames verified in the video: {frame_count_verified}")