import cv2
import numpy as np

def extract_random_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    random_frame = np.random.randint(0, total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
    success, frame = cap.read()
    cap.release()
    return frame, random_frame

# Example usage
video_path = '/home/adamchun/cs231n-project/data/MSRVTT/videos/all/video9997.mp4'
frame, frame_number = extract_random_frame(video_path)
cv2.imwrite('/home/adamchun/cs231n-project/finetune/atp-video-language/src/random_frame.jpg', frame)  # Save the frame as an image
