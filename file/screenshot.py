import cv2
import os

video_path = "C:/Users/USER/yolotracking/video/video25.mp4"
frames_dir = "C:/Users/USER/yolotracking/frame/video25"

os.makedirs(frames_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
frame_id = 0
save_count = 0

while cap.isOpened():
    r , frame = cap.read()
    if not r:
        break

    if frame_id %1 == 0:
        frame_name = os.path.join(frames_dir, f"frame_{save_count:05d}.jpg")
        cv2.imwrite(frame_name, frame)
        save_count += 1

    frame_id +=1    
