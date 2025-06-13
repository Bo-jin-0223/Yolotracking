from ultralytics import YOLO

model= YOLO("C:/Users/USER/yolotracking/model/train/weights/best.pt")

video_path = "C:/Users/USER/yolotracking/video/video19.mp4"

results = model.predict(source=video_path, imgsz=640, save=True)