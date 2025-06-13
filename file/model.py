from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(data="C:/Users/USER/yolotracking/yolo_dataset/data.yaml",
                      epochs=5,
                      batch=64,
                      imgsz=640,
                      project="C:/Users/USER/yolotracking/model",
                      name="train",
                      device=0
                      )