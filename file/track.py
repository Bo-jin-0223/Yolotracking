import os
import sys
from collections import defaultdict
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import json
from json import load
from typing import Dict,List,Tuple,DefaultDict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class yolotracking():
    def __init__(self, config: Dict[str, object]):
        self.MODEL_PATH = config["MODEL_PATH"]
        self.VIDEO_PATH = config["VIDEO_PATH"]
        self.TRACKER = config["TRACKER"]
        self.WRONG_CAR = config["WRONG_CAR"]
        self.MIN_WIDTH = config["MIN_WIDTH"]
        self.MIN_HEIGHT = config["MIN_HEIGHT"]
        self.OUTPUT_DIRS = config["OUTPUT_DIRS"]
        self.OUTPUT_VIDEO = config["OUTPUT_VIDEO"]
        self.IMAGES_DIR = config["IMAGES_DIR"]
        self.LABELS_DIR = config["LABELS_DIR"]
        self.Function_mode=config["Function_mode"]
        
    
    def setup_dirs(self)->Tuple[str,str]:
        for path in self.OUTPUT_DIRS.values():
            os.makedirs(path,exist_ok=True)
        os.makedirs(self.WRONG_CAR,exist_ok=True)
        os.makedirs(self.IMAGES_DIR,exist_ok=True)
        os.makedirs(self.LABELS_DIR,exist_ok=True)

    def set_cap(self)->None:
        cap = cv2.VideoCapture(self.VIDEO_PATH)
        return cap

    def tracking(self,cap:cv2.VideoCapture
                 )->Tuple[YOLO, List[Results],List[np.ndarray],List[Tuple[int, Results, np.ndarray]]]:
        model = YOLO(self.MODEL_PATH)
        results = model.track(source=self.VIDEO_PATH, save=False, tracker=self.TRACKER, stream=True)

        frame_index = 0
        all_img = []
        img_list = []
        for r in results:
            success, orig_img = cap.read()
            if not success:
                break
            all_img.append(orig_img.copy())
            img_list.append((frame_index, r, orig_img))
            frame_index += 1
        return model,all_img,img_list

    def statistics(self,
                    img_list:List[Tuple[int, Results, np.ndarray]]
                    )->Tuple[DefaultDict[int, DefaultDict[int, int]], 
                        DefaultDict[int, List[Tuple[int, int, int, int, int]]]]:
        
        id_class_counter = defaultdict(lambda:defaultdict(int))
        id_boxes = defaultdict(list)

        for frame_index, r ,orig_img in img_list:
            if r.boxes.id is None:
                continue
            ids =  r.boxes.id.int().tolist()
            classes = r.boxes.cls.int().tolist()
            for track_id, cls in zip(ids, classes):
                id_class_counter[track_id][cls] += 1
            for box in r.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_img.shape[1], x2), min(orig_img.shape[0], y2)
                track_id = int(box.id[0])
                id_boxes[track_id].append((frame_index, x1, y1, x2, y2))
        return id_class_counter,id_boxes
    
    def classification(self,model:YOLO,
                       id_boxes:DefaultDict[int, List[Tuple[int, int, int, int, int]]],
                       id_class_counter:defaultdict[int, DefaultDict[int, int]]):
        names = model.names
        most_frequent_category_data={}
        for track_id in id_boxes:
            class_counts = id_class_counter[track_id]
            most_frequent_ID = max(class_counts.items(), key=lambda x: x[1])[0]
            most_frequent_category = names[most_frequent_ID]
            most_frequent_category_data[track_id] = most_frequent_category
        return most_frequent_category_data   
        

    def output_images(self,
                  id_boxes: DefaultDict[int, List[Tuple[int, int, int, int, int]]],
                  most_frequent_category_data: Dict[int, str],
                  all_img: List[np.ndarray])->None:
        image_counter = 1
        for track_id, boxes in id_boxes.items():
            images_most_category_name = most_frequent_category_data.get(track_id, "unknown")
            for frame_id, x1, y1, x2, y2 in boxes:
                width = x2 - x1
                height = y2 - y1
                x3 = x1 + width / 2
                y3 = y2 + height / 2
                orig_img = all_img[frame_id]
                cropped = orig_img[y1:y2, x1:x2]
                if width < self.MIN_WIDTH or height < self.MIN_HEIGHT:
                    save_path = os.path.join(self.WRONG_CAR, f"wrong_car_{frame_id}.jpg")
                else:
                    filename = (
                        f"{images_most_category_name}_id{track_id}_{image_counter}_"
                        f"X{x3}_Y{y3}_W{width}_H{height}.jpg"
                    )
                    save_path = os.path.join(self.OUTPUT_DIRS[images_most_category_name], filename)
                    image_counter += 1

                cv2.imwrite(save_path, cropped)

    def output_video(self,most_frequent_category_data:Dict[int, str],
                all_img:List[np.ndarray],
                id_boxes:DefaultDict[int, List[Tuple[int, int, int, int, int, int]]])->None:

        for track_id, boxes in id_boxes.items():
            video_most_category_name = most_frequent_category_data.get(track_id, "unknown")
            for frame_id, x1, y1, x2, y2 in boxes:
                frame = all_img[frame_id]
                label = f"{video_most_category_name} ID:{track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        h, w = all_img[0].shape[:2]
        out = cv2.VideoWriter(self.OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))

        for frame in all_img:
            out.write(frame)
        out.release()

    def write_in_txt(self,img_list: List[Tuple[int, Results, np.ndarray]])->None:
        frame_count = 0
        for frame_index, r, orig_img in img_list:
            if r.boxes.id is None:
                frame_index += 1
                continue

            classes = r.boxes.cls.int().tolist()
            boxes = r.boxes.xyxy.tolist()
            img_h, img_w = orig_img.shape[:2]

            txt_datas = []
            for (cls_id, (x1, y1, x2, y2)) in zip(classes, boxes):
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(img_w, int(x2))
                y2 = min(img_h, int(y2))

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                txt_data = f"{cls_id} {x_center} {y_center} {w} {h}"
                txt_datas.append(txt_data)

            filename = f"frame_{frame_count:05d}_{classes}.jpg"
            save_path = os.path.join(self.IMAGES_DIR, filename)
            cv2.imwrite(save_path, orig_img)
        
            txt_path = os.path.join(self.LABELS_DIR, f"frame_{frame_count:05d}_{classes}.txt")
            with open(txt_path, "w") as f:
                f.write("\n".join(txt_datas))
                print(f"Saved {txt_path}")

            frame_index += 1
            frame_count += 1    

    def parameter_judgment(self):
        if ".pt" not in config["MODEL_PATH"]:
            print("MODEL_PATH error"); sys.exit(0)
        if ".mp4" not in config["VIDEO_PATH"]:
            print("VIDEO_PATH error"); sys.exit(0)
        if ".mp4" not in config["OUTPUT_VIDEO"]:
            print("OUTPUT_VIDEO error"); sys.exit(0)

        if config["TRACKER"] not in ["botsort.yaml", "bytetrack.yaml"]:
            print("TRACKER error"); sys.exit(0)

        config_type = {
            "WRONG_CAR": str,
            "MIN_WIDTH": int,
            "MIN_HEIGHT": int,
            "OUTPUT_DIRS": dict,
            "IMAGES_DIR": str,
            "LABELS_DIR": str
        }
        for key, config_type in config_type.items():
            if not isinstance(config[key], config_type):
                print(f"{key} error"); sys.exit(0)

        if config["Function_mode"] not in ["classification", "write_in_txt"]:
            print("Function_mode error"); sys.exit(0)

        
    def yolo_identic(self):

        self.setup_dirs()
        cap = self.set_cap()
        model, all_img,img_list = self.tracking(cap)
        if self.Function_mode == "classification":
            id_class_counter,id_boxes = self.statistics(img_list)
            most_frequent_category_data = self.classification(model,id_boxes, id_class_counter)
            self.output_images(id_boxes, most_frequent_category_data, all_img)
            self.output_video(most_frequent_category_data,all_img,id_boxes)
        elif self.Function_mode == "write_in_txt":
            self.write_in_txt(img_list)
        cap.release()
        
    def run(self):
        self.parameter_judgment()
        self.yolo_identic()
            
if __name__ == "__main__":
    with open('parameter.json', 'r') as f:
        config = json.load(f)
    tracker = yolotracking(config)
    tracker.run()    
            
