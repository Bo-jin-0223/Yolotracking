from collections import defaultdict
import unittest
from unittest.mock import MagicMock,patch,mock_open
import numpy as np
import os
import shutil
import torch
import tempfile
from track import yolotracking


class DummyBox:
    def __init__(self, box_id, cls, xyxy):
        self.id = np.array([box_id])
        self.cls = np.array([cls])
        self.xyxy = np.array([xyxy])

class DummyResult:
    def __init__(self, ids, cls, boxes):
        self.boxes = MagicMock()
        self.boxes.id = torch.tensor(ids)
        self.boxes.cls = torch.tensor(cls)
        self.boxes.xyxy = torch.tensor(boxes)

        self.boxes.__iter__.return_value = [
            DummyBox(i, c, b) for i, c, b in zip(ids, cls, boxes)
        ]

class DummyModel:
    def __init__(self, names):
        self.names = names

class TestYoloTracking(unittest.TestCase):

    def setUp(self):
        # 每一個testcase開始前都會被呼叫 -> 前置作業
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "MODEL_PATH": "C:/Users/USER/yolotracking/model/model8/weights/best.pt",  
            "VIDEO_PATH": "C:/Users/USER/yolotracking/video/video25.mp4", 
            "TRACKER" :"botsort.yaml",             
            "WRONG_CAR": "C:/Users/USER/yolotracking/output/Classification/wrong_car",
            "MIN_WIDTH": 30,
            "MIN_HEIGHT": 20,
            "OUTPUT_DIRS": {
                "cars": "C:/Users/USER/yolotracking/output/Classification/cars",
                "motor": "C:/Users/USER/yolotracking/output/Classification/motor",
                "pickedtruck": "C:/Users/USER/yolotracking/output/Classification/pickedtruck",
                "big_truck": "C:/Users/USER/yolotracking/output/Classification/big_truck"
                },
            "OUTPUT_VIDEO": "C:/Users/USER/yolotracking/output/Classification/video.mp4",
            "Function_mode": "write_in_txt"
        }
        self.tracker = yolotracking(self.config)

    def tearDown(self):
        # 每一個testcase結束後會被呼叫 -> 清理善後作業
        shutil.rmtree(self.temp_dir)

    def test_setup_dirs(self):
        self.tracker.setup_dirs()
        for path in self.config["OUTPUT_DIRS"].values():
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(self.config["WRONG_CAR"]))
        self.assertTrue(os.path.exists(self.config["IMAGES_DIR"]))
        self.assertTrue(os.path.exists(self.config["LABELS_DIR"]))
        

    def test_set_cap(self):
        cap = self.tracker.set_cap()
        self.assertTrue(cap.isOpened())

    @patch("track.YOLO")
    def test_tracking(self, mock_yolo_class):
        dummy_results = [DummyResult([1], [0], [[10, 10, 50, 50]]) for _ in range(5)]
        mock_model = MagicMock()
        mock_model.track.return_value = iter(dummy_results)
        mock_yolo_class.return_value = mock_model

        dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        mock_cap = MagicMock()
        mock_cap.read.side_effect = [(True, dummy_img) for _ in range(5)] + [(False, None)]

        model, all_img, img_list = self.tracker.tracking(mock_cap)
     
        self.assertEqual(model, mock_model)
        self.assertEqual(len(all_img), 5)
        self.assertEqual(len(img_list), 5)

        for i, (frame_index, result, img) in enumerate(img_list):
            self.assertEqual(frame_index, i)
            self.assertIsInstance(result, DummyResult)
            np.testing.assert_array_equal(img, dummy_img)

    def test_statistics(self):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

        img_list = [
            (0, DummyResult([1], [0], [[10, 10, 50, 50]]), dummy_img),
            (1, DummyResult([1], [0], [[15, 15, 55, 55]]), dummy_img),
            (2, DummyResult([1], [3], [[20, 20, 60, 60]]), dummy_img),
            (3, DummyResult([2], [1], [[100, 100, 150, 150]]), dummy_img),
            (4, DummyResult([3], [2], [[150, 200, 100, 100]]), dummy_img)
        ]

        id_class_counter, id_boxes = self.tracker.statistics(img_list)

        self.assertEqual(id_class_counter[1][0],2)
        self.assertEqual(id_class_counter[1][3],1)
        self.assertEqual(id_class_counter[2][1],1)
        self.assertEqual(id_class_counter[3][2],1)

        self.assertEqual(len(id_boxes[1]),3)
        self.assertEqual(len(id_boxes[2]),1)
        self.assertEqual(len(id_boxes[3]),1)
    
    def test_classification(self):
        model = DummyModel(["big_truck", "cars", "motor", "pickedtruck"])

        id_class_counter = defaultdict(lambda: defaultdict(int))
        id_class_counter[1][0] = 2
        id_class_counter[1][3] = 1
        id_class_counter[2][1] = 1
        id_class_counter[3][2] = 1
        
        id_boxes = defaultdict(list)
        id_boxes[1] = [(0,10,10,50,50), (1,15,15,55,55),(2,20,20,60,60)]
        id_boxes[2] = [(3,100,100,150,150)]
        id_boxes[3] = [(4,150,200,100,100)]
        result = self.tracker.classification(model, id_boxes, id_class_counter)
        self.assertEqual(result[1], "big_truck")
        self.assertEqual(result[2], "cars")
        self.assertEqual(result[3], "motor")

    def test_output_images(self):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        all_img = [dummy_img] * 5  
        id_boxes = {
            1: [(0, 10, 20, 100, 200)],
            2: [(1, 15, 25, 110, 210)],
            3: [(2, 20, 30, 120, 220)],
            4: [(3, 25, 35, 130, 230)],
            5: [(4, 30, 40, 140, 240)],
        }

        most_frequent_category_data = {
            1: "big_truck",
            2: "cars",
            3: "motor",
            4: "cars",
            5: "cars"
        }

        with patch("cv2.imwrite", return_value=True) as mock_writer:
            self.tracker.output_images(id_boxes, most_frequent_category_data, all_img)
            self.assertEqual(mock_writer.call_count, 5)

    def test_output_video(self):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        all_img = [dummy_img] * 5
        
        most_category_name_data = {1: "car", 2: "truck", 3:"motor"}

        id_boxes = defaultdict(list)
        id_boxes[1] = [(0,10,10,50,50), (1,15,15,55,55),(2,20,20,60,60)]
        id_boxes[2] = [(3,100,100,150,150)]
        id_boxes[3] = [(4,150,200,100,100)]

        with patch("cv2.VideoWriter") as mock_writer:
            mock_out = MagicMock()
            mock_writer.return_value = mock_out
            self.tracker.output_video(most_category_name_data, all_img, id_boxes)
            self.assertEqual(mock_out.write.call_count, 5)
            mock_out.release.assert_called_once()

    def test_write_in_txt(self):
        images_dir = "C:/Users/USER/yolotracking_todo/output/write_in_txt/images"
        labels_dir = "C:/Users/USER/yolotracking_todo/output/write_in_txt/labels"
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        img_list = [
            (0, DummyResult([1], [0], [[10, 10, 50, 50]]), dummy_img),
            (1, DummyResult([1], [0], [[15, 15, 55, 55]]), dummy_img),
            (2, DummyResult([1], [3], [[20, 20, 60, 60]]), dummy_img),
            (3, DummyResult([2], [1], [[100, 100, 150, 150]]), dummy_img),
            (4, DummyResult([3], [2], [[150, 200, 100, 100]]), dummy_img)
        ]

        with patch("cv2.imwrite") as mock_writer, \
        patch("builtins.open", new_callable=mock_open) as mock_file:
            self.tracker.write_in_txt(img_list,images_dir,labels_dir)
            self.assertEqual(mock_writer.call_count, 5)
            self.assertEqual(mock_file.call_count, 5)
    
    def test_parameter_judgment(self):
        self.assertIn(".pt", (self.config["MODEL_PATH"]))
        self.assertIn(".mp4", (self.config["VIDEO_PATH"]))
        self.assertIn(self.config["TRACKER"],["botsort.yaml","bytetrack.yaml"])
        self.assertIsInstance((self.config["WRONG_CAR"]), str)
        self.assertIsInstance((self.config["MIN_WIDTH"]), int)
        self.assertIsInstance((self.config["MIN_HEIGHT"]), int)
        self.assertIsInstance((self.config["OUTPUT_DIRS"]), dict)
        self.assertIn(".mp4", (self.config["OUTPUT_VIDEO"]))
        self.assertIn(self.config["Function_mode"],["classification","write_in_txt"])


if __name__ == '__main__':
    unittest.main()