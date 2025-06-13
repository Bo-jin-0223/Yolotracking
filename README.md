# 專案名稱
**Yolotracking**

# 專案介紹
本專案透過Roboflow創建數據集，利用數據集訓練模型和模型追蹤影片，並將物件進行分類裁切與輸出影片。同時產生新的數據集數據，將新的數據集產生新的模型重新追蹤影片以提高模型追蹤影片的準確率。

## 專案功能
- 將影片以每一楨畫面擷取以方便上傳至Roboflow創建數據集
- 將Roboflow數據集訓練成模型並套用至影片上
- 利用模型追蹤影片並分類物件類別，同時產生具有追蹤框的影片
- 利用模型追蹤影片並產生新的數據集，提高數據集的辨識準確率
- 本專案包含單元測試unittest程式，方便各位測試使用

## 專案結構
```bash
file/
│
├── test/                  # 單元測試資料夾
│   ├── __init__.py
│   ├── __pycache__/
│   └── test_track.py      # 單元測試檔案
│
├── model.py               # 將數據集訓練模型
├── parameter.json         # 紀錄參數的檔案
├── screenshot.py          # 將影片每一楨擷取成圖片
├── track.py               # 將影片追蹤並裁切分類物件，或將影片產生數據集
├── train.py               # 測試模型套用影片效果
├── yolo11n.pt             # 使用的yoloV11套件
```

### 環境需求
- GPU(非強迫，但有GPU速度會更快)
- python 3.10+
- Opencv 4.11.0.86 
- ultralytics 8.3.152

### 前置作業(安裝套件)
- 請先準備一部影片至專案內部
- 建立虛擬環境並安裝必要套件
```bash
python -m venv venv
venv/Scripts/activate   #macOS/Linux 改用 source venv/bin/activate
pip install opencv-python
pip install ultralytics
cd file
```
### 注意事項
1. 使用screenshot.py前請將screenshot.py內的video_path更換為自己存放影片的位置
2. 使用model.py前請先進入[Roboflow 官方網站] https://roboflow.com/ 創建數據集並下載至專案內
3. 使用train.py前請將train.py內的video_path更換為自己存放影片的位置
4. 使用track.py前請將parameter.json內的"VIDEO_PATH"更換為自己存放影片的位置
5. 使用track.py時若想將物件進行分類裁切與輸出影片，請先至parameter.json將Function_mode更換為"classification"
6. 使用track.py時若想產生數據集，請先至parameter.json將Function_mode更換為"write_in_txt"
7. 由於yolo數據集規定圖片必須放入images資料夾，紀錄物件位置的記事本必須放入labels資料夾，因此請不要更換資料夾名稱

### 參考資源
- python官方網站 https://docs.python.org/3/
- opencv https://opencv.org/
- ultralytics https://www.ultralytics.com/zh
- roboflow https://roboflow.com/
- 教學檔案 https://docs.google.com/presentation/d/19blP637dl4qPmJAiD6Hbb1fZpU2iDv8i/edit?usp=drive_link&ouid=100548288520061624838&rtpof=true&sd=true







