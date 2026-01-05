import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

class DrivingDataset:
    def __init__(self, image_dir, label_file):
        self.image_dir = image_dir  # 리사이즈된 이미지가 저장된 경로
        self.label_file = label_file  # 레이블 CSV 파일 경로
        self.labels = pd.read_csv(label_file)  # 레이블 CSV 파일 읽기
        self.image_paths = self.get_image_paths()  # 이미지 파일 경로 얻기

    def get_image_paths(self):
        """이미지 경로를 하위 폴더까지 포함하여 전부 읽어오기"""
        image_paths = []
        # 주어진 경로에서 하위 폴더를 모두 탐색
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):  # 이미지 파일만 필터링
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # 이미지 로딩
        image = Image.open(image_path)
        image = np.array(image)  # 이미지를 NumPy 배열로 변환
        # 이미지 파일명으로 레이블 찾기
        label = self.labels.loc[self.labels['filename'] == os.path.basename(image_path), 'steering_angle'].values[0]
        return image, label

# 예시 사용
image_dir = r"C:\Users\USER\jetson_nano_dataset"  # 리사이즈된 이미지가 저장된 경로 (수정된 경로)
label_file = r"C:\Users\USER\jetson_csv\jetson_data.csv"  # CSV 레이블 파일 경로 (수정된 경로)

dataset = DrivingDataset(image_dir, label_file)

# 데이터셋의 이미지 경로와 레이블 확인
for idx in range(len(dataset.image_paths)):
    image, label = dataset[idx]
    print(f"Image path: {dataset.image_paths[idx]}, Steering angle: {label}")
