import os
import pandas as pd
from PIL import Image
import numpy as np

class DrivingDataset:
    def __init__(self, image_dir, csv_file):
        self.image_dir = image_dir
        self.csv_file = csv_file
        self.labels = pd.read_csv(csv_file)  # 'filename', 'steering_angle'
        
        # 데이터셋에 맞는 이미지 경로 설정
        self.image_paths = [os.path.join(self.image_dir, fname) for fname in self.labels['filename']]

    def __getitem__(self, idx):
        # 이미지 경로 가져오기
        image_path = self.image_paths[idx]

        # 이미지 파일 존재 여부 체크
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
        
        try:
            # 이미지 읽기
            image = Image.open(image_path).convert('RGB')  # RGB로 변환
            image = np.array(image)
        except Exception as e:
            raise IOError(f"이미지 로드 실패: {image_path}. 에러: {e}")
        
        # 이미지 차원 확인 (height, width, 3)
        if image.shape[-1] != 3:
            raise ValueError(f"이미지가 RGB 채널을 가지고 있지 않습니다: {image_path}")
        
        # steering_angle 값 가져오기
        label = self.labels.loc[self.labels['filename'] == os.path.basename(image_path), 'steering_angle']
        
        # 라벨이 없을 경우 기본값 설정
        if label.empty:
            raise ValueError(f"파일명에 해당하는 라벨을 찾을 수 없습니다: {image_path}")
        
        label = label.values[0]
        
        return image, label
    
    def __len__(self):
        return len(self.labels)
