import torch
import os
import pandas as pd

class DrivingDataset:
    def __init__(self, image_dir, csv_file):
        self.image_dir = image_dir
        self.labels = pd.read_csv(csv_file)

    def __getitem__(self, index):
        image_name = self.labels.iloc[index, 0]
        image_path = os.path.join(self.image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"경고: {image_path} 경로에 이미지가 없습니다.")
        
        # 디버깅용 출력
        print(f"파일 이름: {image_name}, 경로: {image_path}")
        
        # 이미지를 불러오는 코드
        image = self.load_image(image_path)

        # 라벨 값 추출
        label = self.labels.loc[self.labels['filename'] == os.path.basename(image_path), 'steering_angle']
        
        # 라벨을 찾을 수 없을 때 경고 메시지 출력
        if label.empty:
            print(f"경고: {os.path.basename(image_path)}의 라벨을 찾을 수 없습니다.")
            label = 0.0  # 기본값 0.0으로 설정하거나 다른 처리를 할 수 있습니다.
        else:
            label = label.values[0]
        
        return image, label

    def load_image(self, image_path):
        # 이미지를 로드하는 함수 (예시로 PIL을 사용)
        from PIL import Image
        return Image.open(image_path)

# 예시로 데이터를 로드하고 사용할 때
image_dir = r"C:\Users\USER\jetson_nano_dataset"
csv_file = r"C:\Users\USER\jetson_csv\jetson_data.csv"

dataset = DrivingDataset(image_dir, csv_file)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    # 모델 학습 코드
    pass
