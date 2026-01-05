import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# ==== 검증 데이터셋 클래스 ====
class LineTrackingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        # 검증 데이터에는 레이블이 필요 없을 수 있음 (예측만 수행)
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]  # 파일명도 반환해서 결과 매칭

# ==== 검증 데이터 전처리 ====
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 학습 시 사용했던 이미지 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==== 모델 불러오기 ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'trained_model.pth'  # 학습된 모델 경로
model = torch.load(model_path)
model = model.to(device)
model.eval()  # 모델을 평가 모드로 설정

# ==== 검증 데이터셋 및 데이터로더 ====
data_dir = 'path_to_validation_images'  # 검증 데이터 경로
batch_size = 8

validation_dataset = LineTrackingDataset(data_dir, transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# ==== 검증 과정 ====
print("Starting validation...")

with torch.no_grad():  # 그래디언트 계산 비활성화
    for images, filenames in validation_loader:
        images = images.to(device)
        outputs = model(images)  # 모델에 이미지 입력
        
        # 예측 결과 처리 (예: 분류 문제라면 소프트맥스 적용)
        _, predicted = torch.max(outputs, 1)
        
        # 결과 출력
        for i in range(len(filenames)):
            print(f"Image: {filenames[i]}, Predicted: {predicted[i].item()}")

print("Validation completed.")
