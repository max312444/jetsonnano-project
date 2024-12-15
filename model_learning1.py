import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

# 1. 데이터 전처리: 이미지 크기 조정
def resize_images(input_path, output_path, target_size=(66, 200)):
    # input_path: 원본 이미지가 저장된 디렉토리
    # output_path: 리사이즈된 이미지가 저장될 디렉토리
    os.makedirs(output_path, exist_ok=True)
    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        resized_img = cv2.resize(img, target_size)
        cv2.imwrite(os.path.join(output_path, img_name), resized_img)

# 아래에서 원본 이미지 폴더와 리사이즈 후 저장할 폴더 경로를 설정하세요.
resize_images(
    input_path= r"C:\jetson_nano_dataset",         # 원본 이미지 폴더 경로 (변경 가능)
    output_path= r"C:\change_image" # 리사이즈된 이미지 저장 경로 (변경 가능)
)

# 2. 데이터 증강 및 정규화
augmentations = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 밝기, 대비, 채도 변경
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
    transforms.RandomRotation(degrees=10),  # 회전
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -1 ~ 1 스케일
])

# 3. 데이터셋 클래스
class DrivingDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        # image_dir: 이미지 파일이 저장된 디렉토리
        # label_file: 레이블 정보가 담긴 CSV 파일 경로
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_file)  # filename, steering_angle
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.labels.iloc[idx, 0])
        image = cv2.imread(img_path)[:, :, ::-1]  # BGR → RGB 변환
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# 아래에서 리사이즈된 이미지 폴더와 레이블 파일 경로를 설정하세요.
dataset = DrivingDataset(
    image_dir= r"C:\change_image", # 리사이즈된 이미지 폴더 경로 (변경 가능)
    label_file= r"C:\jetson_csv",   # 레이블 CSV 파일 경로 (변경 가능)
    transform=augmentations
)

# 4. 데이터 분리
train_data, test_data = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

train_loader = DataLoader(dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(train_data))
val_loader = DataLoader(dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(val_data))
test_loader = DataLoader(dataset, batch_size=32, sampler=torch.utils.data.SubsetRandomSampler(test_data))

# 5. 모델 정의 (NVIDIA 모델)
class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 2 * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # 조향 각도 출력
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

model = NvidiaModel()

# 6. 손실 함수 및 옵티마이저
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. 학습 루프
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # 정확도 계산: 예측된 각도와 실제 각도 차이를 비교
        pred_angles = outputs
        total_predictions += len(labels)
        correct_predictions += torch.sum(torch.abs(pred_angles - labels) < 0.1).item()  # 0.1을 정확도로 사용

    # 검증 루프
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

    # 학습 손실, 정확도 및 검증 손실 출력
    train_accuracy = correct_predictions / total_predictions * 100  # 정확도 계산 (percentage)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}")

# 8. 테스트
model.eval()
test_loss = 0
correct_predictions = 0
total_predictions = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        test_loss += loss.item()

        # 정확도 계산
        pred_angles = outputs
        total_predictions += len(labels)
        correct_predictions += torch.sum(torch.abs(pred_angles - labels) < 0.1).item()

test_accuracy = correct_predictions / total_predictions * 100
print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%")
