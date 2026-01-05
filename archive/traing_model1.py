import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# 원본 이미지 폴더
input_folder = r"C:\Users\USER\jetson_nano_dataset"

# CSV 경로 및 새 CSV 경로
existing_csv_path = r"C:\Users\USER\jetson_csv\steering_data.csv"  # 기존 CSV 파일 경로
new_csv_path = r"C:\Users\USER\new_csv\new_steering_dat.csv"  # 새 CSV 파일 경로

# 새 CSV 파일 저장 (기존 CSV 파일의 데이터 사용)
try:
    df = pd.read_csv(existing_csv_path)
    os.makedirs(os.path.dirname(new_csv_path), exist_ok=True)
    df.to_csv(new_csv_path, index=False)
    print(f"새로운 CSV 파일이 생성되었습니다: {new_csv_path}")
except FileNotFoundError:
    print(f"기존 CSV 파일을 찾을 수 없습니다: {existing_csv_path}")
except Exception as e:
    print(f"오류 발생: {e}")

# 데이터셋 클래스 정의
class SteeringDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None):
        self.image_folder = image_folder
        self.csv_file = csv_file
        self.transform = transform
        
        # CSV 파일 읽기
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 이미지 파일 경로와 각도 정보 가져오기
        img_name = os.path.join(self.image_folder, self.df.iloc[idx, 0])  # 첫 번째 열: 이미지 경로
        image = Image.open(img_name)
        steering_angle = self.df.iloc[idx, 1]  # 두 번째 열: steering_angle
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(steering_angle, dtype=torch.float)

# 이미지 전처리 및 데이터 변환
transform = transforms.Compose([
    transforms.Resize((66, 200)),  # CNN 입력 크기 맞추기
    transforms.ToTensor(),         # 이미지를 텐서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
])

# 데이터셋 준비
dataset = SteeringDataset(input_folder, new_csv_path, transform)

# DataLoader 준비
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# CNN 모델 정의
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2)
        self.fc1 = nn.Linear(48 * 8 * 25, 1000)  # FC layer 1
        self.fc2 = nn.Linear(1000, 1)            # FC layer 2 (output steering angle)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output steering angle
        return x

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 생성 및 GPU로 이동
model = CNNModel().to(device)

# 손실 함수 및 최적화 함수 설정
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)  # 학습률 증가

# 정확도 계산 함수
def calculate_accuracy(outputs, labels, threshold=1.0):
    # 예측과 실제값의 차이가 threshold 이하일 때 정확하다고 판단
    difference = torch.abs(outputs.squeeze() - labels)
    correct = (difference <= threshold).sum().item()
    accuracy = correct / len(labels)
    return accuracy

# 학습 루프
num_epochs = 10  # 학습 에폭 수
for epoch in range(num_epochs):
    model.train()  # 학습 모드
    running_loss = 0.0
    running_accuracy = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 데이터 GPU로 이동

        optimizer.zero_grad()  # 그래디언트 초기화
        
        # 모델 예측
        outputs = model(inputs)
        
        # 손실 계산
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()  # 역전파
        optimizer.step()  # 파라미터 업데이트
        
        running_loss += loss.item()
        
        # 정확도 계산
        accuracy = calculate_accuracy(outputs, labels)
        running_accuracy += accuracy
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Accuracy: {running_accuracy/len(train_loader)}")

# 모델 저장
torch.save(model.state_dict(), "steering_model.pth")
print("모델이 저장되었습니다.")
