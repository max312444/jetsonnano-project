import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

# **1. CSV 파일 경로 및 데이터 로드**
csv_path = r"C:\Users\USER\balanced_training_data.csv"  # 주로 학습에 사용할 CSV 파일
df = pd.read_csv(csv_path)

# **2. SteeringDataset 정의**
class SteeringDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['frame_path']
        angle = row['angle']  # 'angle' 또는 'steering_angle' 컬럼 사용

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지 변환
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(angle, dtype=torch.float32)

# **3. 데이터 변환 및 데이터셋 준비**
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((66, 200)),  # PilotNet의 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 평균/표준편차
])

dataset = SteeringDataset(df, transform=transform)

# **4. 데이터 분할**
train_data, val_test_data = train_test_split(dataset, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# **5. PilotNet 모델 정의**
class PilotNet(nn.Module):
    def __init__(self):
        super(PilotNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # 각도 예측
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# **6. 학습 및 검증 루프**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PilotNet().to(device)
criterion = nn.SmoothL1Loss()  # Huber Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_val_loss = float('inf')
best_model_path = r"C:\Users\USER\best_pilotnet_model.pth"

epochs = 30
for epoch in range(epochs):
    # **학습**
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # **검증**
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # **출력 및 모델 저장**
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    scheduler.step()

# **7. 테스트 정확도 계산**
model.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # 정확도 계산 (허용 오차 내에서 예측 성공 판단)
        correct += torch.sum(torch.abs(outputs - labels) < 0.1).item()
        total += labels.size(0)

test_loss /= len(test_loader)
accuracy = (correct / total) * 100
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.2f}%")
print("Training completed.")
