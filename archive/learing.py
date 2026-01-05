import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# **1. 데이터셋 정의**
class SteeringDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.categories = [30, 60, 90, 120, 150]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['frame_path']
        category = int(row['steering_category'])

        # 경로 구분자 정리
        image_path = os.path.normpath(image_path)  # 경로 구분자 통일

        # 이미지 로드 및 전처리
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"파일이 존재하지 않음: {image_path}")
            
            image = Image.open(image_path).convert("RGB")
            image = image.resize((200, 66))  # 이미지 크기 조정
            image = np.array(image) / 255.0  # 정규화
            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW (PyTorch 입력 형식)
        except Exception as e:
            print(f"Warning: 이미지를 로드할 수 없습니다: {image_path}. 오류: {e}.")
            raise ValueError(f"이미지 로드 실패: {image_path}")  # 이미지를 로드할 수 없으면 예외 발생

        return torch.tensor(image, dtype=torch.float32), torch.tensor(category, dtype=torch.long)

# **2. PilotNet 모델 정의**
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
            nn.Linear(10, 5)  # 5개의 범주로 분류
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# **3. 데이터 로더 설정**
# CSV 파일 경로 (경로 수정)
csv_path = r"C:\Users\USER\OneDrive\바탕 화면\lastdance\augmented_frames\training_data.csv"  # 경로 수정

# 데이터 로드
df = pd.read_csv(csv_path)
df['steering_category'] = df['angle'].apply(
    lambda angle: [30, 60, 90, 120, 150].index(min([30, 60, 90, 120, 150], key=lambda x: abs(x - angle)))
)

train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.3333, random_state=42)

train_data.to_csv(r"C:\Users\USER\OneDrive\바탕 화면\lastdance\augmented_frames\train.csv", index=False)
val_data.to_csv(r"C:\Users\USER\OneDrive\바탕 화면\lastdance\augmented_frames\val.csv", index=False)
test_data.to_csv(r"C:\Users\USER\OneDrive\바탕 화면\lastdance\augmented_frames\test.csv", index=False)

train_dataset = SteeringDataset(r"C:\Users\USER\OneDrive\바탕 화면\lastdance\augmented_frames\train.csv")  # 경로 수정
val_dataset = SteeringDataset(r"C:\Users\USER\OneDrive\바탕 화면\lastdance\augmented_frames\val.csv")  # 경로 수정
test_dataset = SteeringDataset(r"C:\Users\USER\OneDrive\바탕 화면\lastdance\augmented_frames\test.csv")  # 경로 수정

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# **4. 학습 설정**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used for training: {device}")
model = PilotNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# **모델 저장 관련 설정**
best_val_loss = float('inf')
best_model_path = r"C:\Users\USER\OneDrive\바탕 화면\lastdance\augmented_frames\best_pilotnet_model.pth"

# **5. 학습 및 검증**
epochs = 30
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_losses[-1]:.4f}")
    print(f"Validation Loss: {val_losses[-1]:.4f}")

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    scheduler.step()

print("학습 완료. Best Model이 저장되었습니다.")

# **6. 테스트 손실 계산**
print("테스트 데이터 평가 시작...")
model.eval()
test_loss = 0
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

test_loss /= len(test_loader)
test_accuracy = correct_predictions / total_predictions * 100

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

# **7. 학습 결과 시각화**
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label=f'Test Loss: {test_loss:.4f}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training, Validation, and Test Loss')
plt.legend()
plt.grid()
plt.show()

# **8. 저장된 모델 로드**
print("저장된 모델 로드 및 테스트...")
model.load_state_dict(torch.load(best_model_path))
model.eval()
