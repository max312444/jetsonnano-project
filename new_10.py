import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# **1. CSV 파일 경로**
csv_path = r"C:\Users\USER\dataset_0117\lastdance_csv.csv"

# **2. 데이터 로드 및 경로 변경 설정**
df = pd.read_csv(csv_path)
print("CSV 파일 열 이름:", df.columns.tolist())

# 'steering_angle' 열 이름 확인 및 수정
if 'steering_angle' in df.columns:
    angle_column = 'steering_angle'
elif 'angle' in df.columns:
    angle_column = 'angle'
else:
    raise KeyError("CSV 파일에 'steering_angle' 또는 'angle' 열이 없습니다. 열 이름을 확인하세요.")

# 기존 경로를 새로운 경로로 변경 (파일명에 "cropped_" 추가)
def preprocess_and_check(path):
    """전처리 및 누락된 파일 처리"""
    new_base_path = r"C:\Users\USER\mydataset"
    original_path = path  # 이미 절대 경로로 되어 있음
    new_path = os.path.join(new_base_path, f"cropped_{os.path.basename(path)}")

    # 이미지 전처리 수행
    if not os.path.exists(new_path):
        image = cv2.imread(original_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {original_path}")
            return None

        # ROI 설정 (상단 30% 제거)
        height = image.shape[0]
        roi = image[int(height * 0.3):, :]

        # 크기 조정 (200x66)
        resized = cv2.resize(roi, (200, 66))

        # 전처리된 이미지 저장
        cv2.imwrite(new_path, resized)
    return new_path

# 각 경로를 전처리 및 경로 갱신
df['frame_path'] = df['frame_path'].apply(preprocess_and_check)

# **누락된 데이터 확인 및 제거**
df = df.dropna(subset=['frame_path'])  # 누락된 이미지를 제거

# **3. SteeringDataset 정의**
class SteeringDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['frame_path']
        angle = row[angle_column]

        # 이미지 로드 및 전처리
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")

        image = cv2.resize(image, (200, 66))
        image = image / 255.0  # 정규화
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)  # HWC -> CHW

        return image, torch.tensor(angle, dtype=torch.float32)

# **4. 데이터 분할 및 데이터 로더 설정**
if len(df) == 0:
    raise ValueError("유효한 데이터가 없습니다. CSV 파일을 확인하세요.")

train_data, val_test_data = train_test_split(df, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

train_dataset = SteeringDataset(train_data)
val_dataset = SteeringDataset(val_data)
test_dataset = SteeringDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
            nn.Linear(10, 1)  # 회귀 문제로 변경
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# **6. 학습 및 검증 루프 설정**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PilotNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_val_loss = float('inf')
best_model_path = r"C:\Users\USER\best_pilotnet_model.pth"

# 학습
epochs = 30
for epoch in range(epochs):
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
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    scheduler.step()

print("Training completed.")
