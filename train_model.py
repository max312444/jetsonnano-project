import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from driving_dataset import DrivingDataset
from model import DrivingModel

# 파일 경로 설정
image_dir = r"C:\Users\USER\change_image"  # 이미지가 저장된 폴더 경로
csv_file = r"C:\Users\USER\jetson_csv\jetson_data.csv"  # CSV 파일 경로

# 데이터셋 생성
dataset = DrivingDataset(image_dir, csv_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델, 손실 함수 및 최적화 설정
model = DrivingModel()
criterion = torch.nn.MSELoss()  # 회귀 문제에서 MSELoss 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        
        # 모델 출력
        outputs = model(images.float())
        
        # 손실 계산
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
